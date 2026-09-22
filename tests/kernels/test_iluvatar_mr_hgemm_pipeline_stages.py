# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Full-pipeline smoke tests for Iluvatar MR HGEMM.

The isolated stage tests live in dedicated files:

G2S belongs to ``tests/kernels/test_iluvatar_mr_async_cp_device.py``.
S2R belongs to ``tests/kernels/test_iluvatar_mr_s2r_device.py``.
MMA belongs to ``tests/kernels/test_iluvatar_mr_mma_device.py``.
Epilogue belongs to ``tests/kernels/test_iluvatar_mr_epilogue_device.py``.

This file exercises the production ``compile_iluvatar_mr_hgemm`` launch wrapper across:

* ``elem_dtype`` / torch dtype (float16 / bfloat16)
* ``major_pattern`` (nn / tn / nt / tt)
* ``k_atoms`` (BK = 16 * k_atoms, i.e. 32 and 64)
* ``epilogue_store`` (shfl / tiled for ``no_c_read``)
* ``write_c_fp32`` (fp32 write-only store, optional ``n_first``)
* ``crop_m`` on ``write_c_fp32`` and batched ``no_c_read`` with ``ldc``
* single-CTA (256 x 256 x 64, grid 1 x 1) and multi-CTA (512 x 512 x 128, grid 2 x 2)

"""

import os
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.iluvatar_lower]

_REPO_ROOT = Path(__file__).resolve().parents[2]

_PATTERNS = ("nt", "nn", "tn", "tt")
_K_ATOMS_VALUES = (2, 4)
_DTYPE_CASES = (
    ("float16", "Float16"),
    ("bfloat16", "BFloat16"),
)
_EPILOGUE_CASES = (
    ("no_c_read", "tiled"),
    ("no_c_read", "shfl"),
    ("read_c_accum", "tiled"),
)

from tests.unit.iluvatar_mr_hgemm_test_common import (  # noqa: E402
    STAGED_BRICK_M,
    STAGED_BRICK_N,
    STAGED_WARP_ATOMS_M,
    STAGED_WARP_ATOMS_N,
    STAGED_WARPS_M,
    STAGED_WARPS_N,
    remap_gemm_tensors,
)

_SINGLE_CTA_SHAPE = (STAGED_BRICK_M, STAGED_BRICK_N, 64)
_MULTI_CTA_SHAPE = (STAGED_BRICK_M * 2, STAGED_BRICK_N * 2, 128)
_LARGE_SHAPE = (1024, 1024, 1024)
# crop_m requires M == bm. vpr=32 so BK=32 yields one A brick; one warp in M.
_CROP_M = 8
_CROP_BM = 16
_CROP_N = 128
_CROP_K = 64
_CROP_WARPS_M = 1
_CROP_WARPS_N = 1
_CROP_WARP_ATOMS_M = 1
_CROP_WARP_ATOMS_N = 8
_CROP_BATCH = 2
_CROP_LDC = 160


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        pytest.skip(f"torch is required for Iluvatar MR HGEMM staged tests: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA-compatible Iluvatar device is not available")
    return torch


def _configure_iluvatar_env(monkeypatch) -> None:
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


def _require_hgemm_kernel():
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    try:
        import flydsl.expr as fx
        from kernels.gemm.iluvatar.mr.hgemm import (
            EPILOGUE_NO_C_READ,
            EPILOGUE_READ_C_ACCUM,
            EPILOGUE_STORE_SHFL,
            EPILOGUE_STORE_TILED,
            EPILOGUE_WRITE_C_FP32,
            WARP_SIZE,
            compile_iluvatar_mr_hgemm,
        )
    except ModuleNotFoundError as exc:
        pytest.fail(f"failed to import kernels.gemm.iluvatar.mr.hgemm: {exc}")
    return {
        "EPILOGUE_NO_C_READ": EPILOGUE_NO_C_READ,
        "EPILOGUE_READ_C_ACCUM": EPILOGUE_READ_C_ACCUM,
        "EPILOGUE_STORE_SHFL": EPILOGUE_STORE_SHFL,
        "EPILOGUE_STORE_TILED": EPILOGUE_STORE_TILED,
        "EPILOGUE_WRITE_C_FP32": EPILOGUE_WRITE_C_FP32,
        "WARP_SIZE": WARP_SIZE,
        "compile_iluvatar_mr_hgemm": compile_iluvatar_mr_hgemm,
        "fx": fx,
    }


def _torch_dtype(torch, torch_dtype_name: str):
    return getattr(torch, torch_dtype_name)


def _fx_elem_dtype(fx, fx_dtype_name: str):
    return getattr(fx, fx_dtype_name)


def _make_c_tensor(torch, m: int, n: int, epilogue: str, hgemm, torch_dtype, *, seed: int):
    if epilogue == hgemm["EPILOGUE_NO_C_READ"]:
        return torch.zeros(m, n, dtype=torch_dtype, device="cuda")
    if epilogue == hgemm["EPILOGUE_WRITE_C_FP32"]:
        return torch.empty(m, n, dtype=torch.float32, device="cuda")
    torch.manual_seed(seed)
    return torch.randn(m, n, dtype=torch.float32, device="cuda")


def _expected_result(torch, A, B, C_in, epilogue: str, hgemm):
    expected = A.to(torch.float32) @ B.to(torch.float32).T
    if epilogue == hgemm["EPILOGUE_READ_C_ACCUM"]:
        expected = expected + C_in.to(torch.float32)
    return expected


def _compare_atol(k: int, k_atoms: int, torch_dtype_name: str) -> float:
    bk = 16 * k_atoms
    # bf16 has a shorter mantissa; scale the f16 baseline by ~2x (matches mma_pipeline_device).
    scale = 2.0 if torch_dtype_name == "bfloat16" else 1.0
    return 2e-2 * scale * max(1.0, (k / bk) ** 0.5)


def _cta_grid(
    m: int, n: int, k_atoms: int, *, warp_size: int
) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
    warp_m = 16 * STAGED_WARP_ATOMS_M
    warp_n = 16 * STAGED_WARP_ATOMS_N
    bm = warp_m * STAGED_WARPS_M
    bn = warp_n * STAGED_WARPS_N
    bk = 16 * k_atoms
    threads = STAGED_WARPS_M * STAGED_WARPS_N * warp_size
    grid = (m // bm, n // bn, 1)
    block = (threads, 1, 1)
    smem = (bm + bn) * bk * 2 * 2
    return grid, block, smem


def _check_hgemm_pipeline(
    torch,
    hgemm,
    *,
    shape: tuple[int, int, int],
    major_pattern: str,
    epilogue: str,
    epilogue_store: str,
    k_atoms: int,
    torch_dtype_name: str,
    fx_dtype_name: str,
    seed: int = 0,
    n_first: bool = False,
) -> bool:
    m, n, k = shape
    torch_dtype = _torch_dtype(torch, torch_dtype_name)
    elem_dtype = _fx_elem_dtype(hgemm["fx"], fx_dtype_name)
    torch.manual_seed(seed)
    A = torch.randn(m, k, dtype=torch_dtype, device="cuda")
    B = torch.randn(n, k, dtype=torch_dtype, device="cuda")
    C = _make_c_tensor(torch, m, n, epilogue, hgemm, torch_dtype, seed=seed + 1)
    C_in = C.clone()

    launcher = hgemm["compile_iluvatar_mr_hgemm"](
        M=m,
        N=n,
        K=k,
        warps_m=STAGED_WARPS_M,
        warps_n=STAGED_WARPS_N,
        k_atoms=k_atoms,
        warp_atoms_m=STAGED_WARP_ATOMS_M,
        warp_atoms_n=STAGED_WARP_ATOMS_N,
        epilogue=epilogue,
        epilogue_store=epilogue_store,
        major_pattern=major_pattern,
        elem_dtype=elem_dtype,
        n_first=n_first,
    )
    a_dev, b_dev = remap_gemm_tensors(A, B, major_pattern)
    stream = torch.cuda.Stream()
    launcher(a_dev, b_dev, C, stream=stream)
    torch.cuda.synchronize()

    expected = _expected_result(torch, A, B, C_in, epilogue, hgemm)
    got = C if epilogue != hgemm["EPILOGUE_NO_C_READ"] else C.to(torch.float32)
    diff = (got - expected).abs()
    atol = _compare_atol(k, k_atoms, torch_dtype_name)
    ok = torch.allclose(got, expected, atol=atol, rtol=2e-2)
    finite_ok = torch.isfinite(got).all().item()
    grid, block, smem = _cta_grid(m, n, k_atoms, warp_size=hgemm["WARP_SIZE"])
    cta_note = (
        f" cta={STAGED_WARPS_M}x{STAGED_WARPS_N}warps"
        f" atoms={STAGED_WARP_ATOMS_M}x{STAGED_WARP_ATOMS_N}"
        f" threads={block[0]}"
    )
    store_note = f" store={epilogue_store}" if epilogue == hgemm["EPILOGUE_NO_C_READ"] else ""
    print(
        f"[check] dtype={torch_dtype_name} epilogue={epilogue}{store_note} pattern={major_pattern} "
        f"k_atoms={k_atoms} M={m} N={n} K={k}{cta_note} grid={grid} block={block} smem={smem} "
        f"ok={ok} finite={finite_ok} max_abs={diff.max().item():.3e} "
        f"mean_abs={diff.mean().item():.3e} atol={atol:.2e}"
    )
    if not ok:
        print(f"  C[0,0:4]      = {got[0, 0:4].tolist()}")
        print(f"  expect[0,0:4] = {expected[0, 0:4].tolist()}")
    return bool(ok and finite_ok)


@pytest.mark.parametrize("torch_dtype_name,fx_dtype_name", _DTYPE_CASES)
@pytest.mark.parametrize("k_atoms", _K_ATOMS_VALUES)
@pytest.mark.parametrize("major_pattern", _PATTERNS)
@pytest.mark.parametrize("epilogue,epilogue_store", _EPILOGUE_CASES)
def test_iluvatar_mr_hgemm_single_cta_pipeline(
    major_pattern, epilogue, epilogue_store, k_atoms, torch_dtype_name, fx_dtype_name, monkeypatch
):
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    hgemm = _require_hgemm_kernel()

    assert _check_hgemm_pipeline(
        torch,
        hgemm,
        shape=_SINGLE_CTA_SHAPE,
        major_pattern=major_pattern,
        epilogue=epilogue,
        epilogue_store=epilogue_store,
        k_atoms=k_atoms,
        torch_dtype_name=torch_dtype_name,
        fx_dtype_name=fx_dtype_name,
    )


@pytest.mark.parametrize("torch_dtype_name,fx_dtype_name", _DTYPE_CASES)
@pytest.mark.parametrize("k_atoms", _K_ATOMS_VALUES)
@pytest.mark.parametrize("major_pattern", _PATTERNS)
@pytest.mark.parametrize("epilogue_store", ("tiled", "shfl"))
def test_iluvatar_mr_hgemm_multi_cta_pipeline(
    major_pattern, epilogue_store, k_atoms, torch_dtype_name, fx_dtype_name, monkeypatch
):
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    hgemm = _require_hgemm_kernel()

    assert _check_hgemm_pipeline(
        torch,
        hgemm,
        shape=_MULTI_CTA_SHAPE,
        major_pattern=major_pattern,
        epilogue="no_c_read",
        epilogue_store=epilogue_store,
        k_atoms=k_atoms,
        torch_dtype_name=torch_dtype_name,
        fx_dtype_name=fx_dtype_name,
    )


@pytest.mark.large_shape
@pytest.mark.parametrize("torch_dtype_name,fx_dtype_name", _DTYPE_CASES)
@pytest.mark.parametrize("k_atoms", _K_ATOMS_VALUES)
@pytest.mark.parametrize("major_pattern", _PATTERNS)
@pytest.mark.parametrize("epilogue_store", ("shfl",))
def test_iluvatar_mr_hgemm_large_multi_cta_pipeline(
    major_pattern, epilogue_store, k_atoms, torch_dtype_name, fx_dtype_name, monkeypatch
):
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    hgemm = _require_hgemm_kernel()

    assert _check_hgemm_pipeline(
        torch,
        hgemm,
        shape=_LARGE_SHAPE,
        major_pattern=major_pattern,
        epilogue="no_c_read",
        epilogue_store=epilogue_store,
        k_atoms=k_atoms,
        torch_dtype_name=torch_dtype_name,
        fx_dtype_name=fx_dtype_name,
    )


@pytest.mark.parametrize("n_first", (False, True))
def test_iluvatar_mr_hgemm_write_c_fp32_single_cta(n_first, monkeypatch):
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    hgemm = _require_hgemm_kernel()

    assert _check_hgemm_pipeline(
        torch,
        hgemm,
        shape=_SINGLE_CTA_SHAPE,
        major_pattern="tn",
        epilogue=hgemm["EPILOGUE_WRITE_C_FP32"],
        epilogue_store="tiled",
        k_atoms=2,
        torch_dtype_name="bfloat16",
        fx_dtype_name="BFloat16",
        n_first=n_first,
    )


def test_iluvatar_mr_hgemm_write_c_fp32_crop_m(monkeypatch):
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    hgemm = _require_hgemm_kernel()
    fx = hgemm["fx"]
    torch.manual_seed(0)
    a = torch.randn(_CROP_BM, _CROP_K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(_CROP_N, _CROP_K, dtype=torch.bfloat16, device="cuda")
    c = torch.empty(_CROP_M, _CROP_N, dtype=torch.float32, device="cuda")
    expected = (a.float() @ b.float().T)[:_CROP_M]

    launch = hgemm["compile_iluvatar_mr_hgemm"](
        M=_CROP_BM,
        N=_CROP_N,
        K=_CROP_K,
        warps_m=_CROP_WARPS_M,
        warps_n=_CROP_WARPS_N,
        k_atoms=2,
        warp_atoms_m=_CROP_WARP_ATOMS_M,
        warp_atoms_n=_CROP_WARP_ATOMS_N,
        epilogue=hgemm["EPILOGUE_WRITE_C_FP32"],
        elem_dtype=fx.BFloat16,
        crop_m=_CROP_M,
    )
    launch(a, b, c)
    torch.cuda.synchronize()
    atol = _compare_atol(_CROP_K, 2, "bfloat16")
    assert torch.isfinite(c).all()
    assert torch.allclose(c, expected, atol=atol, rtol=2e-2)


def test_iluvatar_mr_hgemm_no_c_read_crop_batched_ldc(monkeypatch):
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    hgemm = _require_hgemm_kernel()
    fx = hgemm["fx"]
    torch.manual_seed(1)
    a = torch.randn(_CROP_BATCH, _CROP_BM, _CROP_K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(_CROP_BATCH, _CROP_N, _CROP_K, dtype=torch.bfloat16, device="cuda")
    c = torch.zeros(_CROP_BATCH, _CROP_M, _CROP_LDC, dtype=torch.bfloat16, device="cuda")

    launch = hgemm["compile_iluvatar_mr_hgemm"](
        M=_CROP_BM,
        N=_CROP_N,
        K=_CROP_K,
        warps_m=_CROP_WARPS_M,
        warps_n=_CROP_WARPS_N,
        k_atoms=2,
        warp_atoms_m=_CROP_WARP_ATOMS_M,
        warp_atoms_n=_CROP_WARP_ATOMS_N,
        epilogue=hgemm["EPILOGUE_NO_C_READ"],
        elem_dtype=fx.BFloat16,
        crop_m=_CROP_M,
        ldc=_CROP_LDC,
        batch=_CROP_BATCH,
        a_batch_stride=_CROP_BM * _CROP_K,
        b_batch_stride=_CROP_N * _CROP_K,
        c_batch_stride=_CROP_M * _CROP_LDC,
    )
    launch(
        a.reshape(_CROP_BATCH * _CROP_BM, _CROP_K),
        b.reshape(_CROP_BATCH * _CROP_N, _CROP_K),
        c.reshape(_CROP_BATCH * _CROP_M, _CROP_LDC),
    )
    torch.cuda.synchronize()
    atol = _compare_atol(_CROP_K, 2, "bfloat16")
    for batch_id in range(_CROP_BATCH):
        expected = (a[batch_id].float() @ b[batch_id].float().T)[:_CROP_M]
        got = c[batch_id, :, :_CROP_N].float()
        assert torch.isfinite(got).all()
        assert torch.allclose(got, expected, atol=atol, rtol=2e-2)
        assert c[batch_id, :, _CROP_N:].abs().max().item() == 0.0
