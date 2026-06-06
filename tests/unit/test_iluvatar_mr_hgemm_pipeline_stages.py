# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Opt-in full-pipeline smoke tests for Iluvatar MR HGEMM.

The isolated stage tests live in dedicated files:

G2S belongs to ``tests/unit/test_iluvatar_mr_async_cp_device.py``.
S2R belongs to ``tests/unit/test_iluvatar_mr_s2r_device.py``.
MMA belongs to ``tests/unit/test_iluvatar_mr_mma_device.py``.
Epilogue belongs to ``tests/unit/test_iluvatar_mr_epilogue_device.py``.

This file exercises the production ``compile_iluvatar_mr_hgemm`` launch wrapper across:

* ``major_pattern`` (nn / tn / nt / tt)
* ``k_rep`` (BK = 16 * k_rep, i.e. 32 and 64)
* ``epilogue_store`` (shfl / tiled for ``no_c_read``)
* single-CTA (256 x 256 x 64, grid 1 x 1) and multi-CTA (512 x 512 x 128, grid 2 x 2)

Set ``FLYDSL_ILUVATAR_RUN_MR_HGEMM_STAGES=1`` to run (needs an Iluvatar device).
"""

import os
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l2_device]

_REPO_ROOT = Path(__file__).resolve().parents[2]

_PATTERNS = ("nt", "nn", "tn", "tt")
_K_REP_VALUES = (2, 4)
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
)

_SINGLE_CTA_SHAPE = (STAGED_BRICK_M, STAGED_BRICK_N, 64)
_MULTI_CTA_SHAPE = (STAGED_BRICK_M * 2, STAGED_BRICK_N * 2, 128)
_LARGE_SHAPE = (4096, 4096, 4096)


def _require_enabled() -> None:
    if os.environ.get("FLYDSL_ILUVATAR_RUN_MR_HGEMM_STAGES", "").lower() not in {"1", "true", "yes", "on"}:
        pytest.skip("set FLYDSL_ILUVATAR_RUN_MR_HGEMM_STAGES=1 to run Iluvatar MR HGEMM staged tests")


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
        from kernels.iluvatar_mr_hgemm import (
            EPILOGUE_NO_C_READ,
            EPILOGUE_STORE_SHFL,
            EPILOGUE_STORE_TILED,
            WARP_SIZE,
            compile_iluvatar_mr_hgemm,
        )
    except ModuleNotFoundError as exc:
        pytest.fail(f"failed to import kernels.iluvatar_mr_hgemm: {exc}")
    return {
        "EPILOGUE_NO_C_READ": EPILOGUE_NO_C_READ,
        "EPILOGUE_STORE_SHFL": EPILOGUE_STORE_SHFL,
        "EPILOGUE_STORE_TILED": EPILOGUE_STORE_TILED,
        "WARP_SIZE": WARP_SIZE,
        "compile_iluvatar_mr_hgemm": compile_iluvatar_mr_hgemm,
    }


def _make_c_tensor(torch, m: int, n: int, epilogue: str, hgemm, *, seed: int):
    if epilogue == hgemm["EPILOGUE_NO_C_READ"]:
        return torch.zeros(m, n, dtype=torch.float16, device="cuda")
    torch.manual_seed(seed)
    return torch.randn(m, n, dtype=torch.float32, device="cuda")


def _expected_result(torch, A, B, C_in, epilogue: str, hgemm):
    expected = A.to(torch.float32) @ B.to(torch.float32).T
    if epilogue != hgemm["EPILOGUE_NO_C_READ"]:
        expected = expected + C_in.to(torch.float32)
    return expected


def _remap_hgemm_tensors_for_pattern(A, B, major_pattern: str):
    """Test-side physical layout adapter for logical A(m,k), B(n,k) inputs."""
    if major_pattern == "nn":
        return A, B.t().contiguous()
    if major_pattern == "tn":
        return A.t().contiguous(), B.t().contiguous()
    if major_pattern == "nt":
        return A, B
    if major_pattern == "tt":
        return A.t().contiguous(), B
    raise ValueError(f"unknown major pattern: {major_pattern}")


def _compare_atol(k: int, k_rep: int) -> float:
    bk = 16 * k_rep
    return 2e-2 * max(1.0, (k / bk) ** 0.5)


def _cta_grid(m: int, n: int, k_rep: int, *, warp_size: int) -> tuple[tuple[int, int, int], tuple[int, int, int], int]:
    warp_m = 16 * STAGED_WARP_ATOMS_M
    warp_n = 16 * STAGED_WARP_ATOMS_N
    bm = warp_m * STAGED_WARPS_M
    bn = warp_n * STAGED_WARPS_N
    bk = 16 * k_rep
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
    k_rep: int,
    seed: int = 0,
) -> bool:
    m, n, k = shape
    torch.manual_seed(seed)
    A = torch.randn(m, k, dtype=torch.float16, device="cuda")
    B = torch.randn(n, k, dtype=torch.float16, device="cuda")
    C = _make_c_tensor(torch, m, n, epilogue, hgemm, seed=seed + 1)
    C_in = C.clone()

    launcher = hgemm["compile_iluvatar_mr_hgemm"](
        M=m,
        N=n,
        K=k,
        warps_m=STAGED_WARPS_M,
        warps_n=STAGED_WARPS_N,
        k_rep=k_rep,
        warp_atoms_m=STAGED_WARP_ATOMS_M,
        warp_atoms_n=STAGED_WARP_ATOMS_N,
        epilogue=epilogue,
        epilogue_store=epilogue_store,
        major_pattern=major_pattern,
    )
    a_dev, b_dev = _remap_hgemm_tensors_for_pattern(A, B, major_pattern)
    stream = torch.cuda.Stream()
    launcher(a_dev, b_dev, C, stream=stream)
    torch.cuda.synchronize()

    expected = _expected_result(torch, A, B, C_in, epilogue, hgemm)
    got = C.to(torch.float32) if epilogue == hgemm["EPILOGUE_NO_C_READ"] else C
    diff = (got - expected).abs()
    atol = _compare_atol(k, k_rep)
    ok = torch.allclose(got, expected, atol=atol, rtol=2e-2)
    finite_ok = torch.isfinite(got).all().item()
    grid, block, smem = _cta_grid(m, n, k_rep, warp_size=hgemm["WARP_SIZE"])
    cta_note = (
        f" cta={STAGED_WARPS_M}x{STAGED_WARPS_N}warps"
        f" atoms={STAGED_WARP_ATOMS_M}x{STAGED_WARP_ATOMS_N}"
        f" threads={block[0]}"
    )
    store_note = f" store={epilogue_store}" if epilogue == hgemm["EPILOGUE_NO_C_READ"] else ""
    print(
        f"[check] epilogue={epilogue}{store_note} pattern={major_pattern} k_rep={k_rep} "
        f"M={m} N={n} K={k}{cta_note} grid={grid} block={block} smem={smem} "
        f"ok={ok} finite={finite_ok} max_abs={diff.max().item():.3e} "
        f"mean_abs={diff.mean().item():.3e} atol={atol:.2e}"
    )
    if not ok:
        print(f"  C[0,0:4]      = {got[0, 0:4].tolist()}")
        print(f"  expect[0,0:4] = {expected[0, 0:4].tolist()}")
    return bool(ok and finite_ok)


@pytest.mark.parametrize("k_rep", _K_REP_VALUES)
@pytest.mark.parametrize("major_pattern", _PATTERNS)
@pytest.mark.parametrize("epilogue,epilogue_store", _EPILOGUE_CASES)
def test_iluvatar_mr_hgemm_single_cta_pipeline(major_pattern, epilogue, epilogue_store, k_rep, monkeypatch):
    _require_enabled()
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
        k_rep=k_rep,
    )


@pytest.mark.parametrize("k_rep", _K_REP_VALUES)
@pytest.mark.parametrize("major_pattern", _PATTERNS)
@pytest.mark.parametrize("epilogue_store", ("tiled", "shfl"))
def test_iluvatar_mr_hgemm_multi_cta_pipeline(major_pattern, epilogue_store, k_rep, monkeypatch):
    _require_enabled()
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
        k_rep=k_rep,
    )


@pytest.mark.large_shape
@pytest.mark.parametrize("k_rep", _K_REP_VALUES)
@pytest.mark.parametrize("major_pattern", _PATTERNS)
@pytest.mark.parametrize("epilogue_store", ("shfl",))
def test_iluvatar_mr_hgemm_large_multi_cta_pipeline(major_pattern, epilogue_store, k_rep, monkeypatch):
    _require_enabled()
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
        k_rep=k_rep,
    )
