# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Device correctness tests for Iluvatar CQ (ivcore30) pipeline HGEMM.

Exercises ``compile_iluvatar_cq_hgemm`` / ``select_swizzle_cta`` on CQ hardware.
Bring-up scope matches the kernel: ``major_pattern=\"tn\"`` only; fp16 / bf16.

**Skip policy:** CI Iluvatar runners are MR (ivcore11). These cases skip when
CUDA is missing or ``cuda:0`` is not a CQ device (see
``tests/iluvatar_cq_device.py``).
"""

import sys
from pathlib import Path

import pytest

from tests.iluvatar_cq_device import require_iluvatar_cq_torch

pytestmark = [pytest.mark.l2_device, pytest.mark.iluvatar_lower]

_REPO_ROOT = Path(__file__).resolve().parents[2]

_DTYPE_CASES = (
    ("float16", "Float16"),
    ("bfloat16", "BFloat16"),
)
_EPILOGUE_CASES = (
    ("no_c_read", "tiled"),
    ("no_c_read", "shfl"),
    ("read_c_accum", "tiled"),
)

# Preset name -> (m, n, k) that fits that CTA tile.
_PRESET_SHAPES = (
    ("256", (64, 64, 64)),
    ("1024", (256, 256, 64)),
    ("2048", (512, 256, 64)),
)
_MULTI_CTA_SHAPE = (512, 512, 128)


def _configure_cq_env(monkeypatch) -> None:
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", "ivcore30")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


def _require_hgemm():
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    try:
        import flydsl.expr as fx
        from kernels.gemm.iluvatar.common import remap_gemm_tensors
        from kernels.gemm.iluvatar.cq.hgemm import (
            EPILOGUE_NO_C_READ,
            SWIZZLE_CTA_PRESETS,
            compile_iluvatar_cq_hgemm,
            select_swizzle_cta,
        )
    except ModuleNotFoundError as exc:
        pytest.fail(f"failed to import kernels.gemm.iluvatar.cq.hgemm: {exc}")
    return {
        "EPILOGUE_NO_C_READ": EPILOGUE_NO_C_READ,
        "SWIZZLE_CTA_PRESETS": SWIZZLE_CTA_PRESETS,
        "compile_iluvatar_cq_hgemm": compile_iluvatar_cq_hgemm,
        "select_swizzle_cta": select_swizzle_cta,
        "remap_gemm_tensors": remap_gemm_tensors,
        "fx": fx,
    }


def _compare_atol(k: int, k_atoms: int, torch_dtype_name: str) -> float:
    bk = 16 * k_atoms
    scale = 2.0 if torch_dtype_name == "bfloat16" else 1.0
    return 2e-2 * scale * max(1.0, (k / bk) ** 0.5)


def _make_c_tensor(torch, m, n, epilogue, hgemm, torch_dtype, *, seed: int):
    if epilogue == hgemm["EPILOGUE_NO_C_READ"]:
        return torch.zeros(m, n, dtype=torch_dtype, device="cuda")
    torch.manual_seed(seed)
    return torch.randn(m, n, dtype=torch.float32, device="cuda")


def _check_hgemm(
    torch,
    hgemm,
    *,
    shape: tuple[int, int, int],
    preset_name: str | None,
    epilogue: str,
    epilogue_store: str,
    torch_dtype_name: str,
    fx_dtype_name: str,
    seed: int = 0,
) -> bool:
    m, n, k = shape
    if preset_name is None:
        preset = hgemm["select_swizzle_cta"](m, n, k)
    else:
        preset = hgemm["SWIZZLE_CTA_PRESETS"][preset_name]

    torch_dtype = getattr(torch, torch_dtype_name)
    elem_dtype = getattr(hgemm["fx"], fx_dtype_name)
    torch.manual_seed(seed)
    A = torch.randn(m, k, dtype=torch_dtype, device="cuda")
    B = torch.randn(n, k, dtype=torch_dtype, device="cuda")
    C = _make_c_tensor(torch, m, n, epilogue, hgemm, torch_dtype, seed=seed + 1)
    C_in = C.clone()

    launcher = hgemm["compile_iluvatar_cq_hgemm"](
        M=m,
        N=n,
        K=k,
        warps_m=preset.warps_m,
        warps_n=preset.warps_n,
        k_atoms=preset.default_k_atoms,
        warp_atoms_m=preset.warp_atoms_m,
        warp_atoms_n=preset.warp_atoms_n,
        epilogue=epilogue,
        epilogue_store=epilogue_store,
        major_pattern="tn",
        elem_dtype=elem_dtype,
    )
    a_dev, b_dev = hgemm["remap_gemm_tensors"](A, B, "tn")
    stream = torch.cuda.Stream()
    launcher(a_dev, b_dev, C, stream=stream)
    torch.cuda.synchronize()

    expected = A.to(torch.float32) @ B.to(torch.float32).T
    if epilogue != hgemm["EPILOGUE_NO_C_READ"]:
        expected = expected + C_in.to(torch.float32)
    got = C.to(torch.float32) if epilogue == hgemm["EPILOGUE_NO_C_READ"] else C
    atol = _compare_atol(k, preset.default_k_atoms, torch_dtype_name)
    ok = bool(torch.allclose(got, expected, atol=atol, rtol=2e-2))
    finite_ok = bool(torch.isfinite(got).all().item())
    diff = (got - expected).abs()
    store_note = f" store={epilogue_store}" if epilogue == hgemm["EPILOGUE_NO_C_READ"] else ""
    print(
        f"[check] cta={preset.name} dtype={torch_dtype_name} epilogue={epilogue}{store_note} "
        f"M={m} N={n} K={k} ok={ok} finite={finite_ok} "
        f"max_abs={diff.max().item():.3e} atol={atol:.2e}"
    )
    return ok and finite_ok


@pytest.mark.parametrize("torch_dtype_name,fx_dtype_name", _DTYPE_CASES)
@pytest.mark.parametrize("epilogue,epilogue_store", _EPILOGUE_CASES)
@pytest.mark.parametrize("preset_name,shape", _PRESET_SHAPES)
def test_iluvatar_cq_hgemm_preset_pipeline(
    preset_name, shape, epilogue, epilogue_store, torch_dtype_name, fx_dtype_name, monkeypatch
):
    torch = require_iluvatar_cq_torch()
    # read_c_accum on the asymmetric 8x4 (2048-thread) CTA is not validated yet
    # (no_c_read paths are covered). Keep MR-style epilogue coverage on 256/1024.
    if preset_name == "2048" and epilogue == "read_c_accum":
        pytest.skip("read_c_accum not yet validated on CQ 2048-thread (8x4) CTA")
    _configure_cq_env(monkeypatch)
    hgemm = _require_hgemm()
    assert _check_hgemm(
        torch,
        hgemm,
        shape=shape,
        preset_name=preset_name,
        epilogue=epilogue,
        epilogue_store=epilogue_store,
        torch_dtype_name=torch_dtype_name,
        fx_dtype_name=fx_dtype_name,
    )


@pytest.mark.parametrize("torch_dtype_name,fx_dtype_name", _DTYPE_CASES)
@pytest.mark.parametrize("epilogue_store", ("tiled", "shfl"))
def test_iluvatar_cq_hgemm_multi_cta_pipeline(
    epilogue_store, torch_dtype_name, fx_dtype_name, monkeypatch
):
    torch = require_iluvatar_cq_torch()
    _configure_cq_env(monkeypatch)
    hgemm = _require_hgemm()
    assert _check_hgemm(
        torch,
        hgemm,
        shape=_MULTI_CTA_SHAPE,
        preset_name="1024",
        epilogue="no_c_read",
        epilogue_store=epilogue_store,
        torch_dtype_name=torch_dtype_name,
        fx_dtype_name=fx_dtype_name,
    )


@pytest.mark.parametrize(
    "shape,expected_cta",
    [
        ((64, 64, 64), "256"),
        ((1024, 1024, 64), "1024"),
        ((2048, 2048, 64), "2048"),
    ],
)
def test_iluvatar_cq_hgemm_auto_select_pipeline(shape, expected_cta, monkeypatch):
    torch = require_iluvatar_cq_torch()
    _configure_cq_env(monkeypatch)
    hgemm = _require_hgemm()
    preset = hgemm["select_swizzle_cta"](*shape)
    assert preset.name == expected_cta
    assert _check_hgemm(
        torch,
        hgemm,
        shape=shape,
        preset_name=None,
        epilogue="no_c_read",
        epilogue_store="shfl",
        torch_dtype_name="float16",
        fx_dtype_name="Float16",
    )
