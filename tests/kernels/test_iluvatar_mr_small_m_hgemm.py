# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Opt-in device tests for Iluvatar MR small-M HGEMM (Path A v1).

Set ``FLYDSL_ILUVATAR_RUN_MR_SMALL_M=1`` to run (needs an Iluvatar device).
"""

import os
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.iluvatar_lower]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DTYPE_CASES = (
    ("float16", "Float16"),
    ("bfloat16", "BFloat16"),
)
_M_VALUES = (1, 8, 15, 16)


def _require_enabled() -> None:
    if os.environ.get("FLYDSL_ILUVATAR_RUN_MR_SMALL_M", "").lower() not in {"1", "true", "yes", "on"}:
        pytest.skip("set FLYDSL_ILUVATAR_RUN_MR_SMALL_M=1 to run Iluvatar MR small-M HGEMM tests")


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        pytest.skip(f"torch is required for Iluvatar MR small-M tests: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA-compatible Iluvatar device is not available")
    return torch


def _configure_iluvatar_env(monkeypatch) -> None:
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


def _require_kernel():
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    try:
        import flydsl.expr as fx
        from kernels.gemm.iluvatar.mr.small_m_hgemm import compile_iluvatar_mr_small_m_hgemm
    except ModuleNotFoundError as exc:
        pytest.fail(f"failed to import small_m_hgemm: {exc}")
    return fx, compile_iluvatar_mr_small_m_hgemm


@pytest.mark.parametrize("torch_dtype_name,fx_dtype_name", _DTYPE_CASES)
@pytest.mark.parametrize("m", _M_VALUES)
def test_iluvatar_mr_small_m_hgemm_path_a(m, torch_dtype_name, fx_dtype_name, monkeypatch):
    _require_enabled()
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    fx, compile_fn = _require_kernel()

    # Auto-pick for N<1024 prefers 16x64x64 (1x2 warps, k_atoms=4).
    n, k = 128, 256
    torch_dtype = getattr(torch, torch_dtype_name)
    elem_dtype = getattr(fx, fx_dtype_name)

    torch.manual_seed(0)
    A = torch.randn(m, k, dtype=torch_dtype, device="cuda")
    B = torch.randn(n, k, dtype=torch_dtype, device="cuda")
    C = torch.empty(m, n, dtype=torch_dtype, device="cuda")

    launch = compile_fn(m=m, N=n, K=k, elem_dtype=elem_dtype)
    stream = torch.cuda.Stream()
    launch(A, B, C, stream=stream)
    torch.cuda.synchronize()

    expected = (A.float() @ B.float().T).to(torch_dtype)
    diff = (C.float() - expected.float()).abs()
    atol = 2e-2 if torch_dtype_name == "float16" else 5e-2
    ok = torch.allclose(C.float(), expected.float(), atol=atol, rtol=2e-2)
    finite_ok = torch.isfinite(C).all().item()
    print(
        f"[small_m] m={m} dtype={torch_dtype_name} N={n} K={k} "
        f"ok={ok} finite={finite_ok} max_abs={diff.max().item():.3e}"
    )
    assert ok and finite_ok


def test_iluvatar_mr_small_m_rejects_bad_shapes(monkeypatch):
    _configure_iluvatar_env(monkeypatch)
    fx, compile_fn = _require_kernel()
    with pytest.raises(ValueError):
        compile_fn(m=0, N=64, K=64, elem_dtype=fx.Float16)
    with pytest.raises(ValueError):
        compile_fn(m=17, N=64, K=64, elem_dtype=fx.Float16)
    with pytest.raises(ValueError):
        compile_fn(m=8, N=65, K=64, elem_dtype=fx.Float16)
    with pytest.raises(ValueError):
        compile_fn(m=8, N=64, K=48, elem_dtype=fx.Float16)
