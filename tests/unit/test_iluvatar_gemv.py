# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Opt-in Iluvatar GEMV V1 correctness tests."""

import os
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l2_device]

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _require_enabled() -> None:
    if os.environ.get("FLYDSL_ILUVATAR_RUN_GEMV", "").lower() not in {"1", "true", "yes", "on"}:
        pytest.skip("set FLYDSL_ILUVATAR_RUN_GEMV=1 to run Iluvatar GEMV tests")


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        pytest.skip(f"torch is required for Iluvatar GEMV tests: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA-compatible Iluvatar device is not available")
    return torch


def _configure_iluvatar_env(monkeypatch) -> None:
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


def _require_kernel_import():
    try:
        from kernels.gemm.iluvatar.gemv import compile_iluvatar_gemv
    except ModuleNotFoundError as exc:
        pytest.skip(f"FlyDSL GEMV kernel module is not importable: {exc}")
    return compile_iluvatar_gemv


@pytest.mark.parametrize("dtype_name,rtol,atol", [("float16", 1e-2, 1e-2), ("bfloat16", 2e-2, 2e-2)])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("input_is_2d", [False, True])
def test_iluvatar_gemv_matches_flinear(monkeypatch, dtype_name, rtol, atol, with_bias, input_is_2d):
    _require_enabled()
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    compile_iluvatar_gemv = _require_kernel_import()

    N = 64
    K = 32
    dtype = getattr(torch, dtype_name)

    x = torch.randn((1, K) if input_is_2d else (K,), device="cuda", dtype=dtype)
    w = torch.randn((N, K), device="cuda", dtype=dtype)
    bias = torch.randn((N,), device="cuda", dtype=dtype) if with_bias else None

    launch = compile_iluvatar_gemv(N=N, K=K)
    out = launch(x, w, bias=bias)
    ref = torch.nn.functional.linear(x, w, bias)
    torch.cuda.synchronize()

    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
    expected_shape = (1, N) if input_is_2d else (N,)
    assert tuple(out.shape) == expected_shape


def test_iluvatar_gemv_rejects_invalid_input_shape(monkeypatch):
    _require_enabled()
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    compile_iluvatar_gemv = _require_kernel_import()

    launch = compile_iluvatar_gemv(N=64, K=32)
    x_bad = torch.randn((2, 32), device="cuda", dtype=torch.float16)
    w = torch.randn((64, 32), device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError, match="x shape mismatch"):
        launch(x_bad, w, bias=None)


def test_iluvatar_gemv_rejects_invalid_dtype(monkeypatch):
    _require_enabled()
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    compile_iluvatar_gemv = _require_kernel_import()

    launch = compile_iluvatar_gemv(N=64, K=32)
    x = torch.randn((32,), device="cuda", dtype=torch.float32)
    w = torch.randn((64, 32), device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="dtype must be fp16/bf16"):
        launch(x, w, bias=None)


def test_iluvatar_gemv_rejects_non_divisible_shape():
    compile_iluvatar_gemv = _require_kernel_import()
    with pytest.raises(ValueError, match="divisible"):
        compile_iluvatar_gemv(N=65, K=32)

    with pytest.raises(ValueError, match="divisible"):
        compile_iluvatar_gemv(N=64, K=30)
