# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar RMSNorm V1 device tests (fp32 forward only)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.iluvatar_lower]

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is None or not torch.cuda.is_available():
    pytest.skip("CUDA-compatible Iluvatar device is not available", allow_module_level=True)

from kernels.norm.iluvatar.rmsnorm_kernel import compile_iluvatar_rmsnorm  # noqa: E402

RTOL = 1e-5
ATOL = 1e-6
EPS = 1e-5


def _configure_iluvatar_env(monkeypatch) -> None:
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


def _reference_rmsnorm_fp32(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    mean_sq = (x * x).mean(dim=1, keepdim=True)
    rrms = torch.rsqrt(mean_sq + eps)
    return x * rrms * gamma


@pytest.mark.parametrize(
    "M,N",
    [
        (1, 1),
        (1, 255),
        (1, 256),
        (7, 511),
        (64, 1024),
        (0, 256),
    ],
)
def test_iluvatar_rmsnorm_forward_v1_fp32(monkeypatch, M, N):
    _configure_iluvatar_env(monkeypatch)
    torch.manual_seed(42)

    launch = compile_iluvatar_rmsnorm(N=N, eps=EPS, dtype="f32")
    x = torch.randn((M, N), device="cuda", dtype=torch.float32).contiguous()
    gamma = torch.randn((N,), device="cuda", dtype=torch.float32).contiguous()
    out = torch.empty((M, N), device="cuda", dtype=torch.float32).contiguous()

    ret = launch(x, gamma, out, M)
    assert ret is out
    torch.cuda.synchronize()

    if M == 0:
        assert out.numel() == 0
        return

    expected = _reference_rmsnorm_fp32(x, gamma, EPS)
    torch.testing.assert_close(out, expected, rtol=RTOL, atol=ATOL)


def test_iluvatar_rmsnorm_m0_noop(monkeypatch):
    _configure_iluvatar_env(monkeypatch)

    launch = compile_iluvatar_rmsnorm(N=256, eps=EPS, dtype="f32")
    x = torch.empty((0, 256), device="cuda", dtype=torch.float32).contiguous()
    gamma = torch.randn((256,), device="cuda", dtype=torch.float32).contiguous()
    out = torch.empty((0, 256), device="cuda", dtype=torch.float32).contiguous()

    ret = launch(x, gamma, out, 0)
    assert ret is out
    assert out.numel() == 0


def test_iluvatar_rmsnorm_compile_time_guards():
    with pytest.raises(ValueError, match="N must be > 0"):
        compile_iluvatar_rmsnorm(N=0, eps=EPS, dtype="f32")
    with pytest.raises(ValueError, match="eps must be > 0"):
        compile_iluvatar_rmsnorm(N=128, eps=0.0, dtype="f32")
    with pytest.raises(ValueError, match="dtype must be 'f32'"):
        compile_iluvatar_rmsnorm(N=128, eps=EPS, dtype="bf16")


def test_iluvatar_rmsnorm_runtime_guards(monkeypatch):
    _configure_iluvatar_env(monkeypatch)

    M, N = 4, 16
    launch = compile_iluvatar_rmsnorm(N=N, eps=EPS, dtype="f32")
    x = torch.randn((M, N), device="cuda", dtype=torch.float32).contiguous()
    gamma = torch.randn((N,), device="cuda", dtype=torch.float32).contiguous()
    out = torch.empty((M, N), device="cuda", dtype=torch.float32).contiguous()

    with pytest.raises(ValueError, match="expected x shape \\(M,N\\)="):
        launch(x, gamma, out, M + 1)

    with pytest.raises(ValueError, match="out must not overlap with x"):
        launch(x, gamma, x, M)

    x_nc = torch.randn((N, M), device="cuda", dtype=torch.float32).t()
    assert tuple(x_nc.shape) == (M, N) and not x_nc.is_contiguous()
    with pytest.raises(ValueError, match="x must be contiguous"):
        launch(x_nc, gamma, out, M)

    gamma_nc = torch.randn((N * 2,), device="cuda", dtype=torch.float32)[::2]
    assert tuple(gamma_nc.shape) == (N,) and not gamma_nc.is_contiguous()
    with pytest.raises(ValueError, match="gamma must be contiguous"):
        launch(x, gamma_nc, out, M)

    out_nc = torch.empty((N, M), device="cuda", dtype=torch.float32).t()
    assert tuple(out_nc.shape) == (M, N) and not out_nc.is_contiguous()
    with pytest.raises(ValueError, match="out must be contiguous"):
        launch(x, gamma, out_nc, M)
