# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar RMSNorm device tests for the ixdnn FP16/BF16 algorithm."""

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

from kernels.norm.iluvatar.rmsnorm_kernel_b16 import compile_iluvatar_rmsnorm_b16  # noqa: E402

EPS = 1e-5
TORCH_DTYPES = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
}


def _configure_iluvatar_env(monkeypatch) -> None:
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


def _reference_ixdnn_rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float):
    x_f32 = x.float()
    gamma_f32 = gamma.float()
    rrms = torch.rsqrt((x_f32 * x_f32).mean(dim=1) + eps)

    if x.shape[1] & 1:
        # Scalar branch in ixdnn converts x and gamma to FP32 before multiply.
        out_f32 = x_f32 * gamma_f32 * rrms[:, None]
    else:
        # Packed branch executes __hmul2 first, then applies FP32 rsigma.
        product_b16 = (x_f32 * gamma_f32).to(x.dtype)
        out_f32 = product_b16.float() * rrms[:, None]
    return out_f32.to(x.dtype), rrms


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize(
    "M,N",
    [
        (1, 1),  # odd scalar branch, partial wave
        (1, 2),  # packed branch, partial wave
        (3, 63),
        (3, 64),
        (7, 255),
        (7, 256),
        (64, 1024),
        (0, 256),
    ],
)
def test_iluvatar_rmsnorm_b16_forward(monkeypatch, dtype, M, N):
    _configure_iluvatar_env(monkeypatch)
    torch.manual_seed(42)

    torch_dtype = TORCH_DTYPES[dtype]
    launch = compile_iluvatar_rmsnorm_b16(N=N, eps=EPS, dtype=dtype)
    x = torch.randn((M, N), device="cuda", dtype=torch_dtype).contiguous()
    gamma = torch.randn((N,), device="cuda", dtype=torch_dtype).contiguous()
    out = torch.empty_like(x)
    rsigma = torch.empty((M,), device="cuda", dtype=torch.float32).contiguous()

    ret_out, ret_rsigma = launch(x, gamma, out, rsigma, M)
    assert ret_out is out and ret_rsigma is rsigma
    torch.cuda.synchronize()

    if M == 0:
        assert out.numel() == 0 and rsigma.numel() == 0
        return

    expected, expected_rsigma = _reference_ixdnn_rmsnorm(x, gamma, EPS)
    out_rtol, out_atol = ((1e-2, 1e-2) if dtype == "f16" else (2e-2, 2e-2))
    torch.testing.assert_close(out, expected, rtol=out_rtol, atol=out_atol)
    torch.testing.assert_close(rsigma, expected_rsigma, rtol=2e-3, atol=1e-5)


def test_iluvatar_rmsnorm_b16_fp16_alias(monkeypatch):
    """The public ``fp16`` spelling aliases the project's canonical ``f16``."""

    _configure_iluvatar_env(monkeypatch)
    M, N = 2, 64
    launch = compile_iluvatar_rmsnorm_b16(N=N, eps=EPS, dtype="fp16")
    x = torch.randn((M, N), device="cuda", dtype=torch.float16).contiguous()
    gamma = torch.randn((N,), device="cuda", dtype=torch.float16).contiguous()
    out = torch.empty_like(x)
    rsigma = torch.empty((M,), device="cuda", dtype=torch.float32)

    ret_out, ret_rsigma = launch(x, gamma, out, rsigma, M)
    assert ret_out is out and ret_rsigma is rsigma
    torch.cuda.synchronize()


def test_iluvatar_rmsnorm_b16_compile_time_guards():
    with pytest.raises(ValueError, match="N must be > 0"):
        compile_iluvatar_rmsnorm_b16(N=0, eps=EPS, dtype="f16")
    with pytest.raises(ValueError, match="eps must be > 0"):
        compile_iluvatar_rmsnorm_b16(N=128, eps=0.0, dtype="f16")
    with pytest.raises(ValueError, match="dtype must be one of"):
        compile_iluvatar_rmsnorm_b16(N=128, eps=EPS, dtype="f32")


def test_iluvatar_rmsnorm_b16_runtime_guards(monkeypatch):
    _configure_iluvatar_env(monkeypatch)

    M, N = 4, 16
    launch = compile_iluvatar_rmsnorm_b16(N=N, eps=EPS, dtype="bf16")
    x = torch.randn((M, N), device="cuda", dtype=torch.bfloat16).contiguous()
    gamma = torch.randn((N,), device="cuda", dtype=torch.bfloat16).contiguous()
    out = torch.empty_like(x)
    rsigma = torch.empty((M,), device="cuda", dtype=torch.float32).contiguous()

    with pytest.raises(ValueError, match=r"expected x shape \(M,N\)="):
        launch(x, gamma, out, rsigma, M + 1)

    with pytest.raises(ValueError, match=r"x dtype must be torch\.bfloat16"):
        launch(x.float(), gamma, out, rsigma, M)

    with pytest.raises(ValueError, match=r"rsigma dtype must be torch\.float32"):
        launch(x, gamma, out, rsigma.to(torch.float16), M)

    x_nc = torch.randn((N, M), device="cuda", dtype=torch.bfloat16).t()
    assert tuple(x_nc.shape) == (M, N) and not x_nc.is_contiguous()
    with pytest.raises(ValueError, match="x must be contiguous"):
        launch(x_nc, gamma, out, rsigma, M)

    with pytest.raises(ValueError, match="out must not overlap with x"):
        launch(x, gamma, x, rsigma, M)

    gamma_as_out = gamma.view(1, N)
    launch_m1 = compile_iluvatar_rmsnorm_b16(N=N, eps=EPS, dtype="bf16")
    with pytest.raises(ValueError, match="out must not overlap with gamma"):
        launch_m1(x[:1].contiguous(), gamma, gamma_as_out, rsigma[:1].contiguous(), 1)
