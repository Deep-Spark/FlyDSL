# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar RMSNorm backward device tests."""

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

from kernels.norm.iluvatar.rmsnorm_bwd_kernel import compile_iluvatar_rmsnorm_bwd  # noqa: E402
from kernels.norm.iluvatar.rmsnorm_kernel import compile_iluvatar_rmsnorm  # noqa: E402

EPS = 1e-5
TORCH_DTYPES = {
    "f32": torch.float32,
    "bf16": torch.bfloat16,
    "f16": torch.float16,
}


def _configure_iluvatar_env(monkeypatch) -> None:
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


def _reference_rmsnorm_bwd(x, gamma, dy):
    x_f32 = x.float()
    gamma_f32 = gamma.float()
    dy_f32 = dy.float()
    rstd = torch.rsqrt((x_f32 * x_f32).mean(dim=1) + EPS)
    x_hat = x_f32 * rstd[:, None]
    wdy = dy_f32 * gamma_f32
    c1 = (x_hat * wdy).mean(dim=1, keepdim=True)
    dx = (wdy - x_hat * c1) * rstd[:, None]
    dweight = (dy_f32 * x_hat).sum(dim=0)
    out = x_hat * gamma_f32
    return out, rstd, dx, dweight


@pytest.mark.parametrize("dtype", ["f32", "bf16", "f16"])
@pytest.mark.parametrize(
    "M,N",
    [
        (1, 1),
        (3, 255),
        (4, 256),
        (7, 511),
        (64, 1024),
        (0, 256),
    ],
)
def test_iluvatar_rmsnorm_backward(monkeypatch, dtype, M, N):
    _configure_iluvatar_env(monkeypatch)
    torch.manual_seed(42)

    torch_dtype = TORCH_DTYPES[dtype]
    launch_fwd = compile_iluvatar_rmsnorm(N=N, eps=EPS, dtype=dtype, store_rstd=True)
    launch_bwd = compile_iluvatar_rmsnorm_bwd(N=N, dtype=dtype)
    x = torch.randn((M, N), device="cuda", dtype=torch_dtype).contiguous()
    gamma = torch.randn((N,), device="cuda", dtype=torch_dtype).contiguous()
    dy = torch.randn((M, N), device="cuda", dtype=torch_dtype).contiguous()
    out = torch.empty_like(x)
    rstd = torch.empty((M,), device="cuda", dtype=torch.float32)
    dx = torch.empty_like(x)
    dweight = torch.zeros((N,), device="cuda", dtype=torch.float32)

    ret_out, ret_rstd = launch_fwd(x, gamma, out, rstd, M)
    assert ret_out is out and ret_rstd is rstd
    ret_dx, ret_dweight = launch_bwd(x, gamma, dy, rstd, dx, dweight, M)
    assert ret_dx is dx and ret_dweight is dweight
    torch.cuda.synchronize()

    if M == 0:
        assert dx.numel() == 0
        torch.testing.assert_close(dweight, torch.zeros_like(dweight), rtol=0, atol=0)
        return

    expected_out, expected_rstd, expected_dx, expected_dweight = _reference_rmsnorm_bwd(x, gamma, dy)
    expected_out = expected_out.to(torch_dtype)
    expected_dx = expected_dx.to(torch_dtype)
    if dtype == "f32":
        out_rtol, out_atol = 2e-5, 2e-6
        dx_rtol, dx_atol = 2e-5, 2e-6
    elif dtype == "f16":
        out_rtol, out_atol = 3e-3, 3e-3
        dx_rtol, dx_atol = 3e-3, 3e-3
    else:
        out_rtol, out_atol = 1e-2, 1e-2
        dx_rtol, dx_atol = 1e-2, 1e-2
    torch.testing.assert_close(out, expected_out, rtol=out_rtol, atol=out_atol)
    torch.testing.assert_close(rstd, expected_rstd, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(dx, expected_dx, rtol=dx_rtol, atol=dx_atol)
    torch.testing.assert_close(dweight, expected_dweight, rtol=2e-4, atol=2e-4)


def test_iluvatar_rmsnorm_backward_fp16_alias():
    launch = compile_iluvatar_rmsnorm_bwd(N=64, dtype="fp16")
    assert callable(launch)


def test_iluvatar_rmsnorm_backward_compile_time_guards():
    with pytest.raises(ValueError, match="N must be > 0"):
        compile_iluvatar_rmsnorm_bwd(N=0, dtype="f32")
    with pytest.raises(ValueError, match="dtype must be one of"):
        compile_iluvatar_rmsnorm_bwd(N=128, dtype="i8")


def test_iluvatar_rmsnorm_backward_runtime_guards(monkeypatch):
    _configure_iluvatar_env(monkeypatch)
    M, N = 2, 16
    launch = compile_iluvatar_rmsnorm_bwd(N=N, dtype="bf16")
    x = torch.randn((M, N), device="cuda", dtype=torch.bfloat16)
    gamma = torch.randn((N,), device="cuda", dtype=torch.bfloat16)
    dy = torch.randn((M, N), device="cuda", dtype=torch.bfloat16)
    rstd = torch.ones((M,), device="cuda", dtype=torch.float32)
    dx = torch.empty_like(x)
    dweight = torch.zeros((N,), device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="expected x shape"):
        launch(x, gamma, dy, rstd, dx, dweight, M + 1)
    with pytest.raises(ValueError, match=r"dy dtype must be torch\.bfloat16"):
        launch(x, gamma, dy.float(), rstd, dx, dweight, M)
    with pytest.raises(ValueError, match="dweight dtype must be torch.float32"):
        launch(x, gamma, dy, rstd, dx, dweight.to(torch.bfloat16), M)

    x_nc = torch.randn((N, M), device="cuda", dtype=torch.bfloat16).t()
    assert tuple(x_nc.shape) == (M, N) and not x_nc.is_contiguous()
    with pytest.raises(ValueError, match="x must be contiguous"):
        launch(x_nc, gamma, dy, rstd, dx, dweight, M)

    with pytest.raises(ValueError, match="dx must not overlap with x"):
        launch(x, gamma, dy, rstd, x, dweight, M)
