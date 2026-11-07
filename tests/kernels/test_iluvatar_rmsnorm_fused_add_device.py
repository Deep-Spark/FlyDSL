# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar RMSNorm V4a fused-add device tests."""

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

from kernels.norm.iluvatar import (  # noqa: E402
    compile_iluvatar_rmsnorm,
    compile_iluvatar_rmsnorm_fused_add,
)

RTOL = 1e-5
ATOL = 1e-6
EPS = 1e-5


def _configure_iluvatar_env(monkeypatch) -> None:
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


def _reference_fused_add_rmsnorm(x, residual_in, gamma, eps):
    residual_out = x + residual_in
    rrms = torch.rsqrt((residual_out * residual_out).mean(dim=1, keepdim=True) + eps)
    return residual_out * rrms * gamma, residual_out


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
def test_iluvatar_rmsnorm_fused_add_forward(monkeypatch, M, N):
    _configure_iluvatar_env(monkeypatch)
    torch.manual_seed(42)

    launch = compile_iluvatar_rmsnorm_fused_add(N=N, eps=EPS)
    x = torch.randn((M, N), device="cuda", dtype=torch.float32).contiguous()
    residual_in = torch.randn((M, N), device="cuda", dtype=torch.float32).contiguous()
    gamma = torch.randn((N,), device="cuda", dtype=torch.float32).contiguous()
    out = torch.empty((M, N), device="cuda", dtype=torch.float32).contiguous()
    residual_out = torch.empty_like(x)

    ret_out, ret_residual = launch(x, residual_in, gamma, out, residual_out, M)
    assert ret_out is out and ret_residual is residual_out

    if M == 0:
        assert out.numel() == 0 and residual_out.numel() == 0
        return

    expected_out, expected_residual = _reference_fused_add_rmsnorm(x, residual_in, gamma, EPS)
    torch.testing.assert_close(residual_out, expected_residual, rtol=0.0, atol=0.0)
    torch.testing.assert_close(out, expected_out, rtol=RTOL, atol=ATOL)


def test_iluvatar_rmsnorm_fused_add_zero_residual_matches_v1(monkeypatch):
    _configure_iluvatar_env(monkeypatch)
    torch.manual_seed(43)
    M, N = 7, 511

    launch_v1 = compile_iluvatar_rmsnorm(N=N, eps=EPS)
    launch_fused = compile_iluvatar_rmsnorm_fused_add(N=N, eps=EPS)
    x = torch.randn((M, N), device="cuda", dtype=torch.float32).contiguous()
    residual_in = torch.zeros_like(x)
    gamma = torch.randn((N,), device="cuda", dtype=torch.float32).contiguous()
    out_v1 = torch.empty_like(x)
    out_fused = torch.empty_like(x)
    residual_out = torch.empty_like(x)

    launch_v1(x, gamma, out_v1, M)
    launch_fused(x, residual_in, gamma, out_fused, residual_out, M)

    assert torch.equal(residual_out, x)
    assert torch.equal(out_fused, out_v1)


def test_iluvatar_rmsnorm_fused_add_zero_added(monkeypatch):
    _configure_iluvatar_env(monkeypatch)
    torch.manual_seed(44)
    M, N = 4, 255

    launch = compile_iluvatar_rmsnorm_fused_add(N=N, eps=EPS)
    x = torch.randn((M, N), device="cuda", dtype=torch.float32).contiguous()
    residual_in = -x
    gamma = torch.randn((N,), device="cuda", dtype=torch.float32).contiguous()
    out = torch.empty_like(x)
    residual_out = torch.empty_like(x)

    launch(x, residual_in, gamma, out, residual_out, M)

    assert torch.count_nonzero(residual_out).item() == 0
    assert torch.count_nonzero(out).item() == 0


def test_iluvatar_rmsnorm_fused_add_m0_noop(monkeypatch):
    _configure_iluvatar_env(monkeypatch)
    N = 256
    launch = compile_iluvatar_rmsnorm_fused_add(N=N, eps=EPS)
    x = torch.empty((0, N), device="cuda", dtype=torch.float32)
    residual_in = torch.empty_like(x)
    gamma = torch.randn((N,), device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    residual_out = torch.empty_like(x)

    ret_out, ret_residual = launch(x, residual_in, gamma, out, residual_out, 0)

    assert ret_out is out and ret_residual is residual_out
    assert out.numel() == 0 and residual_out.numel() == 0


def test_iluvatar_rmsnorm_fused_add_compile_time_guards():
    with pytest.raises(ValueError, match="N must be > 0"):
        compile_iluvatar_rmsnorm_fused_add(N=0, eps=EPS)
    with pytest.raises(ValueError, match="eps must be > 0"):
        compile_iluvatar_rmsnorm_fused_add(N=128, eps=0.0)
    with pytest.raises(ValueError, match="dtype must be 'f32'"):
        compile_iluvatar_rmsnorm_fused_add(N=128, eps=EPS, dtype="bf16")


def test_iluvatar_rmsnorm_fused_add_runtime_guards(monkeypatch):
    _configure_iluvatar_env(monkeypatch)
    M, N = 4, 16
    launch = compile_iluvatar_rmsnorm_fused_add(N=N, eps=EPS)
    x = torch.randn((M, N), device="cuda", dtype=torch.float32).contiguous()
    residual_in = torch.randn_like(x)
    gamma = torch.randn((N,), device="cuda", dtype=torch.float32).contiguous()
    out = torch.empty_like(x)
    residual_out = torch.empty_like(x)

    with pytest.raises(ValueError, match="M must be int"):
        launch(x, residual_in, gamma, out, residual_out, float(M))
    with pytest.raises(ValueError, match="M must be >= 0"):
        launch(x, residual_in, gamma, out, residual_out, -1)
    with pytest.raises(ValueError, match=r"expected x shape \(M,N\)="):
        launch(x, residual_in, gamma, out, residual_out, M + 1)

    bad_residual_shape = torch.empty((M, N + 1), device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match=r"expected residual_in shape \(M,N\)="):
        launch(x, bad_residual_shape, gamma, out, residual_out, M)
    bad_gamma_shape = torch.empty((N + 1,), device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match=r"expected gamma shape \(N,\)="):
        launch(x, residual_in, bad_gamma_shape, out, residual_out, M)

    x_f16 = x.to(torch.float16)
    with pytest.raises(ValueError, match=r"x dtype must be torch\.float32"):
        launch(x_f16, residual_in, gamma, out, residual_out, M)
    residual_f16 = residual_in.to(torch.float16)
    with pytest.raises(ValueError, match=r"residual_in dtype must be torch\.float32"):
        launch(x, residual_f16, gamma, out, residual_out, M)
    out_f16 = out.to(torch.float16)
    with pytest.raises(ValueError, match=r"out dtype must be torch\.float32"):
        launch(x, residual_in, gamma, out_f16, residual_out, M)

    x_nc = torch.randn((N, M), device="cuda", dtype=torch.float32).t()
    assert tuple(x_nc.shape) == (M, N) and not x_nc.is_contiguous()
    with pytest.raises(ValueError, match="x must be contiguous"):
        launch(x_nc, residual_in, gamma, out, residual_out, M)
    residual_nc = torch.randn((N, M), device="cuda", dtype=torch.float32).t()
    assert tuple(residual_nc.shape) == (M, N) and not residual_nc.is_contiguous()
    with pytest.raises(ValueError, match="residual_in must be contiguous"):
        launch(x, residual_nc, gamma, out, residual_out, M)

    residual_cpu = torch.empty((M, N), dtype=torch.float32)
    with pytest.raises(ValueError, match="must be on same device"):
        launch(x, residual_cpu, gamma, out, residual_out, M)

    overlap_cases = [
        (x, residual_out, "out must not overlap with x"),
        (residual_in, residual_out, "out must not overlap with residual_in"),
        (out, x, "residual_out must not overlap with x"),
        (out, residual_in, "residual_out must not overlap with residual_in"),
        (out, out, "out must not overlap with residual_out"),
    ]
    for out_arg, residual_out_arg, message in overlap_cases:
        with pytest.raises(ValueError, match=message):
            launch(x, residual_in, gamma, out_arg, residual_out_arg, M)

    gamma_storage = torch.empty((M * N,), device="cuda", dtype=torch.float32)
    gamma_alias = gamma_storage[:N]
    output_alias = gamma_storage.view(M, N)
    with pytest.raises(ValueError, match="out must not overlap with gamma"):
        launch(x, residual_in, gamma_alias, output_alias, residual_out, M)
    with pytest.raises(ValueError, match="residual_out must not overlap with gamma"):
        launch(x, residual_in, gamma_alias, out, output_alias, M)
