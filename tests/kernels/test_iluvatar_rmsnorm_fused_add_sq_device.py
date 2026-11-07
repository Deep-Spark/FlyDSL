# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar RMSNorm V4b fused-add SmoothQuant device tests."""

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
    compile_iluvatar_rmsnorm_fused_add_dynamicquant,
    compile_iluvatar_rmsnorm_fused_add_smoothquant,
    compile_iluvatar_rmsnorm_smoothquant,
)
from kernels.norm.iluvatar.rmsnorm_kernel import QUANT_I8_MAX  # noqa: E402

EPS = 1e-5
OUTPUT_ATOL_LSB = 1
SCALE_RTOL = 1e-3
SCALE_ATOL = 1e-6


def _configure_iluvatar_env(monkeypatch) -> None:
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


def _reference_fused_add_sq(x, residual_in, gamma, x_scale, eps):
    residual_out = x + residual_in
    mean_sq = (residual_out * residual_out).mean(dim=1, keepdim=True)
    rrms = torch.rsqrt(mean_sq + eps)
    y = residual_out * rrms * gamma * x_scale
    amax = y.abs().max(dim=1, keepdim=True).values
    scale = amax / QUANT_I8_MAX
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.trunc(y / scale).to(torch.int8)
    return q, scale.squeeze(1), residual_out


def _run_fused_sq(monkeypatch, x, residual_in, gamma, x_scale, M, N):
    _configure_iluvatar_env(monkeypatch)
    launch = compile_iluvatar_rmsnorm_fused_add_smoothquant(N=N, eps=EPS)
    out = torch.empty((M, N), device="cuda", dtype=torch.int8).contiguous()
    y_scale = torch.empty((M,), device="cuda", dtype=torch.float32).contiguous()
    residual_out = torch.empty((M, N), device="cuda", dtype=torch.float32).contiguous()
    ret = launch(x, residual_in, gamma, x_scale, out, y_scale, residual_out, M)
    torch.cuda.synchronize()
    assert ret == (out, y_scale, residual_out)
    return out, y_scale, residual_out


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
def test_iluvatar_rmsnorm_fused_add_sq_forward(monkeypatch, M, N):
    torch.manual_seed(42)
    x = torch.randn((M, N), device="cuda", dtype=torch.float32).contiguous()
    residual_in = torch.randn((M, N), device="cuda", dtype=torch.float32).contiguous()
    gamma = torch.randn((N,), device="cuda", dtype=torch.float32).contiguous()
    x_scale = (torch.rand((N,), device="cuda", dtype=torch.float32) + 0.5).contiguous()

    out, y_scale, residual_out = _run_fused_sq(monkeypatch, x, residual_in, gamma, x_scale, M, N)

    if M == 0:
        assert out.numel() == 0 and y_scale.numel() == 0 and residual_out.numel() == 0
        return

    q_ref, scale_ref, residual_ref = _reference_fused_add_sq(x, residual_in, gamma, x_scale, EPS)
    torch.testing.assert_close(residual_out, residual_ref, rtol=0.0, atol=0.0)
    diff = (out.to(torch.int32) - q_ref.to(torch.int32)).abs().max().item()
    assert diff <= OUTPUT_ATOL_LSB, f"i8 out max abs diff = {diff} (> {OUTPUT_ATOL_LSB})"
    torch.testing.assert_close(y_scale, scale_ref, rtol=SCALE_RTOL, atol=SCALE_ATOL)


def test_iluvatar_rmsnorm_fused_add_sq_zero_residual_matches_v3(monkeypatch):
    M, N = 7, 511
    torch.manual_seed(43)
    x = torch.randn((M, N), device="cuda", dtype=torch.float32).contiguous()
    residual_in = torch.zeros_like(x)
    gamma = torch.randn((N,), device="cuda", dtype=torch.float32).contiguous()
    x_scale = (torch.rand((N,), device="cuda", dtype=torch.float32) + 0.5).contiguous()

    out_fused, y_scale_fused, residual_out = _run_fused_sq(monkeypatch, x, residual_in, gamma, x_scale, M, N)

    _configure_iluvatar_env(monkeypatch)
    launch_v3 = compile_iluvatar_rmsnorm_smoothquant(N=N, eps=EPS)
    out_v3 = torch.empty((M, N), device="cuda", dtype=torch.int8).contiguous()
    y_scale_v3 = torch.empty((M,), device="cuda", dtype=torch.float32).contiguous()
    launch_v3(x, gamma, x_scale, out_v3, y_scale_v3, M)
    torch.cuda.synchronize()

    assert torch.equal(residual_out, x)
    diff = (out_fused.to(torch.int32) - out_v3.to(torch.int32)).abs().max().item()
    assert diff <= OUTPUT_ATOL_LSB
    torch.testing.assert_close(y_scale_fused, y_scale_v3, rtol=SCALE_RTOL, atol=SCALE_ATOL)


def test_iluvatar_rmsnorm_fused_add_sq_x_scale_ones_matches_dq(monkeypatch):
    M, N = 8, 256
    torch.manual_seed(7)
    x = torch.randn((M, N), device="cuda", dtype=torch.float32).contiguous()
    residual_in = torch.randn_like(x)
    gamma = torch.randn((N,), device="cuda", dtype=torch.float32).contiguous()
    x_scale = torch.ones((N,), device="cuda", dtype=torch.float32).contiguous()

    out_sq, y_scale_sq, residual_sq = _run_fused_sq(monkeypatch, x, residual_in, gamma, x_scale, M, N)

    _configure_iluvatar_env(monkeypatch)
    launch_dq = compile_iluvatar_rmsnorm_fused_add_dynamicquant(N=N, eps=EPS)
    out_dq = torch.empty((M, N), device="cuda", dtype=torch.int8).contiguous()
    y_scale_dq = torch.empty((M,), device="cuda", dtype=torch.float32).contiguous()
    residual_dq = torch.empty_like(x)
    launch_dq(x, residual_in, gamma, out_dq, y_scale_dq, residual_dq, M)
    torch.cuda.synchronize()

    assert torch.equal(out_sq, out_dq)
    assert torch.equal(residual_sq, residual_dq)
    torch.testing.assert_close(y_scale_sq, y_scale_dq, rtol=0.0, atol=0.0)


def test_iluvatar_rmsnorm_fused_add_sq_x_scale_negative_and_zero(monkeypatch):
    M, N = 4, 128
    torch.manual_seed(11)
    x = torch.randn((M, N), device="cuda", dtype=torch.float32).contiguous()
    residual_in = torch.randn_like(x)
    gamma = torch.randn((N,), device="cuda", dtype=torch.float32).contiguous()
    x_scale = torch.randn((N,), device="cuda", dtype=torch.float32).contiguous()
    x_scale[0] = 0.0
    x_scale[1] = -1.5

    out, y_scale, residual_out = _run_fused_sq(monkeypatch, x, residual_in, gamma, x_scale, M, N)
    q_ref, scale_ref, residual_ref = _reference_fused_add_sq(x, residual_in, gamma, x_scale, EPS)
    torch.testing.assert_close(residual_out, residual_ref, rtol=0.0, atol=0.0)
    diff = (out.to(torch.int32) - q_ref.to(torch.int32)).abs().max().item()
    assert diff <= OUTPUT_ATOL_LSB
    torch.testing.assert_close(y_scale, scale_ref, rtol=SCALE_RTOL, atol=SCALE_ATOL)


def test_iluvatar_rmsnorm_fused_add_sq_zero_added(monkeypatch):
    M, N = 4, 255
    torch.manual_seed(44)
    x = torch.randn((M, N), device="cuda", dtype=torch.float32).contiguous()
    residual_in = -x
    gamma = torch.randn((N,), device="cuda", dtype=torch.float32).contiguous()
    x_scale = (torch.rand((N,), device="cuda", dtype=torch.float32) + 0.5).contiguous()

    out, y_scale, residual_out = _run_fused_sq(monkeypatch, x, residual_in, gamma, x_scale, M, N)

    assert torch.count_nonzero(residual_out).item() == 0
    assert torch.all(out == 0)
    torch.testing.assert_close(
        y_scale,
        torch.ones((M,), device="cuda", dtype=torch.float32),
        rtol=SCALE_RTOL,
        atol=SCALE_ATOL,
    )


def test_iluvatar_rmsnorm_fused_add_sq_m0_noop(monkeypatch):
    _configure_iluvatar_env(monkeypatch)
    N = 256
    launch = compile_iluvatar_rmsnorm_fused_add_smoothquant(N=N, eps=EPS)
    x = torch.empty((0, N), device="cuda", dtype=torch.float32)
    residual_in = torch.empty_like(x)
    gamma = torch.randn((N,), device="cuda", dtype=torch.float32)
    x_scale = torch.randn((N,), device="cuda", dtype=torch.float32)
    out = torch.empty((0, N), device="cuda", dtype=torch.int8)
    y_scale = torch.empty((0,), device="cuda", dtype=torch.float32)
    residual_out = torch.empty_like(x)

    ret = launch(x, residual_in, gamma, x_scale, out, y_scale, residual_out, 0)
    assert ret == (out, y_scale, residual_out)
    assert out.numel() == 0 and y_scale.numel() == 0 and residual_out.numel() == 0


def test_iluvatar_rmsnorm_fused_add_sq_compile_time_guards():
    with pytest.raises(ValueError, match="N must be > 0"):
        compile_iluvatar_rmsnorm_fused_add_smoothquant(N=0, eps=EPS)
    with pytest.raises(ValueError, match="eps must be > 0"):
        compile_iluvatar_rmsnorm_fused_add_smoothquant(N=128, eps=0.0)


def test_iluvatar_rmsnorm_fused_add_sq_runtime_guards(monkeypatch):
    _configure_iluvatar_env(monkeypatch)
    M, N = 4, 16
    launch = compile_iluvatar_rmsnorm_fused_add_smoothquant(N=N, eps=EPS)
    x = torch.randn((M, N), device="cuda", dtype=torch.float32).contiguous()
    residual_in = torch.randn_like(x)
    gamma = torch.randn((N,), device="cuda", dtype=torch.float32).contiguous()
    x_scale = torch.randn((N,), device="cuda", dtype=torch.float32).contiguous()
    out = torch.empty((M, N), device="cuda", dtype=torch.int8).contiguous()
    y_scale = torch.empty((M,), device="cuda", dtype=torch.float32).contiguous()
    residual_out = torch.empty_like(x)

    with pytest.raises(ValueError, match=r"expected x shape \(M,N\)="):
        launch(x, residual_in, gamma, x_scale, out, y_scale, residual_out, M + 1)

    out_f32 = torch.empty((M, N), device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match=r"out dtype must be torch\.int8"):
        launch(x, residual_in, gamma, x_scale, out_f32, y_scale, residual_out, M)

    x_scale_f16 = torch.empty((N,), device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match=r"x_scale dtype must be torch\.float32"):
        launch(x, residual_in, gamma, x_scale_f16, out, y_scale, residual_out, M)

    x_scale_bad = torch.empty((N + 1,), device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match=r"expected x_scale shape \(N,\)="):
        launch(x, residual_in, gamma, x_scale_bad, out, y_scale, residual_out, M)

    x_scale_nc = torch.empty((N * 2,), device="cuda", dtype=torch.float32)[::2]
    with pytest.raises(ValueError, match="x_scale must be contiguous"):
        launch(x, residual_in, gamma, x_scale_nc, out, y_scale, residual_out, M)

    with pytest.raises(ValueError, match="x_scale must not overlap with gamma"):
        launch(x, residual_in, gamma, gamma, out, y_scale, residual_out, M)

    with pytest.raises(ValueError, match="residual_out must not overlap with x_scale"):
        storage = torch.empty((M * N,), device="cuda", dtype=torch.float32)
        residual_alias = storage.view(M, N)
        x_scale_alias = storage[:N]
        launch(x, residual_in, gamma, x_scale_alias, out, y_scale, residual_alias, M)

    x_int8_view = x.view(torch.int8)
    out_alias = torch.as_strided(x_int8_view, (M, N), (N, 1))
    with pytest.raises(ValueError, match="out must not overlap with x"):
        launch(x, residual_in, gamma, x_scale, out_alias, y_scale, residual_out, M)
