# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar RMSNorm V3 SmoothQuant device tests (fp32 + x_scale -> i8)."""

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

from kernels.norm.iluvatar.rmsnorm_kernel import (  # noqa: E402
    QUANT_I8_MAX,
    compile_iluvatar_rmsnorm_dynamicquant,
    compile_iluvatar_rmsnorm_smoothquant,
)

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


def _reference_rmsnorm_sq_fp32(x: torch.Tensor, gamma: torch.Tensor, x_scale: torch.Tensor, eps: float):
    mean_sq = (x * x).mean(dim=1, keepdim=True)
    rrms = torch.rsqrt(mean_sq + eps)
    y = x * rrms * gamma * x_scale
    amax = y.abs().max(dim=1, keepdim=True).values
    scale = amax / QUANT_I8_MAX
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q_f = torch.trunc(y / scale)
    q = q_f.to(torch.int8)
    return q, scale.squeeze(1)


def _run_sq(monkeypatch, x, gamma, x_scale, M, N):
    _configure_iluvatar_env(monkeypatch)
    launch = compile_iluvatar_rmsnorm_smoothquant(N=N, eps=EPS)
    out = torch.empty((M, N), device="cuda", dtype=torch.int8).contiguous()
    y_scale = torch.empty((M,), device="cuda", dtype=torch.float32).contiguous()
    ret_out, ret_scale = launch(x, gamma, x_scale, out, y_scale, M)
    torch.cuda.synchronize()
    assert ret_out is out and ret_scale is y_scale
    return out, y_scale


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
def test_iluvatar_rmsnorm_sq_forward_v3_i8(monkeypatch, M, N):
    torch.manual_seed(42)
    x = torch.randn((M, N), device="cuda", dtype=torch.float32).contiguous()
    gamma = torch.randn((N,), device="cuda", dtype=torch.float32).contiguous()
    x_scale = (torch.rand((N,), device="cuda", dtype=torch.float32) + 0.5).contiguous()

    out, y_scale = _run_sq(monkeypatch, x, gamma, x_scale, M, N)

    if M == 0:
        assert out.numel() == 0 and y_scale.numel() == 0
        return

    q_ref, scale_ref = _reference_rmsnorm_sq_fp32(x, gamma, x_scale, EPS)
    diff = (out.to(torch.int32) - q_ref.to(torch.int32)).abs().max().item()
    assert diff <= OUTPUT_ATOL_LSB, f"i8 out max abs diff = {diff} (> {OUTPUT_ATOL_LSB})"
    torch.testing.assert_close(y_scale, scale_ref, rtol=SCALE_RTOL, atol=SCALE_ATOL)


def test_iluvatar_rmsnorm_sq_x_scale_ones_matches_dq(monkeypatch):
    # Cross-path regression: x_scale=1 must match V2 DQ on the same inputs.
    M, N = 8, 256
    torch.manual_seed(7)
    x = torch.randn((M, N), device="cuda", dtype=torch.float32).contiguous()
    gamma = torch.randn((N,), device="cuda", dtype=torch.float32).contiguous()
    x_scale = torch.ones((N,), device="cuda", dtype=torch.float32).contiguous()

    out_sq, y_scale_sq = _run_sq(monkeypatch, x, gamma, x_scale, M, N)

    _configure_iluvatar_env(monkeypatch)
    launch_dq = compile_iluvatar_rmsnorm_dynamicquant(N=N, eps=EPS)
    out_dq = torch.empty((M, N), device="cuda", dtype=torch.int8).contiguous()
    y_scale_dq = torch.empty((M,), device="cuda", dtype=torch.float32).contiguous()
    launch_dq(x, gamma, out_dq, y_scale_dq, M)
    torch.cuda.synchronize()

    assert torch.equal(out_sq, out_dq)
    torch.testing.assert_close(y_scale_sq, y_scale_dq, rtol=0.0, atol=0.0)


def test_iluvatar_rmsnorm_sq_x_scale_negative_and_zero(monkeypatch):
    M, N = 4, 128
    torch.manual_seed(11)
    x = torch.randn((M, N), device="cuda", dtype=torch.float32).contiguous()
    gamma = torch.randn((N,), device="cuda", dtype=torch.float32).contiguous()
    x_scale = torch.randn((N,), device="cuda", dtype=torch.float32).contiguous()
    x_scale[0] = 0.0
    x_scale[1] = -1.5

    out, y_scale = _run_sq(monkeypatch, x, gamma, x_scale, M, N)
    q_ref, scale_ref = _reference_rmsnorm_sq_fp32(x, gamma, x_scale, EPS)
    diff = (out.to(torch.int32) - q_ref.to(torch.int32)).abs().max().item()
    assert diff <= OUTPUT_ATOL_LSB, f"i8 out max abs diff = {diff} (> {OUTPUT_ATOL_LSB})"
    torch.testing.assert_close(y_scale, scale_ref, rtol=SCALE_RTOL, atol=SCALE_ATOL)


def test_iluvatar_rmsnorm_sq_all_zero_row(monkeypatch):
    M, N = 3, 128
    x = torch.zeros((M, N), device="cuda", dtype=torch.float32).contiguous()
    gamma = torch.randn((N,), device="cuda", dtype=torch.float32).contiguous()
    x_scale = (torch.rand((N,), device="cuda", dtype=torch.float32) + 0.5).contiguous()

    out, y_scale = _run_sq(monkeypatch, x, gamma, x_scale, M, N)

    assert torch.all(out == 0), f"expected all-zero i8 out, got nonzero count={int((out != 0).sum())}"
    torch.testing.assert_close(
        y_scale,
        torch.ones((M,), device="cuda", dtype=torch.float32),
        rtol=SCALE_RTOL,
        atol=SCALE_ATOL,
    )


def test_iluvatar_rmsnorm_sq_m0_noop(monkeypatch):
    _configure_iluvatar_env(monkeypatch)

    launch = compile_iluvatar_rmsnorm_smoothquant(N=256, eps=EPS)
    x = torch.empty((0, 256), device="cuda", dtype=torch.float32).contiguous()
    gamma = torch.randn((256,), device="cuda", dtype=torch.float32).contiguous()
    x_scale = torch.randn((256,), device="cuda", dtype=torch.float32).contiguous()
    out = torch.empty((0, 256), device="cuda", dtype=torch.int8).contiguous()
    y_scale = torch.empty((0,), device="cuda", dtype=torch.float32).contiguous()

    ret_out, ret_scale = launch(x, gamma, x_scale, out, y_scale, 0)
    assert ret_out is out and ret_scale is y_scale
    assert out.numel() == 0 and y_scale.numel() == 0


def test_iluvatar_rmsnorm_sq_compile_time_guards():
    with pytest.raises(ValueError, match="N must be > 0"):
        compile_iluvatar_rmsnorm_smoothquant(N=0, eps=EPS)
    with pytest.raises(ValueError, match="eps must be > 0"):
        compile_iluvatar_rmsnorm_smoothquant(N=128, eps=0.0)


def test_iluvatar_rmsnorm_sq_runtime_guards(monkeypatch):
    _configure_iluvatar_env(monkeypatch)

    M, N = 4, 16
    launch = compile_iluvatar_rmsnorm_smoothquant(N=N, eps=EPS)
    x = torch.randn((M, N), device="cuda", dtype=torch.float32).contiguous()
    gamma = torch.randn((N,), device="cuda", dtype=torch.float32).contiguous()
    x_scale = torch.randn((N,), device="cuda", dtype=torch.float32).contiguous()
    out = torch.empty((M, N), device="cuda", dtype=torch.int8).contiguous()
    y_scale = torch.empty((M,), device="cuda", dtype=torch.float32).contiguous()

    with pytest.raises(ValueError, match="expected x shape \\(M,N\\)="):
        launch(x, gamma, x_scale, out, y_scale, M + 1)

    out_f32 = torch.empty((M, N), device="cuda", dtype=torch.float32).contiguous()
    with pytest.raises(ValueError, match=r"out dtype must be torch\.int8"):
        launch(x, gamma, x_scale, out_f32, y_scale, M)

    x_scale_f16 = torch.empty((N,), device="cuda", dtype=torch.float16).contiguous()
    with pytest.raises(ValueError, match=r"x_scale dtype must be torch\.float32"):
        launch(x, gamma, x_scale_f16, out, y_scale, M)

    x_scale_bad = torch.empty((N + 1,), device="cuda", dtype=torch.float32).contiguous()
    with pytest.raises(ValueError, match=r"expected x_scale shape \(N,\)="):
        launch(x, gamma, x_scale_bad, out, y_scale, M)

    x_scale_nc = torch.empty((N * 2,), device="cuda", dtype=torch.float32)[::2]
    assert tuple(x_scale_nc.shape) == (N,) and not x_scale_nc.is_contiguous()
    with pytest.raises(ValueError, match="x_scale must be contiguous"):
        launch(x, gamma, x_scale_nc, out, y_scale, M)

    # overlap: x_scale aliases gamma storage
    with pytest.raises(ValueError, match="x_scale must not overlap with gamma"):
        launch(x, gamma, gamma, out, y_scale, M)

    x_int8_view = x.view(torch.int8)
    out_alias = torch.as_strided(x_int8_view, (M, N), (N, 1))
    assert out_alias.data_ptr() == x.data_ptr() and out_alias.is_contiguous()
    with pytest.raises(ValueError, match="out must not overlap with x"):
        launch(x, gamma, x_scale, out_alias, y_scale, M)
