# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar RMSNorm V2 dynamic-quant device tests (fp32 input → i8 output)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l2_device]

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
)

EPS = 1e-5
# fp reduction (sumsq, amax) reorder → up to 1-LSB drift in i8 space; scale drift
# stays well under 1e-3 relative for the shapes tested here.
OUTPUT_ATOL_LSB = 1
SCALE_RTOL = 1e-3
SCALE_ATOL = 1e-6


def _configure_iluvatar_env(monkeypatch) -> None:
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


def _reference_rmsnorm_dq_fp32(x: torch.Tensor, gamma: torch.Tensor, eps: float):
    # Semantic mirror of the kernel: 3-pass f32 rmsnorm → per-row amax → sym scale
    # → truncate-toward-zero cast. Kept in float64 for the amax/scale step so any
    # divergence from the kernel comes from reduction reorder, not the reference.
    mean_sq = (x * x).mean(dim=1, keepdim=True)
    rrms = torch.rsqrt(mean_sq + eps)
    y = x * rrms * gamma
    amax = y.abs().max(dim=1, keepdim=True).values
    scale = amax / QUANT_I8_MAX
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    # torch.trunc + .to(int8) exactly matches the kernel's arith.fptosi. Using
    # ``.to(torch.int8)`` directly would rely on PyTorch's cast being truncating,
    # which is true today but not part of the public contract for OOB inputs.
    q_f = torch.trunc(y / scale)
    q = q_f.to(torch.int8)
    return q, scale.squeeze(1)


def _run_kernel(monkeypatch, x, gamma, M, N):
    _configure_iluvatar_env(monkeypatch)
    launch = compile_iluvatar_rmsnorm_dynamicquant(N=N, eps=EPS)
    out = torch.empty((M, N), device="cuda", dtype=torch.int8).contiguous()
    y_scale = torch.empty((M,), device="cuda", dtype=torch.float32).contiguous()
    ret_out, ret_scale = launch(x, gamma, out, y_scale, M)
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
def test_iluvatar_rmsnorm_dq_forward_v2_i8(monkeypatch, M, N):
    torch.manual_seed(42)
    x = torch.randn((M, N), device="cuda", dtype=torch.float32).contiguous()
    gamma = torch.randn((N,), device="cuda", dtype=torch.float32).contiguous()

    out, y_scale = _run_kernel(monkeypatch, x, gamma, M, N)

    if M == 0:
        assert out.numel() == 0 and y_scale.numel() == 0
        return

    q_ref, scale_ref = _reference_rmsnorm_dq_fp32(x, gamma, EPS)
    diff = (out.to(torch.int32) - q_ref.to(torch.int32)).abs().max().item()
    assert diff <= OUTPUT_ATOL_LSB, f"i8 out max abs diff = {diff} (> {OUTPUT_ATOL_LSB})"
    torch.testing.assert_close(y_scale, scale_ref, rtol=SCALE_RTOL, atol=SCALE_ATOL)


def test_iluvatar_rmsnorm_dq_all_zero_row(monkeypatch):
    # Degenerate row: scale would divide by zero → kernel pins final_scale=1
    # and stores out=0. Verifies the ``select(scale==0, 1, scale)`` protection.
    M, N = 3, 128
    x = torch.zeros((M, N), device="cuda", dtype=torch.float32).contiguous()
    gamma = torch.randn((N,), device="cuda", dtype=torch.float32).contiguous()

    out, y_scale = _run_kernel(monkeypatch, x, gamma, M, N)

    assert torch.all(out == 0), f"expected all-zero i8 out, got nonzero count={int((out != 0).sum())}"
    torch.testing.assert_close(
        y_scale,
        torch.ones((M,), device="cuda", dtype=torch.float32),
        rtol=SCALE_RTOL,
        atol=SCALE_ATOL,
    )


def test_iluvatar_rmsnorm_dq_m0_noop(monkeypatch):
    _configure_iluvatar_env(monkeypatch)

    launch = compile_iluvatar_rmsnorm_dynamicquant(N=256, eps=EPS)
    x = torch.empty((0, 256), device="cuda", dtype=torch.float32).contiguous()
    gamma = torch.randn((256,), device="cuda", dtype=torch.float32).contiguous()
    out = torch.empty((0, 256), device="cuda", dtype=torch.int8).contiguous()
    y_scale = torch.empty((0,), device="cuda", dtype=torch.float32).contiguous()

    ret_out, ret_scale = launch(x, gamma, out, y_scale, 0)
    assert ret_out is out and ret_scale is y_scale
    assert out.numel() == 0 and y_scale.numel() == 0


def test_iluvatar_rmsnorm_dq_compile_time_guards():
    with pytest.raises(ValueError, match="N must be > 0"):
        compile_iluvatar_rmsnorm_dynamicquant(N=0, eps=EPS)
    with pytest.raises(ValueError, match="eps must be > 0"):
        compile_iluvatar_rmsnorm_dynamicquant(N=128, eps=0.0)


def test_iluvatar_rmsnorm_dq_runtime_guards(monkeypatch):
    _configure_iluvatar_env(monkeypatch)

    M, N = 4, 16
    launch = compile_iluvatar_rmsnorm_dynamicquant(N=N, eps=EPS)
    x = torch.randn((M, N), device="cuda", dtype=torch.float32).contiguous()
    gamma = torch.randn((N,), device="cuda", dtype=torch.float32).contiguous()
    out = torch.empty((M, N), device="cuda", dtype=torch.int8).contiguous()
    y_scale = torch.empty((M,), device="cuda", dtype=torch.float32).contiguous()

    # M mismatch
    with pytest.raises(ValueError, match="expected x shape \\(M,N\\)="):
        launch(x, gamma, out, y_scale, M + 1)

    # dtype guards
    out_f32 = torch.empty((M, N), device="cuda", dtype=torch.float32).contiguous()
    with pytest.raises(ValueError, match=r"out dtype must be torch\.int8"):
        launch(x, gamma, out_f32, y_scale, M)

    y_scale_f16 = torch.empty((M,), device="cuda", dtype=torch.float16).contiguous()
    with pytest.raises(ValueError, match=r"y_scale dtype must be torch\.float32"):
        launch(x, gamma, out, y_scale_f16, M)

    # y_scale shape
    y_scale_bad = torch.empty((M + 1,), device="cuda", dtype=torch.float32).contiguous()
    with pytest.raises(ValueError, match=r"expected y_scale shape \(M,\)="):
        launch(x, gamma, out, y_scale_bad, M)

    # contiguity: y_scale
    y_scale_nc = torch.empty((M * 2,), device="cuda", dtype=torch.float32)[::2]
    assert tuple(y_scale_nc.shape) == (M,) and not y_scale_nc.is_contiguous()
    with pytest.raises(ValueError, match="y_scale must be contiguous"):
        launch(x, gamma, out, y_scale_nc, M)

    # contiguity: x/gamma/out (mirrors V1 coverage but on the DQ launcher)
    x_nc = torch.randn((N, M), device="cuda", dtype=torch.float32).t()
    assert tuple(x_nc.shape) == (M, N) and not x_nc.is_contiguous()
    with pytest.raises(ValueError, match="x must be contiguous"):
        launch(x_nc, gamma, out, y_scale, M)

    gamma_nc = torch.randn((N * 2,), device="cuda", dtype=torch.float32)[::2]
    assert tuple(gamma_nc.shape) == (N,) and not gamma_nc.is_contiguous()
    with pytest.raises(ValueError, match="gamma must be contiguous"):
        launch(x, gamma_nc, out, y_scale, M)

    out_nc = torch.empty((N, M), device="cuda", dtype=torch.int8).t()
    assert tuple(out_nc.shape) == (M, N) and not out_nc.is_contiguous()
    with pytest.raises(ValueError, match="out must be contiguous"):
        launch(x, gamma, out_nc, y_scale, M)

    # overlaps: build a contiguous int8 tensor that aliases x's storage exactly.
    # x is (M, N) f32 = M*N*4 bytes; out_alias is a (M, N) i8 view over the
    # same buffer's low bytes. Passes shape/dtype/contiguity checks so the
    # overlap guard is what fires.
    x_int8_view = x.view(torch.int8)
    out_alias = torch.as_strided(x_int8_view, (M, N), (N, 1))
    assert out_alias.data_ptr() == x.data_ptr() and out_alias.is_contiguous()
    with pytest.raises(ValueError, match="out must not overlap with x"):
        launch(x, gamma, out_alias, y_scale, M)
