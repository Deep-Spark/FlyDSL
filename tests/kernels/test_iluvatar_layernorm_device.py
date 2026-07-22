# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar LayerNorm device tests (basic / fused-add / dynamic-quant / smooth-quant)."""

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

import flydsl.compiler as flyc  # noqa: E402
from kernels.norm.iluvatar.layernorm_kernel import (  # noqa: E402
    build_fused_add_layernorm_dynamicquant_module,
    build_fused_add_layernorm_module,
    build_fused_add_layernorm_smoothquant_module,
    build_layernorm_dynamicquant_module,
    build_layernorm_module,
    build_layernorm_smoothquant_module,
)

EPS = 1e-5
ATOL = {"f32": 1e-4, "f16": 1e-2, "bf16": 2e-2}
TORCH_DTYPE = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}


def _configure_iluvatar_env(monkeypatch) -> None:
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


def _reference_layernorm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    mean = xf.mean(dim=1, keepdim=True)
    var = xf.var(dim=1, keepdim=True, unbiased=False)
    return (xf - mean) / torch.sqrt(var + EPS) * gamma.float() + beta.float()


def _reference_quant(x, gamma, beta, xscale=None):
    y = _reference_layernorm(x, gamma, beta)
    if xscale is not None:
        y = y * xscale.float()
    yscale = y.abs().amax(dim=1) / 127.0
    yscale = torch.where(yscale == 0, torch.ones_like(yscale), yscale)
    q = torch.clamp(torch.trunc(y / yscale.unsqueeze(1)), -127, 127).to(torch.int8)
    return q, yscale


def _randn(M: int, N: int, dtype: str) -> torch.Tensor:
    return torch.randn((M, N), device="cuda", dtype=torch.float32).to(TORCH_DTYPE[dtype]).contiguous()


def _rand_row(N: int, dtype: str, offset: float = 0.0) -> torch.Tensor:
    return (torch.rand((N,), device="cuda", dtype=torch.float32) + offset).to(TORCH_DTYPE[dtype]).contiguous()


@pytest.mark.parametrize(
    "M,N,dtype",
    [
        (64, 256, "f32"),
        (32, 128, "f16"),
        (8, 8192, "bf16"),
    ],
)
def test_iluvatar_layernorm_basic(monkeypatch, M, N, dtype):
    _configure_iluvatar_env(monkeypatch)
    torch.manual_seed(0)

    launch_fn = build_layernorm_module(N, dtype)
    x = _randn(M, N, dtype)
    gamma = _rand_row(N, dtype)
    beta = _rand_row(N, dtype)
    out = torch.empty((M, N), device="cuda", dtype=TORCH_DTYPE[dtype])
    expected = _reference_layernorm(x, gamma, beta)

    stream = torch.cuda.current_stream()
    compiled = flyc.compile(launch_fn, x, gamma, beta, out, M, stream)
    compiled(x, gamma, beta, out, M, stream)
    torch.cuda.synchronize()

    err = (out.float() - expected).abs().max().item()
    assert err <= ATOL[dtype], f"max_abs={err}"


@pytest.mark.parametrize(
    "M,N,dtype",
    [
        (32, 256, "f16"),
        (16, 2000, "f32"),
        (4, 8192, "bf16"),
    ],
)
def test_iluvatar_layernorm_fused_add(monkeypatch, M, N, dtype):
    _configure_iluvatar_env(monkeypatch)
    torch.manual_seed(1)

    launch_fn = build_fused_add_layernorm_module(N, dtype)
    x = _randn(M, N, dtype)
    residual = _randn(M, N, dtype)
    gamma = _rand_row(N, dtype)
    beta = _rand_row(N, dtype)
    out = torch.empty((M, N), device="cuda", dtype=TORCH_DTYPE[dtype])
    residual_out = torch.empty((M, N), device="cuda", dtype=TORCH_DTYPE[dtype])
    added = x.float() + residual.float()
    expected = _reference_layernorm(added, gamma, beta)

    stream = torch.cuda.current_stream()
    compiled = flyc.compile(launch_fn, x, residual, gamma, beta, out, residual_out, M, stream)
    compiled(x, residual, gamma, beta, out, residual_out, M, stream)
    torch.cuda.synchronize()

    err = (out.float() - expected).abs().max().item()
    rerr = (residual_out.float() - added).abs().max().item()
    assert err <= ATOL[dtype], f"max_abs={err}"
    assert rerr <= ATOL[dtype], f"residual_abs={rerr}"


@pytest.mark.parametrize(
    "M,N,dtype,is_smooth",
    [
        (32, 256, "f16", False),
        (16, 512, "bf16", True),
        (4, 8192, "f16", False),
    ],
)
def test_iluvatar_layernorm_quant(monkeypatch, M, N, dtype, is_smooth):
    _configure_iluvatar_env(monkeypatch)
    torch.manual_seed(2)

    builder = build_layernorm_smoothquant_module if is_smooth else build_layernorm_dynamicquant_module
    launch_fn = builder(N, dtype)
    x = _randn(M, N, dtype)
    gamma = _rand_row(N, dtype)
    beta = _rand_row(N, dtype)
    xscale = _rand_row(N, dtype, offset=0.5) if is_smooth else None
    out = torch.empty((M, N), device="cuda", dtype=torch.int8)
    yscale = torch.empty((M,), device="cuda", dtype=torch.float32)
    q_ref, yscale_ref = _reference_quant(x, gamma, beta, xscale=xscale)

    stream = torch.cuda.current_stream()
    if is_smooth:
        compiled = flyc.compile(launch_fn, x, gamma, beta, xscale, out, yscale, M, stream)
        compiled(x, gamma, beta, xscale, out, yscale, M, stream)
    else:
        compiled = flyc.compile(launch_fn, x, gamma, beta, out, yscale, M, stream)
        compiled(x, gamma, beta, out, yscale, M, stream)
    torch.cuda.synchronize()

    qdiff = (out.to(torch.int16) - q_ref.to(torch.int16)).abs().max().item()
    sdiff = (yscale - yscale_ref).abs().max().item()
    assert qdiff <= 1, f"quant_diff={qdiff}"
    assert sdiff < 1e-3, f"scale_diff={sdiff}"


@pytest.mark.parametrize(
    "M,N,dtype,is_smooth",
    [
        (16, 256, "f16", False),
        (4, 8192, "bf16", True),
    ],
)
def test_iluvatar_layernorm_fused_add_quant(monkeypatch, M, N, dtype, is_smooth):
    _configure_iluvatar_env(monkeypatch)
    torch.manual_seed(3)

    builder = (
        build_fused_add_layernorm_smoothquant_module if is_smooth else build_fused_add_layernorm_dynamicquant_module
    )
    launch_fn = builder(N, dtype)
    x = _randn(M, N, dtype)
    residual = _randn(M, N, dtype)
    gamma = _rand_row(N, dtype)
    beta = _rand_row(N, dtype)
    xscale = _rand_row(N, dtype, offset=0.5) if is_smooth else None
    out = torch.empty((M, N), device="cuda", dtype=torch.int8)
    residual_out = torch.empty((M, N), device="cuda", dtype=TORCH_DTYPE[dtype])
    yscale = torch.empty((M,), device="cuda", dtype=torch.float32)
    added = x.float() + residual.float()
    q_ref, yscale_ref = _reference_quant(added, gamma, beta, xscale=xscale)

    stream = torch.cuda.current_stream()
    if is_smooth:
        compiled = flyc.compile(launch_fn, x, residual, gamma, beta, xscale, out, residual_out, yscale, M, stream)
        compiled(x, residual, gamma, beta, xscale, out, residual_out, yscale, M, stream)
    else:
        compiled = flyc.compile(launch_fn, x, residual, gamma, beta, out, residual_out, yscale, M, stream)
        compiled(x, residual, gamma, beta, out, residual_out, yscale, M, stream)
    torch.cuda.synchronize()

    qdiff = (out.to(torch.int16) - q_ref.to(torch.int16)).abs().max().item()
    sdiff = (yscale - yscale_ref).abs().max().item()
    rerr = (residual_out.float() - added).abs().max().item()
    assert qdiff <= 1, f"quant_diff={qdiff}"
    assert sdiff < 1e-3, f"scale_diff={sdiff}"
    assert rerr <= ATOL[dtype], f"residual_abs={rerr}"
