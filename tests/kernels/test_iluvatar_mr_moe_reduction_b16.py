# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Device tests for the Iluvatar BF16/FP16 MoE reduction kernel."""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.iluvatar_lower]


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        pytest.skip(f"torch is required: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA-compatible Iluvatar device is not available")
    return torch


def _configure_iluvatar_env(monkeypatch) -> None:
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize("use_valid_mask", [False, True])
def test_reduction_correctness(monkeypatch, dtype, use_valid_mask):
    _configure_iluvatar_env(monkeypatch)
    torch = _require_torch()
    from kernels.moe.iluvatar.mr.moe_gemm_2stage.reduction_b16 import (
        compile_iluvatar_mr_moe_reduction_b16,
    )

    tokens, topk, model_dim = 5, 3, 1037
    td = torch.float16 if dtype == "f16" else torch.bfloat16
    torch.manual_seed(13)
    x = torch.randn(tokens, topk, model_dim, device="cuda").to(td)
    mask = torch.randint(0, 2, (tokens, topk), dtype=torch.uint8, device="cuda")
    out = torch.empty(tokens, model_dim, dtype=td, device="cuda")
    launch = compile_iluvatar_mr_moe_reduction_b16(
        topk=topk,
        model_dim=model_dim,
        dtype=dtype,
        use_valid_mask=use_valid_mask,
    )
    launch(x, out, mask if use_valid_mask else None, tokens)
    torch.cuda.synchronize()
    ref = (x.float() * mask[:, :, None].float() if use_valid_mask else x.float()).sum(dim=1)
    tol = 2e-2 if dtype == "f16" else 5e-2
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)
