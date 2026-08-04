# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Device tests for Iluvatar MR BF16/FP16 two-stage MoE kernels."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.iluvatar_lower]

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


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


def _sort_routes(topk_ids, topk_weights, *, experts: int, block_size: int):
    """Build the packed sorting buffers consumed by the grouped GEMMs."""
    torch = _require_torch()
    tokens, topk = topk_ids.shape
    max_rows = int(tokens * topk + experts * block_size - topk)
    max_blocks = (max_rows + block_size - 1) // block_size
    sentinel = (int(topk) << 24) | int(tokens)
    sorted_ids = torch.full((max_rows,), sentinel, dtype=torch.int32, device=topk_ids.device)
    sorted_weights = torch.zeros((max_rows,), dtype=torch.float32, device=topk_ids.device)
    expert_ids = torch.full((max_blocks,), -1, dtype=torch.int32, device=topk_ids.device)
    row_begin = 0
    block_begin = 0
    for expert in range(experts):
        token, slot = torch.where(topk_ids == expert)
        count = int(token.numel())
        blocks = (count + block_size - 1) // block_size
        padded = blocks * block_size
        if count:
            sorted_ids[row_begin : row_begin + count] = (slot.to(torch.int32) << 24) | token.to(torch.int32)
            sorted_weights[row_begin : row_begin + count] = topk_weights[token, slot]
        if blocks:
            expert_ids[block_begin : block_begin + blocks] = expert
        row_begin += padded
        block_begin += blocks
    num_valid_ids = torch.tensor([row_begin, tokens], dtype=torch.int32, device=topk_ids.device)
    return (
        sorted_ids[:row_begin].contiguous(),
        sorted_weights[:row_begin].contiguous(),
        expert_ids[:block_begin].contiguous(),
        num_valid_ids,
        block_begin,
    )


def _reference_stage1(x, w1, topk_ids, topk_weights, apply_route_weight):
    torch = _require_torch()
    tokens, topk = topk_ids.shape
    inter_dim = w1.shape[1] // 2
    out = torch.empty((tokens, topk, inter_dim), dtype=torch.float32, device=x.device)
    for token in range(tokens):
        for slot in range(topk):
            expert = int(topk_ids[token, slot])
            projected = x[token].float() @ w1[expert].float().T
            gate, up = projected.split(inter_dim)
            value = torch.nn.functional.silu(gate) * up
            if apply_route_weight:
                value = value * topk_weights[token, slot]
            out[token, slot] = value
    return out


def _reference_stage2(x, w2, topk_ids, topk_weights, apply_route_weight):
    torch = _require_torch()
    tokens, topk = topk_ids.shape
    model_dim = w2.shape[1]
    per_slot = torch.empty((tokens, topk, model_dim), dtype=torch.float32, device=x.device)
    for token in range(tokens):
        for slot in range(topk):
            expert = int(topk_ids[token, slot])
            value = x[token, slot].float() @ w2[expert].float().T
            if apply_route_weight:
                value = value * topk_weights[token, slot]
            per_slot[token, slot] = value
    return per_slot


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"dtype": "f32"}, "dtype"),
        ({"topk": 0}, "positive"),
        ({"topk": 256}, "8-bit"),
        ({"model_dim": 48}, "divisible"),
        ({"stages": 3}, "two-stage"),
        ({"warp_atoms_n": 1}, "even"),
    ],
)
def test_gemm1_compile_guards(monkeypatch, kwargs, match):
    _configure_iluvatar_env(monkeypatch)
    from kernels.moe.iluvatar.mr.moe_gemm_2stage import (
        compile_iluvatar_mr_moe_gemm1_b16,
    )

    options = dict(model_dim=64, inter_dim=64, experts=4, topk=2, dtype="bf16")
    options.update(kwargs)
    with pytest.raises(ValueError, match=match):
        compile_iluvatar_mr_moe_gemm1_b16(**options)


def test_gemm2_rejects_unreliable_b16_atomics(monkeypatch):
    _configure_iluvatar_env(monkeypatch)
    from kernels.moe.iluvatar.mr.moe_gemm_2stage import (
        MoeGemm2Mode,
        compile_iluvatar_mr_moe_gemm2_b16,
        compile_iluvatar_mr_moe_gemm2_b16_ex,
    )

    options = dict(model_dim=64, inter_dim=64, experts=4, topk=2, dtype="bf16")
    with pytest.raises(ValueError, match="atomic"):
        compile_iluvatar_mr_moe_gemm2_b16(accumulate=True, **options)
    with pytest.raises(ValueError, match="ATOMIC"):
        compile_iluvatar_mr_moe_gemm2_b16_ex(mode=MoeGemm2Mode.ATOMIC, **options)


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize("apply_route_weight", [False, True])
def test_gemm1_correctness(monkeypatch, dtype, apply_route_weight):
    _configure_iluvatar_env(monkeypatch)
    torch = _require_torch()
    from kernels.moe.iluvatar.mr.moe_gemm_2stage import (
        compile_iluvatar_mr_moe_gemm1_b16,
    )

    tokens, experts, topk, model_dim, inter_dim = 8, 4, 2, 64, 64
    td = torch.float16 if dtype == "f16" else torch.bfloat16
    torch.manual_seed(7)
    x = (torch.randn(tokens, model_dim, device="cuda") * 0.25).to(td)
    w1 = (torch.randn(experts, 2 * inter_dim, model_dim, device="cuda") * 0.05).to(td)
    topk_ids = torch.stack([torch.randperm(experts, device="cuda")[:topk] for _ in range(tokens)]).to(torch.int32)
    topk_weights = torch.rand(tokens, topk, device="cuda", dtype=torch.float32)
    sorted_ids, sorted_weights, expert_ids, num_valid_ids, num_blocks = _sort_routes(
        topk_ids,
        topk_weights,
        experts=experts,
        block_size=32,
    )
    out = torch.zeros(tokens, topk, inter_dim, device="cuda", dtype=td)
    workspace = torch.zeros(tokens, topk, 2 * inter_dim, device="cuda", dtype=td)
    launch = compile_iluvatar_mr_moe_gemm1_b16(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        dtype=dtype,
        apply_route_weight=apply_route_weight,
    )
    stream = torch.cuda.Stream()
    launch(
        out,
        workspace,
        x,
        w1,
        sorted_ids,
        expert_ids,
        sorted_weights,
        num_valid_ids,
        tokens,
        num_blocks,
        stream=stream,
    )
    stream.synchronize()

    ref = _reference_stage1(x, w1, topk_ids, topk_weights, apply_route_weight)
    # MR MMA changes the FP32 accumulation order relative to torch.matmul.
    tol = 4e-2 if dtype == "f16" else 6e-2
    torch.testing.assert_close(out.float(), ref, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_gemm2_per_slot_and_reduction(monkeypatch, dtype):
    _configure_iluvatar_env(monkeypatch)
    torch = _require_torch()
    from kernels.moe.iluvatar.mr.moe_gemm_2stage import (
        MoeGemm2Mode,
        compile_iluvatar_mr_moe_gemm2_b16,
        compile_iluvatar_mr_moe_gemm2_b16_ex,
    )

    tokens, experts, topk, model_dim, inter_dim = 8, 4, 2, 64, 64
    td = torch.float16 if dtype == "f16" else torch.bfloat16
    torch.manual_seed(11)
    x = (torch.randn(tokens, topk, inter_dim, device="cuda") * 0.1).to(td)
    w2 = (torch.randn(experts, model_dim, inter_dim, device="cuda") * 0.05).to(td)
    topk_ids = torch.stack([torch.randperm(experts, device="cuda")[:topk] for _ in range(tokens)]).to(torch.int32)
    topk_weights = torch.rand(tokens, topk, device="cuda", dtype=torch.float32)
    sorted_ids, sorted_weights, expert_ids, num_valid_ids, num_blocks = _sort_routes(
        topk_ids,
        topk_weights,
        experts=experts,
        block_size=32,
    )
    per_slot = torch.zeros(tokens, topk, model_dim, device="cuda", dtype=td)
    gemm2 = compile_iluvatar_mr_moe_gemm2_b16(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        dtype=dtype,
        apply_route_weight=True,
        accumulate=False,
    )
    gemm2(
        per_slot,
        x,
        w2,
        sorted_ids,
        expert_ids,
        sorted_weights,
        num_valid_ids,
        tokens,
        num_blocks,
    )
    torch.cuda.synchronize()

    ref_slot = _reference_stage2(x, w2, topk_ids, topk_weights, apply_route_weight=True)
    # MR MMA changes the FP32 accumulation order relative to torch.matmul.
    tol = 4e-2 if dtype == "f16" else 6e-2
    torch.testing.assert_close(per_slot.float(), ref_slot, rtol=tol, atol=tol)

    out = torch.empty(tokens, model_dim, device="cuda", dtype=td)
    workspace = torch.empty_like(per_slot)
    combined = compile_iluvatar_mr_moe_gemm2_b16_ex(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        dtype=dtype,
        apply_route_weight=True,
        mode=MoeGemm2Mode.REDUCE,
    )
    combined(
        out,
        workspace,
        x,
        w2,
        sorted_ids,
        expert_ids,
        sorted_weights,
        num_valid_ids,
        tokens,
        num_blocks,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref_slot.sum(dim=1), rtol=tol, atol=tol)
