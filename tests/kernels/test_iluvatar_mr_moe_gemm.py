# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar MR MoE grouped GEMM device + guard tests."""

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


def _moe_sorting_torch(topk_ids, topk_weights, *, num_experts: int, block_size: int):
    """Host sorting matching test_moe_gemm.moe_sorting_torch_native."""
    torch = _require_torch()
    device = topk_ids.device
    M, topk = topk_ids.shape
    max_num_tokens_padded = int(topk_ids.numel() + int(num_experts) * int(block_size) - int(topk))
    max_num_m_blocks = int((max_num_tokens_padded + int(block_size) - 1) // int(block_size))
    init_val = (int(topk) << 24) | int(M)
    sorted_ids = torch.full((max_num_tokens_padded,), init_val, dtype=torch.int32, device=device)
    sorted_weights = torch.empty((max_num_tokens_padded,), dtype=torch.float32, device=device)
    sorted_expert_ids = torch.full((max_num_m_blocks,), -1, dtype=torch.int32, device=device)

    sorted_ids_begin = 0
    sorted_expert_ids_begin = 0
    for expert_id in range(int(num_experts)):
        token_id, topk_id = torch.where(topk_ids == expert_id)
        tokens_num = int(token_id.numel())
        n_blocks = int((tokens_num + int(block_size) - 1) // int(block_size))
        tokens_pad = int(n_blocks * int(block_size))
        sorted_ids[sorted_ids_begin : sorted_ids_begin + tokens_num] = (
            topk_id.to(torch.int32) << 24
        ) | token_id.to(torch.int32)
        sorted_weights[sorted_ids_begin : sorted_ids_begin + tokens_num] = topk_weights[
            token_id, topk_id
        ].to(torch.float32)
        sorted_ids_begin += tokens_pad
        sorted_expert_ids[sorted_expert_ids_begin : sorted_expert_ids_begin + n_blocks] = int(expert_id)
        sorted_expert_ids_begin += n_blocks

    return (
        sorted_ids[:sorted_ids_begin].contiguous(),
        sorted_weights[:sorted_ids_begin].contiguous(),
        sorted_expert_ids[:sorted_expert_ids_begin].contiguous(),
        sorted_expert_ids_begin,
    )


def _pertoken_quant(x_fp32, quant_dtype):
    torch = _require_torch()
    amax = torch.amax(torch.abs(x_fp32), dim=-1, keepdim=True)
    scale = amax / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = (x_fp32 / scale).to(quant_dtype)
    return q, scale


def _ref_moe_grouped_gemm(
    *,
    x_q,
    w_q,
    scale_x,
    scale_w,
    topk_ids,
    topk_weights,
    quant_mode: str,
    apply_route_weight: bool,
):
    """Simple grouped projection reference: Out[t,s] = (x @ W[e].T) * sx * sw [* w]."""
    torch = _require_torch()
    tokens, topk = topk_ids.shape
    E, N, K = w_q.shape
    out = torch.zeros((tokens, topk, N), device=x_q.device, dtype=torch.float32)
    w_f = w_q.to(torch.float32) * scale_w.reshape(E, N, 1).to(torch.float32)

    for t in range(tokens):
        for s in range(topk):
            e = int(topk_ids[t, s].item())
            if quant_mode == "int8smooth":
                x_row = x_q[s * tokens + t].to(torch.float32) * float(scale_x[s * tokens + t].item())
            else:
                x_row = x_q[t].to(torch.float32) * float(scale_x[t].item())
            y = x_row @ w_f[e].T
            if apply_route_weight:
                y = y * float(topk_weights[t, s].item())
            out[t, s] = y
    return out


@pytest.mark.parametrize(
    "bad_kwargs,match",
    [
        ({"major_pattern": "nn"}, "tn"),
        ({"quant_mode": "fp8"}, "quant_mode"),
        ({"out_dtype": "i8"}, "out_dtype"),
        ({"K": 63}, "multiple"),
        ({"stages": 3}, "stages"),
    ],
)
def test_compile_guards(monkeypatch, bad_kwargs, match):
    _configure_iluvatar_env(monkeypatch)
    from kernels.moe.iluvatar.mr import compile_iluvatar_mr_moe_gemm

    kwargs = dict(N=128, K=64, topk=2, quant_mode="int8", out_dtype="f16")
    kwargs.update(bad_kwargs)
    with pytest.raises(ValueError, match=match):
        compile_iluvatar_mr_moe_gemm(**kwargs)


def test_apply_route_weight_requires_weights(monkeypatch):
    _configure_iluvatar_env(monkeypatch)
    torch = _require_torch()
    from kernels.moe.iluvatar.mr import compile_iluvatar_mr_moe_gemm

    launch = compile_iluvatar_mr_moe_gemm(
        N=128, K=64, topk=2, quant_mode="int8", out_dtype="f16", apply_route_weight=True
    )
    device = "cuda"
    Out = torch.empty((1, 2, 128), device=device, dtype=torch.float16)
    X = torch.zeros((1, 64), device=device, dtype=torch.int8)
    W = torch.zeros((1, 128, 64), device=device, dtype=torch.int8)
    sx = torch.ones((1,), device=device, dtype=torch.float32)
    sw = torch.ones((1, 128), device=device, dtype=torch.float32)
    ids = torch.zeros((32,), device=device, dtype=torch.int32)
    eids = torch.zeros((1,), device=device, dtype=torch.int32)
    with pytest.raises(ValueError, match="sorted_weights"):
        launch(Out, X, W, sx, sw, ids, eids, None, 1, 1)


@pytest.mark.parametrize("quant_mode", ["int8", "int8smooth"])
@pytest.mark.parametrize("apply_route_weight", [False, True])
@pytest.mark.parametrize("N", [128, 96])
def test_moe_gemm_correctness(monkeypatch, quant_mode, apply_route_weight, N):
    _run_moe_correctness(
        monkeypatch,
        quant_mode=quant_mode,
        apply_route_weight=apply_route_weight,
        N=N,
        out_dtype="f16",
        torch_dtype="float16",
    )


@pytest.mark.parametrize("out_dtype,torch_dtype", [("bf16", "bfloat16"), ("f32", "float32")])
def test_moe_gemm_out_dtypes(monkeypatch, out_dtype, torch_dtype):
    _run_moe_correctness(
        monkeypatch,
        quant_mode="int8",
        apply_route_weight=False,
        N=128,
        out_dtype=out_dtype,
        torch_dtype=torch_dtype,
    )


def _run_moe_correctness(monkeypatch, *, quant_mode, apply_route_weight, N, out_dtype, torch_dtype):
    _configure_iluvatar_env(monkeypatch)
    torch = _require_torch()
    from kernels.moe.iluvatar.mr import compile_iluvatar_mr_moe_gemm

    tokens, E, topk, K = 8, 4, 2, 64
    device = "cuda"
    torch.manual_seed(0)

    x_fp32 = torch.randn((tokens, K), device=device, dtype=torch.float32)
    w_fp32 = torch.randn((E, N, K), device=device, dtype=torch.float32) * 0.05
    topk_ids = torch.randint(0, E, (tokens, topk), device=device, dtype=torch.int32)
    topk_weights = torch.rand((tokens, topk), device=device, dtype=torch.float32)

    if quant_mode == "int8":
        x_q, scale_x = _pertoken_quant(x_fp32, torch.int8)
        scale_x = scale_x.reshape(tokens)
    else:
        smooth = 0.75 + 0.5 * torch.rand((E, K), device=device, dtype=torch.float32)
        x_route = x_fp32[:, None, :] * smooth[topk_ids.to(torch.int64)]
        amax = torch.amax(torch.abs(x_route), dim=-1, keepdim=True)
        scale_x = amax / 127.0
        scale_x = torch.where(scale_x == 0, torch.ones_like(scale_x), scale_x)
        x_q = (x_route / scale_x).to(torch.int8)
        # Slot-major [topk*tokens, K]
        x_q = x_q.permute(1, 0, 2).contiguous().reshape(topk * tokens, K)
        scale_x = scale_x.permute(1, 0, 2).contiguous().reshape(topk * tokens)

    w_q, scale_w = _pertoken_quant(w_fp32.reshape(E * N, K), torch.int8)
    w_q = w_q.reshape(E, N, K)
    scale_w = scale_w.reshape(E, N)

    bm = 32
    sorted_ids, sorted_weights, sorted_expert_ids, num_blocks = _moe_sorting_torch(
        topk_ids, topk_weights, num_experts=E, block_size=bm
    )

    td = getattr(torch, torch_dtype)
    Out = torch.zeros((tokens, topk, N), device=device, dtype=td)

    launch = compile_iluvatar_mr_moe_gemm(
        N=N,
        K=K,
        topk=topk,
        quant_mode=quant_mode,
        out_dtype=out_dtype,
        apply_route_weight=apply_route_weight,
    )
    launch(
        Out,
        x_q,
        w_q,
        scale_x,
        scale_w,
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        tokens,
        num_blocks,
    )
    torch.cuda.synchronize()

    ref = _ref_moe_grouped_gemm(
        x_q=x_q,
        w_q=w_q,
        scale_x=scale_x,
        scale_w=scale_w,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        quant_mode=quant_mode,
        apply_route_weight=apply_route_weight,
    )
    got = Out.float()
    torch.testing.assert_close(got, ref.to(got.dtype), rtol=2e-2, atol=5e-2)
