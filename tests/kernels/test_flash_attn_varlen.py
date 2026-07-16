#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for the Iluvatar varlen FlashAttention kernel."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if not torch.cuda.is_available():
    pytest.skip("CUDA-compatible device is not available", allow_module_level=True)

from kernels.attention.iluvatar.flash_attn_varlen import flash_attn_varlen_func_flydsl  # noqa: E402


def _reference_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k):
    """Bottom-right causal varlen attention reference for GQA inputs."""
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    head_dim = q.shape[2]
    out = torch.empty_like(q)

    for batch_idx in range(cu_seqlens_q.numel() - 1):
        q_start, q_end = (int(cu_seqlens_q[batch_idx]), int(cu_seqlens_q[batch_idx + 1]))
        k_start, k_end = (int(cu_seqlens_k[batch_idx]), int(cu_seqlens_k[batch_idx + 1]))
        q_seq = q[q_start:q_end].float()
        k_seq = k[k_start:k_end].float().repeat_interleave(num_heads // num_kv_heads, dim=1)
        v_seq = v[k_start:k_end].float().repeat_interleave(num_heads // num_kv_heads, dim=1)
        scores = torch.einsum("qhd,khd->qhk", q_seq, k_seq) / math.sqrt(head_dim)
        q_len, k_len = q_end - q_start, k_end - k_start
        causal_mask = torch.arange(k_len, device=q.device)[None, :] <= (
            torch.arange(q_len, device=q.device)[:, None] + k_len - q_len
        )
        scores = scores.masked_fill(~causal_mask[:, None, :], float("-inf"))
        out[q_start:q_end] = torch.einsum("qhk,khd->qhd", torch.softmax(scores, dim=-1), v_seq).to(q.dtype)

    return out


@pytest.mark.parametrize(
    "paged,kv_cache_layout",
    [(False, "NHD"), (True, "NHD"), (True, "HND")],
    ids=["dense", "paged-nhd", "paged-hnd-irregular-blocks"],
)
def test_flash_attn_varlen_func_flydsl(paged: bool, kv_cache_layout: str):
    torch.manual_seed(20260720)
    device = "cuda"
    dtype = torch.bfloat16
    batch, num_heads, num_kv_heads, head_dim, page_size = 2, 16, 2, 128, 16
    q_lengths = [129, 67]
    k_lengths = [187, 128]
    cu_seqlens_q = torch.tensor([0, *torch.tensor(q_lengths).cumsum(0).tolist()], device=device, dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, *torch.tensor(k_lengths).cumsum(0).tolist()], device=device, dtype=torch.int32)
    seqused_k = torch.tensor(k_lengths, device=device, dtype=torch.int32)

    q = torch.randn(sum(q_lengths), num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(sum(k_lengths), num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    ref = _reference_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k)
    out = torch.empty_like(q)

    if paged:
        max_blocks = max(math.ceil(length / page_size) for length in k_lengths)
        num_blocks = batch * max_blocks
        # Exercise the actual page indirection instead of a logical == physical
        # table.  The HND variant specifically covers page-boundary gathers.
        block_table = torch.randperm(num_blocks, device=device, dtype=torch.int32).reshape(
            batch, max_blocks
        )
        if kv_cache_layout == "HND":
            k_arg = torch.zeros(
                num_blocks, num_kv_heads, page_size, head_dim, device=device, dtype=dtype
            )
        else:
            k_arg = torch.zeros(
                num_blocks, page_size, num_kv_heads, head_dim, device=device, dtype=dtype
            )
        v_arg = torch.zeros_like(k_arg)
        for batch_idx, k_len in enumerate(k_lengths):
            start = int(cu_seqlens_k[batch_idx])
            for token in range(k_len):
                physical_block = int(block_table[batch_idx, token // page_size])
                if kv_cache_layout == "HND":
                    k_arg[physical_block, :, token % page_size] = k[start + token]
                    v_arg[physical_block, :, token % page_size] = v[start + token]
                else:
                    k_arg[physical_block, token % page_size] = k[start + token]
                    v_arg[physical_block, token % page_size] = v[start + token]
    else:
        k_arg, v_arg, block_table = k, v, cu_seqlens_k

    result = flash_attn_varlen_func_flydsl(
        q,
        k_arg,
        v_arg,
        cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        causal=True,
        block_table=block_table if paged else None,
        seqused_k=seqused_k,
        kv_cache_layout=kv_cache_layout,
        out=out,
    )
    torch.cuda.synchronize()

    assert result is out
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)
