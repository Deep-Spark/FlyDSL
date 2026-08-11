#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for the Iluvatar varlen FlashAttention kernel."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.l2_device, pytest.mark.iluvatar_lower]

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if not torch.cuda.is_available():
    pytest.skip("CUDA-compatible device is not available", allow_module_level=True)
if os.environ.get("FLYDSL_COMPILE_BACKEND") != "iluvatar" and "Iluvatar" not in torch.cuda.get_device_name():
    pytest.skip("Iluvatar device test", allow_module_level=True)

import flydsl.expr as fx  # noqa: E402
from kernels.attention.iluvatar.flash_attn_varlen import flash_attn_varlen_func  # noqa: E402


@pytest.fixture(autouse=True)
def _configure_iluvatar_env(monkeypatch):
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


def _reference_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, *, causal, seqused_k=None):
    """Varlen attention reference for MHA, GQA, and MQA inputs."""
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    head_dim = q.shape[2]
    out = torch.empty_like(q)

    for batch_idx in range(cu_seqlens_q.numel() - 1):
        q_start, q_end = (int(cu_seqlens_q[batch_idx]), int(cu_seqlens_q[batch_idx + 1]))
        k_start = int(cu_seqlens_k[batch_idx])
        k_end = (
            k_start + int(seqused_k[batch_idx])
            if seqused_k is not None
            else int(cu_seqlens_k[batch_idx + 1])
        )
        q_seq = q[q_start:q_end].float()
        k_seq = k[k_start:k_end].float().repeat_interleave(num_heads // num_kv_heads, dim=1)
        v_seq = v[k_start:k_end].float().repeat_interleave(num_heads // num_kv_heads, dim=1)
        scores = torch.einsum("qhd,khd->qhk", q_seq, k_seq) / math.sqrt(head_dim)
        if causal:
            q_len, k_len = q_end - q_start, k_end - k_start
            causal_mask = torch.arange(k_len, device=q.device)[None, :] <= (
                torch.arange(q_len, device=q.device)[:, None] + k_len - q_len
            )
            scores = scores.masked_fill(~causal_mask[:, None, :], float("-inf"))
        probs = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        out[q_start:q_end] = torch.einsum("qhk,khd->qhd", probs, v_seq).to(q.dtype)

    return out


@pytest.mark.parametrize(
    "paged,kv_cache_layout,causal,num_heads,num_kv_heads,q_lengths,k_lengths",
    [
        (False, "NHD", True, 16, 2, [129, 67], [187, 128]),
        (True, "NHD", True, 16, 2, [129, 67], [187, 128]),
        (True, "HND", True, 16, 2, [129, 67], [187, 128]),
        (False, "NHD", False, 8, 8, [17, 3], [15, 16]),
        (True, "HND", False, 8, 1, [1, 17], [16, 17]),
        (True, "HND", True, 8, 2, [17], [3]),
        (True, "HND", True, 8, 2, [16], [16]),
        (True, "NHD", True, 8, 2, [1, 1], [17, 16]),
    ],
    ids=[
        "dense-gqa-causal",
        "paged-nhd-gqa-causal",
        "paged-hnd-gqa-causal",
        "dense-mha-noncausal-tail",
        "paged-hnd-mqa-noncausal-page-boundary",
        "paged-causal-fully-masked-prefix",
        "paged-single-sequence-partial-bm",
        "paged-nhd-decode",
    ],
)
def test_flash_attn_varlen_func(
    paged: bool,
    kv_cache_layout: str,
    causal: bool,
    num_heads: int,
    num_kv_heads: int,
    q_lengths: list[int],
    k_lengths: list[int],
):
    torch.manual_seed(20260720)
    device = "cuda"
    dtype = torch.bfloat16
    batch, head_dim, page_size = len(q_lengths), 128, 16
    cu_seqlens_q = torch.tensor([0, *torch.tensor(q_lengths).cumsum(0).tolist()], device=device, dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, *torch.tensor(k_lengths).cumsum(0).tolist()], device=device, dtype=torch.int32)
    seqused_k = torch.tensor(k_lengths, device=device, dtype=torch.int32)

    q = torch.randn(sum(q_lengths), num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(sum(k_lengths), num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    ref = _reference_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, causal=causal)
    out = torch.empty_like(q)

    if paged:
        max_blocks = max(math.ceil(length / page_size) for length in k_lengths)
        num_blocks = batch * max_blocks
        # Exercise the actual page indirection instead of a logical == physical
        # table.  The HND variant specifically covers page-boundary gathers.
        block_table = torch.randperm(num_blocks, device=device, dtype=torch.int32).reshape(batch, max_blocks)
        if kv_cache_layout == "HND":
            k_arg = torch.zeros(num_blocks, num_kv_heads, page_size, head_dim, device=device, dtype=dtype)
        else:
            k_arg = torch.zeros(num_blocks, page_size, num_kv_heads, head_dim, device=device, dtype=dtype)
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

    result = flash_attn_varlen_func(
        q,
        k_arg,
        v_arg,
        cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        causal=causal,
        block_table=block_table if paged else None,
        seqused_k=seqused_k,
        kv_cache_layout=kv_cache_layout,
        out=out,
    )
    torch.cuda.synchronize()

    assert result is out
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)


def test_flash_attn_varlen_dense_seqused_k_limits_visible_prefix():
    torch.manual_seed(20260812)
    device, dtype = "cuda", torch.bfloat16
    q_lengths = [3, 2]
    k_spans = [7, 9]
    visible_k = [3, 4]
    num_heads, num_kv_heads, head_dim = 8, 2, 128
    cu_q = torch.tensor([0, 3, 5], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, 7, 16], device=device, dtype=torch.int32)
    seqused_k = torch.tensor(visible_k, device=device, dtype=torch.int32)
    q = torch.randn(sum(q_lengths), num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(sum(k_spans), num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    ref = _reference_varlen(
        q,
        k,
        v,
        cu_q,
        cu_k,
        causal=False,
        seqused_k=seqused_k,
    )

    out = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_seqlens_k=cu_k,
        seqused_k=seqused_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(visible_k),
        causal=False,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize(
    ("q_lengths", "k_lengths", "causal", "strided_cache"),
    [
        ([7, 13], [19, 29], True, False),
        ([128], [128], True, False),
        ([127, 33], [128, 47], False, False),
        ([512], [512], True, False),
        ([512], [512], True, True),
        ([1024], [1024], True, False),
    ],
    ids=[
        "batch-tail",
        "ctx128",
        "batch-tail-noncausal",
        "ctx512",
        "ctx512-strided-cache",
        "ctx1024",
    ],
)
def test_flash_attn_varlen_paged_hnd_qwen35_head_dim_256(
    q_lengths: list[int],
    k_lengths: list[int],
    causal: bool,
    strided_cache: bool,
):
    """Qwen3.5 full-attention shape: Hq=16, Hkv=4, D=256."""
    torch.manual_seed(20260804)
    device = "cuda"
    dtype = torch.bfloat16
    num_heads, num_kv_heads, head_dim, page_size = 16, 4, 256, 16
    batch = len(q_lengths)
    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(q_lengths).cumsum(0).tolist()],
        device=device,
        dtype=torch.int32,
    )
    cu_seqlens_k = torch.tensor(
        [0, *torch.tensor(k_lengths).cumsum(0).tolist()],
        device=device,
        dtype=torch.int32,
    )
    seqused_k = torch.tensor(k_lengths, device=device, dtype=torch.int32)

    q = torch.randn(
        sum(q_lengths), num_heads, head_dim, device=device, dtype=dtype
    )
    k = torch.randn(
        sum(k_lengths), num_kv_heads, head_dim, device=device, dtype=dtype
    )
    v = torch.randn_like(k)
    ref = _reference_varlen(
        q, k, v, cu_seqlens_q, cu_seqlens_k, causal=causal
    )

    max_blocks = max(math.ceil(length / page_size) for length in k_lengths)
    num_blocks = batch * max_blocks
    block_table = torch.randperm(
        num_blocks, device=device, dtype=torch.int32
    ).reshape(batch, max_blocks)
    if strided_cache:
        k_storage = torch.zeros(
            num_blocks,
            num_kv_heads,
            page_size,
            2,
            head_dim,
            device=device,
            dtype=dtype,
        )
        v_storage = torch.zeros_like(k_storage)
        k_cache = k_storage[:, :, :, 0, :]
        v_cache = v_storage[:, :, :, 0, :]
    else:
        k_cache = torch.zeros(
            num_blocks,
            num_kv_heads,
            page_size,
            head_dim,
            device=device,
            dtype=dtype,
        )
        v_cache = torch.zeros_like(k_cache)
    for batch_idx, k_len in enumerate(k_lengths):
        start = int(cu_seqlens_k[batch_idx])
        for token in range(k_len):
            physical_block = int(block_table[batch_idx, token // page_size])
            k_cache[physical_block, :, token % page_size] = k[start + token]
            v_cache[physical_block, :, token % page_size] = v[start + token]

    out = torch.empty_like(q)
    result = flash_attn_varlen_func(
        q,
        k_cache,
        v_cache,
        cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(q_lengths),
        max_seqlen_k=max(k_lengths),
        causal=causal,
        block_table=block_table,
        seqused_k=seqused_k,
        kv_cache_layout="HND",
        out=out,
    )
    torch.cuda.synchronize()

    assert result is out
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("strict_checks", [True, False])
def test_flash_attn_varlen_zero_length_paged_kv(strict_checks):
    device = "cuda"
    dtype = torch.bfloat16
    q = torch.randn(1, 8, 128, device=device, dtype=dtype)
    k_cache = torch.zeros(1, 2, 16, 128, device=device, dtype=dtype)
    v_cache = torch.zeros_like(k_cache)
    cu_seqlens_q = torch.tensor([0, 1], device=device, dtype=torch.int32)
    seqused_k = torch.tensor([0], device=device, dtype=torch.int32)
    block_table = torch.tensor([[-1]], device=device, dtype=torch.int32)

    out = flash_attn_varlen_func(
        q,
        k_cache,
        v_cache,
        cu_seqlens_q,
        max_seqlen_q=1,
        max_seqlen_k=0,
        causal=True,
        block_table=block_table,
        seqused_k=seqused_k,
        kv_cache_layout="HND",
        strict_checks=strict_checks,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)


def test_flash_attn_varlen_zero_length_query_returns_without_launch():
    q = torch.empty(0, 8, 128, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.zeros(1, 2, 16, 128, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    cu_seqlens_q = torch.tensor([0, 0], device="cuda", dtype=torch.int32)
    seqused_k = torch.tensor([1], device="cuda", dtype=torch.int32)
    block_table = torch.tensor([[0]], device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)

    result = flash_attn_varlen_func(
        q,
        k_cache,
        v_cache,
        cu_seqlens_q,
        causal=True,
        block_table=block_table,
        seqused_k=seqused_k,
        kv_cache_layout="HND",
        out=out,
    )

    assert result is out
    assert result.shape == (0, 8, 128)


def test_flash_attn_varlen_dense_tail_isolates_next_sequence_nan():
    device = "cuda"
    dtype = torch.bfloat16
    q = torch.randn(2, 8, 128, device=device, dtype=dtype)
    k = torch.randn(18, 2, 128, device=device, dtype=dtype)
    v = torch.randn_like(k)
    v[1].copy_(v[0])
    v[2:].fill_(float("nan"))
    cu_q = torch.tensor([0, 1, 2], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, 2, 18], device=device, dtype=torch.int32)
    seqused_k = torch.tensor([2, 16], device=device, dtype=torch.int32)

    out = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=1,
        max_seqlen_k=16,
        causal=True,
        seqused_k=seqused_k,
    )
    torch.cuda.synchronize()

    assert torch.isfinite(out[0]).all()
    torch.testing.assert_close(
        out[0].float(),
        v[0].repeat_interleave(4, dim=0).float(),
        atol=3e-2,
        rtol=3e-2,
    )


def test_flash_attn_varlen_rejects_noncontiguous_dense_kv():
    q = torch.randn(2, 8, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 2, 128, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    v = torch.randn_like(k)
    cu = torch.tensor([0, 2], device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="dense k/v must be contiguous"):
        flash_attn_varlen_func(
            q,
            k,
            v,
            cu,
            cu_seqlens_k=cu,
            max_seqlen_q=2,
            max_seqlen_k=2,
        )


def test_flash_attn_varlen_rejects_noncurrent_stream():
    q = torch.empty(1, 2, 128, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.empty(1, 1, 16, 128, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.empty_like(k_cache)
    cu_seqlens_q = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    seqused_k = torch.tensor([1], device="cuda", dtype=torch.int32)
    block_table = torch.tensor([[0]], device="cuda", dtype=torch.int32)

    with pytest.raises(NotImplementedError, match="current PyTorch stream"):
        flash_attn_varlen_func(
            q,
            k_cache,
            v_cache,
            cu_seqlens_q,
            block_table=block_table,
            seqused_k=seqused_k,
            kv_cache_layout="HND",
            stream=fx.Stream(torch.cuda.Stream()),
        )
