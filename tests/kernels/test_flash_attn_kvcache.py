#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for FlyDSL flash_attn_with_kvcache."""

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

from kernels.attention.iluvatar.flash_attn_kvcache import flash_attn_with_kvcache_flydsl  # noqa: E402


def _make_rotary_tables(max_pos: int, head_dim: int, dtype: torch.dtype, device: str):
    half = head_dim // 2
    pos = torch.arange(max_pos, device=device, dtype=torch.float32)[:, None]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    freqs = pos * inv_freq[None, :]
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    interleaved: bool,
):
    head_dim = x.shape[-1]
    cos_pos = cos[positions.long()].unsqueeze(-2).to(x.dtype)
    sin_pos = sin[positions.long()].unsqueeze(-2).to(x.dtype)
    if interleaved:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x_even * cos_pos - x_odd * sin_pos
        out[..., 1::2] = x_odd * cos_pos + x_even * sin_pos
        return out
    x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2 :]
    return torch.cat([x1 * cos_pos - x2 * sin_pos, x2 * cos_pos + x1 * sin_pos], dim=-1)


def _dense_to_paged(cache: torch.Tensor, block_table: torch.Tensor, block_size: int, num_blocks: int):
    bsz, heads, seqlen, head_dim = cache.shape
    paged = torch.empty(num_blocks, heads, block_size, head_dim, device=cache.device, dtype=cache.dtype)
    paged.zero_()
    for b in range(bsz):
        for tok in range(seqlen):
            logical_block = tok // block_size
            block_off = tok % block_size
            phys_block = int(block_table[b, logical_block])
            paged[phys_block, :, block_off, :] = cache[b, :, tok, :]
    return paged


def _dense_to_paged_upstream(cache: torch.Tensor, block_table: torch.Tensor, block_size: int, num_blocks: int):
    bsz, seqlen, heads, head_dim = cache.shape
    paged = torch.empty(num_blocks, block_size, heads, head_dim, device=cache.device, dtype=cache.dtype)
    paged.zero_()
    for b in range(bsz):
        for tok in range(seqlen):
            logical_block = tok // block_size
            block_off = tok % block_size
            phys_block = int(block_table[b, logical_block])
            paged[phys_block, block_off, :, :] = cache[b, tok, :, :]
    return paged


def _paged_to_dense(cache: torch.Tensor, block_table: torch.Tensor, seqlen: int):
    bsz, max_blocks = block_table.shape
    _, heads, block_size, head_dim = cache.shape
    dense = torch.empty(bsz, heads, seqlen, head_dim, device=cache.device, dtype=cache.dtype)
    for b in range(bsz):
        for tok in range(seqlen):
            logical_block = tok // block_size
            assert logical_block < max_blocks
            block_off = tok % block_size
            phys_block = int(block_table[b, logical_block])
            dense[b, :, tok, :] = cache[phys_block, :, block_off, :]
    return dense


def _paged_to_dense_upstream(cache: torch.Tensor, block_table: torch.Tensor, seqlen: int):
    bsz, max_blocks = block_table.shape
    _, block_size, heads, head_dim = cache.shape
    dense = torch.empty(bsz, seqlen, heads, head_dim, device=cache.device, dtype=cache.dtype)
    for b in range(bsz):
        for tok in range(seqlen):
            logical_block = tok // block_size
            assert logical_block < max_blocks
            block_off = tok % block_size
            phys_block = int(block_table[b, logical_block])
            dense[b, tok, :, :] = cache[phys_block, block_off, :, :]
    return dense


def _attention_ref(q_used, dense_k_bshd, dense_v_bshd, visible_lens, *, causal, window_size):
    bsz, seqlen_q, num_heads, head_dim = q_used.shape
    num_kv_heads = dense_k_bshd.shape[2]
    repeat = num_heads // num_kv_heads
    k_rep = dense_k_bshd.repeat_interleave(repeat, dim=2).float()
    v_rep = dense_v_bshd.repeat_interleave(repeat, dim=2).float()
    out = torch.empty(bsz, seqlen_q, num_heads, head_dim, device=q_used.device, dtype=q_used.dtype)
    scale = 1.0 / math.sqrt(head_dim)
    left, right = window_size
    if causal:
        right = 0

    for b in range(bsz):
        cache_len = int(visible_lens[b])
        for s in range(seqlen_q):
            q_pos = cache_len - seqlen_q + s
            keep = torch.arange(cache_len, device=q_used.device)
            mask = torch.ones(cache_len, device=q_used.device, dtype=torch.bool)
            if causal:
                mask &= keep <= q_pos
            if left >= 0:
                mask &= keep >= q_pos - left
            if right >= 0:
                mask &= keep <= q_pos + right
            for h in range(num_heads):
                scores = torch.matmul(k_rep[b, :cache_len, h], q_used[b, s, h].float()) * scale
                scores = scores.masked_fill(~mask, float("-inf"))
                probs = torch.softmax(scores, dim=0)
                if torch.isfinite(scores).any():
                    out[b, s, h] = torch.matmul(probs, v_rep[b, :cache_len, h]).to(q_used.dtype)
                else:
                    out[b, s, h].zero_()
    return out


def _reference(qkv, k_cache, v_cache, cache_seqlens, cos, sin, *, causal, window_size, block_table=None):
    bsz, seqlen_q, packed_heads, head_dim = qkv.shape
    num_kv_heads = k_cache.shape[1]
    num_heads = packed_heads - 2 * num_kv_heads
    q, k_new, v_new = qkv.split([num_heads, num_kv_heads, num_kv_heads], dim=2)

    if block_table is not None:
        max_seqlen = block_table.shape[1] * k_cache.shape[2]
        dense_k = _paged_to_dense(k_cache, block_table, max_seqlen)
        dense_v = _paged_to_dense(v_cache, block_table, max_seqlen)
    else:
        dense_k = k_cache.clone()
        dense_v = v_cache.clone()

    positions = cache_seqlens[:, None] - seqlen_q + torch.arange(seqlen_q, device=qkv.device)[None, :]
    q_rot = _apply_rope(q, cos, sin, positions, interleaved=False)
    k_rot = _apply_rope(k_new, cos, sin, positions, interleaved=False)

    for b in range(bsz):
        for s in range(seqlen_q):
            pos = int(cache_seqlens[b] - 1) if seqlen_q == 1 else s
            dense_k[b, :, pos, :] = k_rot[b, s]
            dense_v[b, :, pos, :] = v_new[b, s]

    out = _attention_ref(q_rot, dense_k.transpose(1, 2), dense_v.transpose(1, 2), cache_seqlens, causal=causal, window_size=window_size)
    return out, dense_k, dense_v


def _reference_upstream(
    q,
    k_cache,
    v_cache,
    cache_seqlens,
    cos,
    sin,
    *,
    k=None,
    v=None,
    causal,
    window_size,
    block_table=None,
    rotary_interleaved=True,
):
    if block_table is not None:
        max_seqlen = block_table.shape[1] * k_cache.shape[1]
        dense_k = _paged_to_dense_upstream(k_cache, block_table, max_seqlen)
        dense_v = _paged_to_dense_upstream(v_cache, block_table, max_seqlen)
    else:
        dense_k = k_cache.clone()
        dense_v = v_cache.clone()

    if k is None:
        q_used = q
        visible_lens = cache_seqlens
    else:
        positions = cache_seqlens[:, None] + torch.arange(k.shape[1], device=q.device)[None, :]
        q_used = _apply_rope(q, cos, sin, positions, interleaved=rotary_interleaved)
        k_rot = _apply_rope(k, cos, sin, positions, interleaved=rotary_interleaved)
        for b in range(q.shape[0]):
            for s in range(k.shape[1]):
                pos = int(cache_seqlens[b] + s)
                dense_k[b, pos] = k_rot[b, s]
                dense_v[b, pos] = v[b, s]
        visible_lens = cache_seqlens + k.shape[1]

    out = _attention_ref(q_used, dense_k, dense_v, visible_lens, causal=causal, window_size=window_size)
    return out, dense_k, dense_v


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("seqlen_q", [1, 3])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
def test_flash_attn_with_kvcache_flydsl(paged: bool, seqlen_q: int, dtype: torch.dtype):
    torch.manual_seed(1234 + seqlen_q + int(paged))
    device = "cuda"
    bsz, num_heads, num_kv_heads, head_dim = 2, 4, 2, 16
    max_seqlen = 8
    block_size = 16
    max_blocks = 1
    cache_seqlens = torch.tensor([5, 7], device=device, dtype=torch.int32)
    if seqlen_q > 1:
        cache_seqlens = torch.tensor([seqlen_q, seqlen_q], device=device, dtype=torch.int32)

    qkv = torch.empty(
        bsz, seqlen_q, num_heads + 2 * num_kv_heads, head_dim, device=device, dtype=dtype
    ).uniform_(-1, 1)
    k_dense = torch.empty(bsz, num_kv_heads, max_seqlen, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    v_dense = torch.empty_like(k_dense).uniform_(-1, 1)
    cos, sin = _make_rotary_tables(max_seqlen + 8, head_dim, dtype, device)

    block_table = None
    k_arg, v_arg = k_dense.clone(), v_dense.clone()
    if paged:
        block_table = torch.arange(bsz * max_blocks, device=device, dtype=torch.int32).reshape(bsz, max_blocks)
        num_blocks = bsz * max_blocks
        k_arg = _dense_to_paged(k_dense, block_table, block_size, num_blocks)
        v_arg = _dense_to_paged(v_dense, block_table, block_size, num_blocks)

    ref, ref_k, ref_v = _reference(
        qkv,
        k_arg.clone(),
        v_arg.clone(),
        cache_seqlens,
        cos,
        sin,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
    )
    out = flash_attn_with_kvcache_flydsl(
        qkv,
        k_arg,
        v_arg,
        rotary_cos=cos,
        rotary_sin=sin,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        causal=True,
        rotary_interleaved=False,
        is_qkv_packed=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)
    if paged:
        got_k = _paged_to_dense(k_arg, block_table, ref_k.shape[2])
        got_v = _paged_to_dense(v_arg, block_table, ref_v.shape[2])
    else:
        got_k, got_v = k_arg, v_arg
    torch.testing.assert_close(got_k.float(), ref_k.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(got_v.float(), ref_v.float(), atol=3e-2, rtol=3e-2)


def test_flash_attn_with_kvcache_flydsl_split_kv_decode():
    torch.manual_seed(2468)
    device = "cuda"
    dtype = torch.bfloat16
    bsz, seqlen_q, num_heads, num_kv_heads, head_dim = 1, 1, 2, 1, 16
    max_seqlen = 128
    cache_seqlens = torch.tensor([97], device=device, dtype=torch.int32)

    qkv = torch.empty(
        bsz, seqlen_q, num_heads + 2 * num_kv_heads, head_dim, device=device, dtype=dtype
    ).uniform_(-1, 1)
    k_cache = torch.empty(bsz, num_kv_heads, max_seqlen, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    v_cache = torch.empty_like(k_cache).uniform_(-1, 1)
    cos, sin = _make_rotary_tables(max_seqlen + 8, head_dim, dtype, device)

    ref, ref_k, ref_v = _reference(
        qkv,
        k_cache.clone(),
        v_cache.clone(),
        cache_seqlens,
        cos,
        sin,
        causal=True,
        window_size=(-1, -1),
    )
    out = flash_attn_with_kvcache_flydsl(
        qkv,
        k_cache,
        v_cache,
        rotary_cos=cos,
        rotary_sin=sin,
        cache_seqlens=cache_seqlens,
        causal=True,
        rotary_interleaved=False,
        is_qkv_packed=True,
        num_splits=4,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(k_cache.float(), ref_k.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(v_cache.float(), ref_v.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 2), (4, 4), (8, 1)])
def test_flash_attn_with_kvcache_flydsl_mma_decode(paged: bool, num_heads: int, num_kv_heads: int):
    """HEAD_DIM=128 bf16 decode path that dispatches to the MMA decode kernel."""
    torch.manual_seed(7 + num_heads + int(paged))
    device = "cuda"
    dtype = torch.bfloat16
    bsz, seqlen_q, head_dim = 2, 1, 128
    max_seqlen = 128
    # Use a single 128-wide page so the (separately) validated cache-update path
    # stays on its block-0 fast path; the MMA multi-block paged gather is covered
    # by test_flash_attn_with_kvcache_flydsl_mma_decode_paged_multiblock.
    block_size = 128
    max_blocks = max_seqlen // block_size
    cache_seqlens = torch.tensor([128, 100], device=device, dtype=torch.int32)

    qkv = torch.empty(
        bsz, seqlen_q, num_heads + 2 * num_kv_heads, head_dim, device=device, dtype=dtype
    ).uniform_(-1, 1)
    k_dense = torch.empty(bsz, num_kv_heads, max_seqlen, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    v_dense = torch.empty_like(k_dense).uniform_(-1, 1)
    cos, sin = _make_rotary_tables(max_seqlen + 8, head_dim, dtype, device)

    block_table = None
    k_arg, v_arg = k_dense.clone(), v_dense.clone()
    if paged:
        block_table = torch.arange(bsz * max_blocks, device=device, dtype=torch.int32).reshape(bsz, max_blocks)
        num_blocks = bsz * max_blocks
        k_arg = _dense_to_paged(k_dense, block_table, block_size, num_blocks)
        v_arg = _dense_to_paged(v_dense, block_table, block_size, num_blocks)

    ref, ref_k, ref_v = _reference(
        qkv,
        k_arg.clone(),
        v_arg.clone(),
        cache_seqlens,
        cos,
        sin,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
    )
    out = flash_attn_with_kvcache_flydsl(
        qkv,
        k_arg,
        v_arg,
        rotary_cos=cos,
        rotary_sin=sin,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        causal=True,
        rotary_interleaved=False,
        is_qkv_packed=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)
    if paged:
        got_k = _paged_to_dense(k_arg, block_table, ref_k.shape[2])
        got_v = _paged_to_dense(v_arg, block_table, ref_v.shape[2])
    else:
        got_k, got_v = k_arg, v_arg
    torch.testing.assert_close(got_k.float(), ref_k.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(got_v.float(), ref_v.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 2), (4, 4)])
def test_flash_attn_with_kvcache_flydsl_mma_decode_paged_multiblock(num_heads: int, num_kv_heads: int):
    """Attention-only (no cache update) multi-block paged MMA decode read."""
    torch.manual_seed(99 + num_heads)
    device = "cuda"
    dtype = torch.bfloat16
    bsz, seqlen_q, head_dim = 2, 1, 128
    max_seqlen = 256
    block_size = 16
    max_blocks = max_seqlen // block_size
    cache_seqlens = torch.tensor([256, 200], device=device, dtype=torch.int32)

    # upstream attention-only path: separate q, no k/v -> update_cache disabled.
    q = torch.empty(bsz, seqlen_q, num_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    k_dense = torch.empty(bsz, max_seqlen, num_kv_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    v_dense = torch.empty_like(k_dense).uniform_(-1, 1)
    cos, sin = _make_rotary_tables(max_seqlen + 8, head_dim, dtype, device)

    block_table = torch.arange(bsz * max_blocks, device=device, dtype=torch.int32).reshape(bsz, max_blocks)
    num_blocks = bsz * max_blocks
    k_arg = _dense_to_paged_upstream(k_dense, block_table, block_size, num_blocks)
    v_arg = _dense_to_paged_upstream(v_dense, block_table, block_size, num_blocks)

    ref, _, _ = _reference_upstream(
        q,
        k_arg.clone(),
        v_arg.clone(),
        cache_seqlens,
        cos,
        sin,
        k=None,
        v=None,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
    )
    out = flash_attn_with_kvcache_flydsl(
        q,
        k_arg,
        v_arg,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        causal=True,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("num_splits", [2, 3, 5, 9, 15, 64, 114])
def test_flash_attn_with_kvcache_flydsl_v5_hnd_uneven_group(num_splits: int):
    """V5 HND fast path with ixinfer-aligned split-group reductions."""
    torch.manual_seed(211)
    device = "cuda"
    dtype = torch.bfloat16
    bsz, seqlen_q, num_heads, num_kv_heads, head_dim = 1, 1, 16, 2, 128
    max_seqlen = 2048
    block_size = 16
    cache_seqlens = torch.tensor([2001], device=device, dtype=torch.int32)

    q = torch.empty(
        bsz, seqlen_q, num_heads, head_dim, device=device, dtype=dtype
    ).uniform_(-1, 1)
    k_dense = torch.empty(
        bsz, num_kv_heads, max_seqlen, head_dim, device=device, dtype=dtype
    ).uniform_(-1, 1)
    v_dense = torch.empty_like(k_dense).uniform_(-1, 1)
    max_blocks = max_seqlen // block_size
    block_table = torch.arange(
        bsz * max_blocks, device=device, dtype=torch.int32
    ).reshape(bsz, max_blocks)
    k_cache = _dense_to_paged(
        k_dense, block_table, block_size, bsz * max_blocks)
    v_cache = _dense_to_paged(
        v_dense, block_table, block_size, bsz * max_blocks)

    ref = _attention_ref(
        q,
        k_dense.transpose(1, 2),
        v_dense.transpose(1, 2),
        cache_seqlens,
        causal=True,
        window_size=(-1, -1),
    )
    out = flash_attn_with_kvcache_flydsl(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        causal=True,
        force_upstream_cache_layout=False,
        num_splits=num_splits,
    )
    torch.cuda.synchronize()
    # Two wide splits accumulate many more BF16 values per group than the V5
    # operating point, so retain the established BF16 relative tolerance while
    # allowing its observed absolute rounding envelope.
    atol = 5e-2 if num_splits < 16 else 3e-2
    torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=3e-2)


def test_mma_decode_num_splits_qwen3_shape():
    from kernels.attention.iluvatar.mma_decode_splits import (
        compute_mma_decode_num_splits,
        compute_v5_decode_config,
    )

    assert compute_v5_decode_config(
        batch_size=1, seqlen_q=1, num_heads=16, num_kv_heads=2,
        head_dim=128, max_seqlen_k=32768,
    ) == (64, 512, 1)
    assert compute_mma_decode_num_splits(
        batch_size=1, seqlen_q=1, num_heads=16, num_kv_heads=8,
        head_dim=128, max_seqlen_k=512, block_n=32,
    ) == 16
    assert compute_mma_decode_num_splits(
        batch_size=128, seqlen_q=1, num_heads=8, num_kv_heads=2,
        head_dim=128, max_seqlen_k=8192, block_n=32,
    ) == 1
    assert compute_mma_decode_num_splits(
        batch_size=1, seqlen_q=1, num_heads=16, num_kv_heads=2,
        head_dim=128, max_seqlen_k=32768, block_n=32,
    ) == 32


@pytest.mark.parametrize("num_splits", [4, 8])
def test_flash_attn_with_kvcache_flydsl_mma_decode_split(num_splits: int):
    """Flash-decoding (split-KV) MMA decode path + reduce kernel."""
    torch.manual_seed(31 + num_splits)
    device = "cuda"
    dtype = torch.bfloat16
    bsz, seqlen_q, num_heads, num_kv_heads, head_dim = 2, 1, 8, 2, 128
    max_seqlen = 512
    block_size = 16
    max_blocks = max_seqlen // block_size
    cache_seqlens = torch.tensor([512, 401], device=device, dtype=torch.int32)

    q = torch.empty(bsz, seqlen_q, num_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    k_dense = torch.empty(bsz, max_seqlen, num_kv_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    v_dense = torch.empty_like(k_dense).uniform_(-1, 1)
    cos, sin = _make_rotary_tables(max_seqlen + 8, head_dim, dtype, device)

    block_table = torch.arange(bsz * max_blocks, device=device, dtype=torch.int32).reshape(bsz, max_blocks)
    num_blocks = bsz * max_blocks
    k_arg = _dense_to_paged_upstream(k_dense, block_table, block_size, num_blocks)
    v_arg = _dense_to_paged_upstream(v_dense, block_table, block_size, num_blocks)

    ref, _, _ = _reference_upstream(
        q, k_arg.clone(), v_arg.clone(), cache_seqlens, cos, sin,
        k=None, v=None, causal=True, window_size=(-1, -1), block_table=block_table,
    )
    out = flash_attn_with_kvcache_flydsl(
        q, k_arg, v_arg,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        causal=True,
        num_splits=num_splits,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("with_update", [False, True])
def test_flash_attn_with_kvcache_flydsl_upstream_api(paged: bool, with_update: bool):
    torch.manual_seed(4321 + int(paged) + 10 * int(with_update))
    device = "cuda"
    dtype = torch.bfloat16
    bsz, seqlen_q, num_heads, num_kv_heads, head_dim = 2, 1, 4, 2, 16
    max_seqlen = 8
    block_size = 16
    max_blocks = 1
    cache_seqlens = torch.tensor([4, 6], device=device, dtype=torch.int32)

    q = torch.empty(bsz, seqlen_q, num_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    k = torch.empty(bsz, seqlen_q, num_kv_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    v = torch.empty_like(k).uniform_(-1, 1)
    k_dense = torch.empty(bsz, max_seqlen, num_kv_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    v_dense = torch.empty_like(k_dense).uniform_(-1, 1)
    cos, sin = _make_rotary_tables(max_seqlen + 8, head_dim, dtype, device)

    block_table = None
    k_arg, v_arg = k_dense.clone(), v_dense.clone()
    if paged:
        block_table = torch.arange(bsz * max_blocks, device=device, dtype=torch.int32).reshape(bsz, max_blocks)
        num_blocks = bsz * max_blocks
        k_arg = _dense_to_paged_upstream(k_dense, block_table, block_size, num_blocks)
        v_arg = _dense_to_paged_upstream(v_dense, block_table, block_size, num_blocks)

    ref, ref_k, ref_v = _reference_upstream(
        q,
        k_arg.clone(),
        v_arg.clone(),
        cache_seqlens,
        cos,
        sin,
        k=k if with_update else None,
        v=v if with_update else None,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        rotary_interleaved=True,
    )
    out = flash_attn_with_kvcache_flydsl(
        q,
        k_arg,
        v_arg,
        k=k if with_update else None,
        v=v if with_update else None,
        rotary_cos=cos if with_update else None,
        rotary_sin=sin if with_update else None,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        causal=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)
    if paged:
        got_k = _paged_to_dense_upstream(k_arg, block_table, ref_k.shape[1])
        got_v = _paged_to_dense_upstream(v_arg, block_table, ref_v.shape[1])
    else:
        got_k, got_v = k_arg, v_arg
    torch.testing.assert_close(got_k.float(), ref_k.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(got_v.float(), ref_v.float(), atol=3e-2, rtol=3e-2)

