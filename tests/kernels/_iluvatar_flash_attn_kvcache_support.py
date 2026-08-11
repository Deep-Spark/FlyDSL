#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Shared references and fixtures for KV-cache FlashAttention tests."""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.l2_device]

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if not torch.cuda.is_available():
    pytest.skip("CUDA-compatible device is not available", allow_module_level=True)
if os.environ.get("FLYDSL_COMPILE_BACKEND") != "iluvatar" and "Iluvatar" not in torch.cuda.get_device_name():
    pytest.skip("Iluvatar device test", allow_module_level=True)

import flydsl.expr as fx  # noqa: E402, F401
import kernels.attention.iluvatar.flash_attn_kvcache as flash_attn_kvcache_module  # noqa: E402, F401
from kernels.attention.iluvatar.flash_attn_kvcache import flash_attn_with_kvcache  # noqa: E402, F401
from kernels.attention.iluvatar.flash_attn_kvcache_planner import clear_kvcache_caches  # noqa: E402


@pytest.fixture(autouse=True)
def _configure_iluvatar_env(monkeypatch):
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)
    clear_kvcache_caches()
    yield
    clear_kvcache_caches()


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


def _attention_ref(
    q_used, dense_k_bshd, dense_v_bshd, visible_lens, *, causal, window_size, softmax_scale=None, softcap=0.0
):
    bsz, seqlen_q, num_heads, head_dim = q_used.shape
    num_kv_heads = dense_k_bshd.shape[2]
    repeat = num_heads // num_kv_heads
    k_rep = dense_k_bshd.repeat_interleave(repeat, dim=2).float()
    v_rep = dense_v_bshd.repeat_interleave(repeat, dim=2).float()
    out = torch.empty(bsz, seqlen_q, num_heads, head_dim, device=q_used.device, dtype=q_used.dtype)
    scale = 1.0 / math.sqrt(head_dim) if softmax_scale is None else softmax_scale
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
                if softcap > 0.0:
                    scores = softcap * torch.tanh(scores / softcap)
                scores = scores.masked_fill(~mask, float("-inf"))
                probs = torch.softmax(scores, dim=0)
                if torch.isfinite(scores).any():
                    out[b, s, h] = torch.matmul(probs, v_rep[b, :cache_len, h]).to(q_used.dtype)
                else:
                    out[b, s, h].zero_()
    return out


def _reference(
    qkv,
    k_cache,
    v_cache,
    cache_seqlens,
    cos,
    sin,
    *,
    causal,
    window_size,
    block_table=None,
    rotary_interleaved=False,
):
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
    q_rot = _apply_rope(q, cos, sin, positions, interleaved=rotary_interleaved)
    k_rot = _apply_rope(k_new, cos, sin, positions, interleaved=rotary_interleaved)

    for b in range(bsz):
        for s in range(seqlen_q):
            pos = int(cache_seqlens[b] - 1) if seqlen_q == 1 else s
            dense_k[b, :, pos, :] = k_rot[b, s]
            dense_v[b, :, pos, :] = v_new[b, s]

    out = _attention_ref(
        q_rot, dense_k.transpose(1, 2), dense_v.transpose(1, 2), cache_seqlens, causal=causal, window_size=window_size
    )
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
        token_offsets = torch.arange(k.shape[1], device=q.device)[None, :]
        k_positions = cache_seqlens[:, None] + token_offsets
        q_positions = (
            k_positions if causal or window_size != (-1, -1) else cache_seqlens[:, None].expand_as(k_positions)
        )
        q_used = _apply_rope(q, cos, sin, q_positions, interleaved=rotary_interleaved)
        k_rot = _apply_rope(k, cos, sin, k_positions, interleaved=rotary_interleaved)
        for b in range(q.shape[0]):
            for s in range(k.shape[1]):
                pos = int(cache_seqlens[b] + s)
                dense_k[b, pos] = k_rot[b, s]
                dense_v[b, pos] = v[b, s]
        visible_lens = cache_seqlens + k.shape[1]

    out = _attention_ref(q_used, dense_k, dense_v, visible_lens, causal=causal, window_size=window_size)
    return out, dense_k, dense_v
