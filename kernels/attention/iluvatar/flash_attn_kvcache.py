# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Correctness-first FlyDSL ``flash_attn_with_kvcache`` implementation.

This module mirrors the MR flash-attention KV-cache contract closely enough for
native FlyDSL testing:

* upstream input:   q [B, S_q, H_q, D], optional k/v [B, S_new, H_kv, D]
* upstream cache:   dense [B, S_cache, H_kv, D] or paged [num_blocks, page, H_kv, D]
* MR input:         packed QKV [B, S_q, H_q + 2 * H_kv, D]
* MR cache:         dense [B, H_kv, S_cache, D] or paged [num_blocks, H_kv, page, D]
* cache_seqlens follows upstream when k/v are separate (old lengths) and MR when packed
  (visible lengths after the packed K/V tokens are present).

The kernels are intentionally simple.  A first launch rotates/copies Q and
updates K/V cache; a second launch computes one attention row per CTA.
"""

import functools
import math
import os
from typing import Optional, Tuple

os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "iluvatar")
os.environ.setdefault("FLYDSL_RUNTIME_KIND", "iluvatar")
os.environ.setdefault("ARCH", "ivcore11")

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import scf
from flydsl.expr import arith, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from flydsl.expr.utils.arith import _to_raw
from flydsl.expr.vector import full

from kernels.attention.iluvatar.flash_attn_kvcache_mma import (
    build_mma_decode_attention_kernel,
)

ATTN_THREADS = 256
WARP_SIZE = 64
RED_SLOTS = ATTN_THREADS // WARP_SIZE
_LOG2E = 1.4426950408889634
_COMPILED_LAUNCH_CACHE: dict = {}


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.bfloat16:
        return "bf16"
    if dtype is torch.float16:
        return "f16"
    raise TypeError(f"flash_attn_with_kvcache_flydsl only supports bf16/f16, got {dtype}")


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _normalise_window(causal: bool, window_size: Tuple[int, int], max_seqlen: int) -> Tuple[int, int]:
    left, right = window_size
    if causal:
        right = 0
    if left >= max_seqlen:
        left = -1
    if right >= max_seqlen:
        right = -1
    return left, right


def _apply_interleaved_rotary_torch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, positions: torch.Tensor):
    cos_pos = cos[positions.long()].unsqueeze(-2).to(dtype=x.dtype, device=x.device)
    sin_pos = sin[positions.long()].unsqueeze(-2).to(dtype=x.dtype, device=x.device)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos_pos - x_odd * sin_pos
    out[..., 1::2] = x_odd * cos_pos + x_even * sin_pos
    return out


@functools.lru_cache(maxsize=None)
def build_flash_attn_with_kvcache_module(
    *,
    batch_size: int,
    seqlen_q: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seqlen_k: int,
    dtype_str: str = "bf16",
    paged: bool = False,
    page_block_size: int = 16,
    has_rotary: bool = False,
    rotary_cols: int = 0,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    rotary_interleaved: bool = True,
    update_cache: bool = True,
    num_splits: int = 1,
    upstream_cache_layout: bool = False,
    use_mma_decode: bool = False,
    use_v5_decode: bool = False,
    mma_block_n: int = 128,
):
    if dtype_str not in ("bf16", "f16"):
        raise ValueError(f"dtype_str must be 'bf16' or 'f16', got {dtype_str!r}")
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    if paged and (page_block_size % 16 != 0 or page_block_size > 256):
        raise ValueError("paged cache block size must be divisible by 16 and <= 256")
    if has_rotary and rotary_cols not in (head_dim, head_dim // 2):
        raise ValueError("rotary_cos/sin last dimension must be head_dim or head_dim // 2")
    if num_splits < 1:
        raise ValueError("num_splits must be >= 1")

    repeats = num_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)
    red_slots = RED_SLOTS
    reduce_threads = max(WARP_SIZE, min(ATTN_THREADS, 1 << (head_dim - 1).bit_length()))
    window_left, window_right = _normalise_window(causal, window_size, max_seqlen_k)
    elem_dtype = fx.BFloat16 if dtype_str == "bf16" else fx.Float16
    max_blocks_per_seq = max(1, _ceil_div(max_seqlen_k, page_block_size))
    split_chunk = _ceil_div(max_seqlen_k, num_splits)
    prob_slots = split_chunk
    use_vec2_qk = head_dim % 2 == 0 and head_dim <= WARP_SIZE * 2 and max_seqlen_k >= 2048
    use_vec2_pv = (
        upstream_cache_layout
        and paged
        and 4 <= max_blocks_per_seq <= 16
        and head_dim % 2 == 0
        and head_dim <= WARP_SIZE * 2
    )
    k_repeats = max(1, head_dim // WARP_SIZE)
    use_krepeat_pv = num_splits > 1 and head_dim % WARP_SIZE == 0 and head_dim <= WARP_SIZE * 4
    use_block_table_cache = paged and 1 < max_blocks_per_seq <= 4

    @fx.struct
    class SharedStorage:
        s_red: fx.Array[fx.Float32, RED_SLOTS, 16]
        s_prob: fx.Array[fx.Float32, prob_slots]
        s_q: fx.Array[fx.Float32, head_dim]
        s_blocks: fx.Array[fx.Int32, max_blocks_per_seq]

    # Match ixinfer's D=128 reducer dispatch: use the smallest power-of-two
    # warp count that covers the split groups, capped at 16.
    v5_reduce_warps = min(16, 1 << (num_splits - 1).bit_length())
    v5_reduce_slots = max(1, v5_reduce_warps // 2)

    @fx.struct
    class V5ReduceStorage:
        max_values: fx.Array[fx.Float32, v5_reduce_slots]
        exp_sums: fx.Array[fx.Float32, v5_reduce_slots]
        partials: fx.Array[fx.Float32, v5_reduce_slots * head_dim]

    @flyc.kernel
    def update_cache_kernel(
        QKV: fx.Tensor,
        QWork: fx.Tensor,
        KCache: fx.Tensor,
        VCache: fx.Tensor,
        CacheSeqLens: fx.Tensor,
        BlockTable: fx.Tensor,
        RotaryCos: fx.Tensor,
        RotarySin: fx.Tensor,
    ):
        head = fx.block_idx.x
        sq = fx.Int32(fx.block_idx.y)
        b = fx.Int32(fx.block_idx.z)
        tid = fx.thread_idx.x
        copy_atom = fx.make_copy_atom(fx.UniversalCopy16b(), elem_dtype)

        def _load_elem(tensor):
            r = fx.make_rmem_tensor(1, elem_dtype)
            fx.copy_atom_call(copy_atom, tensor, r)
            return fx.memref_load_vec(r)[0]

        copy_atom_i32 = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)

        def _load_i32(tensor):
            r = fx.make_rmem_tensor(1, fx.Int32)
            fx.copy_atom_call(copy_atom_i32, tensor, r)
            return fx.memref_load_vec(r)[0]

        def _store_elem(tensor, value):
            if not hasattr(value, "to"):
                value = fx.Float32(value)
            r = fx.make_rmem_tensor(1, elem_dtype)
            fx.memref_store_vec(full(1, value.to(elem_dtype), elem_dtype), r)
            fx.copy_atom_call(copy_atom, r, tensor)

        def _rotary_index(d):
            if rotary_cols == head_dim:
                return d
            if fx.const_expr(rotary_interleaved):
                return d // fx.Int32(2)
            return d % fx.Int32(head_dim // 2)

        def _load_cos_sin(pos, d):
            cos_row = fx.slice(RotaryCos, (pos, None))
            sin_row = fx.slice(RotarySin, (pos, None))
            cos_div = fx.logical_divide(cos_row, fx.make_layout(1, 1))
            sin_div = fx.logical_divide(sin_row, fx.make_layout(1, 1))
            ridx = _rotary_index(d)
            cos_v = _load_elem(fx.slice(cos_div, (None, ridx))).to(fx.Float32)
            sin_v = _load_elem(fx.slice(sin_div, (None, ridx))).to(fx.Float32)
            return cos_v, sin_v

        def _qkv_scalar(row_head, d):
            row = fx.slice(QKV, (b, sq, row_head, None))
            row_div = fx.logical_divide(row, fx.make_layout(1, 1))
            return _load_elem(fx.slice(row_div, (None, d))).to(fx.Float32)

        def _load_cache_len():
            seq_div = fx.logical_divide(CacheSeqLens, fx.make_layout(1, 1))
            return _load_i32(fx.slice(seq_div, (None, b)))

        def _store_q(head_q, d, value):
            row = fx.slice(QWork, (b, sq, head_q, None))
            row_div = fx.logical_divide(row, fx.make_layout(1, 1))
            _store_elem(fx.slice(row_div, (None, d)), value)

        def _cache_indices(pos):
            if paged:
                block = pos // fx.Int32(page_block_size)
                block_off = pos % fx.Int32(page_block_size)
                table_row = fx.slice(BlockTable, (b, None))
                table_div = fx.logical_divide(table_row, fx.make_layout(1, 1))
                phys_block = _load_i32(fx.slice(table_div, (None, block)))
                return phys_block, block_off
            return b, pos

        def _store_kv(kv_head, d, k_value, v_value):
            cache_len = _load_cache_len()
            pos = cache_len - fx.Int32(seqlen_q) + sq
            outer, token = _cache_indices(pos)
            if fx.const_expr(upstream_cache_layout):
                k_row = fx.slice(KCache, (outer, token, kv_head, None))
                v_row = fx.slice(VCache, (outer, token, kv_head, None))
            else:
                k_row = fx.slice(KCache, (outer, kv_head, token, None))
                v_row = fx.slice(VCache, (outer, kv_head, token, None))
            k_div = fx.logical_divide(k_row, fx.make_layout(1, 1))
            v_div = fx.logical_divide(v_row, fx.make_layout(1, 1))
            _store_elem(fx.slice(k_div, (None, d)), k_value)
            _store_elem(fx.slice(v_div, (None, d)), v_value)

        if tid < head_dim:
            d = fx.Int32(tid)
            half = fx.Int32(head_dim // 2)
            if fx.const_expr(rotary_interleaved):
                is_first = (d % fx.Int32(2)) == fx.Int32(0)
                pair_d = is_first.select(d + fx.Int32(1), d - fx.Int32(1))
            else:
                is_first = d < half
                pair_d = is_first.select(d + half, d - half)
            if head < num_heads:
                q_val = _qkv_scalar(head, d)
                if has_rotary:
                    cache_len = _load_cache_len()
                    pos = cache_len - fx.Int32(seqlen_q) + sq
                    pair_val = _qkv_scalar(head, pair_d)
                    cos_v, sin_v = _load_cos_sin(pos, d)
                    q_val = q_val * cos_v + is_first.select(-pair_val * sin_v, pair_val * sin_v)
                _store_q(head, d, q_val)
            if head < num_kv_heads:
                k_head = fx.Int32(num_heads) + head
                v_head = fx.Int32(num_heads + num_kv_heads) + head
                k_val = _qkv_scalar(k_head, d)
                v_val = _qkv_scalar(v_head, d)
                if has_rotary:
                    cache_len = _load_cache_len()
                    pos = cache_len - fx.Int32(seqlen_q) + sq
                    k_pair = _qkv_scalar(k_head, pair_d)
                    cos_v, sin_v = _load_cos_sin(pos, d)
                    k_val = k_val * cos_v + is_first.select(-k_pair * sin_v, k_pair * sin_v)
                _store_kv(head, d, k_val, v_val)

    @flyc.kernel(known_block_size=[ATTN_THREADS, 1, 1])
    def attention_kernel(
        QWork: fx.Tensor,
        KCache: fx.Tensor,
        VCache: fx.Tensor,
        CacheSeqLens: fx.Tensor,
        BlockTable: fx.Tensor,
        Out: fx.Tensor,
    ):
        row = fx.Int32(fx.block_idx.x)
        b = fx.Int32(fx.block_idx.y)
        hq = fx.Int32(fx.block_idx.z)
        tid = fx.thread_idx.x
        sq = row
        kv_head = hq // fx.Int32(repeats)
        copy_atom = fx.make_copy_atom(fx.UniversalCopy16b(), elem_dtype)
        fm_fast = arith.FastMathFlags.fast
        zero_raw = arith.constant(0.0, type=T.f32)
        neg_inf_raw = arith.constant(float("-inf"), type=T.f32)
        zero = fx.Float32(zero_raw)
        neg_inf = fx.Float32(neg_inf_raw)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_red = lds.s_red.view(fx.make_layout(red_slots, 1))
        s_prob = lds.s_prob.view(fx.make_layout(prob_slots, 1))
        s_q = lds.s_q.view(fx.make_layout(head_dim, 1))
        s_blocks = lds.s_blocks.view(fx.make_layout(max_blocks_per_seq, 1))

        def _load_elem(tensor):
            r = fx.make_rmem_tensor(1, elem_dtype)
            fx.copy_atom_call(copy_atom, tensor, r)
            return fx.memref_load_vec(r)[0]

        copy_atom_i32 = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
        copy_atom_vec2 = fx.make_copy_atom(fx.UniversalCopy32b(), elem_dtype)

        def _load_i32(tensor):
            r = fx.make_rmem_tensor(1, fx.Int32)
            fx.copy_atom_call(copy_atom_i32, tensor, r)
            return fx.memref_load_vec(r)[0]

        if fx.const_expr(use_block_table_cache):
            if tid < max_blocks_per_seq:
                table_row = fx.slice(BlockTable, (b, None))
                table_div = fx.logical_divide(table_row, fx.make_layout(1, 1))
                phys_block = _load_i32(fx.slice(table_div, (None, tid)))
                fx.memref_store(phys_block, s_blocks, tid)
            gpu.barrier()

        def _store_elem(tensor, value):
            if not hasattr(value, "to"):
                value = fx.Float32(value)
            r = fx.make_rmem_tensor(1, elem_dtype)
            fx.memref_store_vec(full(1, value.to(elem_dtype), elem_dtype), r)
            fx.copy_atom_call(copy_atom, r, tensor)

        def _cache_indices(tok):
            if paged:
                block = tok // fx.Int32(page_block_size)
                block_off = tok % fx.Int32(page_block_size)
                if fx.const_expr(use_block_table_cache):
                    phys_block = fx.memref_load(s_blocks, block)
                else:
                    table_row = fx.slice(BlockTable, (b, None))
                    table_div = fx.logical_divide(table_row, fx.make_layout(1, 1))
                    phys_block = _load_i32(fx.slice(table_div, (None, block)))
                return phys_block, block_off
            return b, tok

        def _load_q(d):
            q_row = fx.slice(QWork, (b, sq, hq, None))
            q_div = fx.logical_divide(q_row, fx.make_layout(1, 1))
            return _load_elem(fx.slice(q_div, (None, d))).to(fx.Float32)

        def _load_q_cached(d):
            return fx.memref_load(s_q, d)

        def _load_k(tok, d):
            outer, token = _cache_indices(tok)
            if fx.const_expr(upstream_cache_layout):
                k_row = fx.slice(KCache, (outer, token, kv_head, None))
            else:
                k_row = fx.slice(KCache, (outer, kv_head, token, None))
            k_div = fx.logical_divide(k_row, fx.make_layout(1, 1))
            return _load_elem(fx.slice(k_div, (None, d))).to(fx.Float32)

        def _load_k2(tok, d):
            outer, token = _cache_indices(tok)
            if fx.const_expr(upstream_cache_layout):
                k_row = fx.slice(KCache, (outer, token, kv_head, None))
            else:
                k_row = fx.slice(KCache, (outer, kv_head, token, None))
            k_div = fx.logical_divide(k_row, fx.make_layout(1, 1))
            r = fx.make_rmem_tensor(2, elem_dtype)
            fx.copy_atom_call(copy_atom_vec2, fx.slice(k_div, (None, d)), r)
            vals = fx.memref_load_vec(r)
            return vals[0].to(fx.Float32), vals[1].to(fx.Float32)

        def _load_v(tok, d):
            outer, token = _cache_indices(tok)
            if fx.const_expr(upstream_cache_layout):
                v_row = fx.slice(VCache, (outer, token, kv_head, None))
            else:
                v_row = fx.slice(VCache, (outer, kv_head, token, None))
            v_div = fx.logical_divide(v_row, fx.make_layout(1, 1))
            return _load_elem(fx.slice(v_div, (None, d))).to(fx.Float32)

        def _load_v2(tok, d):
            outer, token = _cache_indices(tok)
            if fx.const_expr(upstream_cache_layout):
                v_row = fx.slice(VCache, (outer, token, kv_head, None))
            else:
                v_row = fx.slice(VCache, (outer, kv_head, token, None))
            v_div = fx.logical_divide(v_row, fx.make_layout(1, 1))
            r = fx.make_rmem_tensor(2, elem_dtype)
            fx.copy_atom_call(copy_atom_vec2, fx.slice(v_div, (None, d)), r)
            vals = fx.memref_load_vec(r)
            return vals[0].to(fx.Float32), vals[1].to(fx.Float32)

        def _store_out(d, value):
            out_row = fx.slice(Out, (b, sq, hq, None))
            out_div = fx.logical_divide(out_row, fx.make_layout(1, 1))
            _store_elem(fx.slice(out_div, (None, d)), value)

        seq_div = fx.logical_divide(CacheSeqLens, fx.make_layout(1, 1))
        cache_len = _load_i32(fx.slice(seq_div, (None, b)))
        q_pos = cache_len - fx.Int32(seqlen_q) + sq
        if tid < head_dim:
            fx.memref_store(_load_q(fx.Int32(tid)), s_q, tid)
        gpu.barrier()

        def _keep_token(tok):
            keep = tok < cache_len
            if causal:
                keep = keep & (tok <= q_pos)
            if window_left >= 0:
                keep = keep & (tok >= (q_pos - fx.Int32(window_left)))
            if window_right >= 0:
                keep = keep & (tok <= (q_pos + fx.Int32(window_right)))
            return keep

        def _wave_reduce(value, mode):
            acc = value
            for _sh_exp in range_constexpr(6):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = acc.shuffle_xor(off, WARP_SIZE)
                if mode == "max":
                    acc = arith.maxnumf(acc, peer)
                else:
                    acc = acc.addf(peer, fastmath=fm_fast)
            return acc

        def _score_coop(tok):
            lane = tid % WARP_SIZE
            dot = zero
            if fx.const_expr(use_vec2_qk):
                d0 = lane * fx.Int32(2)
                if d0 < head_dim:
                    k0, k1 = _load_k2(tok, d0)
                    dot = dot + _load_q_cached(d0) * k0
                    dot = dot + _load_q_cached(d0 + fx.Int32(1)) * k1
            else:
                for d_chunk in range_constexpr(_ceil_div(head_dim, WARP_SIZE)):
                    d = lane + fx.Int32(d_chunk * WARP_SIZE)
                    if d < head_dim:
                        dot = dot + _load_q_cached(d) * _load_k(tok, d)
            return _wave_reduce(dot, "sum") * scale

        def _block_reduce(value, mode):
            if red_slots == 1:
                return _wave_reduce(value, mode)
            lane = tid % WARP_SIZE
            warp = tid // WARP_SIZE
            neutral = neg_inf if mode == "max" else zero
            warp_value = _wave_reduce(value, mode)
            if lane == 0:
                fx.memref_store(warp_value, s_red, warp)
            gpu.barrier()
            if warp == 0:
                in_range = lane < red_slots
                lane_safe = in_range.select(lane, fx.Int32(0))
                partial = fx.memref_load(s_red, lane_safe)
                total = _wave_reduce(in_range.select(partial, neutral), mode)
                if lane == 0:
                    fx.memref_store(total, s_red, 0)
            gpu.barrier()
            return fx.memref_load(s_red, 0)

        c0_idx = arith.constant(0, type=T.index)
        c1_idx = arith.constant(1, type=T.index)
        c_red_idx = arith.constant(RED_SLOTS, type=T.index)
        cache_len_idx = arith.index_cast(T.index, _to_raw(cache_len))
        score_lane = tid % WARP_SIZE
        score_warp = tid // WARP_SIZE
        score_start_idx = arith.index_cast(T.index, _to_raw(score_warp))

        thread_max = neg_inf_raw
        for tok, iter_args, loop_results in scf.for_(score_start_idx, cache_len_idx, c_red_idx, iter_args=[thread_max]):
            thread_max = fx.Float32(iter_args)
            tok_i = fx.Int32(tok)
            keep = _keep_token(tok_i)
            token_score = keep.select(_score_coop(tok_i), neg_inf)
            if score_lane == 0:
                fx.memref_store(token_score, s_prob, tok_i)
                thread_max = arith.maxnumf(thread_max, token_score)
            yield [_to_raw(thread_max)]
        thread_max = fx.Float32(loop_results)
        max_score = _block_reduce(thread_max, "max")

        thread_sum = zero_raw
        for tok, iter_args, loop_results in scf.for_(score_start_idx, cache_len_idx, c_red_idx, iter_args=[thread_sum]):
            thread_sum = fx.Float32(iter_args)
            tok_i = fx.Int32(tok)
            if score_lane == 0:
                score_v = fx.memref_load(s_prob, tok_i)
                exp_v = fmath.exp2((score_v - max_score) * _LOG2E, fastmath=fm_fast)
                exp_safe = (score_v > neg_inf).select(exp_v, zero)
                fx.memref_store(exp_safe, s_prob, tok_i)
                thread_sum = thread_sum + exp_safe
            yield [_to_raw(thread_sum)]
        thread_sum = fx.Float32(loop_results)
        denom = _block_reduce(thread_sum, "sum")
        has_tokens = denom > zero

        if fx.const_expr(use_vec2_pv):
            if tid < head_dim // 2:
                d0 = fx.Int32(tid * fx.Int32(2))
                acc0 = zero_raw
                acc1 = zero_raw
                for tok, iter_args, loop_results in scf.for_(c0_idx, cache_len_idx, c1_idx, iter_args=[acc0, acc1]):
                    acc0 = fx.Float32(iter_args[0])
                    acc1 = fx.Float32(iter_args[1])
                    tok_i = fx.Int32(tok)
                    prob = fx.memref_load(s_prob, tok_i) / denom
                    v0, v1 = _load_v2(tok_i, d0)
                    acc0 = acc0 + prob * v0
                    acc1 = acc1 + prob * v1
                    yield [_to_raw(acc0), _to_raw(acc1)]
                acc0 = fx.Float32(loop_results[0])
                acc1 = fx.Float32(loop_results[1])
                _store_out(d0, has_tokens.select(acc0, zero))
                _store_out(d0 + fx.Int32(1), has_tokens.select(acc1, zero))
        else:
            if tid < head_dim:
                d = fx.Int32(tid)
                acc = zero_raw
                for tok, iter_args, loop_results in scf.for_(c0_idx, cache_len_idx, c1_idx, iter_args=[acc]):
                    acc = fx.Float32(iter_args)
                    tok_i = fx.Int32(tok)
                    prob = fx.memref_load(s_prob, tok_i) / denom
                    acc = acc + prob * _load_v(tok_i, d)
                    yield [_to_raw(acc)]
                acc = fx.Float32(loop_results)
                _store_out(d, has_tokens.select(acc, zero))

    @flyc.kernel(known_block_size=[ATTN_THREADS, 1, 1])
    def split_attention_kernel(
        QWork: fx.Tensor,
        KCache: fx.Tensor,
        VCache: fx.Tensor,
        CacheSeqLens: fx.Tensor,
        BlockTable: fx.Tensor,
        GroupMax: fx.Tensor,
        GroupSum: fx.Tensor,
        PartialOut: fx.Tensor,
    ):
        row = fx.Int32(fx.block_idx.x)
        b = fx.Int32(fx.block_idx.y)
        split_head = fx.Int32(fx.block_idx.z)
        split = split_head % fx.Int32(num_splits)
        hq = split_head // fx.Int32(num_splits)
        tid = fx.thread_idx.x
        sq = row
        kv_head = hq // fx.Int32(repeats)
        copy_atom = fx.make_copy_atom(fx.UniversalCopy16b(), elem_dtype)
        fm_fast = arith.FastMathFlags.fast
        zero_raw = arith.constant(0.0, type=T.f32)
        neg_inf_raw = arith.constant(float("-inf"), type=T.f32)
        zero = fx.Float32(zero_raw)
        neg_inf = fx.Float32(neg_inf_raw)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_red = lds.s_red.view(fx.make_layout(red_slots, 1))
        s_prob = lds.s_prob.view(fx.make_layout(prob_slots, 1))
        s_q = lds.s_q.view(fx.make_layout(head_dim, 1))
        s_blocks = lds.s_blocks.view(fx.make_layout(max_blocks_per_seq, 1))

        def _load_elem(tensor):
            r = fx.make_rmem_tensor(1, elem_dtype)
            fx.copy_atom_call(copy_atom, tensor, r)
            return fx.memref_load_vec(r)[0]

        copy_atom_i32 = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
        copy_atom_vec2 = fx.make_copy_atom(fx.UniversalCopy32b(), elem_dtype)

        def _load_i32(tensor):
            r = fx.make_rmem_tensor(1, fx.Int32)
            fx.copy_atom_call(copy_atom_i32, tensor, r)
            return fx.memref_load_vec(r)[0]

        if fx.const_expr(use_block_table_cache):
            if tid < max_blocks_per_seq:
                table_row = fx.slice(BlockTable, (b, None))
                table_div = fx.logical_divide(table_row, fx.make_layout(1, 1))
                phys_block = _load_i32(fx.slice(table_div, (None, tid)))
                fx.memref_store(phys_block, s_blocks, tid)
            gpu.barrier()

        def _cache_indices(tok):
            if paged:
                block = tok // fx.Int32(page_block_size)
                block_off = tok % fx.Int32(page_block_size)
                if fx.const_expr(use_block_table_cache):
                    phys_block = fx.memref_load(s_blocks, block)
                else:
                    table_row = fx.slice(BlockTable, (b, None))
                    table_div = fx.logical_divide(table_row, fx.make_layout(1, 1))
                    phys_block = _load_i32(fx.slice(table_div, (None, block)))
                return phys_block, block_off
            return b, tok

        def _load_q(d):
            q_row = fx.slice(QWork, (b, sq, hq, None))
            q_div = fx.logical_divide(q_row, fx.make_layout(1, 1))
            return _load_elem(fx.slice(q_div, (None, d))).to(fx.Float32)

        def _load_q_cached(d):
            return fx.memref_load(s_q, d)

        def _load_k(tok, d):
            outer, token = _cache_indices(tok)
            if fx.const_expr(upstream_cache_layout):
                k_row = fx.slice(KCache, (outer, token, kv_head, None))
            else:
                k_row = fx.slice(KCache, (outer, kv_head, token, None))
            k_div = fx.logical_divide(k_row, fx.make_layout(1, 1))
            return _load_elem(fx.slice(k_div, (None, d))).to(fx.Float32)

        def _load_k2(tok, d):
            outer, token = _cache_indices(tok)
            if fx.const_expr(upstream_cache_layout):
                k_row = fx.slice(KCache, (outer, token, kv_head, None))
            else:
                k_row = fx.slice(KCache, (outer, kv_head, token, None))
            k_div = fx.logical_divide(k_row, fx.make_layout(1, 1))
            r = fx.make_rmem_tensor(2, elem_dtype)
            fx.copy_atom_call(copy_atom_vec2, fx.slice(k_div, (None, d)), r)
            vals = fx.memref_load_vec(r)
            return vals[0].to(fx.Float32), vals[1].to(fx.Float32)

        def _load_v(tok, d):
            outer, token = _cache_indices(tok)
            if fx.const_expr(upstream_cache_layout):
                v_row = fx.slice(VCache, (outer, token, kv_head, None))
            else:
                v_row = fx.slice(VCache, (outer, kv_head, token, None))
            v_div = fx.logical_divide(v_row, fx.make_layout(1, 1))
            return _load_elem(fx.slice(v_div, (None, d))).to(fx.Float32)

        def _load_v2(tok, d):
            outer, token = _cache_indices(tok)
            if fx.const_expr(upstream_cache_layout):
                v_row = fx.slice(VCache, (outer, token, kv_head, None))
            else:
                v_row = fx.slice(VCache, (outer, kv_head, token, None))
            v_div = fx.logical_divide(v_row, fx.make_layout(1, 1))
            r = fx.make_rmem_tensor(2, elem_dtype)
            fx.copy_atom_call(copy_atom_vec2, fx.slice(v_div, (None, d)), r)
            vals = fx.memref_load_vec(r)
            return vals[0].to(fx.Float32), vals[1].to(fx.Float32)

        def _store_group_value(tensor, value):
            row_view = fx.slice(tensor, (b, row, hq, None))
            row_div = fx.logical_divide(row_view, fx.make_layout(1, 1))
            fx.memref_store(value, row_div, split)

        def _store_partial(d, value):
            row_view = fx.slice(PartialOut, (b, row, hq, split, None))
            row_div = fx.logical_divide(row_view, fx.make_layout(1, 1))
            fx.memref_store(value, row_div, d)

        seq_div = fx.logical_divide(CacheSeqLens, fx.make_layout(1, 1))
        cache_len = _load_i32(fx.slice(seq_div, (None, b)))
        q_pos = cache_len - fx.Int32(seqlen_q) + sq
        split_start = split * fx.Int32(split_chunk)
        split_end_raw = split_start + fx.Int32(split_chunk)
        split_end = (split_end_raw < cache_len).select(split_end_raw, cache_len)
        has_split_tokens = split_start < split_end
        if tid < head_dim:
            fx.memref_store(_load_q(fx.Int32(tid)), s_q, tid)
        gpu.barrier()

        def _keep_token(tok):
            keep = (tok >= split_start) & (tok < split_end)
            if causal:
                keep = keep & (tok <= q_pos)
            if window_left >= 0:
                keep = keep & (tok >= (q_pos - fx.Int32(window_left)))
            if window_right >= 0:
                keep = keep & (tok <= (q_pos + fx.Int32(window_right)))
            return keep

        def _wave_reduce(value, mode):
            acc = value
            for _sh_exp in range_constexpr(6):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = acc.shuffle_xor(off, WARP_SIZE)
                if mode == "max":
                    acc = arith.maxnumf(acc, peer)
                else:
                    acc = acc.addf(peer, fastmath=fm_fast)
            return acc

        def _score_coop(tok):
            lane = tid % WARP_SIZE
            dot = zero
            if fx.const_expr(use_vec2_qk):
                d0 = lane * fx.Int32(2)
                if d0 < head_dim:
                    k0, k1 = _load_k2(tok, d0)
                    dot = dot + _load_q_cached(d0) * k0
                    dot = dot + _load_q_cached(d0 + fx.Int32(1)) * k1
            else:
                for d_chunk in range_constexpr(_ceil_div(head_dim, WARP_SIZE)):
                    d = lane + fx.Int32(d_chunk * WARP_SIZE)
                    if d < head_dim:
                        dot = dot + _load_q_cached(d) * _load_k(tok, d)
            return _wave_reduce(dot, "sum") * scale

        def _block_reduce(value, mode):
            if red_slots == 1:
                return _wave_reduce(value, mode)
            lane = tid % WARP_SIZE
            warp = tid // WARP_SIZE
            neutral = neg_inf if mode == "max" else zero
            warp_value = _wave_reduce(value, mode)
            if lane == 0:
                fx.memref_store(warp_value, s_red, warp)
            gpu.barrier()
            if warp == 0:
                in_range = lane < red_slots
                lane_safe = in_range.select(lane, fx.Int32(0))
                partial = fx.memref_load(s_red, lane_safe)
                total = _wave_reduce(in_range.select(partial, neutral), mode)
                if lane == 0:
                    fx.memref_store(total, s_red, 0)
            gpu.barrier()
            return fx.memref_load(s_red, 0)

        c1_idx = arith.constant(1, type=T.index)
        c_red_idx = arith.constant(RED_SLOTS, type=T.index)
        split_start_idx = arith.index_cast(T.index, _to_raw(split_start))
        split_end_idx = arith.index_cast(T.index, _to_raw(split_end))
        score_lane = tid % WARP_SIZE
        score_warp = tid // WARP_SIZE
        score_start = split_start + score_warp
        score_start_idx = arith.index_cast(T.index, _to_raw(score_start))

        thread_max = neg_inf_raw
        for tok, iter_args, loop_results in scf.for_(score_start_idx, split_end_idx, c_red_idx, iter_args=[thread_max]):
            thread_max = fx.Float32(iter_args)
            tok_i = fx.Int32(tok)
            keep = _keep_token(tok_i)
            token_score = keep.select(_score_coop(tok_i), neg_inf)
            if score_lane == 0:
                local_tok = tok_i - split_start
                fx.memref_store(token_score, s_prob, local_tok)
                thread_max = arith.maxnumf(thread_max, token_score)
            yield [_to_raw(thread_max)]
        thread_max = fx.Float32(loop_results)
        max_score = has_split_tokens.select(_block_reduce(thread_max, "max"), neg_inf)

        thread_sum = zero_raw
        for tok, iter_args, loop_results in scf.for_(score_start_idx, split_end_idx, c_red_idx, iter_args=[thread_sum]):
            thread_sum = fx.Float32(iter_args)
            tok_i = fx.Int32(tok)
            if score_lane == 0:
                local_tok = tok_i - split_start
                score_v = fx.memref_load(s_prob, local_tok)
                exp_v = fmath.exp2((score_v - max_score) * _LOG2E, fastmath=fm_fast)
                exp_safe = (score_v > neg_inf).select(exp_v, zero)
                fx.memref_store(exp_safe, s_prob, local_tok)
                thread_sum = thread_sum + exp_safe
            yield [_to_raw(thread_sum)]
        thread_sum = fx.Float32(loop_results)
        denom = _block_reduce(thread_sum, "sum")

        if tid == 0:
            _store_group_value(GroupMax, max_score)
            _store_group_value(GroupSum, denom)

        if fx.const_expr(use_krepeat_pv):
            if tid < WARP_SIZE:
                lane = fx.Int32(tid)
                accs = [zero_raw for _ in range_constexpr(k_repeats)]
                for tok, iter_args, loop_results in scf.for_(
                    split_start_idx,
                    split_end_idx,
                    c1_idx,
                    iter_args=accs,
                ):
                    tok_i = fx.Int32(tok)
                    local_tok = tok_i - split_start
                    prob_num = fx.memref_load(s_prob, local_tok)
                    next_accs = []
                    for repeat_const in range_constexpr(k_repeats):
                        acc = fx.Float32(iter_args[repeat_const])
                        d = lane + fx.Int32(repeat_const * WARP_SIZE)
                        acc = acc + prob_num * _load_v(tok_i, d)
                        next_accs.append(acc)
                    yield [_to_raw(acc) for acc in next_accs]
                for repeat_const in range_constexpr(k_repeats):
                    d = lane + fx.Int32(repeat_const * WARP_SIZE)
                    _store_partial(d, fx.Float32(loop_results[repeat_const]))
        elif fx.const_expr(use_vec2_pv):
            if tid < head_dim // 2:
                d0 = fx.Int32(tid * fx.Int32(2))
                acc0 = zero_raw
                acc1 = zero_raw
                for tok, iter_args, loop_results in scf.for_(split_start_idx, split_end_idx, c1_idx, iter_args=[acc0, acc1]):
                    acc0 = fx.Float32(iter_args[0])
                    acc1 = fx.Float32(iter_args[1])
                    tok_i = fx.Int32(tok)
                    local_tok = tok_i - split_start
                    prob_num = fx.memref_load(s_prob, local_tok)
                    v0, v1 = _load_v2(tok_i, d0)
                    acc0 = acc0 + prob_num * v0
                    acc1 = acc1 + prob_num * v1
                    yield [_to_raw(acc0), _to_raw(acc1)]
                acc0 = fx.Float32(loop_results[0])
                acc1 = fx.Float32(loop_results[1])
                _store_partial(d0, acc0)
                _store_partial(d0 + fx.Int32(1), acc1)
        else:
            if tid < head_dim:
                d = fx.Int32(tid)
                acc = zero_raw
                for tok, iter_args, loop_results in scf.for_(split_start_idx, split_end_idx, c1_idx, iter_args=[acc]):
                    acc = fx.Float32(iter_args)
                    tok_i = fx.Int32(tok)
                    local_tok = tok_i - split_start
                    prob_num = fx.memref_load(s_prob, local_tok)
                    acc = acc + prob_num * _load_v(tok_i, d)
                    yield [_to_raw(acc)]
                acc = fx.Float32(loop_results)
                _store_partial(d, acc)

    @flyc.kernel(known_block_size=[reduce_threads, 1, 1])
    def split_reduce_kernel(
        GroupMax: fx.Tensor,
        GroupSum: fx.Tensor,
        PartialOut: fx.Tensor,
        Out: fx.Tensor,
    ):
        row = fx.Int32(fx.block_idx.x)
        b = fx.Int32(fx.block_idx.y)
        hq = fx.Int32(fx.block_idx.z)
        tid = fx.thread_idx.x
        lane = tid % fx.Int32(WARP_SIZE)
        copy_atom = fx.make_copy_atom(fx.UniversalCopy16b(), elem_dtype)
        fm_fast = arith.FastMathFlags.fast
        zero_raw = arith.constant(0.0, type=T.f32)
        neg_inf_raw = arith.constant(float("-inf"), type=T.f32)
        zero = fx.Float32(zero_raw)
        neg_inf = fx.Float32(neg_inf_raw)

        def _store_elem(tensor, value):
            if not hasattr(value, "to"):
                value = fx.Float32(value)
            r = fx.make_rmem_tensor(1, elem_dtype)
            fx.memref_store_vec(full(1, value.to(elem_dtype), elem_dtype), r)
            fx.copy_atom_call(copy_atom, r, tensor)

        def _store_out(d, value):
            out_row = fx.slice(Out, (b, row, hq, None))
            out_div = fx.logical_divide(out_row, fx.make_layout(1, 1))
            _store_elem(fx.slice(out_div, (None, d)), value)

        def _load_group_value(tensor, split):
            row_view = fx.slice(tensor, (b, row, hq, None))
            row_div = fx.logical_divide(row_view, fx.make_layout(1, 1))
            return fx.memref_load(row_div, split)

        def _load_partial(split, d):
            row_view = fx.slice(PartialOut, (b, row, hq, split, None))
            row_div = fx.logical_divide(row_view, fx.make_layout(1, 1))
            return fx.memref_load(row_div, d)

        # Group statistics are identical for every output element.  Compute
        # them once per warp, then broadcast from lane 0 instead of issuing the
        # same global loads from every lane.
        max_score = neg_inf
        if lane == fx.Int32(0):
            for split_const in range_constexpr(num_splits):
                split = fx.Int32(split_const)
                group_max = _load_group_value(GroupMax, split)
                max_score = arith.maxnumf(max_score, group_max)
        max_score = fx.Float32(max_score).shuffle_idx(
            fx.Int32(0), fx.Int32(WARP_SIZE))

        denom = zero
        if lane == fx.Int32(0):
            for split_const in range_constexpr(num_splits):
                split = fx.Int32(split_const)
                group_max = _load_group_value(GroupMax, split)
                group_sum = _load_group_value(GroupSum, split)
                scale_group = fmath.exp2((group_max - max_score) * _LOG2E, fastmath=fm_fast)
                denom = denom + group_sum * scale_group
        denom = fx.Float32(denom).shuffle_idx(
            fx.Int32(0), fx.Int32(WARP_SIZE))
        has_tokens = denom > zero

        d = fx.Int32(tid)
        acc = zero
        for split_const in range_constexpr(num_splits):
            split = fx.Int32(split_const)
            group_max = neg_inf
            if lane == fx.Int32(0):
                group_max = _load_group_value(GroupMax, split)
            group_max = fx.Float32(group_max).shuffle_idx(
                fx.Int32(0), fx.Int32(WARP_SIZE))
            scale_group = fmath.exp2((group_max - max_score) * _LOG2E, fastmath=fm_fast)
            if tid < head_dim:
                partial = _load_partial(split, d)
                acc = acc + partial * scale_group
        if tid < head_dim:
            _store_out(d, has_tokens.select(acc / denom, zero))

    @flyc.kernel(known_block_size=[v5_reduce_warps * WARP_SIZE, 1, 1])
    def v5_split_reduce_kernel(
        GroupMax: fx.Tensor,
        GroupSum: fx.Tensor,
        PartialOut: fx.Tensor,
        Out: fx.Tensor,
    ):
        """ixInfer-style 16-warp reduction for V5's [split, head_dim] workspace."""
        row = fx.Int32(fx.block_idx.x)
        b = fx.Int32(fx.block_idx.y)
        hq = fx.Int32(fx.block_idx.z)
        tid = fx.thread_idx.x
        warp = tid // fx.Int32(WARP_SIZE)
        lane = tid % fx.Int32(WARP_SIZE)
        d0 = lane
        d1 = lane + fx.Int32(WARP_SIZE)
        copy_atom = fx.make_copy_atom(fx.UniversalCopy16b(), elem_dtype)
        fm_fast = arith.FastMathFlags.fast
        zero_raw = arith.constant(0.0, type=T.f32)
        # Keep empty warps mergeable when num_splits < v5_reduce_warps.
        # A finite sentinel avoids the -inf - -inf NaN that otherwise occurs
        # before an empty warp is merged with a warp that has valid groups.
        neg_inf_raw = arith.constant(-3.40282e38, type=T.f32)
        zero = fx.Float32(zero_raw)
        neg_inf = fx.Float32(neg_inf_raw)

        lds = fx.SharedAllocator().allocate(V5ReduceStorage).peek()
        s_max = lds.max_values.view(fx.make_layout(v5_reduce_slots, 1))
        s_sum = lds.exp_sums.view(fx.make_layout(v5_reduce_slots, 1))
        s_partial = lds.partials.view(fx.make_layout(v5_reduce_slots * head_dim, 1))

        def _store_elem(tensor, value):
            r = fx.make_rmem_tensor(1, elem_dtype)
            fx.memref_store_vec(full(1, value.to(elem_dtype), elem_dtype), r)
            fx.copy_atom_call(copy_atom, r, tensor)

        def _store_out(d, value):
            out_row = fx.slice(Out, (b, row, hq, None))
            out_div = fx.logical_divide(out_row, fx.make_layout(1, 1))
            _store_elem(fx.slice(out_div, (None, d)), value)

        def _load_group_value(tensor, split):
            row_view = fx.slice(tensor, (b, row, hq, None))
            row_div = fx.logical_divide(row_view, fx.make_layout(1, 1))
            return fx.memref_load(row_div, split)

        def _load_partial(split, d):
            row_view = fx.slice(PartialOut, (b, row, hq, split, None))
            row_div = fx.logical_divide(row_view, fx.make_layout(1, 1))
            return fx.memref_load(row_div, d)

        # Each warp reduces an interleaved subset of split groups.  This is the
        # same group-to-warp mapping used by ixinfer's pageAttentionReduce.
        warp_max = neg_inf
        warp_sum = zero
        warp_out0 = zero
        warp_out1 = zero
        for group_iter in fx.range_constexpr(_ceil_div(num_splits, v5_reduce_warps)):
            split = warp + fx.Int32(group_iter * v5_reduce_warps)
            group_max = neg_inf
            group_sum = zero
            part0 = zero
            part1 = zero
            if split < fx.Int32(num_splits):
                if lane == fx.Int32(0):
                    group_max = _load_group_value(GroupMax, split)
                    group_sum = _load_group_value(GroupSum, split)
                group_max = fx.Float32(group_max).shuffle_idx(fx.Int32(0), fx.Int32(WARP_SIZE))
                group_sum = fx.Float32(group_sum).shuffle_idx(fx.Int32(0), fx.Int32(WARP_SIZE))
                part0 = _load_partial(split, d0)
                part1 = _load_partial(split, d1)
            new_max = arith.maxnumf(warp_max, group_max)
            old_scale = fmath.exp2((warp_max - new_max) * _LOG2E, fastmath=fm_fast)
            group_scale = fmath.exp2((group_max - new_max) * _LOG2E, fastmath=fm_fast)
            warp_sum = warp_sum * old_scale + group_sum * group_scale
            warp_out0 = warp_out0 * old_scale + part0 * group_scale
            warp_out1 = warp_out1 * old_scale + part1 * group_scale
            warp_max = new_max

        if lane == fx.Int32(0):
            fx.memref_store(warp_max, s_max, warp)
            fx.memref_store(warp_sum, s_sum, warp)
        fx.memref_store(warp_out0, s_partial, warp * fx.Int32(head_dim) + d0)
        fx.memref_store(warp_out1, s_partial, warp * fx.Int32(head_dim) + d1)
        gpu.barrier()

        # Mirror ixinfer's LDS binary tree: upper-half warps spill their
        # partials, then lower-half warps merge one partner per stage.
        for stage in fx.range_constexpr(v5_reduce_warps.bit_length() - 1):
            part_warps = v5_reduce_warps >> stage
            middle_warp = part_warps // 2
            if warp >= fx.Int32(middle_warp) and warp < fx.Int32(part_warps):
                partner_slot = warp - fx.Int32(middle_warp)
                if lane == fx.Int32(0):
                    fx.memref_store(warp_max, s_max, partner_slot)
                    fx.memref_store(warp_sum, s_sum, partner_slot)
                fx.memref_store(
                    warp_out0, s_partial, partner_slot * fx.Int32(head_dim) + d0
                )
                fx.memref_store(
                    warp_out1, s_partial, partner_slot * fx.Int32(head_dim) + d1
                )
            gpu.barrier()

            if warp < fx.Int32(middle_warp):
                partner_max = neg_inf
                partner_sum = zero
                if lane == fx.Int32(0):
                    partner_max = fx.memref_load(s_max, warp)
                    partner_sum = fx.memref_load(s_sum, warp)
                partner_max = fx.Float32(partner_max).shuffle_idx(fx.Int32(0), fx.Int32(WARP_SIZE))
                partner_sum = fx.Float32(partner_sum).shuffle_idx(fx.Int32(0), fx.Int32(WARP_SIZE))
                new_max = arith.maxnumf(warp_max, partner_max)
                old_scale = fmath.exp2((warp_max - new_max) * _LOG2E, fastmath=fm_fast)
                partner_scale = fmath.exp2((partner_max - new_max) * _LOG2E, fastmath=fm_fast)
                warp_sum = warp_sum * old_scale + partner_sum * partner_scale
                partial_base = warp * fx.Int32(head_dim)
                warp_out0 = (
                    warp_out0 * old_scale
                    + fx.memref_load(s_partial, partial_base + d0) * partner_scale
                )
                warp_out1 = (
                    warp_out1 * old_scale
                    + fx.memref_load(s_partial, partial_base + d1) * partner_scale
                )
                warp_max = new_max
            gpu.barrier()

        if warp == fx.Int32(0):
            has_tokens = warp_sum > zero
            _store_out(d0, has_tokens.select(warp_out0 / warp_sum, zero))
            _store_out(d1, has_tokens.select(warp_out1 / warp_sum, zero))

    if use_v5_decode:
        from kernels.attention.iluvatar.flash_attn_kvcache_v5 import (
            build_v5_decode_attention_kernel,
        )

        mma_decode_kernel, mma_threads, mma_smem, mma_grid = (
            build_v5_decode_attention_kernel(
                batch_size=batch_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                max_seqlen_k=max_seqlen_k,
                page_block_size=page_block_size,
                num_groups=num_splits,
            )
        )
    elif use_mma_decode:
        mma_decode_kernel, mma_threads, mma_smem, mma_grid = build_mma_decode_attention_kernel(
            batch_size=batch_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seqlen_k=max_seqlen_k,
            paged=paged,
            page_block_size=page_block_size,
            upstream_cache_layout=upstream_cache_layout,
            num_splits=num_splits,
            block_n=mma_block_n,
        )

    @flyc.jit
    def launch_flash_attn_with_kvcache(
        QKV: fx.Tensor,
        QWork: fx.Tensor,
        KCache: fx.Tensor,
        VCache: fx.Tensor,
        CacheSeqLens: fx.Tensor,
        BlockTable: fx.Tensor,
        RotaryCos: fx.Tensor,
        RotarySin: fx.Tensor,
        Out: fx.Tensor,
        GroupMax: fx.Tensor,
        GroupSum: fx.Tensor,
        PartialOut: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        if update_cache:
            update_cache_kernel(
                QKV,
                QWork,
                KCache,
                VCache,
                CacheSeqLens,
                BlockTable,
                RotaryCos,
                RotarySin,
            ).launch(
                grid=(max(num_heads, num_kv_heads), seqlen_q, batch_size),
                block=(max(head_dim, 1), 1, 1),
                stream=stream,
            )
        if use_mma_decode or use_v5_decode:
            mma_decode_kernel(
                QWork, KCache, VCache, CacheSeqLens, BlockTable, Out, GroupMax, GroupSum, PartialOut
            ).launch(
                grid=mma_grid,
                block=(mma_threads, 1, 1),
                smem=mma_smem,
                stream=stream,
            )
            if num_splits > 1:
                if use_v5_decode:
                    v5_split_reduce_kernel(GroupMax, GroupSum, PartialOut, Out).launch(
                        grid=(seqlen_q, batch_size, num_heads),
                        block=(v5_reduce_warps * WARP_SIZE, 1, 1),
                        stream=stream,
                    )
                else:
                    split_reduce_kernel(GroupMax, GroupSum, PartialOut, Out).launch(
                        grid=(seqlen_q, batch_size, num_heads),
                        block=(reduce_threads, 1, 1),
                        stream=stream,
                    )
        elif num_splits == 1:
            attention_kernel(QWork, KCache, VCache, CacheSeqLens, BlockTable, Out).launch(
                grid=(seqlen_q, batch_size, num_heads),
                block=(ATTN_THREADS, 1, 1),
                stream=stream,
            )
        else:
            split_attention_kernel(
                QWork, KCache, VCache, CacheSeqLens, BlockTable, GroupMax, GroupSum, PartialOut
            ).launch(
                grid=(seqlen_q, batch_size, num_heads * num_splits),
                block=(ATTN_THREADS, 1, 1),
                stream=stream,
            )
            split_reduce_kernel(GroupMax, GroupSum, PartialOut, Out).launch(
                grid=(seqlen_q, batch_size, num_heads),
                block=(reduce_threads, 1, 1),
                stream=stream,
            )

    return launch_flash_attn_with_kvcache


def flash_attn_with_kvcache_flydsl(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[torch.Tensor | int] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    alibi_slopes=None,
    num_splits: int = 0,
    return_softmax_lse: bool = False,
    alibi_mode: int = 0,
    is_qkv_packed: bool = False,
    force_upstream_cache_layout: Optional[bool] = None,
    stream=None,
):
    """FlyDSL-native forward-only KV-cache attention.

    Unsupported optional arguments are rejected instead of silently falling
    back, so callers can see exactly which MR-compatible subset is active.

    ``force_upstream_cache_layout`` overrides the cache-layout inference: by
    default the layout follows ``is_qkv_packed`` (packed -> MR/HND, separate ->
    upstream/NHD).  Pass ``True`` to force NHD ``[blocks, page, Hkv, D]`` or
    ``False`` to force HND ``[blocks, Hkv, page, D]`` regardless of packing.
    """
    del alibi_mode
    if (k is None) != (v is None):
        raise ValueError("k and v must be provided together")
    if cache_batch_idx is not None:
        raise NotImplementedError("cache_batch_idx is not supported in the FlyDSL native path")
    if cache_leftpad is not None:
        raise NotImplementedError("cache_leftpad is not supported in the FlyDSL native path")
    if alibi_slopes is not None:
        raise NotImplementedError("ALiBi is not supported in the FlyDSL native path")
    if return_softmax_lse:
        raise NotImplementedError("return_softmax_lse is not supported in the FlyDSL native path")
    if softcap != 0.0:
        raise NotImplementedError("softcap is not supported in the FlyDSL native path")
    if num_splits < 0:
        raise ValueError("num_splits must be >= 0")
    if softmax_scale is not None and softmax_scale != q.shape[-1] ** (-0.5):
        raise NotImplementedError("custom softmax_scale is not supported yet")
    if q.ndim != 4:
        raise ValueError("q must be 4D")
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError("k_cache and v_cache must be 4D tensors")
    if q.device != k_cache.device or q.device != v_cache.device:
        raise ValueError("q, k_cache, and v_cache must be on the same device")
    if q.dtype != k_cache.dtype or q.dtype != v_cache.dtype:
        raise ValueError("q, k_cache, and v_cache must have the same dtype")
    if q.stride(-1) != 1 or k_cache.stride(-1) != 1 or v_cache.stride(-1) != 1:
        raise ValueError("q, k_cache, and v_cache must have contiguous last dimension")

    batch_size, seqlen_q = q.shape[0], q.shape[1]
    head_dim = q.shape[-1]
    update_cache = True
    upstream_layout = (not is_qkv_packed) if force_upstream_cache_layout is None \
        else bool(force_upstream_cache_layout)
    upstream_cache_layout = upstream_layout
    rotary_applied_in_wrapper = False

    if is_qkv_packed:
        if k is not None or v is not None:
            raise NotImplementedError("Separate k/v inputs are not supported when is_qkv_packed=True")
        num_kv_heads = k_cache.shape[1]
        num_heads = q.shape[2] - 2 * num_kv_heads
        if rotary_interleaved and (rotary_cos is not None or rotary_sin is not None):
            if rotary_cos is None or rotary_sin is None:
                raise ValueError("rotary_cos and rotary_sin must be provided together")
            positions = cache_seqlens[:, None] - seqlen_q + torch.arange(seqlen_q, device=q.device)[None, :]
            q_part, k_part, v_part = q.split([num_heads, num_kv_heads, num_kv_heads], dim=2)
            q_part = _apply_interleaved_rotary_torch(q_part, rotary_cos, rotary_sin, positions)
            k_part = _apply_interleaved_rotary_torch(k_part, rotary_cos, rotary_sin, positions)
            qkv = torch.cat([q_part, k_part, v_part], dim=2).contiguous()
            rotary_applied_in_wrapper = True
        else:
            qkv = q.contiguous()
        k_cache_view = k_cache
        v_cache_view = v_cache
        cache_len_delta = 0
    else:
        num_heads = q.shape[2]
        if k is None:
            update_cache = False
            # NHD [blocks/B, page/S, Hkv, D] -> Hkv at dim 2;
            # HND [blocks/B, Hkv, page/S, D] -> Hkv at dim 1.
            num_kv_heads = k_cache.shape[2] if upstream_cache_layout else k_cache.shape[1]
            qkv = q.contiguous()
            cache_len_delta = 0
        else:
            if k.ndim != 4 or v.ndim != 4:
                raise ValueError("k and v must be [B, S_new, H_kv, D]")
            if k.shape != v.shape:
                raise ValueError("k and v must have matching shapes")
            if k.shape[0] != batch_size or k.shape[1] != seqlen_q or k.shape[-1] != head_dim:
                raise NotImplementedError("FlyDSL currently requires k/v to have [B, S_q, H_kv, D]")
            if k.dtype != q.dtype or v.dtype != q.dtype or k.device != q.device or v.device != q.device:
                raise ValueError("q, k, and v must have matching dtype and device")
            num_kv_heads = k.shape[2]
            if rotary_interleaved and (rotary_cos is not None or rotary_sin is not None):
                if rotary_cos is None or rotary_sin is None:
                    raise ValueError("rotary_cos and rotary_sin must be provided together")
                if cache_seqlens is None:
                    raise ValueError("cache_seqlens is required when applying rotary to separate k/v")
                base_lens = (
                    torch.full((batch_size,), cache_seqlens, dtype=torch.int32, device=q.device)
                    if isinstance(cache_seqlens, int)
                    else cache_seqlens.contiguous()
                )
                token_offsets = torch.arange(seqlen_q, device=q.device, dtype=torch.int32)[None, :]
                if causal or window_size != (-1, -1):
                    positions = base_lens[:, None] + token_offsets
                else:
                    positions = base_lens[:, None].expand(batch_size, seqlen_q)
                q = _apply_interleaved_rotary_torch(q, rotary_cos, rotary_sin, positions)
                k = _apply_interleaved_rotary_torch(k, rotary_cos, rotary_sin, positions)
                rotary_applied_in_wrapper = True
            qkv = torch.empty(
                (batch_size, seqlen_q, num_heads + 2 * num_kv_heads, head_dim),
                device=q.device,
                dtype=q.dtype,
            )
            qkv[:, :, :num_heads, :].copy_(q)
            qkv[:, :, num_heads : num_heads + num_kv_heads, :].copy_(k)
            qkv[:, :, num_heads + num_kv_heads :, :].copy_(v)
            cache_len_delta = seqlen_q
        # Upstream cache layout is dense [B, S, Hkv, D] or paged
        # [blocks, page, Hkv, D]. The kernels handle this layout directly.
        k_cache_view = k_cache
        v_cache_view = v_cache

    if num_heads <= 0:
        raise ValueError("number of query heads must be positive")
    if num_heads % num_kv_heads != 0:
        raise ValueError("number of Q heads must be divisible by number of KV heads")
    if upstream_cache_layout:
        if v_cache_view.shape[2] != num_kv_heads or v_cache_view.shape[-1] != head_dim:
            raise ValueError("upstream v_cache shape must match k_cache heads and head_dim")
    elif v_cache_view.shape[1] != num_kv_heads or v_cache_view.shape[-1] != head_dim:
        raise ValueError("MR v_cache shape must match k_cache heads and head_dim")

    paged = block_table is not None
    kernel_paged = paged
    if paged:
        if block_table.dtype != torch.int32 or not block_table.is_cuda:
            raise ValueError("block_table must be a CUDA int32 tensor")
        if block_table.shape[0] != batch_size or block_table.stride(-1) != 1:
            raise ValueError("block_table must be [B, max_blocks] with contiguous last dimension")
        page_block_size = k_cache_view.shape[1] if upstream_cache_layout else k_cache_view.shape[2]
        max_seqlen_k = block_table.shape[1] * page_block_size
        if v_cache_view.shape[:3] != k_cache_view.shape[:3]:
            raise ValueError("paged k_cache/v_cache must have matching leading dimensions")
        block_table_for_kernel = block_table
    else:
        if k_cache_view.shape[0] != batch_size or v_cache_view.shape[0] != batch_size:
            raise ValueError("dense cache batch must match q batch")
        page_block_size = 16
        max_seqlen_k = k_cache_view.shape[1] if upstream_cache_layout else k_cache_view.shape[2]
        block_table_for_kernel = torch.empty((batch_size, 1), device=q.device, dtype=torch.int32)

    if cache_seqlens is None:
        cache_seqlens = torch.full((batch_size,), max_seqlen_k, dtype=torch.int32, device=q.device)
    elif isinstance(cache_seqlens, int):
        cache_seqlens = torch.full((batch_size,), cache_seqlens, dtype=torch.int32, device=q.device)
    else:
        cache_seqlens = cache_seqlens.contiguous()
    if cache_seqlens.shape != (batch_size,) or cache_seqlens.dtype != torch.int32:
        raise ValueError("cache_seqlens must be an int or a [B] int32 tensor")
    if cache_len_delta:
        cache_seqlens = cache_seqlens + torch.full_like(cache_seqlens, cache_len_delta)

    if rotary_applied_in_wrapper:
        rotary_cos = None
        rotary_sin = None
        rotary_interleaved_for_kernel = False
    else:
        rotary_interleaved_for_kernel = rotary_interleaved

    has_rotary = rotary_cos is not None or rotary_sin is not None
    if has_rotary:
        if rotary_cos is None or rotary_sin is None:
            raise ValueError("rotary_cos and rotary_sin must be provided together")
        if rotary_cos.shape != rotary_sin.shape or rotary_cos.ndim != 2:
            raise ValueError("rotary_cos and rotary_sin must have matching [S, C] shape")
        if rotary_cos.dtype != q.dtype or rotary_sin.dtype != q.dtype:
            rotary_cos = rotary_cos.to(dtype=q.dtype)
            rotary_sin = rotary_sin.to(dtype=q.dtype)
        rotary_cos = rotary_cos.contiguous()
        rotary_sin = rotary_sin.contiguous()
        rotary_cols = rotary_cos.shape[1]
    else:
        rotary_cols = 0
        rotary_cos = torch.empty((1, head_dim), device=q.device, dtype=q.dtype)
        rotary_sin = torch.empty((1, head_dim), device=q.device, dtype=q.dtype)

    # The MMA decode kernel handles only the tensor-core friendly decode shape
    # (single query step, HEAD_DIM=128, bf16, GQA group <= 16, 128-aligned
    # cache). Everything else stays on the scalar path.
    mma_decode_env = os.environ.get("FLYDSL_KVCACHE_MMA_DECODE", "1") == "1"
    # Default BN=32 matches ixinfer CHUNK_SIZE=16 with two chunks per flash-decoding
    # group (fixLength=32 when Sk=512 and groups=16). Override via FLYDSL_KVCACHE_MMA_BN.
    mma_block_n = int(os.environ.get("FLYDSL_KVCACHE_MMA_BN", "32"))
    use_v5_decode = (
        mma_decode_env
        and head_dim == 128
        and q.dtype is torch.bfloat16
        and seqlen_q == 1
        and 1 <= (num_heads // num_kv_heads) <= 16
        and kernel_paged
        and not upstream_cache_layout
        and page_block_size == 16
        and max_seqlen_k >= 1024
        and max_seqlen_k % 16 == 0
        and window_size == (-1, -1)
        and k_cache_view.is_contiguous()
        and v_cache_view.is_contiguous()
    )
    use_mma_decode = (
        not use_v5_decode
        and mma_decode_env
        and head_dim == 128
        and q.dtype is torch.bfloat16
        and seqlen_q == 1
        and 1 <= (num_heads // num_kv_heads) <= 16
        and max_seqlen_k % 128 == 0
        and window_size == (-1, -1)
        and k_cache_view.is_contiguous()
        and v_cache_view.is_contiguous()
    )

    if use_mma_decode or use_v5_decode:
        # Tail-pad q_work so the kernel's unguarded 16-row Q async-load stays
        # in-bounds when the last GQA group has fewer than 16 heads.
        q_elems = batch_size * seqlen_q * num_heads * head_dim
        q_storage = torch.empty(
            q_elems + 16 * head_dim, device=q.device, dtype=q.dtype
        )
        q_work = q_storage[:q_elems].view(batch_size, seqlen_q, num_heads, head_dim)
        if not update_cache:
            q_work.copy_(q)
        out = torch.empty(
            (batch_size, seqlen_q, num_heads, head_dim), device=q.device, dtype=q.dtype
        )
    elif not update_cache:
        q_work = q.contiguous()
        out = torch.empty_like(q_work)
    else:
        q_work = torch.empty((batch_size, seqlen_q, num_heads, head_dim), device=q.device, dtype=q.dtype)
        out = torch.empty_like(q_work)
    dtype_str = _dtype_name(q.dtype)
    if use_v5_decode:
        from kernels.attention.iluvatar.mma_decode_splits import compute_v5_decode_config

        if num_splits == 0:
            effective_num_splits, _, _ = compute_v5_decode_config(
                batch_size=batch_size,
                seqlen_q=seqlen_q,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                max_seqlen_k=max_seqlen_k,
            )
        else:
            effective_num_splits = min(
                max(1, num_splits), max_seqlen_k // 16)
    elif use_mma_decode:
        from kernels.attention.iluvatar.mma_decode_splits import (
            compute_mma_decode_num_splits,
        )

        if num_splits == 0:
            effective_num_splits = compute_mma_decode_num_splits(
                batch_size=batch_size,
                seqlen_q=seqlen_q,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                max_seqlen_k=max_seqlen_k,
                block_n=mma_block_n,
            )
        elif num_splits >= 2:
            effective_num_splits = min(num_splits, max_seqlen_k // mma_block_n)
            while (
                effective_num_splits > 1
                and (max_seqlen_k // mma_block_n) % effective_num_splits != 0
            ):
                effective_num_splits -= 1
        else:
            effective_num_splits = 1
    elif num_splits == 0:
        # Flash-decoding style split-KV is useful when one decode row launches too
        # few CTAs to occupy the device. Keep short-cache cases on the cheaper
        # single-kernel path.
        low_parallelism = batch_size * seqlen_q * num_heads < 64
        effective_num_splits = 8 if max_seqlen_k >= 2048 else 4
        if seqlen_q != 1 or max_seqlen_k < 1024 or not low_parallelism:
            effective_num_splits = 1
    else:
        effective_num_splits = num_splits
    effective_num_splits = max(1, min(effective_num_splits, max_seqlen_k))
    if effective_num_splits > 1:
        group_max = torch.empty(
            (batch_size, seqlen_q, num_heads, effective_num_splits),
            device=q.device,
            dtype=torch.float32,
        )
        group_sum = torch.empty_like(group_max)
        partial_out = torch.empty(
            (batch_size, seqlen_q, num_heads, effective_num_splits, head_dim),
            device=q.device,
            dtype=torch.float32,
        )
    else:
        group_max = torch.empty((1, 1, 1, 1), device=q.device, dtype=torch.float32)
        group_sum = torch.empty_like(group_max)
        partial_out = torch.empty((1, 1, 1, 1, 1), device=q.device, dtype=torch.float32)
    launcher = build_flash_attn_with_kvcache_module(
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_seqlen_k=max_seqlen_k,
        dtype_str=dtype_str,
        paged=kernel_paged,
        page_block_size=page_block_size,
        has_rotary=has_rotary,
        rotary_cols=rotary_cols,
        causal=causal,
        window_size=window_size,
        rotary_interleaved=rotary_interleaved_for_kernel,
        update_cache=update_cache,
        num_splits=effective_num_splits,
        upstream_cache_layout=upstream_cache_layout,
        use_mma_decode=use_mma_decode,
        use_v5_decode=use_v5_decode,
        mma_block_n=mma_block_n,
    )
    launch_args = (
        qkv,
        q_work,
        k_cache_view,
        v_cache_view,
        cache_seqlens,
        block_table_for_kernel,
        rotary_cos,
        rotary_sin,
        out,
        group_max,
        group_sum,
        partial_out,
        stream,
    )
    compile_key = (launcher, q.device)
    compiled = _COMPILED_LAUNCH_CACHE.get(compile_key)
    if compiled is None:
        compiled = flyc.compile(launcher, *launch_args)
        _COMPILED_LAUNCH_CACHE[compile_key] = compiled
    compiled(*launch_args)
    return out

