# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Scalar update and attention kernels for Iluvatar KV-cache FlashAttention."""

import math
from typing import Optional, Tuple

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.ixdl as ixdl
from flydsl.expr import arith, gpu, range_constexpr
from flydsl.expr import math as fmath

ATTN_THREADS = 256
WARP_SIZE = 64
RED_SLOTS = ATTN_THREADS // WARP_SIZE
_LOG2E = 1.4426950408889634


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _kv_elem_offset(
    outer,
    head,
    token,
    d,
    *,
    paged: bool,
    upstream_cache_layout: bool,
    page_block_size: int,
    max_seqlen_k: int,
    num_kv_heads: int,
    head_dim: int,
    cache_strides: tuple[int, int, int, int] | None = None,
):
    """Flat element offset into an HND / upstream KV cache."""
    seq_cap = page_block_size if paged else max_seqlen_k
    if cache_strides is not None:
        if upstream_cache_layout:
            block_stride, token_stride, head_stride, _ = cache_strides
        else:
            block_stride, head_stride, token_stride, _ = cache_strides
        return (
            outer * fx.Int32(block_stride)
            + head * fx.Int32(head_stride)
            + token * fx.Int32(token_stride)
            + d
        )
    if upstream_cache_layout:
        return ((outer * fx.Int32(seq_cap) + token) * fx.Int32(num_kv_heads) + head) * fx.Int32(head_dim) + d
    return ((outer * fx.Int32(num_kv_heads) + head) * fx.Int32(seq_cap) + token) * fx.Int32(head_dim) + d


def _kv_elem_ptr(cache, outer, head, token, d, *, elem_dtype, **offset_kwargs):
    elem = _kv_elem_offset(outer, head, token, d, **offset_kwargs)
    ptr = fx.recast_iter(
        fx.PointerType.get(elem_dtype.ir_type, fx.AddressSpace.Global),
        fx.get_iter(cache),
    )
    return fx.add_offset(ptr, fx.make_int_tuple(elem))


def _normalise_window(causal: bool, window_size: Tuple[int, int], max_seqlen: int) -> Tuple[int, int]:
    left, right = window_size
    if causal:
        right = 0
    if left >= max_seqlen:
        left = -1
    if right >= max_seqlen:
        right = -1
    return left, right


def build_scalar_kvcache_kernels(
    *,
    seqlen_q: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seqlen_k: int,
    dtype_str: str,
    paged: bool,
    page_block_size: int,
    has_rotary: bool,
    rotary_cols: int,
    causal: bool,
    window_size: Tuple[int, int],
    rotary_interleaved: bool,
    num_splits: int,
    upstream_cache_layout: bool,
    softmax_scale: Optional[float],
    softcap: float,
    cache_strides: tuple[int, int, int, int] | None = None,
):
    repeats = num_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim) if softmax_scale is None else softmax_scale
    use_softcap = softcap > 0.0
    red_slots = RED_SLOTS
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
    if cache_strides is not None and cache_strides[-1] != 1:
        raise ValueError("cache_strides last dimension must be contiguous")
    kv_offset_kwargs = dict(
        paged=paged,
        upstream_cache_layout=upstream_cache_layout,
        page_block_size=page_block_size,
        max_seqlen_k=max_seqlen_k,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        cache_strides=cache_strides,
    )

    @fx.struct
    class SharedStorage:
        s_red: fx.Array[fx.Float32, RED_SLOTS, 16]
        s_prob: fx.Array[fx.Float32, prob_slots]
        s_q: fx.Array[fx.Float32, head_dim]
        s_blocks: fx.Array[fx.Int32, max_blocks_per_seq]

    @flyc.kernel
    def update_cache_kernel(
        QKV: fx.Tensor,
        QWork: fx.Tensor,
        KCache: fx.Tensor,
        VCache: fx.Tensor,
        CacheSeqLens: fx.Tensor,
        BlockTable: fx.Tensor,
        CacheLeftpad: fx.Tensor,
        RotaryCos: fx.Tensor,
        RotarySin: fx.Tensor,
    ):
        head = fx.block_idx.x
        sq = fx.Int32(fx.block_idx.y)
        b = fx.Int32(fx.block_idx.z)
        tid = fx.thread_idx.x

        def _as_elem(value):
            if not hasattr(value, "to"):
                value = fx.Float32(value)
            return value.to(elem_dtype)

        def _rotary_index(d):
            if rotary_cols == head_dim:
                return d
            if fx.const_expr(rotary_interleaved):
                return d // fx.Int32(2)
            return d % fx.Int32(head_dim // 2)

        def _load_cos_sin(pos, d):
            ridx = _rotary_index(d)
            return RotaryCos[pos, ridx].to(fx.Float32), RotarySin[pos, ridx].to(fx.Float32)

        def _qkv_scalar(row_head, d):
            return QKV[b, sq, row_head, d].to(fx.Float32)

        def _load_cache_len():
            return CacheSeqLens[b]

        def _load_cache_leftpad():
            return CacheLeftpad[b]

        def _store_q(head_q, d, value):
            QWork[b, sq, head_q, d] = _as_elem(value)

        # Dynamic ``BlockTable[b, idx]`` does not lower reliably on this path;
        # use the same SGPR ptr_load as MMA / SIMT decode.
        table_row = fx.slice(BlockTable, (b, None))
        table_base = fx.inttoptr(
            fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global),
            ixdl.readfirstlane(fx.ptrtoint(fx.get_iter(table_row))),
        )

        def _load_phys_block(logical_block):
            # update_cache issues one logical block per active thread; do not
            # readfirstlane the index or every lane collapses to lane0's block.
            loaded = fx.ptr_load(
                fx.add_offset(table_base, fx.make_int_tuple(logical_block)),
                fx.Int32,
            )
            return loaded

        def _cache_indices(pos):
            if fx.const_expr(paged):
                block = pos // fx.Int32(page_block_size)
                block_off = pos % fx.Int32(page_block_size)
                return _load_phys_block(block), block_off
            return b, pos + _load_cache_leftpad()

        def _store_kv(kv_head, d, k_value, v_value):
            cache_len = _load_cache_len()
            pos = cache_len - fx.Int32(seqlen_q) + sq
            outer, token = _cache_indices(pos)
            k_elem = _as_elem(k_value)
            v_elem = _as_elem(v_value)
            fx.ptr_store(
                k_elem,
                _kv_elem_ptr(KCache, outer, kv_head, token, d, elem_dtype=elem_dtype, **kv_offset_kwargs),
            )
            fx.ptr_store(
                v_elem,
                _kv_elem_ptr(VCache, outer, kv_head, token, d, elem_dtype=elem_dtype, **kv_offset_kwargs),
            )

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
                    if fx.const_expr(causal or window_size != (-1, -1)):
                        pos = cache_len - fx.Int32(seqlen_q) + sq
                    else:
                        pos = cache_len - fx.Int32(seqlen_q)
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
        CacheLeftpad: fx.Tensor,
        Out: fx.Tensor,
    ):
        row = fx.Int32(fx.block_idx.x)
        b = fx.Int32(fx.block_idx.y)
        hq = fx.Int32(fx.block_idx.z)
        tid = fx.thread_idx.x
        sq = row
        kv_head = hq // fx.Int32(repeats)
        fm_fast = arith.FastMathFlags.fast
        zero = fx.Float32(0.0)
        neg_inf = fx.Float32(float("-inf"))

        smem = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_red = smem.s_red.view(fx.make_layout(red_slots, 1))
        s_prob = smem.s_prob.view(fx.make_layout(prob_slots, 1))
        s_q = smem.s_q.view(fx.make_layout(head_dim, 1))
        s_blocks = smem.s_blocks.view(fx.make_layout(max_blocks_per_seq, 1))

        # Dynamic ``BlockTable[b, idx]`` does not lower reliably on this path;
        # use the same SGPR ptr_load as MMA / SIMT decode.
        table_row = fx.slice(BlockTable, (b, None))
        table_base = fx.inttoptr(
            fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global),
            ixdl.readfirstlane(fx.ptrtoint(fx.get_iter(table_row))),
        )

        def _load_phys_block_lane(logical_block):
            # Per-thread fill of s_blocks: each lane owns a distinct index.
            return fx.ptr_load(
                fx.add_offset(table_base, fx.make_int_tuple(logical_block)),
                fx.Int32,
            )

        def _load_phys_block(logical_block):
            # Warp-uniform token path: broadcast the block id / value to SGPRs.
            logical_block = ixdl.readfirstlane(logical_block)
            loaded = fx.ptr_load(
                fx.add_offset(table_base, fx.make_int_tuple(logical_block)),
                fx.Int32,
            )
            return ixdl.readfirstlane(loaded)

        if fx.const_expr(use_block_table_cache):
            if tid < max_blocks_per_seq:
                fx.memref_store(_load_phys_block_lane(tid), s_blocks, tid)
            gpu.barrier()

        def _cache_indices(tok):
            if fx.const_expr(paged):
                block = tok // fx.Int32(page_block_size)
                block_off = tok % fx.Int32(page_block_size)
                if fx.const_expr(use_block_table_cache):
                    phys_block = s_blocks[block]
                else:
                    phys_block = _load_phys_block(block)
                return phys_block, block_off
            return b, tok + CacheLeftpad[b]

        def _load_q(d):
            return QWork[b, sq, hq, d].to(fx.Float32)

        def _load_q_cached(d):
            return s_q[d]

        def _load_kv_elem(cache, tok, d):
            # Tensor indexing with a dynamic physical-block outer dim does not
            # lower correctly here; address through a flat gmem pointer instead.
            outer, token = _cache_indices(tok)
            ptr = _kv_elem_ptr(
                cache, outer, kv_head, token, d, elem_dtype=elem_dtype, **kv_offset_kwargs
            )
            return fx.ptr_load(ptr, elem_dtype).to(fx.Float32)

        def _load_kv_pair(cache, tok, d):
            outer, token = _cache_indices(tok)
            ptr = _kv_elem_ptr(
                cache, outer, kv_head, token, d, elem_dtype=elem_dtype, **kv_offset_kwargs
            )
            ptr_i32 = fx.recast_iter(
                fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global),
                ptr,
            )
            word = fx.ptr_load(ptr_i32, fx.Int32)
            vals = fx.Vector.from_elements([word], fx.Int32).bitcast(elem_dtype)
            return vals[0].to(fx.Float32), vals[1].to(fx.Float32)

        def _load_k(tok, d):
            return _load_kv_elem(KCache, tok, d)

        def _load_k2(tok, d):
            return _load_kv_pair(KCache, tok, d)

        def _load_v(tok, d):
            return _load_kv_elem(VCache, tok, d)

        def _load_v2(tok, d):
            return _load_kv_pair(VCache, tok, d)

        def _store_out(d, value):
            if not hasattr(value, "to"):
                value = fx.Float32(value)
            Out[b, sq, hq, d] = value.to(elem_dtype)

        cache_len = CacheSeqLens[b]
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

        def _warp_reduce(value, mode):
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
            score = _warp_reduce(dot, "sum") * scale
            if fx.const_expr(use_softcap):
                score = fmath.tanh(score / fx.Float32(softcap), fastmath=fm_fast) * fx.Float32(softcap)
            return score

        def _block_reduce(value, mode):
            if red_slots == 1:
                return _warp_reduce(value, mode)
            lane = tid % WARP_SIZE
            warp = tid // WARP_SIZE
            neutral = neg_inf if mode == "max" else zero
            warp_value = _warp_reduce(value, mode)
            if lane == 0:
                fx.memref_store(warp_value, s_red, warp)
            gpu.barrier()
            if warp == 0:
                in_range = lane < red_slots
                lane_safe = in_range.select(lane, fx.Int32(0))
                partial = s_red[lane_safe]
                total = _warp_reduce(in_range.select(partial, neutral), mode)
                if lane == 0:
                    fx.memref_store(total, s_red, 0)
            gpu.barrier()
            return s_red[0]

        c0 = fx.Int32(0)
        c1 = fx.Int32(1)
        c_red = fx.Int32(RED_SLOTS)
        score_lane = tid % WARP_SIZE
        score_warp = tid // WARP_SIZE
        score_start = fx.Int32(score_warp)

        thread_max = neg_inf
        for tok in range(score_start, cache_len, c_red):
            tok_i = fx.Int32(tok)
            keep = _keep_token(tok_i)
            token_score = keep.select(_score_coop(tok_i), neg_inf)
            if score_lane == 0:
                fx.memref_store(token_score, s_prob, tok_i)
                thread_max = arith.maxnumf(thread_max, token_score)
        max_score = _block_reduce(thread_max, "max")

        thread_sum = zero
        for tok in range(score_start, cache_len, c_red):
            tok_i = fx.Int32(tok)
            if score_lane == 0:
                score_v = s_prob[tok_i]
                exp_v = fmath.exp2((score_v - max_score) * _LOG2E, fastmath=fm_fast)
                exp_safe = (score_v > neg_inf).select(exp_v, zero)
                fx.memref_store(exp_safe, s_prob, tok_i)
                thread_sum = thread_sum + exp_safe
        denom = _block_reduce(thread_sum, "sum")
        has_tokens = denom > zero

        if fx.const_expr(use_vec2_pv):
            if tid < head_dim // 2:
                d0 = fx.Int32(tid * fx.Int32(2))
                acc0 = zero
                acc1 = zero
                for tok in range(c0, cache_len, c1):
                    tok_i = fx.Int32(tok)
                    prob = s_prob[tok_i] / denom
                    v0, v1 = _load_v2(tok_i, d0)
                    acc0 = acc0 + prob * v0
                    acc1 = acc1 + prob * v1
                _store_out(d0, has_tokens.select(acc0, zero))
                _store_out(d0 + fx.Int32(1), has_tokens.select(acc1, zero))
        else:
            if tid < head_dim:
                d = fx.Int32(tid)
                acc = zero
                for tok in range(c0, cache_len, c1):
                    tok_i = fx.Int32(tok)
                    prob = s_prob[tok_i] / denom
                    acc = acc + prob * _load_v(tok_i, d)
                _store_out(d, has_tokens.select(acc, zero))

    @flyc.kernel(known_block_size=[ATTN_THREADS, 1, 1])
    def split_attention_kernel(
        QWork: fx.Tensor,
        KCache: fx.Tensor,
        VCache: fx.Tensor,
        CacheSeqLens: fx.Tensor,
        BlockTable: fx.Tensor,
        CacheLeftpad: fx.Tensor,
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
        fm_fast = arith.FastMathFlags.fast
        zero = fx.Float32(0.0)
        neg_inf = fx.Float32(float("-inf"))
        smem = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_red = smem.s_red.view(fx.make_layout(red_slots, 1))
        s_prob = smem.s_prob.view(fx.make_layout(prob_slots, 1))
        s_q = smem.s_q.view(fx.make_layout(head_dim, 1))
        s_blocks = smem.s_blocks.view(fx.make_layout(max_blocks_per_seq, 1))

        # Dynamic ``BlockTable[b, idx]`` does not lower reliably on this path;
        # use the same SGPR ptr_load as MMA / SIMT decode.
        table_row = fx.slice(BlockTable, (b, None))
        table_base = fx.inttoptr(
            fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global),
            ixdl.readfirstlane(fx.ptrtoint(fx.get_iter(table_row))),
        )

        def _load_phys_block_lane(logical_block):
            # Per-thread fill of s_blocks: each lane owns a distinct index.
            return fx.ptr_load(
                fx.add_offset(table_base, fx.make_int_tuple(logical_block)),
                fx.Int32,
            )

        def _load_phys_block(logical_block):
            # Warp-uniform token path: broadcast the block id / value to SGPRs.
            logical_block = ixdl.readfirstlane(logical_block)
            loaded = fx.ptr_load(
                fx.add_offset(table_base, fx.make_int_tuple(logical_block)),
                fx.Int32,
            )
            return ixdl.readfirstlane(loaded)

        if fx.const_expr(use_block_table_cache):
            if tid < max_blocks_per_seq:
                fx.memref_store(_load_phys_block_lane(tid), s_blocks, tid)
            gpu.barrier()

        def _cache_indices(tok):
            if fx.const_expr(paged):
                block = tok // fx.Int32(page_block_size)
                block_off = tok % fx.Int32(page_block_size)
                if fx.const_expr(use_block_table_cache):
                    phys_block = s_blocks[block]
                else:
                    phys_block = _load_phys_block(block)
                return phys_block, block_off
            return b, tok + CacheLeftpad[b]

        def _load_q(d):
            return QWork[b, sq, hq, d].to(fx.Float32)

        def _load_q_cached(d):
            return s_q[d]

        def _load_kv_elem(cache, tok, d):
            # Tensor indexing with a dynamic physical-block outer dim does not
            # lower correctly here; address through a flat gmem pointer instead.
            outer, token = _cache_indices(tok)
            ptr = _kv_elem_ptr(
                cache, outer, kv_head, token, d, elem_dtype=elem_dtype, **kv_offset_kwargs
            )
            return fx.ptr_load(ptr, elem_dtype).to(fx.Float32)

        def _load_kv_pair(cache, tok, d):
            outer, token = _cache_indices(tok)
            ptr = _kv_elem_ptr(
                cache, outer, kv_head, token, d, elem_dtype=elem_dtype, **kv_offset_kwargs
            )
            ptr_i32 = fx.recast_iter(
                fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global),
                ptr,
            )
            word = fx.ptr_load(ptr_i32, fx.Int32)
            vals = fx.Vector.from_elements([word], fx.Int32).bitcast(elem_dtype)
            return vals[0].to(fx.Float32), vals[1].to(fx.Float32)

        def _load_k(tok, d):
            return _load_kv_elem(KCache, tok, d)

        def _load_k2(tok, d):
            return _load_kv_pair(KCache, tok, d)

        def _load_v(tok, d):
            return _load_kv_elem(VCache, tok, d)

        def _load_v2(tok, d):
            return _load_kv_pair(VCache, tok, d)

        def _store_group_value(tensor, value):
            tensor[b, row, hq, split] = value

        def _store_partial(d, value):
            PartialOut[b, row, hq, split, d] = value

        cache_len = CacheSeqLens[b]
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

        def _warp_reduce(value, mode):
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
            score = _warp_reduce(dot, "sum") * scale
            if fx.const_expr(use_softcap):
                score = fmath.tanh(score / fx.Float32(softcap), fastmath=fm_fast) * fx.Float32(softcap)
            return score

        def _block_reduce(value, mode):
            if red_slots == 1:
                return _warp_reduce(value, mode)
            lane = tid % WARP_SIZE
            warp = tid // WARP_SIZE
            neutral = neg_inf if mode == "max" else zero
            warp_value = _warp_reduce(value, mode)
            if lane == 0:
                fx.memref_store(warp_value, s_red, warp)
            gpu.barrier()
            if warp == 0:
                in_range = lane < red_slots
                lane_safe = in_range.select(lane, fx.Int32(0))
                partial = s_red[lane_safe]
                total = _warp_reduce(in_range.select(partial, neutral), mode)
                if lane == 0:
                    fx.memref_store(total, s_red, 0)
            gpu.barrier()
            return s_red[0]

        c1 = fx.Int32(1)
        c_red = fx.Int32(RED_SLOTS)
        score_lane = tid % WARP_SIZE
        score_warp = tid // WARP_SIZE
        score_start = split_start + score_warp

        thread_max = neg_inf
        for tok in range(score_start, split_end, c_red):
            tok_i = fx.Int32(tok)
            keep = _keep_token(tok_i)
            token_score = keep.select(_score_coop(tok_i), neg_inf)
            if score_lane == 0:
                local_tok = tok_i - split_start
                fx.memref_store(token_score, s_prob, local_tok)
                thread_max = arith.maxnumf(thread_max, token_score)
        max_score = has_split_tokens.select(_block_reduce(thread_max, "max"), neg_inf)

        thread_sum = zero
        for tok in range(score_start, split_end, c_red):
            tok_i = fx.Int32(tok)
            if score_lane == 0:
                local_tok = tok_i - split_start
                score_v = s_prob[local_tok]
                exp_v = fmath.exp2((score_v - max_score) * _LOG2E, fastmath=fm_fast)
                exp_safe = (score_v > neg_inf).select(exp_v, zero)
                fx.memref_store(exp_safe, s_prob, local_tok)
                thread_sum = thread_sum + exp_safe
        denom = _block_reduce(thread_sum, "sum")

        if tid == 0:
            _store_group_value(GroupMax, max_score)
            _store_group_value(GroupSum, denom)

        if fx.const_expr(use_krepeat_pv):
            if tid < WARP_SIZE:
                lane = fx.Int32(tid)
                accs = [zero for _ in range_constexpr(k_repeats)]
                for tok in range(split_start, split_end, c1):
                    tok_i = fx.Int32(tok)
                    local_tok = tok_i - split_start
                    prob_num = s_prob[local_tok]
                    next_accs = []
                    for repeat_const in range_constexpr(k_repeats):
                        d = lane + fx.Int32(repeat_const * WARP_SIZE)
                        next_accs.append(accs[repeat_const] + prob_num * _load_v(tok_i, d))
                    accs = next_accs
                for repeat_const in range_constexpr(k_repeats):
                    d = lane + fx.Int32(repeat_const * WARP_SIZE)
                    _store_partial(d, fx.Float32(accs[repeat_const]))
        elif fx.const_expr(use_vec2_pv):
            if tid < head_dim // 2:
                d0 = fx.Int32(tid * fx.Int32(2))
                acc0 = zero
                acc1 = zero
                for tok in range(split_start, split_end, c1):
                    tok_i = fx.Int32(tok)
                    local_tok = tok_i - split_start
                    prob_num = s_prob[local_tok]
                    v0, v1 = _load_v2(tok_i, d0)
                    acc0 = acc0 + prob_num * v0
                    acc1 = acc1 + prob_num * v1
                _store_partial(d0, acc0)
                _store_partial(d0 + fx.Int32(1), acc1)
        else:
            if tid < head_dim:
                d = fx.Int32(tid)
                acc = zero
                for tok in range(split_start, split_end, c1):
                    tok_i = fx.Int32(tok)
                    local_tok = tok_i - split_start
                    prob_num = s_prob[local_tok]
                    acc = acc + prob_num * _load_v(tok_i, d)
                _store_partial(d, acc)

    return update_cache_kernel, attention_kernel, split_attention_kernel
