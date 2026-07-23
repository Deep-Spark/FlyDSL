# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Scalar update and attention kernels for Iluvatar KV-cache FlashAttention."""

import math
from typing import Optional, Tuple

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from flydsl.expr.typing import full

ATTN_THREADS = 256
WARP_SIZE = 64
RED_SLOTS = ATTN_THREADS // WARP_SIZE
_LOG2E = 1.4426950408889634


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

        def _load_cache_leftpad():
            leftpad_div = fx.logical_divide(CacheLeftpad, fx.make_layout(1, 1))
            return _load_i32(fx.slice(leftpad_div, (None, b)))

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
            return b, pos + _load_cache_leftpad()

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
        copy_atom = fx.make_copy_atom(fx.UniversalCopy16b(), elem_dtype)
        fm_fast = arith.FastMathFlags.fast
        zero = fx.Float32(0.0)
        neg_inf = fx.Float32(float("-inf"))

        smem = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_red = smem.s_red.view(fx.make_layout(red_slots, 1))
        s_prob = smem.s_prob.view(fx.make_layout(prob_slots, 1))
        s_q = smem.s_q.view(fx.make_layout(head_dim, 1))
        s_blocks = smem.s_blocks.view(fx.make_layout(max_blocks_per_seq, 1))

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
                    phys_block = fx.Int32(
                        fx.ptr_load(
                            fx.add_offset(fx.get_iter(table_row), fx.make_int_tuple(block)),
                            T.i32,
                        )
                    )
                return phys_block, block_off
            leftpad_div = fx.logical_divide(CacheLeftpad, fx.make_layout(1, 1))
            return b, tok + _load_i32(fx.slice(leftpad_div, (None, b)))

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
                partial = fx.memref_load(s_red, lane_safe)
                total = _warp_reduce(in_range.select(partial, neutral), mode)
                if lane == 0:
                    fx.memref_store(total, s_red, 0)
            gpu.barrier()
            return fx.memref_load(s_red, 0)

        c0 = fx.Int32(0)
        c1 = fx.Int32(1)
        c_red = fx.Int32(RED_SLOTS)
        score_lane = tid % WARP_SIZE
        score_warp = tid // WARP_SIZE
        score_start = fx.Int32(score_warp)

        loop_results = [neg_inf]
        for tok, state in range(score_start, cache_len, c_red, init=loop_results):
            thread_max = fx.Float32(state[0])
            tok_i = fx.Int32(tok)
            keep = _keep_token(tok_i)
            token_score = keep.select(_score_coop(tok_i), neg_inf)
            if score_lane == 0:
                fx.memref_store(token_score, s_prob, tok_i)
                thread_max = arith.maxnumf(thread_max, token_score)
            loop_results = yield [thread_max]
        thread_max = fx.Float32(loop_results)
        max_score = _block_reduce(thread_max, "max")

        loop_results = [zero]
        for tok, state in range(score_start, cache_len, c_red, init=loop_results):
            thread_sum = fx.Float32(state[0])
            tok_i = fx.Int32(tok)
            if score_lane == 0:
                score_v = fx.memref_load(s_prob, tok_i)
                exp_v = fmath.exp2((score_v - max_score) * _LOG2E, fastmath=fm_fast)
                exp_safe = (score_v > neg_inf).select(exp_v, zero)
                fx.memref_store(exp_safe, s_prob, tok_i)
                thread_sum = thread_sum + exp_safe
            loop_results = yield [thread_sum]
        thread_sum = fx.Float32(loop_results)
        denom = _block_reduce(thread_sum, "sum")
        has_tokens = denom > zero

        if fx.const_expr(use_vec2_pv):
            if tid < head_dim // 2:
                d0 = fx.Int32(tid * fx.Int32(2))
                loop_results = [zero, zero]
                for tok, state in range(c0, cache_len, c1, init=loop_results):
                    acc0 = fx.Float32(state[0])
                    acc1 = fx.Float32(state[1])
                    tok_i = fx.Int32(tok)
                    prob = fx.memref_load(s_prob, tok_i) / denom
                    v0, v1 = _load_v2(tok_i, d0)
                    acc0 = acc0 + prob * v0
                    acc1 = acc1 + prob * v1
                    loop_results = yield [acc0, acc1]
                acc0 = fx.Float32(loop_results[0])
                acc1 = fx.Float32(loop_results[1])
                _store_out(d0, has_tokens.select(acc0, zero))
                _store_out(d0 + fx.Int32(1), has_tokens.select(acc1, zero))
        else:
            if tid < head_dim:
                d = fx.Int32(tid)
                loop_results = [zero]
                for tok, state in range(c0, cache_len, c1, init=loop_results):
                    acc = fx.Float32(state[0])
                    tok_i = fx.Int32(tok)
                    prob = fx.memref_load(s_prob, tok_i) / denom
                    acc = acc + prob * _load_v(tok_i, d)
                    loop_results = yield [acc]
                acc = fx.Float32(loop_results)
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
        copy_atom = fx.make_copy_atom(fx.UniversalCopy16b(), elem_dtype)
        fm_fast = arith.FastMathFlags.fast
        zero = fx.Float32(0.0)
        neg_inf = fx.Float32(float("-inf"))

        smem = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_red = smem.s_red.view(fx.make_layout(red_slots, 1))
        s_prob = smem.s_prob.view(fx.make_layout(prob_slots, 1))
        s_q = smem.s_q.view(fx.make_layout(head_dim, 1))
        s_blocks = smem.s_blocks.view(fx.make_layout(max_blocks_per_seq, 1))

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
                    phys_block = fx.Int32(
                        fx.ptr_load(
                            fx.add_offset(fx.get_iter(table_row), fx.make_int_tuple(block)),
                            T.i32,
                        )
                    )
                return phys_block, block_off
            leftpad_div = fx.logical_divide(CacheLeftpad, fx.make_layout(1, 1))
            return b, tok + _load_i32(fx.slice(leftpad_div, (None, b)))

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
                partial = fx.memref_load(s_red, lane_safe)
                total = _warp_reduce(in_range.select(partial, neutral), mode)
                if lane == 0:
                    fx.memref_store(total, s_red, 0)
            gpu.barrier()
            return fx.memref_load(s_red, 0)

        c1 = fx.Int32(1)
        c_red = fx.Int32(RED_SLOTS)
        score_lane = tid % WARP_SIZE
        score_warp = tid // WARP_SIZE
        score_start = split_start + score_warp

        max_loop_results = [neg_inf]
        for tok, state in range(score_start, split_end, c_red, init=max_loop_results):
            thread_max = fx.Float32(state[0])
            tok_i = fx.Int32(tok)
            keep = _keep_token(tok_i)
            token_score = keep.select(_score_coop(tok_i), neg_inf)
            if score_lane == 0:
                local_tok = tok_i - split_start
                fx.memref_store(token_score, s_prob, local_tok)
                thread_max = arith.maxnumf(thread_max, token_score)
            max_loop_results = yield [thread_max]
        thread_max = fx.Float32(max_loop_results)
        max_score = has_split_tokens.select(_block_reduce(thread_max, "max"), neg_inf)

        sum_loop_results = [zero]
        for tok, state in range(score_start, split_end, c_red, init=sum_loop_results):
            thread_sum = fx.Float32(state[0])
            tok_i = fx.Int32(tok)
            if score_lane == 0:
                local_tok = tok_i - split_start
                score_v = fx.memref_load(s_prob, local_tok)
                exp_v = fmath.exp2((score_v - max_score) * _LOG2E, fastmath=fm_fast)
                exp_safe = (score_v > neg_inf).select(exp_v, zero)
                fx.memref_store(exp_safe, s_prob, local_tok)
                thread_sum = thread_sum + exp_safe
            sum_loop_results = yield [thread_sum]
        thread_sum = fx.Float32(sum_loop_results)
        denom = _block_reduce(thread_sum, "sum")

        if tid == 0:
            _store_group_value(GroupMax, max_score)
            _store_group_value(GroupSum, denom)

        if fx.const_expr(use_krepeat_pv):
            if tid < WARP_SIZE:
                lane = fx.Int32(tid)
                pv_loop_results = [zero for _ in range_constexpr(k_repeats)]
                for tok, state in range(split_start, split_end, c1, init=pv_loop_results):
                    tok_i = fx.Int32(tok)
                    local_tok = tok_i - split_start
                    prob_num = fx.memref_load(s_prob, local_tok)
                    next_accs = []
                    for repeat_const in range_constexpr(k_repeats):
                        acc = fx.Float32(state[repeat_const])
                        d = lane + fx.Int32(repeat_const * WARP_SIZE)
                        acc = acc + prob_num * _load_v(tok_i, d)
                        next_accs.append(acc)
                    pv_loop_results = yield next_accs
                for repeat_const in range_constexpr(k_repeats):
                    d = lane + fx.Int32(repeat_const * WARP_SIZE)
                    _store_partial(d, fx.Float32(pv_loop_results[repeat_const]))
        elif fx.const_expr(use_vec2_pv):
            if tid < head_dim // 2:
                d0 = fx.Int32(tid * fx.Int32(2))
                pv_loop_results = [zero, zero]
                for tok, state in range(split_start, split_end, c1, init=pv_loop_results):
                    acc0 = fx.Float32(state[0])
                    acc1 = fx.Float32(state[1])
                    tok_i = fx.Int32(tok)
                    local_tok = tok_i - split_start
                    prob_num = fx.memref_load(s_prob, local_tok)
                    v0, v1 = _load_v2(tok_i, d0)
                    acc0 = acc0 + prob_num * v0
                    acc1 = acc1 + prob_num * v1
                    pv_loop_results = yield [acc0, acc1]
                acc0 = fx.Float32(pv_loop_results[0])
                acc1 = fx.Float32(pv_loop_results[1])
                _store_partial(d0, acc0)
                _store_partial(d0 + fx.Int32(1), acc1)
        else:
            if tid < head_dim:
                d = fx.Int32(tid)
                pv_loop_results = [zero]
                for tok, state in range(split_start, split_end, c1, init=pv_loop_results):
                    acc = fx.Float32(state[0])
                    tok_i = fx.Int32(tok)
                    local_tok = tok_i - split_start
                    prob_num = fx.memref_load(s_prob, local_tok)
                    acc = acc + prob_num * _load_v(tok_i, d)
                    pv_loop_results = yield [acc]
                acc = fx.Float32(pv_loop_results)
                _store_partial(d, acc)

    return update_cache_kernel, attention_kernel, split_attention_kernel
