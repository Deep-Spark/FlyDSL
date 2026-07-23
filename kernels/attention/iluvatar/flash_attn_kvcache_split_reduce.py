# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Split-reduction kernels for Iluvatar KV-cache FlashAttention."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import full

ATTN_THREADS = 256
WARP_SIZE = 64
_LOG2E = 1.4426950408889634


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _pipelined_reduce_warp_count(num_splits: int) -> int:
    required_warps = _ceil_div(num_splits, 2)
    return min(16, 1 << (required_warps - 1).bit_length())


def build_split_reduce_kernels(*, head_dim: int, num_splits: int, dtype_str: str):
    elem_dtype = fx.BFloat16 if dtype_str == "bf16" else fx.Float16
    reduce_threads = max(WARP_SIZE, min(ATTN_THREADS, 1 << (head_dim - 1).bit_length()))
    pipelined_reduce_warps = _pipelined_reduce_warp_count(num_splits)
    pipelined_reduce_slots = max(1, pipelined_reduce_warps // 2)

    @fx.struct
    class PipelinedReduceStorage:
        max_values: fx.Array[fx.Float32, pipelined_reduce_slots]
        exp_sums: fx.Array[fx.Float32, pipelined_reduce_slots]
        partials: fx.Array[fx.Float32, pipelined_reduce_slots * head_dim]

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
        zero = fx.Float32(0.0)
        neg_inf = fx.Float32(float("-inf"))

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
        max_score = fx.Float32(max_score).shuffle_idx(fx.Int32(0), fx.Int32(WARP_SIZE))

        denom = zero
        if lane == fx.Int32(0):
            for split_const in range_constexpr(num_splits):
                split = fx.Int32(split_const)
                group_max = _load_group_value(GroupMax, split)
                group_sum = _load_group_value(GroupSum, split)
                scale_group = fmath.exp2((group_max - max_score) * _LOG2E, fastmath=fm_fast)
                denom = denom + group_sum * scale_group
        denom = fx.Float32(denom).shuffle_idx(fx.Int32(0), fx.Int32(WARP_SIZE))
        has_tokens = denom > zero

        d = fx.Int32(tid)
        acc = zero
        for split_const in range_constexpr(num_splits):
            split = fx.Int32(split_const)
            group_max = neg_inf
            if lane == fx.Int32(0):
                group_max = _load_group_value(GroupMax, split)
            group_max = fx.Float32(group_max).shuffle_idx(fx.Int32(0), fx.Int32(WARP_SIZE))
            scale_group = fmath.exp2((group_max - max_score) * _LOG2E, fastmath=fm_fast)
            if tid < head_dim:
                partial = _load_partial(split, d)
                acc = acc + partial * scale_group
        if tid < head_dim:
            _store_out(d, has_tokens.select(acc / denom, zero))

    @flyc.kernel(known_block_size=[pipelined_reduce_warps * WARP_SIZE, 1, 1])
    def pipelined_split_reduce_kernel(
        GroupMax: fx.Tensor,
        GroupSum: fx.Tensor,
        PartialOut: fx.Tensor,
        Out: fx.Tensor,
    ):
        """ixInfer-style reduction for the pipelined MMA split workspace."""
        row = fx.Int32(fx.block_idx.x)
        b = fx.Int32(fx.block_idx.y)
        hq = fx.Int32(fx.block_idx.z)
        tid = fx.thread_idx.x
        warp = tid // fx.Int32(WARP_SIZE)
        lane = tid % fx.Int32(WARP_SIZE)
        d0 = lane
        d1 = lane + fx.Int32(WARP_SIZE)
        d2 = lane + fx.Int32(2 * WARP_SIZE)
        d3 = lane + fx.Int32(3 * WARP_SIZE)
        copy_atom = fx.make_copy_atom(fx.UniversalCopy16b(), elem_dtype)
        fm_fast = arith.FastMathFlags.fast
        zero = fx.Float32(0.0)
        # Keep empty warps mergeable when num_splits < pipelined_reduce_warps.
        # A finite sentinel avoids the -inf - -inf NaN that otherwise occurs
        # before an empty warp is merged with a warp that has valid groups.
        neg_inf = fx.Float32(-3.40282e38)

        smem = fx.SharedAllocator().allocate(PipelinedReduceStorage).peek()
        s_max = smem.max_values.view(fx.make_layout(pipelined_reduce_slots, 1))
        s_sum = smem.exp_sums.view(fx.make_layout(pipelined_reduce_slots, 1))
        s_partial = smem.partials.view(fx.make_layout(pipelined_reduce_slots * head_dim, 1))

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
        # same group-to-warp mapping used by library's pageAttentionReduce.
        warp_max = neg_inf
        warp_sum = zero
        warp_out0 = zero
        warp_out1 = zero
        warp_out2 = zero
        warp_out3 = zero
        for group_iter in fx.range_constexpr(_ceil_div(num_splits, pipelined_reduce_warps)):
            split = warp + fx.Int32(group_iter * pipelined_reduce_warps)
            group_max = neg_inf
            group_sum = zero
            part0 = zero
            part1 = zero
            part2 = zero
            part3 = zero
            if split < fx.Int32(num_splits):
                if lane == fx.Int32(0):
                    group_max = _load_group_value(GroupMax, split)
                    group_sum = _load_group_value(GroupSum, split)
                group_max = fx.Float32(group_max).shuffle_idx(fx.Int32(0), fx.Int32(WARP_SIZE))
                group_sum = fx.Float32(group_sum).shuffle_idx(fx.Int32(0), fx.Int32(WARP_SIZE))
                part0 = _load_partial(split, d0)
                part1 = _load_partial(split, d1)
                if fx.const_expr(head_dim > 128):
                    part2 = _load_partial(split, d2)
                    part3 = _load_partial(split, d3)
            new_max = arith.maxnumf(warp_max, group_max)
            old_scale = fmath.exp2((warp_max - new_max) * _LOG2E, fastmath=fm_fast)
            group_scale = fmath.exp2((group_max - new_max) * _LOG2E, fastmath=fm_fast)
            warp_sum = warp_sum * old_scale + group_sum * group_scale
            warp_out0 = warp_out0 * old_scale + part0 * group_scale
            warp_out1 = warp_out1 * old_scale + part1 * group_scale
            if fx.const_expr(head_dim > 128):
                warp_out2 = warp_out2 * old_scale + part2 * group_scale
                warp_out3 = warp_out3 * old_scale + part3 * group_scale
            warp_max = new_max

        if lane == fx.Int32(0):
            fx.memref_store(warp_max, s_max, warp)
            fx.memref_store(warp_sum, s_sum, warp)
        fx.memref_store(warp_out0, s_partial, warp * fx.Int32(head_dim) + d0)
        fx.memref_store(warp_out1, s_partial, warp * fx.Int32(head_dim) + d1)
        if fx.const_expr(head_dim > 128):
            fx.memref_store(warp_out2, s_partial, warp * fx.Int32(head_dim) + d2)
            fx.memref_store(warp_out3, s_partial, warp * fx.Int32(head_dim) + d3)
        gpu.barrier()

        # Mirror library's LDS binary tree: upper-half warps spill their
        # partials, then lower-half warps merge one partner per stage.
        for stage in fx.range_constexpr(pipelined_reduce_warps.bit_length() - 1):
            part_warps = pipelined_reduce_warps >> stage
            middle_warp = part_warps // 2
            if warp >= fx.Int32(middle_warp) and warp < fx.Int32(part_warps):
                partner_slot = warp - fx.Int32(middle_warp)
                if lane == fx.Int32(0):
                    fx.memref_store(warp_max, s_max, partner_slot)
                    fx.memref_store(warp_sum, s_sum, partner_slot)
                fx.memref_store(warp_out0, s_partial, partner_slot * fx.Int32(head_dim) + d0)
                fx.memref_store(warp_out1, s_partial, partner_slot * fx.Int32(head_dim) + d1)
                if fx.const_expr(head_dim > 128):
                    fx.memref_store(warp_out2, s_partial, partner_slot * fx.Int32(head_dim) + d2)
                    fx.memref_store(warp_out3, s_partial, partner_slot * fx.Int32(head_dim) + d3)
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
                warp_out0 = warp_out0 * old_scale + fx.memref_load(s_partial, partial_base + d0) * partner_scale
                warp_out1 = warp_out1 * old_scale + fx.memref_load(s_partial, partial_base + d1) * partner_scale
                if fx.const_expr(head_dim > 128):
                    warp_out2 = (
                        warp_out2 * old_scale + fx.memref_load(s_partial, partial_base + d2) * partner_scale
                    )
                    warp_out3 = (
                        warp_out3 * old_scale + fx.memref_load(s_partial, partial_base + d3) * partner_scale
                    )
                warp_max = new_max
            gpu.barrier()

        if warp == fx.Int32(0):
            has_tokens = warp_sum > zero
            _store_out(d0, has_tokens.select(warp_out0 / warp_sum, zero))
            _store_out(d1, has_tokens.select(warp_out1 / warp_sum, zero))
            if fx.const_expr(head_dim > 128):
                _store_out(d2, has_tokens.select(warp_out2 / warp_sum, zero))
                _store_out(d3, has_tokens.select(warp_out3 / warp_sum, zero))

    return split_reduce_kernel, pipelined_split_reduce_kernel, reduce_threads, pipelined_reduce_warps
