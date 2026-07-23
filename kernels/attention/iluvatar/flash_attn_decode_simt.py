"""SIMT paged decode for small-GQA Qwen models.

This is intentionally separate from the pipelined MMAD kernel: for Hq/Hkv == 2,
using a 16-row MMAD tile masks fourteen rows.  Each K warp owns a strided set
of 16-token pages for short contexts. Long contexts use one-warp split CTAs
and a separate online-softmax reduction, matching ixinfer's launch topology.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.ixdl as ixdl
from flydsl.expr import arith, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.ixdl.ops import bfdot2
from flydsl.expr.typing import full

WARP_SIZE = 64
PAGE_SIZE = 16
SME_COLS = 32
SME_BRICK_ELEMS = PAGE_SIZE * SME_COLS
_LOG2E = 1.4426950408889634


def build_simt_decode_attention_kernel(
    *,
    batch_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seqlen_k: int,
    page_block_size: int,
    num_splits: int,
    k_warps: int,
    cache_strides: tuple[int, int, int, int] | None = None,
):
    """Build the HND paged SIMT decode kernel for GQA ratio two."""
    assert head_dim in (128, 256)
    gqa_ratio = num_heads // num_kv_heads
    d256_gqa4 = head_dim == 256
    assert (not d256_gqa4 and gqa_ratio == 2) or (d256_gqa4 and gqa_ratio == 4)
    assert max_seqlen_k % PAGE_SIZE == 0
    assert page_block_size % PAGE_SIZE == 0
    assert num_splits >= 1
    assert k_warps in (1, 2, 4, 8, 16)
    if d256_gqa4:
        # One CTA owns two query heads. Two sibling CTAs cover the four query
        # heads sharing a KV head without a cross-warp shared-memory hazard.
        assert num_splits > 1
        assert k_warps == 1
    elif num_splits > 1:
        assert k_warps == 1

    split_mode = num_splits > 1
    q_warps = 2 if d256_gqa4 else 1
    output_elems = head_dim // 4
    dot_repeats = head_dim // 8
    sme_bricks_per_page = head_dim // SME_COLS
    chunks_per_block = page_block_size // PAGE_SIZE
    partial_stride = 2 * (2 + head_dim)
    sme_stage_bytes = 2 * k_warps * sme_bricks_per_page * SME_BRICK_ELEMS * 2
    max_pages = max_seqlen_k // PAGE_SIZE
    pages_per_split = (max_pages + num_splits - 1) // num_splits
    canonical_cache_strides = (
        num_kv_heads * page_block_size * head_dim,
        page_block_size * head_dim,
        head_dim,
        1,
    )
    if cache_strides is None:
        cache_strides = canonical_cache_strides
    assert cache_strides[-1] == 1
    canonical_cache_layout = cache_strides == canonical_cache_strides
    cache_block_stride, cache_head_stride, cache_page_stride, _ = cache_strides

    @flyc.kernel(known_block_size=[q_warps * k_warps * WARP_SIZE, 1, 1])
    def simt_decode_kernel(
        QWork: fx.Tensor,
        KCache: fx.Tensor,
        VCache: fx.Tensor,
        CacheSeqLens: fx.Tensor,
        BlockTable: fx.Tensor,
        Out: fx.Tensor,
        GroupMax: fx.Tensor,
        GroupSum: fx.Tensor,
        PartialOut: fx.Tensor,
    ):
        tid = fx.thread_idx.x
        lane_id = fx.Int32(fx.lane_id)
        worker_warp = tid // WARP_SIZE
        k_warp = worker_warp
        q_warp = fx.Int32(0)
        if fx.const_expr(d256_gqa4):
            k_warp = fx.Int32(0)
            q_warp = fx.Int32(worker_warp)
        b = fx.Int32(fx.block_idx.x)
        split = fx.Int32(fx.block_idx.y)
        block_head = fx.Int32(fx.block_idx.z)
        hkv = block_head
        x = lane_id % fx.Int32(16)
        y = lane_id // fx.Int32(16)
        qh0 = hkv * fx.Int32(2)
        if fx.const_expr(d256_gqa4):
            qh0 = hkv * fx.Int32(4) + q_warp * fx.Int32(2)
        qh1 = qh0 + fx.Int32(1)
        zero = fx.Float32(0.0)
        neg_inf = fx.Float32(float("-inf"))
        reduce_neg_inf = fx.Float32(-3.40282e38)
        scale = fx.Float32(1.0 / math.sqrt(head_dim))
        fm_fast = arith.FastMathFlags.fast
        copy_bf16 = fx.make_copy_atom(fx.UniversalCopy16b(), fx.BFloat16)
        copy_bf16x2 = fx.make_copy_atom(fx.UniversalCopy32b(), fx.BFloat16)
        copy_i32 = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
        sme_col = fx.make_copy_atom(ixdl.MRAsyncCpCol(), fx.BFloat16)
        sme_ptr = fx.get_dyn_shared()
        sme_tile = fx.make_tile(PAGE_SIZE, SME_COLS)

        def load_bf16(tensor):
            reg = fx.make_rmem_tensor(1, fx.BFloat16)
            fx.copy_atom_call(copy_bf16, tensor, reg)
            return fx.memref_load_vec(reg)[0].to(fx.Float32)

        def load_i32(tensor):
            reg = fx.make_rmem_tensor(1, fx.Int32)
            fx.copy_atom_call(copy_i32, tensor, reg)
            return fx.memref_load_vec(reg)[0]

        def load_bf16x2_vec(tensor):
            reg = fx.make_rmem_tensor(2, fx.BFloat16)
            fx.copy_atom_call(copy_bf16x2, tensor, reg)
            return fx.memref_load_vec(reg)

        def load_q2_vec(qh, d):
            row = fx.slice(QWork, (b, fx.Int32(0), qh, None))
            div = fx.logical_divide(row, fx.make_layout(1, 1))
            return load_bf16x2_vec(fx.slice(div, (None, d)))

        def sme_view(elem_offset):
            ptr = fx.recast_iter(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Shared), sme_ptr)
            ptr = fx.add_offset(ptr, fx.make_int_tuple(elem_offset))
            return fx.Tensor(
                fx.make_view(
                    ptr,
                    ixdl.make_sme_shared_layout(
                        ixdl.SMESwizzle.Col,
                        fx.BFloat16,
                        major=ixdl.SMEMajor.K,
                    ),
                )
            )

        def issue_sme_page(cache, page, operand):
            table = fx.slice(BlockTable, (b, None))
            table_div = fx.logical_divide(table, fx.make_layout(1, 1))
            logical_block = page // fx.Int32(chunks_per_block)
            chunk_in_block = page % fx.Int32(chunks_per_block)
            block = load_i32(fx.slice(table_div, (None, logical_block)))
            if fx.const_expr(canonical_cache_layout):
                elem_base = (
                    (block * fx.Int32(num_kv_heads) + hkv) * fx.Int32(page_block_size * head_dim)
                    + chunk_in_block * fx.Int32(PAGE_SIZE * head_dim)
                )
                page_stride = head_dim
            else:
                elem_base = (
                    block * fx.Int32(cache_block_stride)
                    + hkv * fx.Int32(cache_head_stride)
                    + chunk_in_block * fx.Int32(PAGE_SIZE * cache_page_stride)
                )
                page_stride = cache_page_stride
            ptr = fx.add_offset(fx.get_iter(cache), fx.make_int_tuple(elem_base))
            gmem = fx.Tensor(
                fx.make_view(
                    ptr,
                    fx.make_layout((PAGE_SIZE, head_dim), (page_stride, 1)),
                )
            )
            gmem_sme = ixdl.make_sme_gmem_tensor(gmem, leading_stride=page_stride)
            gmem_div = fx.zipped_divide(gmem_sme, sme_tile)
            operand_base = operand * fx.Int32(k_warps * sme_bricks_per_page * SME_BRICK_ELEMS) + k_warp * fx.Int32(
                sme_bricks_per_page * SME_BRICK_ELEMS
            )
            for brick in range_constexpr(sme_bricks_per_page):
                fx.copy_atom_call(
                    sme_col,
                    fx.slice(gmem_div, (None, (0, brick))),
                    sme_view(operand_base + fx.Int32(brick * SME_BRICK_ELEMS)),
                )
            ixdl.cp_async_commit_group()

        def load_sme_pair(operand, brick, pair):
            # Match ixinfer's keySMEMOffset[4] physical Col-swizzle readback:
            # one packed i32 holds the two bf16 values consumed by bfdot2.
            smem_i32 = fx.recast_iter(fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared), sme_ptr)
            row = pair * fx.Int32(4) + y
            row_idx = (row % fx.Int32(4)) * fx.Int32(16)
            row_idx = row_idx + (row % fx.Int32(8) // fx.Int32(4)) * fx.Int32(4)
            row_idx = row_idx + (row // fx.Int32(8)) * fx.Int32(8)
            row_idx = row_idx ^ (x // fx.Int32(4) * fx.Int32(4))
            key_offset = row_idx + (x // fx.Int32(4)) * fx.Int32(64) + x % fx.Int32(4)
            operand_base = operand * fx.Int32(k_warps * sme_bricks_per_page * 256) + k_warp * fx.Int32(
                sme_bricks_per_page * 256
            )
            word = fx.ptr_load(
                fx.add_offset(
                    smem_i32,
                    fx.make_int_tuple(operand_base + brick * fx.Int32(256) + key_offset),
                )
            )
            return fx.Vector.from_elements([word], fx.Int32).bitcast(fx.BFloat16)

        def warp_sum(value):
            acc = value
            for mask in [32, 16, 8, 4, 2, 1]:
                acc = acc + acc.shuffle_xor(fx.Int32(mask), fx.Int32(WARP_SIZE))
            return acc

        def warp_max(value):
            acc = value
            for mask in [32, 16, 8, 4, 2, 1]:
                acc = arith.maxnumf(acc, acc.shuffle_xor(fx.Int32(mask), fx.Int32(WARP_SIZE)))
            return acc

        seq_div = fx.logical_divide(CacheSeqLens, fx.make_layout(1, 1))
        cache_len = load_i32(fx.slice(seq_div, (None, b)))
        q0_packed, q1_packed = [], []
        for j in range_constexpr(head_dim // 32):
            for k in range_constexpr(4):
                d = fx.Int32(j * 32 + k * 8) + y * fx.Int32(2)
                q0_packed.append(load_q2_vec(qh0, d))
                q1_packed.append(load_q2_vec(qh1, d))

        second_head_base = 2 + output_elems
        loop_results = (
            [neg_inf, zero]
            + [zero for _ in range(output_elems)]
            + [neg_inf, zero]
            + [zero for _ in range(output_elems)]
        )
        pages = (cache_len + fx.Int32(PAGE_SIZE - 1)) // fx.Int32(PAGE_SIZE)
        if fx.const_expr(split_mode):
            page_start = split * fx.Int32(pages_per_split)
            page_end_raw = page_start + fx.Int32(pages_per_split)
            page_end = (page_end_raw < pages).select(page_end_raw, pages)
            page_step = fx.Int32(1)
        else:
            page_start = k_warp
            page_end = pages
            page_step = fx.Int32(k_warps)
        for page, state in range(page_start, page_end, page_step, init=loop_results):
            m0, l0 = fx.Float32(state[0]), fx.Float32(state[1])
            o0 = [fx.Float32(state[2 + i]) for i in range(output_elems)]
            m1, l1 = fx.Float32(state[second_head_base]), fx.Float32(state[second_head_base + 1])
            o1 = [fx.Float32(state[second_head_base + 2 + i]) for i in range(output_elems)]
            if fx.const_expr(d256_gqa4):
                if q_warp == fx.Int32(0):
                    issue_sme_page(KCache, fx.Int32(page), fx.Int32(0))
                    issue_sme_page(VCache, fx.Int32(page), fx.Int32(1))
                    ixdl.sl_waitmem(g2s=sme_bricks_per_page)
                gpu.barrier()
            else:
                issue_sme_page(KCache, fx.Int32(page), fx.Int32(0))
                issue_sme_page(VCache, fx.Int32(page), fx.Int32(1))
                ixdl.sl_waitmem(g2s=sme_bricks_per_page)
            tok = fx.Int32(page) * fx.Int32(PAGE_SIZE) + x
            valid = tok < cache_len
            score0, score1 = zero, zero
            for r in range_constexpr(dot_repeats):
                k_packed = load_sme_pair(fx.Int32(0), fx.Int32(r // 4), fx.Int32(r % 4))
                score0 = bfdot2(q0_packed[r], k_packed, score0)
                score1 = bfdot2(q1_packed[r], k_packed, score1)
            for mask in [16, 32]:
                score0 = score0 + score0.shuffle_xor(fx.Int32(mask), fx.Int32(WARP_SIZE))
                score1 = score1 + score1.shuffle_xor(fx.Int32(mask), fx.Int32(WARP_SIZE))
            score0 = valid.select(score0 * scale, neg_inf)
            score1 = valid.select(score1 * scale, neg_inf)
            page_m0, page_m1 = score0, score1
            for mask in [8, 4, 2, 1]:
                page_m0 = arith.maxnumf(page_m0, page_m0.shuffle_xor(fx.Int32(mask), fx.Int32(WARP_SIZE)))
                page_m1 = arith.maxnumf(page_m1, page_m1.shuffle_xor(fx.Int32(mask), fx.Int32(WARP_SIZE)))
            next_m0, next_m1 = arith.maxnumf(m0, page_m0), arith.maxnumf(m1, page_m1)
            c0 = fmath.exp2((m0 - next_m0) * fx.Float32(_LOG2E), fastmath=fm_fast)
            c1 = fmath.exp2((m1 - next_m1) * fx.Float32(_LOG2E), fastmath=fm_fast)
            if fx.const_expr(d256_gqa4):
                if q_warp == fx.Int32(0):
                    ixdl.sl_waitmem(g2s=0)
                gpu.barrier()
            else:
                ixdl.sl_waitmem(g2s=0)
            p0 = valid.select(fmath.exp2((score0 - next_m0) * fx.Float32(_LOG2E), fastmath=fm_fast), zero)
            p1 = valid.select(fmath.exp2((score1 - next_m1) * fx.Float32(_LOG2E), fastmath=fm_fast), zero)
            for r in range_constexpr(dot_repeats):
                v_packed = load_sme_pair(fx.Int32(1), fx.Int32(r // 4), fx.Int32(r % 4))
                v0, v1 = v_packed[0].to(fx.Float32), v_packed[1].to(fx.Float32)
                o0[2 * r] = o0[2 * r] * c0 + p0 * v0
                o0[2 * r + 1] = o0[2 * r + 1] * c0 + p0 * v1
                o1[2 * r] = o1[2 * r] * c1 + p1 * v0
                o1[2 * r + 1] = o1[2 * r + 1] * c1 + p1 * v1
            l0, l1, m0, m1 = l0 * c0 + p0, l1 * c1 + p1, next_m0, next_m1
            if fx.const_expr(d256_gqa4):
                gpu.barrier()
            loop_results = yield [m0, l0] + o0 + [m1, l1] + o1

        states = loop_results
        final_m0, final_l0 = fx.Float32(states[0]), fx.Float32(states[1])
        final_o0 = [fx.Float32(states[2 + i]) for i in range(output_elems)]
        final_m1, final_l1 = fx.Float32(states[second_head_base]), fx.Float32(states[second_head_base + 1])
        final_o1 = [fx.Float32(states[second_head_base + 2 + i]) for i in range(output_elems)]
        # Each x lane_id accumulates a disjoint token stream. Only reduce these
        # partial online-softmax results once all pages have been processed.
        for mask in [8, 4, 2, 1]:
            final_l0 = final_l0 + final_l0.shuffle_xor(fx.Int32(mask), fx.Int32(WARP_SIZE))
            final_l1 = final_l1 + final_l1.shuffle_xor(fx.Int32(mask), fx.Int32(WARP_SIZE))
            for r in range_constexpr(output_elems):
                final_o0[r] = final_o0[r] + final_o0[r].shuffle_xor(fx.Int32(mask), fx.Int32(WARP_SIZE))
                final_o1[r] = final_o1[r] + final_o1[r].shuffle_xor(fx.Int32(mask), fx.Int32(WARP_SIZE))
        if fx.const_expr(split_mode):
            if x == fx.Int32(0) and y == fx.Int32(0):
                gm0 = fx.logical_divide(fx.slice(GroupMax, (b, fx.Int32(0), qh0, None)), fx.make_layout(1, 1))
                gm1 = fx.logical_divide(fx.slice(GroupMax, (b, fx.Int32(0), qh1, None)), fx.make_layout(1, 1))
                gs0 = fx.logical_divide(fx.slice(GroupSum, (b, fx.Int32(0), qh0, None)), fx.make_layout(1, 1))
                gs1 = fx.logical_divide(fx.slice(GroupSum, (b, fx.Int32(0), qh1, None)), fx.make_layout(1, 1))
                fx.memref_store((final_l0 > zero).select(final_m0, reduce_neg_inf), gm0, split)
                fx.memref_store((final_l1 > zero).select(final_m1, reduce_neg_inf), gm1, split)
                fx.memref_store(final_l0, gs0, split)
                fx.memref_store(final_l1, gs1, split)
            if x == fx.Int32(0):
                po0 = fx.logical_divide(
                    fx.slice(PartialOut, (b, fx.Int32(0), qh0, split, None)),
                    fx.make_layout(1, 1),
                )
                po1 = fx.logical_divide(
                    fx.slice(PartialOut, (b, fx.Int32(0), qh1, split, None)),
                    fx.make_layout(1, 1),
                )
                for r in range_constexpr(dot_repeats):
                    d = fx.Int32((r // 4) * 32 + (r % 4) * 8) + y * fx.Int32(2)
                    fx.memref_store(final_o0[2 * r], po0, d)
                    fx.memref_store(final_o0[2 * r + 1], po0, d + fx.Int32(1))
                    fx.memref_store(final_o1[2 * r], po1, d)
                    fx.memref_store(final_o1[2 * r + 1], po1, d + fx.Int32(1))
        else:
            # Reuse the K/V LDS allocation for short-context K-warp reduction.
            gpu.barrier()
            partial_ptr = fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Shared), sme_ptr)
            partials = fx.Tensor(fx.make_view(partial_ptr, fx.make_layout(k_warps * partial_stride, 1)))
            base = k_warp * fx.Int32(partial_stride)
            if x == fx.Int32(0) and y == fx.Int32(0):
                fx.memref_store(final_m0, partials, base)
                fx.memref_store(final_l0, partials, base + fx.Int32(1))
                fx.memref_store(final_m1, partials, base + fx.Int32(2 + head_dim))
                fx.memref_store(final_l1, partials, base + fx.Int32(3 + head_dim))
            if x == fx.Int32(0):
                for r in range_constexpr(16):
                    d = fx.Int32((r // 4) * 32 + (r % 4) * 8) + y * fx.Int32(2)
                    fx.memref_store(final_o0[2 * r], partials, base + fx.Int32(2) + d)
                    fx.memref_store(final_o0[2 * r + 1], partials, base + fx.Int32(3) + d)
                    fx.memref_store(final_o1[2 * r], partials, base + fx.Int32(4 + head_dim) + d)
                    fx.memref_store(final_o1[2 * r + 1], partials, base + fx.Int32(5 + head_dim) + d)
            gpu.barrier()

            def merge_stats(head):
                max_value = neg_inf
                for w in range_constexpr(k_warps):
                    max_value = arith.maxnumf(
                        max_value,
                        fx.memref_load(partials, fx.Int32(w * partial_stride + head * (2 + head_dim))),
                    )
                denom = zero
                for w in range_constexpr(k_warps):
                    pbase = fx.Int32(w * partial_stride + head * (2 + head_dim))
                    local_max = fx.memref_load(partials, pbase)
                    weight = fmath.exp2((local_max - max_value) * fx.Float32(_LOG2E), fastmath=fm_fast)
                    denom = denom + fx.memref_load(partials, pbase + fx.Int32(1)) * weight
                return max_value, denom

            def merge_value(head, d, max_value):
                value = zero
                for w in range_constexpr(k_warps):
                    pbase = fx.Int32(w * partial_stride + head * (2 + head_dim))
                    local_max = fx.memref_load(partials, pbase)
                    weight = fmath.exp2((local_max - max_value) * fx.Float32(_LOG2E), fastmath=fm_fast)
                    value = value + fx.memref_load(partials, pbase + fx.Int32(2) + d) * weight
                return value

            def store_out(qh, d, value, denom):
                row = fx.slice(Out, (b, fx.Int32(0), qh, None))
                div = fx.logical_divide(row, fx.make_layout(1, 1))
                reg = fx.make_rmem_tensor(1, fx.BFloat16)
                fx.memref_store_vec(full(1, (value / denom).to(fx.BFloat16), fx.BFloat16), reg)
                fx.copy_atom_call(copy_bf16, reg, fx.slice(div, (None, d)))

            if k_warp == fx.Int32(0) and x == fx.Int32(0):
                max0, den0 = merge_stats(0)
                max1, den1 = merge_stats(1)
                for r in range_constexpr(dot_repeats):
                    d = fx.Int32((r // 4) * 32 + (r % 4) * 8) + y * fx.Int32(2)
                    store_out(qh0, d, merge_value(0, d, max0), den0)
                    store_out(qh0, d + fx.Int32(1), merge_value(0, d + fx.Int32(1), max0), den0)
                    store_out(qh1, d, merge_value(1, d, max1), den1)
                    store_out(qh1, d + fx.Int32(1), merge_value(1, d + fx.Int32(1), max1), den1)

    grid_heads = num_kv_heads
    grid = (batch_size, num_splits, grid_heads)
    return simt_decode_kernel, q_warps * k_warps * WARP_SIZE, sme_stage_bytes, grid
