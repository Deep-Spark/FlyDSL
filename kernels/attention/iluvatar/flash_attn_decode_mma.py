# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MMA-based decode kernel for ``flash_attn_with_kvcache``.

This module hosts a tensor-core (MMAD) decode attention kernel that mirrors the
structure of ``iluvatar_mr_flash_attn.py`` (the proven HEAD_DIM=128 prefill
kernel) but specialises it for the KV-cache *decode* case:

* ``seqlen_q == 1`` (one decode step per sequence).
* The ``repeats = num_heads // num_kv_heads`` query heads that share one KV head
  are packed into the MMAD's M (query-row) dimension, so a single 64-lane warp
  computes the whole GQA group at once with tensor cores.
* One CTA owns one ``(split, batch, kv_head)`` triple in flash-decoding mode
  (library-style ``grid=(groups, seqs, kv_heads)``), or the full KV stream when
  ``num_splits == 1``.

The kernel covers bf16 ``HEAD_DIM == 128`` decode for dense and paged HND/NHD
caches, including split-KV. Unsupported shapes stay on the scalar or varlen
prefill paths selected by the host planner.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.ixdl as ixdl
from flydsl._mlir.dialects import vector
from flydsl.expr import arith, gpu
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw

from .flash_attn_mma_common import (
    _LOG2E,
    ATOM_K,
    ATOM_M,
    ATOM_N,
    BRICK_ELEMS,
    FRAG_ELEMS,
    HEAD_DIM,
    SME_BF16_PER_ROW,
    SME_ROWS,
    WARP_SIZE,
    sme_view,
)

# Tile shape for the decode kernel.  A single warp owns the whole GQA group, so
# the query tile is one 16-row MMAD atom and one KV tile spans BN keys.
BM = ATOM_M  # 16 query rows (repeats valid, rest masked)
BN = 128  # keys per KV tile
BK = HEAD_DIM  # one K stage covers the full head dim
NUM_WARPS = 1
WARP_M = ATOM_M


_sme_view_dyn = sme_view


def mma_decode_smem_bytes() -> int:
    v_smem_elems = HEAD_DIM * BN
    k_stage_elems = BN * BK
    return (v_smem_elems + k_stage_elems) * 2


def build_mma_decode_attention_kernel(
    *,
    batch_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seqlen_k: int,
    paged: bool = False,
    page_block_size: int = 16,
    upstream_cache_layout: bool = False,
    num_splits: int = 1,
    block_n: int = BN,
):
    """Build the bf16 HEAD_DIM=128 decode attention kernel.

    Supports dense and paged caches in both the MR (``[*, Hkv, S, D]``) and
    upstream (``[*, S, Hkv, D]``) layouts.

    When ``num_splits > 1`` the kernel runs in flash-decoding mode: each CTA
    covers a contiguous slice of the KV cache and writes per-split
    ``(max, sum, unnormalized-O)`` partials into ``GroupMax``/``GroupSum``/
    ``PartialOut`` (matching the scalar ``split_reduce_kernel`` convention),
    which a separate reduce kernel then combines.  Otherwise it writes the
    normalized output directly into ``Out``.

    Returns ``(kernel, threads, smem_bytes, grid)``.
    """
    # `BN` (keys per KV tile) is tunable: smaller tiles use less shared memory
    # per CTA, allowing many more CTAs/warps to be co-resident on each SM.  For
    # this memory-bound decode kernel (negligible compute, all latency is in the
    # K/V streaming) that extra memory-level parallelism is the dominant lever.
    BN = block_n
    assert head_dim == HEAD_DIM, "MMA decode kernel only supports head_dim == 128"
    assert num_heads % num_kv_heads == 0
    assert BN % ATOM_M == 0, "block_n must be a multiple of 16"
    assert max_seqlen_k % BN == 0, "max_seqlen_k must be a multiple of block_n for the MMA decode path"
    assert num_splits >= 1
    if paged:
        assert page_block_size % SME_ROWS == 0, "page_block_size must be a multiple of 16"

    split_mode = num_splits > 1

    repeat = num_heads // num_kv_heads
    assert 1 <= repeat <= ATOM_M, "repeats (num_heads/num_kv_heads) must be <= 16"

    threads = NUM_WARPS * WARP_SIZE  # 64
    scale_log2e = (1.0 / math.sqrt(HEAD_DIM)) * _LOG2E

    WARP_ATOMS_M = BN // ATOM_M  # 8 key atoms per row
    K_STEPS_QK = HEAD_DIM // ATOM_K  # 8
    k_steps_pv = BN // ATOM_K  # 8
    d_atoms = HEAD_DIM // ATOM_N  # 8
    k_rep = BK // ATOM_K  # 8

    cta_atoms_k = BK // SME_BF16_PER_ROW  # 4
    cta_atoms_k_q = HEAD_DIM // SME_BF16_PER_ROW  # 4

    k_atoms_m = BN // SME_ROWS  # 8
    q_atoms_m = BM // SME_ROWS  # 1
    q_atoms_total = q_atoms_m * cta_atoms_k_q  # 4
    q_per_warp = q_atoms_total // NUM_WARPS  # 4

    v_smem_elems = HEAD_DIM * BN  # 16384 bf16
    k_smem_offset = v_smem_elems
    k_stage_elems = BN * BK
    smem_bytes = (v_smem_elems + k_stage_elems) * 2

    total_smem_elems = v_smem_elems + k_stage_elems
    TC_WORDS_PER_WARP = 512
    tc_total_bytes = NUM_WARPS * TC_WORDS_PER_WARP * 4
    assert tc_total_bytes <= total_smem_elems * 2

    num_kv_tiles = max_seqlen_k // BN
    tiles_per_split = (num_kv_tiles + num_splits - 1) // num_splits
    inv_log2e = 1.0 / _LOG2E

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def mma_decode_kernel(
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
        split = fx.Int32(fx.block_idx.x)
        b = fx.Int32(fx.block_idx.y)
        hkv = fx.Int32(fx.block_idx.z)
        warp_id = tid // WARP_SIZE
        lane_id = fx.Int32(fx.lane_id)

        if fx.const_expr(split_mode):
            tile_start = split * fx.Int32(tiles_per_split)
            tile_end_raw = tile_start + fx.Int32(tiles_per_split)
            tile_end = (tile_end_raw < fx.Int32(num_kv_tiles)).select(tile_end_raw, fx.Int32(num_kv_tiles))
        else:
            tile_start = fx.Int32(0)
            tile_end = fx.Int32(num_kv_tiles)

        c_num_heads = fx.Int32(num_heads)
        c_repeat = fx.Int32(repeat)

        # ---- Per-warp MMA / copy setup ----
        mma_atom = fx.make_mma_atom(ixdl.MRMma(ATOM_M, ATOM_N, ATOM_K, fx.BFloat16, fx.BFloat16, fx.Float32))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (1, 1, 1)))
        thr_mma = tiled_mma.thr_slice(lane_id)

        sme_atom_K = fx.make_copy_atom(ixdl.MRAsyncCpRow16b(), fx.BFloat16)
        sme_atom_col = fx.make_copy_atom(ixdl.MRAsyncCpCol(), fx.BFloat16)

        copy_atom_s2r = fx.make_copy_atom(fx.UniversalCopy32b(), fx.BFloat16)
        tiled_copy_A = fx.make_tiled_copy_A(copy_atom_s2r, tiled_mma)
        tiled_copy_B = fx.make_tiled_copy_B(copy_atom_s2r, tiled_mma)
        thr_copy_A = tiled_copy_A.get_slice(lane_id)
        thr_copy_B = tiled_copy_B.get_slice(lane_id)

        copy_atom_i32 = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)

        def _load_i32(tensor):
            r = fx.make_rmem_tensor(1, fx.Int32)
            fx.copy_atom_call(copy_atom_i32, tensor, r)
            return fx.memref_load_vec(r)[0]

        def _load_cache_len():
            seq_div = fx.logical_divide(CacheSeqLens, fx.make_layout(1, 1))
            return _load_i32(fx.slice(seq_div, (None, b)))

        def _load_block(logical_block):
            table_row = fx.slice(BlockTable, (b, None))
            table_div = fx.logical_divide(table_row, fx.make_layout(1, 1))
            return _load_i32(fx.slice(table_div, (None, logical_block)))

        cache_len = _load_cache_len()

        c_Hkv = fx.Int32(num_kv_heads)
        c_S = fx.Int32(max_seqlen_k)
        c_page = fx.Int32(page_block_size)
        c_D = fx.Int32(HEAD_DIM)

        def _kv_group_base(pos_group):
            """Element base + row stride for a 16-key group starting at pos_group."""
            if fx.const_expr(paged):
                logical_block = pos_group // c_page
                block_off = pos_group % c_page
                phys = _load_block(logical_block)
                if fx.const_expr(upstream_cache_layout):
                    base = ((phys * c_page + block_off) * c_Hkv + hkv) * c_D
                    stride = num_kv_heads * HEAD_DIM
                else:
                    base = ((phys * c_Hkv + hkv) * c_page + block_off) * c_D
                    stride = HEAD_DIM
            else:
                if fx.const_expr(upstream_cache_layout):
                    base = ((b * c_S + pos_group) * c_Hkv + hkv) * c_D
                    stride = num_kv_heads * HEAD_DIM
                else:
                    base = ((b * c_Hkv + hkv) * c_S + pos_group) * c_D
                    stride = HEAD_DIM
            return base, stride

        tile_sme = fx.make_tile(SME_ROWS, SME_BF16_PER_ROW)
        tile_atom_A = fx.make_tile(ATOM_M, ATOM_K)
        tile_atom_B = fx.make_tile(ATOM_N, ATOM_K)

        smem_ptr = fx.get_dyn_shared()

        def _make_bf16_gmem_tile(tensor, elem_base, shape, stride):
            ptr = fx.add_offset(fx.get_iter(tensor), fx.make_int_tuple(elem_base))
            view = fx.Tensor(fx.make_view(ptr, fx.make_layout(shape, stride)))
            return ixdl.make_sme_gmem_tensor(view, leading_stride=stride[0])

        # C-accumulators are plain per-thread register arrays: FRAG_ELEMS f32 values,
        # shaped (VAL, M-rest, N-rest) = (FRAG_ELEMS, 1, 1) as fx.gemm expects.
        acc_lyt = fx.make_layout((FRAG_ELEMS, 1, 1), (1, 0, 0))
        accs = []
        for jm in fx.range_constexpr(WARP_ATOMS_M):
            frag = fx.make_rmem_tensor(acc_lyt, fx.Float32)
            frag.fill(0)
            accs.append(frag)

        pv_accs = []
        for nt in fx.range_constexpr(d_atoms):
            frag = fx.make_rmem_tensor(acc_lyt, fx.Float32)
            frag.fill(0)
            pv_accs.append(frag)

        c_zero = fx.Float32(0.0)
        c_neg_inf = fx.Float32(float("-inf"))
        c_scale_log2e = fx.Float32(scale_log2e)
        m_running = c_neg_inf
        l_running = c_zero

        warp_q_start = warp_id * q_per_warp

        def _sync_arrive():
            ixdl.cp_async_wait_group(0)
            gpu.barrier()

        def _make_k_atoms(stage_base):
            k_atoms_s = []
            for jm in fx.range_constexpr(WARP_ATOMS_M):
                row = []
                for ki in fx.range_constexpr(cta_atoms_k):
                    off = fx.Int32(k_smem_offset + stage_base + (jm * cta_atoms_k + ki) * BRICK_ELEMS)
                    row.append(fx.zipped_divide(_sme_view_dyn(smem_ptr, fx.BFloat16, off, False), tile_atom_A))
                k_atoms_s.append(row)
            return k_atoms_s

        k_atoms_list = [_make_k_atoms(0)]

        # ---- Load Q (the GQA group) through SMEM into registers ----
        # For seqlen_q == 1 the `repeat` query heads of group hkv are contiguous
        # in QWork [B, 1, Hq, D] with row stride HEAD_DIM.  Rows >= repeat read
        # padding tail (allocated by the wrapper) and are masked out later.
        q_elem_base = (b * c_num_heads + hkv * c_repeat) * fx.Int32(HEAD_DIM)
        q_sme_tile = _make_bf16_gmem_tile(QWork, q_elem_base, (BM, HEAD_DIM), (HEAD_DIM, 1))
        q_div = fx.zipped_divide(q_sme_tile, tile_sme)
        for t in fx.range_constexpr(q_per_warp):
            atom_idx = warp_q_start + t
            mi = atom_idx // cta_atoms_k_q
            ki_q = atom_idx % cta_atoms_k_q
            q_off = atom_idx * fx.Int32(BRICK_ELEMS)
            fx.copy_atom_call(
                sme_atom_col, fx.slice(q_div, (None, (mi, ki_q))), _sme_view_dyn(smem_ptr, fx.BFloat16, q_off, True)
            )
        ixdl.cp_async_commit_group()
        _sync_arrive()

        q_reg_frags = []
        for k_step in fx.range_constexpr(K_STEPS_QK):
            ki_q = k_step // 2
            kq_sub = k_step % 2
            q_off = (warp_id * fx.Int32(cta_atoms_k_q) + fx.Int32(ki_q)) * fx.Int32(BRICK_ELEMS)
            q_view = _sme_view_dyn(smem_ptr, fx.BFloat16, q_off, True)
            q_atoms = fx.zipped_divide(q_view, tile_atom_B)
            q_tile_s = fx.slice(q_atoms, (None, kq_sub))
            frag_Q = thr_mma.make_fragment_B(q_tile_s)
            fx.copy(copy_atom_s2r, thr_copy_B.partition_S(q_tile_s), thr_copy_B.retile(frag_Q), pred=None)
            q_reg_frags.append(frag_Q)

        gpu.barrier()

        # Output view is i32 (two packed bf16 values per store).
        out_i32_ptr = fx.recast_iter(fx.PointerType.get(T.i32, fx.AddressSpace.Global), fx.get_iter(Out))
        out_i32_elem_base = (b * c_num_heads + hkv * c_repeat) * fx.Int32(HEAD_DIM // 2)
        out_i32_tile = fx.Tensor(
            fx.make_view(
                fx.add_offset(out_i32_ptr, fx.make_int_tuple(out_i32_elem_base)),
                fx.make_layout((BM, HEAD_DIM // 2), (HEAD_DIM // 2, 1)),
            )
        )

        # Each KV tile (BN=128 keys) is staged 16 keys at a time so paged caches
        # whose block size is < BN are gathered block-by-block via block_table.
        def issue_k_stage(kv_tile):
            for ni in fx.range_constexpr(k_atoms_m):
                pos_group = fx.Int32(kv_tile) * fx.Int32(BN) + fx.Int32(ni * SME_ROWS)
                base, stride = _kv_group_base(pos_group)
                k_grp = _make_bf16_gmem_tile(KCache, base, (SME_ROWS, HEAD_DIM), (stride, 1))
                k_div = fx.zipped_divide(k_grp, tile_sme)
                for ki in fx.range_constexpr(cta_atoms_k):
                    atom_idx = ni * cta_atoms_k + ki
                    k_off = fx.Int32(k_smem_offset) + fx.Int32(atom_idx) * fx.Int32(BRICK_ELEMS)
                    fx.copy_atom_call(
                        sme_atom_K, fx.slice(k_div, (None, (0, ki))), _sme_view_dyn(smem_ptr, fx.BFloat16, k_off, False)
                    )
            ixdl.cp_async_commit_group()

        def issue_v_sme(kv_tile):
            for ni in fx.range_constexpr(k_atoms_m):
                pos_group = fx.Int32(kv_tile) * fx.Int32(BN) + fx.Int32(ni * SME_ROWS)
                base, stride = _kv_group_base(pos_group)
                v_grp = _make_bf16_gmem_tile(VCache, base, (SME_ROWS, HEAD_DIM), (stride, 1))
                v_div = fx.zipped_divide(v_grp, tile_sme)
                for ki in fx.range_constexpr(cta_atoms_k):
                    atom_idx = ni * cta_atoms_k + ki
                    v_off = fx.Int32(atom_idx) * fx.Int32(BRICK_ELEMS)
                    fx.copy_atom_call(
                        sme_atom_col,
                        fx.slice(v_div, (None, (0, ki))),
                        _sme_view_dyn(smem_ptr, fx.BFloat16, v_off, True),
                    )
            ixdl.cp_async_commit_group()

        def load_k_frags(sK, kk):
            ki = kk // 2
            kk_in = kk % 2
            fQ = q_reg_frags[kk]
            kFs = []
            for jm in fx.range_constexpr(WARP_ATOMS_M):
                k_tile_s = fx.slice(sK[jm][ki], (None, kk_in))
                frag_K = thr_mma.make_fragment_A(k_tile_s)
                fx.copy(copy_atom_s2r, thr_copy_A.partition_S(k_tile_s), thr_copy_A.retile(frag_K), pred=None)
                kFs.append(frag_K)
            return fQ, kFs

        def mma_frags(frag_Q, k_frags):
            for jm in fx.range_constexpr(WARP_ATOMS_M):
                fx.gemm(mma_atom, accs[jm], k_frags[jm], frag_Q, accs[jm])

        def compute_qk_stage(sK):
            for kk in fx.range_constexpr(k_rep):
                fQ_cur, kFs_cur = load_k_frags(sK, kk)
                mma_frags(fQ_cur, kFs_cur)

        # Constants used by the in-register C(f32) -> B(bf16) pack.
        _c16_p = fx.Int32(16)
        _cmask_p = fx.Int32(-65536)

        # Prologue: stage the first K tile of this CTA's KV slice.
        issue_k_stage(tile_start)
        _sync_arrive()

        loop_results = [m_running, l_running]
        for kv_tile, state in range(tile_start, tile_end, fx.Int32(1), init=loop_results):
            m_running = fx.Float32(state[0])
            l_running = fx.Float32(state[1])

            for jm in fx.range_constexpr(WARP_ATOMS_M):
                accs[jm].fill(0)

            issue_v_sme(kv_tile)
            compute_qk_stage(k_atoms_list[0])

            all_vals = []
            for jm in fx.range_constexpr(WARP_ATOMS_M):
                all_vals.append(accs[jm])

            score_lane_col = lane_id % fx.Int32(16)  # query row
            score_lane_row = lane_id // fx.Int32(16)
            q_row_valid = score_lane_col < c_repeat

            local_max = c_neg_inf
            for jm in fx.range_constexpr(WARP_ATOMS_M):
                for v in fx.range_constexpr(FRAG_ELEMS):
                    k_col_abs = fx.Int32(kv_tile) * fx.Int32(BN) + fx.Int32(jm * ATOM_N + v * 4) + score_lane_row
                    k_col_valid = k_col_abs < cache_len
                    score_valid = q_row_valid & k_col_valid
                    score_val = score_valid.select(all_vals[jm][v], c_neg_inf)
                    local_max = arith.maxnumf(fx.Float32(local_max), score_val)

            for mask in [16, 32]:
                peer = fx.Float32(local_max).shuffle_xor(fx.Int32(mask), fx.Int32(WARP_SIZE))
                local_max = arith.maxnumf(fx.Float32(local_max), fx.Float32(peer))

            local_max = fx.Float32(local_max) * c_scale_log2e

            m_old = m_running
            m_new = arith.maxnumf(fx.Float32(m_old), fx.Float32(local_max))
            corr_exp = fx.Float32(m_old - m_new).exp2()
            is_first_kv = fx.Int32(kv_tile) == tile_start
            tile_is_empty = fx.Int32(kv_tile) * fx.Int32(BN) >= cache_len
            reset_corr = is_first_kv | tile_is_empty
            corr = reset_corr.select(c_zero, corr_exp)

            for nt in fx.range_constexpr(d_atoms):
                pv_v = pv_accs[nt]
                elems = []
                for i in fx.range_constexpr(FRAG_ELEMS):
                    elems.append(pv_v[i] * corr)
                pv_accs[nt].store(
                    vector.from_elements(T.vec(FRAG_ELEMS, T.f32), [_to_raw(elem) for elem in elems])
                )

            l_running_old = l_running
            local_sum = fx.Float32(0.0)
            neg_m_new = -m_new
            for jm in fx.range_constexpr(WARP_ATOMS_M):
                p_elems = []
                for v in fx.range_constexpr(FRAG_ELEMS):
                    p_raw = fx.fma(all_vals[jm][v], c_scale_log2e, neg_m_new).exp2()
                    k_col_abs = fx.Int32(kv_tile) * fx.Int32(BN) + fx.Int32(jm * ATOM_N + v * 4) + score_lane_row
                    k_col_valid = k_col_abs < cache_len
                    score_valid = q_row_valid & k_col_valid
                    p = score_valid.select(p_raw, c_zero)
                    p_elems.append(p)
                    local_sum = local_sum + p
                accs[jm].store(
                    vector.from_elements(T.vec(FRAG_ELEMS, T.f32), [_to_raw(elem) for elem in p_elems])
                )

            l_running = fx.fma(l_running_old, corr, local_sum)
            m_running = m_new

            _sync_arrive()

            kv_next = fx.Int32(kv_tile) + fx.Int32(1)
            if fx.Int32(kv_tile) != tile_end - fx.Int32(1):
                issue_k_stage(kv_next)

            gpu.barrier()

            # ---- PV MMAD: V as A, P as B -> O^T[d, m] ----
            packed_p = []
            for ki in fx.range_constexpr(k_steps_pv):
                p_vals = accs[ki]
                pk = []
                for v in fx.range_constexpr(FRAG_ELEMS // 2):
                    a = p_vals[v * 2].bitcast(fx.Int32)
                    bb = p_vals[v * 2 + 1].bitcast(fx.Int32)
                    pk.append((bb & _cmask_p) | a.shrui(_c16_p))
                packed_p.append(Vec.from_elements(pk, fx.Int32).bitcast(fx.BFloat16))

            for ki in fx.range_constexpr(k_steps_pv):
                frag_P = fx.make_fragment_like(q_reg_frags[0])
                frag_P.store(packed_p[ki])
                for nt in fx.range_constexpr(d_atoms):
                    slb_tcu_idx = ki * d_atoms + nt
                    brick_off = fx.Int32((slb_tcu_idx // 2) * BRICK_ELEMS)
                    d_sub = slb_tcu_idx % 2
                    v_view = _sme_view_dyn(smem_ptr, fx.BFloat16, brick_off, True)
                    # SME Col stores this brick logically as V[key, d]. Compose
                    # with (d, k)->(key, d) so the source layout carries the
                    # permutation instead of explicit lane-address math.
                    v_pv_layout = fx.composition(
                        fx.get_layout(v_view),
                        fx.make_layout(
                            (SME_BF16_PER_ROW, (2, 2, 2, 2)),
                            (SME_ROWS, (4, 1, 2, 8)),
                        ),
                    )
                    v_pv_view = fx.make_view(fx.get_iter(v_view), v_pv_layout)
                    v_atoms = fx.zipped_divide(v_pv_view, tile_atom_A)
                    v_tile_s = fx.slice(v_atoms, (None, d_sub))
                    frag_V = thr_mma.make_fragment_A(v_tile_s)
                    fx.copy(
                        copy_atom_s2r,
                        thr_copy_A.partition_S(v_tile_s),
                        thr_copy_A.retile(frag_V),
                        pred=None,
                    )
                    fx.gemm(mma_atom, pv_accs[nt], frag_V, frag_P, pv_accs[nt])

            _sync_arrive()
            loop_results = yield [m_running, l_running]

        m_running = fx.Float32(loop_results[0])
        l_running = fx.Float32(loop_results[1])

        for mask in [16, 32]:
            ps = fx.Float32(l_running).shuffle_xor(fx.Int32(mask), fx.Int32(WARP_SIZE))
            l_running = l_running + ps

        if fx.const_expr(split_mode):
            # ---- Split epilogue: write per-split (max, sum, unnormalized O) ----
            # m_running is in log2 domain; the reduce kernel expects natural-log
            # max scores, so divide out LOG2E here.
            c_inv_log2e = fx.Float32(inv_log2e)
            head = lane_id % fx.Int32(16)
            head_q = hkv * c_repeat + head
            head_valid = head < c_repeat
            lane_row = lane_id // fx.Int32(16)

            if lane_row == fx.Int32(0):
                if head_valid:
                    m_nat = fx.Float32(m_running) * c_inv_log2e
                    gm_row = fx.slice(GroupMax, (b, fx.Int32(0), head_q, None))
                    gm_div = fx.logical_divide(gm_row, fx.make_layout(1, 1))
                    fx.memref_store(m_nat, gm_div, split)
                    gs_row = fx.slice(GroupSum, (b, fx.Int32(0), head_q, None))
                    gs_div = fx.logical_divide(gs_row, fx.make_layout(1, 1))
                    fx.memref_store(fx.Float32(l_running), gs_div, split)

            # O^T accumulator: pv_accs[nt][v] = O^T[d = nt*16 + lane_row + v*4,
            # head = lane%16].  Scatter each lane's 32 (d) values to PartialOut.
            for nt in fx.range_constexpr(d_atoms):
                pv = pv_accs[nt]
                for v in fx.range_constexpr(FRAG_ELEMS):
                    d = fx.Int32(nt * ATOM_N) + lane_row + fx.Int32(v * 4)
                    if head_valid:
                        po_row = fx.slice(PartialOut, (b, fx.Int32(0), head_q, split, None))
                        po_div = fx.logical_divide(po_row, fx.make_layout(1, 1))
                        fx.memref_store(fx.Float32(pv[v]), po_div, d)
        else:
            # ---- Epilogue: normalise + TransposeCToB16 via layouts / fx.copy ----
            has_visible_key = l_running > fx.Float32(0.0)
            inv_l = has_visible_key.select(fx.Float32(1.0) / l_running, fx.Float32(0.0))

            i32_smem_ptr = fx.recast_iter(fx.PointerType.get(T.i32, fx.AddressSpace.Shared), smem_ptr)
            tc_warp_base = warp_id * fx.Int32(TC_WORDS_PER_WARP)
            tc_smem_iter = fx.add_offset(i32_smem_ptr, fx.make_int_tuple(tc_warp_base))

            c_16 = fx.Int32(16)
            lo_mask = fx.Int32(0xFFFF)
            hi_mask = fx.Int32(-65536)

            seq_q_g_base = warp_id * fx.Int32(WARP_M)
            out_tile_iter = fx.get_iter(out_i32_tile)
            # Tile is already base-offset to this GQA group; row stride is D/2.
            out_row_stride_i32 = HEAD_DIM // 2

            tc_write_lyt = fx.make_composed_layout(
                fx.static(fx.SwizzleType.get(3, 1, 5)),
                fx.make_layout((8, (2, 2, 2, 2, 2, 2)), (2, (16, 32, 64, 128, 256, 1))),
            )
            tc_read_lyt = fx.make_composed_layout(
                fx.static(fx.SwizzleType.get(3, 1, 5)),
                fx.make_layout((8, 64), (64, 1)),
            )
            out_gmem_lyt = fx.make_layout(
                (4, (16, 4)), (4 * out_row_stride_i32, (1, out_row_stride_i32))
            )

            tiled_tc = fx.make_tiled_copy_tv(
                copy_atom_i32,
                fx.make_ordered_layout((1, 64), order=(0, 1)),
                fx.make_ordered_layout((8, 1), order=(0, 1)),
            )
            thr_tc = tiled_tc.get_slice(lane_id)
            tiled_out = fx.make_tiled_copy_tv(
                copy_atom_i32,
                fx.make_ordered_layout((1, 64), order=(0, 1)),
                fx.make_ordered_layout((4, 1), order=(0, 1)),
            )
            thr_out = tiled_out.get_slice(lane_id)

            s_w = fx.make_view(tc_smem_iter, tc_write_lyt)
            s_r = fx.make_view(tc_smem_iter, tc_read_lyt)

            for ni_quad in fx.range_constexpr(d_atoms // 4):
                nt0 = ni_quad * 4
                nt1 = ni_quad * 4 + 1
                nt2 = ni_quad * 4 + 2
                nt3 = ni_quad * 4 + 3

                pv_v0 = pv_accs[nt0]
                pv_v1 = pv_accs[nt1]
                pv_v2 = pv_accs[nt2]
                pv_v3 = pv_accs[nt3]

                def _norm_trunc(val):
                    return fx.BFloat16(fx.Float32(val) * inv_l)

                def _pack_bf16x2(bf16_lo, bf16_hi):
                    return Vec.from_elements([bf16_lo, bf16_hi], fx.BFloat16).bitcast(fx.Int32)[0]

                pack_elems = []
                for v in fx.range_constexpr(FRAG_ELEMS):
                    pack_elems.append(_pack_bf16x2(_norm_trunc(pv_v0[v]), _norm_trunc(pv_v2[v])))
                for v in fx.range_constexpr(FRAG_ELEMS):
                    pack_elems.append(_pack_bf16x2(_norm_trunc(pv_v1[v]), _norm_trunc(pv_v3[v])))

                thr_w = thr_tc.partition_D(s_w)
                r_pack = fx.make_fragment_like(thr_w)
                r_pack.store(Vec.from_elements(pack_elems, fx.Int32))
                fx.copy(copy_atom_i32, r_pack, thr_w, pred=None)

                thr_r = thr_tc.partition_S(s_r)
                r_load = fx.make_fragment_like(thr_r)
                fx.copy(copy_atom_i32, thr_r, r_load, pred=None)

                out0_vals = []
                out1_vals = []
                for i in fx.range_constexpr(FRAG_ELEMS):
                    v0 = fx.Int32(r_load[i])
                    v1 = fx.Int32(r_load[i + 4])
                    bp0 = (v0 & lo_mask) | ((v1 & lo_mask) << c_16)
                    bp1 = v0.shrui(c_16) | (v1 & hi_mask)
                    out0_vals.append(bp0)
                    out1_vals.append(bp1)

                col_base0 = fx.Int32(ni_quad * 4 * ATOM_N // 2)
                col_base1 = fx.Int32((ni_quad * 4 + 2) * ATOM_N // 2)
                warp_elem = seq_q_g_base * fx.Int32(out_row_stride_i32)
                g0 = fx.make_view(
                    fx.add_offset(out_tile_iter, fx.make_int_tuple(warp_elem + col_base0)), out_gmem_lyt
                )
                g1 = fx.make_view(
                    fx.add_offset(out_tile_iter, fx.make_int_tuple(warp_elem + col_base1)), out_gmem_lyt
                )

                lane_row = lane_id // fx.Int32(16)
                for vals, g in ((out0_vals, g0), (out1_vals, g1)):
                    thr_d = thr_out.partition_D(g)
                    r_o = fx.make_fragment_like(thr_d)
                    r_o.store(Vec.from_elements(vals, fx.Int32))
                    pred = fx.make_fragment_like(thr_d, dtype=fx.Boolean)
                    for ei in fx.range_constexpr(FRAG_ELEMS):
                        row_abs = seq_q_g_base + fx.Int32(ei * 4) + lane_row
                        pred[ei] = row_abs < c_repeat
                    fx.copy(copy_atom_i32, r_o, thr_d, pred=pred)

    grid = (num_splits, batch_size, num_kv_heads)
    return mma_decode_kernel, threads, smem_bytes, grid
