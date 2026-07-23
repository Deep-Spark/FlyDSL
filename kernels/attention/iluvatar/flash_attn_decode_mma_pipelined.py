# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Double-buffered MMA implementation used by the HND paged decode fast path.

The KV-cache assembly constrains the production path to a 16-token HND page.
The KV-streaming loop mirrors how the library's
page-attention kernel keeps the global->shared SME DMA engine busy:

* **Double-buffered (ping-pong) K and V** shared memory (``[V0][V1][K0][K1]``)
  so tile ``t+1`` can be prefetched into the idle buffer while tile ``t`` is
  consumed.  A different producer/consumer buffer is what makes the overlap
  correct (single-buffer prefetch races unless a full drain is issued).
* **Bounded G2S waits in the loop.** The baseline's workgroup barriers lower to
  ``sl_wait g2scnt(0)`` and fully drain every SME request.  This path uses
  ``sl_waitmem(g2s=4)`` directly: K is issued before V, so QK waits only for K;
  then V is consumed while the next K remains pending.  ixcc requires the wait
  threshold to remain an immediate, so the final iteration issues one harmless
  dummy K prefetch and drains it once after the loop.
* **Dual prefetch**: the *next* tile's K and V are both issued during the current
  tile's compute (library issues ``FetchKeyFromGlobal`` / ``FetchValueFromGlobal``
  one block ahead), not just K.

The production HND configuration uses ``block_n=16`` and an occupancy-capped
group count; larger blocks remain supported only for experiments.

This module hosts a tensor-core (MMAD) decode attention kernel that mirrors the
structure of ``iluvatar_mr_flash_attn.py`` (the proven HEAD_DIM=128 prefill
kernel) but specialises it for the KV-cache *decode* case:

* ``seqlen_q == 1`` (one decode step per sequence).
* The ``repeats = num_heads // num_kv_heads`` query heads that share one KV head
  are packed into the MMAD's M (query-row) dimension, so a single 64-lane warp
  computes the whole GQA group at once with tensor cores.
* One CTA owns one ``(batch, kv_head)`` pair and streams the paged HND cache for
  that head through the QK and PV MMADs with online softmax.

The production adapter constrains this kernel to HND paged-cache, bf16, and
``HEAD_DIM == 128``. Everything else stays on the baseline MMA, SIMT, or
scalar fallback selected by the host planner.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.ixdl as ixdl
from flydsl._mlir.dialects import math as _math_dialect
from flydsl._mlir.dialects.arith import (
    AndIOp,
    BitcastOp,
    ExtUIOp,
    OrIOp,
    ShLIOp,
    ShRUIOp,
    TruncFOp,
)
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
    # Double-buffered: two V buffers + two K buffers.
    return 2 * (v_smem_elems + k_stage_elems) * 2


def build_pipelined_mma_decode_attention_kernel(
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
    cache_bytes = batch_size * max_seqlen_k * num_kv_heads * head_dim * 2
    use_i64_kv_offset = cache_bytes > (1 << 31)

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
    k_atoms_total = k_atoms_m * cta_atoms_k  # 32
    q_atoms_m = BM // SME_ROWS  # 1
    q_atoms_total = q_atoms_m * cta_atoms_k_q  # 4
    q_per_warp = q_atoms_total // NUM_WARPS  # 4

    v_smem_elems = HEAD_DIM * BN  # 16384 bf16
    k_stage_elems = BN * BK
    # Double-buffered (ping-pong) smem: [V0][V1][K0][K1].  Correctness needs the
    # producer (next-tile SME prefetch) to write a *different* buffer than the
    # consumer, so the backend's auto-inserted waitcnt at the s2r only blocks on
    # the current buffer -- the prefetch stays in flight (library-style overlap).
    # To avoid the occupancy hit of 2x smem, pair this with a *halved* block_n
    # (e.g. BN=32) so per-CTA smem matches the single-buffer baseline at BN=64.
    NUM_BUF = 2
    k_smem_offset = NUM_BUF * v_smem_elems
    smem_bytes = NUM_BUF * (v_smem_elems + k_stage_elems) * 2

    total_smem_elems = NUM_BUF * (v_smem_elems + k_stage_elems)
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
        active_kv_tiles = (cache_len + fx.Int32(BN - 1)) // fx.Int32(BN)
        tile_end = (tile_end < active_kv_tiles).select(tile_end, active_kv_tiles)

        c_Hkv = fx.Int32(num_kv_heads)
        c_S = fx.Int32(max_seqlen_k)
        c_page = fx.Int32(page_block_size)
        c_D = fx.Int32(HEAD_DIM)

        def _kv_group_base(pos_group):
            """Element base + row stride for a 16-key group starting at pos_group."""
            if fx.const_expr(use_i64_kv_offset):
                pos_group_wide = fx.Int64(pos_group)
                hkv_wide = fx.Int64(hkv)
                c_Hkv_wide = fx.Int64(num_kv_heads)
                c_S_wide = fx.Int64(max_seqlen_k)
                c_page_wide = fx.Int64(page_block_size)
                c_D_wide = fx.Int64(HEAD_DIM)
                if fx.const_expr(paged):
                    logical_block = pos_group // c_page
                    block_off = pos_group_wide % c_page_wide
                    phys = fx.Int64(_load_block(logical_block))
                    if fx.const_expr(upstream_cache_layout):
                        base = ((phys * c_page_wide + block_off) * c_Hkv_wide + hkv_wide) * c_D_wide
                        stride = num_kv_heads * HEAD_DIM
                    else:
                        base = ((phys * c_Hkv_wide + hkv_wide) * c_page_wide + block_off) * c_D_wide
                        stride = HEAD_DIM
                else:
                    if fx.const_expr(upstream_cache_layout):
                        base = ((fx.Int64(b) * c_S_wide + pos_group_wide) * c_Hkv_wide + hkv_wide) * c_D_wide
                        stride = num_kv_heads * HEAD_DIM
                    else:
                        base = ((fx.Int64(b) * c_Hkv_wide + hkv_wide) * c_S_wide + pos_group_wide) * c_D_wide
                        stride = HEAD_DIM
            else:
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

        # QK accumulators: warp owns WARP_ATOMS_M key-column atoms over 16 rows.
        dummy_ptr = fx.recast_iter(fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Shared), smem_ptr)
        dummy_tile = fx.Tensor(fx.make_view(dummy_ptr, fx.make_layout((WARP_M, BN), (BN, 1))))
        dummy_atoms = fx.flat_divide(dummy_tile, (ATOM_M, ATOM_N))

        accs = []
        for jm in fx.range_constexpr(WARP_ATOMS_M):
            c_tile = fx.slice(dummy_atoms, (None, None, 0, jm))
            frag = thr_mma.make_fragment_C(c_tile)
            frag.fill(0)
            accs.append(frag)

        dummy_pv = fx.Tensor(fx.make_view(dummy_ptr, fx.make_layout((ATOM_M, ATOM_N), (ATOM_N, 1))))
        pv_accs = []
        for nt in fx.range_constexpr(d_atoms):
            frag = thr_mma.make_fragment_C(dummy_pv)
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
            ixdl.sl_waitmem(g2s=k_atoms_total)

        def _make_k_atoms(k_base):
            # `k_base` is the element offset of the K ping-pong buffer to read.
            k_atoms_s = []
            for jm in fx.range_constexpr(WARP_ATOMS_M):
                row = []
                for ki in fx.range_constexpr(cta_atoms_k):
                    off = k_base + fx.Int32((jm * cta_atoms_k + ki) * BRICK_ELEMS)
                    row.append(fx.zipped_divide(_sme_view_dyn(smem_ptr, fx.BFloat16, off, False), tile_atom_A))
                k_atoms_s.append(row)
            return k_atoms_s

        # Ping-pong buffer bases (element offsets) for stage in {0, 1}.
        def _k_buf_base(stage):
            return fx.Int32(k_smem_offset) + stage * fx.Int32(k_stage_elems)

        def _v_buf_base(stage):
            return stage * fx.Int32(v_smem_elems)

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
        def issue_k_stage(kv_tile, k_base):
            for ni in fx.range_constexpr(k_atoms_m):
                pos_group = fx.Int32(kv_tile) * fx.Int32(BN) + fx.Int32(ni * SME_ROWS)
                base, stride = _kv_group_base(pos_group)
                k_grp = _make_bf16_gmem_tile(KCache, base, (SME_ROWS, HEAD_DIM), (stride, 1))
                k_div = fx.zipped_divide(k_grp, tile_sme)
                for ki in fx.range_constexpr(cta_atoms_k):
                    atom_idx = ni * cta_atoms_k + ki
                    k_off = k_base + fx.Int32(atom_idx * BRICK_ELEMS)
                    fx.copy_atom_call(
                        sme_atom_K, fx.slice(k_div, (None, (0, ki))), _sme_view_dyn(smem_ptr, fx.BFloat16, k_off, False)
                    )
            ixdl.cp_async_commit_group()

        def issue_v_sme(kv_tile, v_base):
            for ni in fx.range_constexpr(k_atoms_m):
                pos_group = fx.Int32(kv_tile) * fx.Int32(BN) + fx.Int32(ni * SME_ROWS)
                base, stride = _kv_group_base(pos_group)
                v_grp = _make_bf16_gmem_tile(VCache, base, (SME_ROWS, HEAD_DIM), (stride, 1))
                v_div = fx.zipped_divide(v_grp, tile_sme)
                for ki in fx.range_constexpr(cta_atoms_k):
                    atom_idx = ni * cta_atoms_k + ki
                    v_off = v_base + fx.Int32(atom_idx * BRICK_ELEMS)
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

        dummy_bf16_ptr = fx.recast_iter(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Shared), smem_ptr)
        dummy_b_tile = fx.Tensor(fx.make_view(dummy_bf16_ptr, fx.make_layout((ATOM_N, ATOM_K), (ATOM_K, 1))))

        # Prologue: prefetch the first tile's K and V into the buffer matching
        # tile_start's parity.  No drain here -- the loop's bounded waits enforce
        # ordering while keeping the freshest SME group in flight.
        start_par = fx.Int32(tile_start) & fx.Int32(1)
        issue_k_stage(tile_start, _k_buf_base(start_par))
        issue_v_sme(tile_start, _v_buf_base(start_par))

        loop_results = [m_running, l_running]
        for kv_tile, state in range(tile_start, tile_end, fx.Int32(1), init=loop_results):
            m_running = fx.Float32(state[0])
            l_running = fx.Float32(state[1])

            tile_i = fx.Int32(kv_tile)
            cur = tile_i & fx.Int32(1)
            nxt = fx.Int32(1) - cur
            k_base_cur = _k_buf_base(cur)
            v_base_cur = _v_buf_base(cur)
            has_next = tile_i != tile_end - fx.Int32(1)
            kv_next = tile_i + fx.Int32(1)

            for jm in fx.range_constexpr(WARP_ATOMS_M):
                accs[jm].fill(0)

            # K was issued before V.  Drain only the four older K SME requests;
            # keep V in flight while QK executes.  This is the same bounded
            # G2S-count protocol used by library and avoids the full g2scnt(0)
            # drain that a workgroup barrier inserts.
            ixdl.sl_waitmem(g2s=k_atoms_total)
            compute_qk_stage(_make_k_atoms(k_base_cur))

            # Prefetch the next tile's K into the idle buffer during softmax.
            # The final iteration reissues its current K into the idle buffer:
            # this keeps the bounded wait immediate constant (required by ixcc)
            # and costs only four redundant SME requests once per split.
            kv_prefetch = has_next.select(kv_next, tile_i)
            issue_k_stage(kv_prefetch, _k_buf_base(nxt))

            all_vals = []
            for jm in fx.range_constexpr(WARP_ATOMS_M):
                all_vals.append(Vec(accs[jm].load()))

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
                pv_v = Vec(pv_accs[nt].load())
                elems = []
                for i in fx.range_constexpr(FRAG_ELEMS):
                    elems.append(pv_v[i] * corr)
                pv_accs[nt].store(vector.from_elements(T.vec(FRAG_ELEMS, T.f32), elems))

            l_running_old = l_running
            local_sum = fx.Float32(0.0)
            neg_m_new = -m_new
            for jm in fx.range_constexpr(WARP_ATOMS_M):
                p_elems = []
                for v in fx.range_constexpr(FRAG_ELEMS):
                    fma_val = _math_dialect.fma(_to_raw(all_vals[jm][v]), _to_raw(c_scale_log2e), _to_raw(neg_m_new))
                    p_raw = fx.Float32(fma_val).exp2()
                    k_col_abs = fx.Int32(kv_tile) * fx.Int32(BN) + fx.Int32(jm * ATOM_N + v * 4) + score_lane_row
                    k_col_valid = k_col_abs < cache_len
                    score_valid = q_row_valid & k_col_valid
                    p = score_valid.select(p_raw, c_zero)
                    p_elems.append(p)
                    local_sum = local_sum + p
                accs[jm].store(vector.from_elements(T.vec(FRAG_ELEMS, T.f32), p_elems))

            l_running = fx.Float32(_math_dialect.fma(_to_raw(l_running_old), _to_raw(corr), _to_raw(local_sum)))
            m_running = m_new

            # V[cur] precedes K[next] (or the final dummy K) in the G2S queue.
            # Leave those four K requests pending while PV consumes V[cur].
            gpu.barrier()

            # ---- PV MMAD: V as A, P as B -> O^T[d, m] ----
            d_atoms_per_chunk = BK // ATOM_N
            _c16_p = fx.Int32(16)
            _cmask_p = fx.Int32(-65536)
            v_smem_i32 = fx.Tensor(
                fx.make_view(
                    fx.recast_iter(fx.PointerType.get(T.i32, fx.AddressSpace.Shared), smem_ptr),
                    fx.make_layout((total_smem_elems // 2,), (1,)),
                )
            )
            v_lane0 = lane_id & fx.Int32(1)
            v_lane12 = (lane_id >> fx.Int32(1)) & fx.Int32(3)
            v_lane3 = (lane_id >> fx.Int32(3)) & fx.Int32(1)
            v_lane45 = lane_id >> fx.Int32(4)
            v_sload_base = v_lane3 * fx.Int32(128) + v_lane0 * fx.Int32(64) + v_lane45 * fx.Int32(16) + v_lane12
            v_sload_key = v_lane3 * fx.Int32(2) + v_lane0
            for ki in fx.range_constexpr(k_steps_pv):
                p_vals = Vec(accs[ki].load())
                pk = []
                for v in fx.range_constexpr(FRAG_ELEMS // 2):
                    a = p_vals[v * 2].bitcast(fx.Int32)
                    bb = p_vals[v * 2 + 1].bitcast(fx.Int32)
                    shifted = fx.Int32(ShRUIOp(_to_raw(a), _to_raw(_c16_p)).result)
                    pk.append((bb & _cmask_p) | shifted)
                p_vec = vector.bitcast(
                    T.vec(FRAG_ELEMS, T.bf16), vector.from_elements(T.vec(FRAG_ELEMS // 2, T.i32), pk)
                )

                frag_P = thr_mma.make_fragment_B(dummy_b_tile)
                frag_P.store(p_vec)

                for chunk_id in fx.range_constexpr(1):
                    chunk_sme_base = 0
                    for nt_loc in fx.range_constexpr(d_atoms_per_chunk):
                        nt = chunk_id * d_atoms_per_chunk + nt_loc
                        jn = ki
                        d_brick = nt_loc // 2
                        d_sub = nt_loc % 2
                        brick_off = v_base_cur + fx.Int32(chunk_sme_base + (jn * cta_atoms_k + d_brick) * BRICK_ELEMS)
                        v_view = _sme_view_dyn(smem_ptr, fx.BFloat16, brick_off, True)
                        v_atoms = fx.zipped_divide(v_view, tile_atom_B)
                        v_tile_s = fx.slice(v_atoms, (None, d_sub))
                        frag_V = thr_mma.make_fragment_B(v_tile_s)
                        v_i32_base = (brick_off >> fx.Int32(1)) + v_sload_base
                        v_em0 = fx.Int32(d_sub * 2)
                        v_em1 = fx.Int32(d_sub * 2 + 1)
                        v_i0 = v_smem_i32[v_i32_base + ((v_sload_key ^ v_em0) * fx.Int32(4))]
                        v_i1 = v_smem_i32[v_i32_base + ((v_sload_key ^ v_em1) * fx.Int32(4))]
                        frag_V.store(
                            vector.bitcast(
                                T.vec(FRAG_ELEMS, T.bf16),
                                vector.from_elements(T.vec(FRAG_ELEMS // 2, T.i32), [v_i0, v_i1]),
                            )
                        )
                        fx.gemm(mma_atom, pv_accs[nt], frag_V, frag_P, pv_accs[nt])

            # Prefetch the next tile's V into the idle buffer for the next iter.
            if has_next:
                issue_v_sme(kv_next, _v_buf_base(nxt))
            loop_results = yield [m_running, l_running]

        m_running = fx.Float32(loop_results[0])
        l_running = fx.Float32(loop_results[1])
        ixdl.sl_waitmem(g2s=0)

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
                pv = Vec(pv_accs[nt].load())
                for v in fx.range_constexpr(FRAG_ELEMS):
                    d = fx.Int32(nt * ATOM_N) + lane_row + fx.Int32(v * 4)
                    if head_valid:
                        po_row = fx.slice(PartialOut, (b, fx.Int32(0), head_q, split, None))
                        po_div = fx.logical_divide(po_row, fx.make_layout(1, 1))
                        fx.memref_store(fx.Float32(pv[v]), po_div, d)
        else:
            # ---- Epilogue: TransposeCToB16 (HEAD_DIM=128) ----
            inv_l = fx.Float32(1.0) / l_running

            i32_smem_ptr = fx.recast_iter(fx.PointerType.get(T.i32, fx.AddressSpace.Shared), smem_ptr)
            tc_warp_base = warp_id * fx.Int32(TC_WORDS_PER_WARP)
            tc_smem = fx.Tensor(
                fx.make_view(
                    fx.add_offset(i32_smem_ptr, fx.make_int_tuple(tc_warp_base)),
                    fx.make_layout((TC_WORDS_PER_WARP,), (1,)),
                )
            )

            lane5 = lane_id >> fx.Int32(5)
            lane04 = lane_id & fx.Int32(31)
            lane24 = lane04 >> fx.Int32(2)
            laneCol = lane_id % fx.Int32(16)
            laneRow = lane_id // fx.Int32(16)

            c_16_raw = _to_raw(fx.Int32(16))
            lo_mask_raw = _to_raw(fx.Int32(0xFFFF))
            hi_mask_raw = _to_raw(fx.Int32(-65536))

            seq_q_g_base = warp_id * fx.Int32(WARP_M)

            for ni_quad in fx.range_constexpr(d_atoms // 4):
                nt0 = ni_quad * 4
                nt1 = ni_quad * 4 + 1
                nt2 = ni_quad * 4 + 2
                nt3 = ni_quad * 4 + 3

                pv_v0 = Vec(pv_accs[nt0].load())
                pv_v1 = Vec(pv_accs[nt1].load())
                pv_v2 = Vec(pv_accs[nt2].load())
                pv_v3 = Vec(pv_accs[nt3].load())

                def _norm_trunc(val):
                    normed = val * inv_l
                    raw = normed._ir_value if hasattr(normed, "_ir_value") else _to_raw(normed)
                    return TruncFOp(T.bf16, raw).result

                def _pack_bf16x2(bf16_lo, bf16_hi):
                    i16_lo = BitcastOp(T.i16, bf16_lo).result
                    i16_hi = BitcastOp(T.i16, bf16_hi).result
                    i32_lo = ExtUIOp(T.i32, i16_lo).result
                    i32_hi = ShLIOp(ExtUIOp(T.i32, i16_hi).result, c_16_raw).result
                    return OrIOp(i32_lo, i32_hi).result

                vr0_vals = []
                vr1_vals = []
                for v in fx.range_constexpr(FRAG_ELEMS):
                    h0 = _norm_trunc(pv_v0[v])
                    h1 = _norm_trunc(pv_v1[v])
                    h2 = _norm_trunc(pv_v2[v])
                    h3 = _norm_trunc(pv_v3[v])
                    vr0_vals.append(_pack_bf16x2(h0, h2))
                    vr1_vals.append(_pack_bf16x2(h1, h3))

                for i in fx.range_constexpr(FRAG_ELEMS):
                    idx0 = lane04 * fx.Int32(16) + ((fx.Int32(i) ^ lane24) * fx.Int32(2)) + lane5
                    idx1 = lane04 * fx.Int32(16) + ((fx.Int32(i + 4) ^ lane24) * fx.Int32(2)) + lane5
                    tc_smem[idx0] = vr0_vals[i]
                    tc_smem[idx1] = vr1_vals[i]

                out0_vals = []
                out1_vals = []
                for i in fx.range_constexpr(FRAG_ELEMS):
                    load_idx0 = fx.Int32(i) * fx.Int32(64) + (lane_id ^ (fx.Int32(i) * fx.Int32(2)))
                    load_idx1 = fx.Int32(i + 4) * fx.Int32(64) + (lane_id ^ (fx.Int32(i + 4) * fx.Int32(2)))
                    val0_raw = _to_raw(tc_smem[load_idx0])
                    val1_raw = _to_raw(tc_smem[load_idx1])
                    bp0 = OrIOp(
                        AndIOp(val0_raw, lo_mask_raw).result,
                        ShLIOp(AndIOp(val1_raw, lo_mask_raw).result, c_16_raw).result,
                    ).result
                    bp1 = OrIOp(ShRUIOp(val0_raw, c_16_raw).result, AndIOp(val1_raw, hi_mask_raw).result).result
                    out0_vals.append(bp0)
                    out1_vals.append(bp1)

                for ei in fx.range_constexpr(FRAG_ELEMS):
                    row_g = seq_q_g_base + fx.Int32(ei * 4) + laneRow
                    col_g0 = fx.Int32(ni_quad * 4 * ATOM_N // 2) + laneCol
                    col_g1 = fx.Int32((ni_quad * 4 + 2) * ATOM_N // 2) + laneCol
                    if row_g < c_repeat:
                        out_i32_tile[row_g, col_g0] = out0_vals[ei]
                        out_i32_tile[row_g, col_g1] = out1_vals[ei]

    grid = (num_splits, batch_size, num_kv_heads)
    return mma_decode_kernel, threads, smem_bytes, grid
