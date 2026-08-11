# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Variable-length (varlen) prefill FlashAttention for ``flash_attn_varlen_func``.

This is an independent FlyDSL re-implementation of the varlen forward that
vLLM v1 drives through ``flash_attn_varlen_func`` (the unified prefill /
chunked-prefill / prefix path).  It targets the vLLM shape:

* Query is *packed* varlen: ``Q [total_q, num_heads, head_dim]`` indexed by
  ``cu_seqlens_q [batch + 1]``.  Sequence ``b`` owns query rows
  ``[cu_seqlens_q[b], cu_seqlens_q[b + 1])``.
* KV lives in a *paged* cache (vLLM NHD layout, obtained by ``kv_cache.unbind(1)``):
    key_cache / value_cache : [num_blocks, page_block_size, num_kv_heads, head_dim]
  addressed per sequence via ``block_table [batch, max_blocks]`` and the real
  KV length ``seqused_k [batch]``.
* Causal masking is aligned to the bottom-right corner (FlashAttention
  convention): query at local position ``i`` (of a sequence of length ``sq``)
  attends to key ``j`` (of a KV run of length ``sk``) iff
  ``j <= i + sk - sq``.
* GQA: ``repeat = num_heads // num_kv_heads`` query heads share one KV head.

Only bf16, ``head_dim in (128, 256)`` is handled. Both variants use the same
16x16x16 tensor-core atom; the number of head-dimension bricks is static in
each compiled specialization.

Grid / tiling
-------------
``grid = (max_q_tiles, batch, num_heads)`` with ``BM = 128`` query rows and
``BN = 128`` keys per KV tile.  Each CTA (8 warps) owns one
``(q_tile, batch, q_head)`` triple, streams the sequence's KV with online
softmax, and writes ``BM`` output rows.  Tiles beyond a sequence's query length
early-exit (zero KV iterations, no stores).  With ``causal`` the KV loop upper
bound is truncated to the last key the tile can see.

SME brick swizzle, QK/PV MMAD, and the TransposeCToB16 epilogue match the
other MMA attention kernels. Varlen-specific pieces are Q/KV addressing, the
per-16-row paged gather, causal + length masking, and the runtime KV-tile
loop bound.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.ixdl as ixdl
from flydsl._mlir.dialects import llvm as _llvm
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

BM = 128  # query rows per CTA (8 warps x 16)
BN = 128  # keys per KV tile
BK = HEAD_DIM  # Backwards-compatible default; builders specialize BK=head_dim.
NUM_WARPS = 8
WARP_M = ATOM_M  # 16 query rows per warp


_sme_view_dyn = sme_view


def flash_attn_varlen_smem_bytes(head_dim: int = HEAD_DIM) -> int:
    v_smem_elems = head_dim * BN
    k_stage_elems = BN * head_dim
    return (v_smem_elems + k_stage_elems) * 2


def select_num_warps(max_seqlen_q: int, head_dim: int = HEAD_DIM) -> int:
    """Pick the CTA size: the wide 16-warp (BM=256) tile amortises KV traffic
    on long prefills; short ones keep the 8-warp (BM=128) tile."""
    del head_dim
    return 16 if max_seqlen_q >= 256 else 8


def bm_for(num_warps: int) -> int:
    return WARP_M * num_warps


def varlen_grid(causal_balance: str, num_heads: int, batch: int, max_q_tiles: int):
    """Launch grid for a given causal-balance scheme.

    * ``head_fastest``: (num_heads, batch, max_q_tiles) -- head varies fastest
      so each dispatch warp holds one q-tile across heads (equal causal work).
    * ``head_parity``: (max_q_tiles, batch, num_heads) -- q-block varies fastest;
      even heads reverse the q-block order so adjacent head pairs have
      complementary triangular work.
    """
    if causal_balance == "head_parity":
        return (max_q_tiles, batch, num_heads)
    return (num_heads, batch, max_q_tiles)


def max_q_tiles_for(max_seqlen_q: int, num_warps: int = NUM_WARPS) -> int:
    bm = bm_for(num_warps)
    return max(1, (max_seqlen_q + bm - 1) // bm)


def build_flash_attn_varlen_kernel(
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_block_size: int = 16,
    causal: bool = True,
    upstream_cache_layout: bool = True,
    softmax_scale: float = None,
    num_warps: int = 8,
    causal_balance: str = "head_fastest",
    paged: bool = True,
    seqused_is_cumulative: bool = False,
    cache_strides: tuple[int, int, int, int] | None = None,
):
    """Build the bf16 D128/D256 varlen prefill attention kernel.

    ``paged=True`` (default): KCache/VCache are a paged cache addressed by
    ``BlockTable`` (HND or NHD, see ``upstream_cache_layout``) with per-sequence
    KV length in ``SequsedK``.

    ``paged=False`` (dense): KCache/VCache are *packed* varlen tensors
    ``[total_k, num_kv_heads, head_dim]`` (like Q), and the ``BlockTable`` arg
    slot instead carries ``cu_seqlens_k [batch + 1]`` -- both the per-sequence KV
    start offset and length are derived from it.  This is the non-paged
    ``flash_attn_varlen_func`` path (fresh prefill without a prefix cache).

    Returns ``(kernel, threads, smem_bytes, (BM, BN, BK))``.  The kernel is
    independent of the (runtime) sequence lengths / batch; the launcher sets
    ``grid = (max_q_tiles, batch, num_heads)``.

    Kernel arguments (all flattened 1-D except where noted):
      Q          : [total_q, num_heads, head_dim]        bf16
      KCache     : [num_blocks, page, num_kv_heads, head_dim]  bf16 (upstream)
      VCache     : same layout as KCache
      CuSeqlensQ : [batch + 1]  int32
      SequsedK   : [batch]      int32   (real KV length per sequence)
      BlockTable : [batch, max_blocks] int32
      O          : [total_q, num_heads, head_dim]        bf16
    """
    assert head_dim in (128, 256), "varlen kernel only supports head_dim in (128, 256)"
    assert num_heads % num_kv_heads == 0
    assert page_block_size % SME_ROWS == 0, "page_block_size must be a multiple of 16"
    assert causal_balance in ("head_fastest", "head_parity"), f"unknown causal_balance {causal_balance!r}"

    repeat = num_heads // num_kv_heads
    if cache_strides is None:
        if upstream_cache_layout:
            cache_strides = (
                page_block_size * num_kv_heads * head_dim,
                num_kv_heads * head_dim,
                head_dim,
                1,
            )
        else:
            cache_strides = (
                num_kv_heads * page_block_size * head_dim,
                page_block_size * head_dim,
                head_dim,
                1,
            )
    assert cache_strides[-1] == 1
    cache_block_stride = cache_strides[0]
    cache_head_stride = cache_strides[2] if upstream_cache_layout else cache_strides[1]
    cache_page_stride = cache_strides[1] if upstream_cache_layout else cache_strides[2]
    # ``page_block_size`` is a compile-time constant.  Power-of-two pages
    # (vLLM default 16) turn ``pos // page`` / ``pos % page`` into shift/mask.
    # ``pos_group`` is always >= 0, so arithmetic and logical shifts match and
    # we avoid signed-division sign correction.
    _page_pow2 = (page_block_size & (page_block_size - 1)) == 0
    _page_log2 = page_block_size.bit_length() - 1
    # CTA size: 8 warps (BM=128) for short prefills, 16 warps (BM=256) for long
    # ones.  The larger tile reuses each staged K/V over more query rows, so
    # KV bytes per FLOP drop.
    NUM_WARPS = num_warps
    BK = head_dim
    BM = WARP_M * NUM_WARPS  # 128 (8 warps) or 256 (16 warps)
    threads = NUM_WARPS * WARP_SIZE  # 512 or 1024
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    scale_log2e = softmax_scale * _LOG2E

    WARP_ATOMS_M = BN // ATOM_M  # 8 key atoms along QK N
    K_STEPS_QK = head_dim // ATOM_K
    k_steps_pv = BN // ATOM_K  # 8
    d_atoms = head_dim // ATOM_N
    k_rep = BK // ATOM_K  # 8

    cta_atoms_k = BK // SME_BF16_PER_ROW  # 4
    cta_atoms_k_q = head_dim // SME_BF16_PER_ROW

    k_atoms_m = BN // SME_ROWS  # 8
    k_atoms_total = k_atoms_m * cta_atoms_k  # 32
    q_atoms_m = BM // SME_ROWS  # 8 or 16
    q_atoms_total = q_atoms_m * cta_atoms_k_q  # 32 or 64
    assert k_atoms_total % NUM_WARPS == 0
    assert q_atoms_total % NUM_WARPS == 0
    k_per_warp = k_atoms_total // NUM_WARPS  # 4 (8 warps) or 2 (16 warps)
    q_per_warp = q_atoms_total // NUM_WARPS  # 4
    # Paged KV gather: a 16-row group is ``cta_atoms_k`` column bricks wide.
    # With 8 warps one warp owns a whole group (k_per_warp == cta_atoms_k); with
    # 16 warps two warps split a group's bricks (k_per_warp == cta_atoms_k / 2).
    # ``ni_warp = warp_k_start // cta_atoms_k`` and ``ki = atom_idx % cta_atoms_k``
    # keep the group base + brick indexing correct in both cases.
    assert cta_atoms_k % k_per_warp == 0, "each 16-row KV group's bricks must be split across a whole number of warps"

    v_smem_elems = head_dim * BN
    k_smem_offset = v_smem_elems
    k_stage_elems = BN * BK
    smem_bytes = (v_smem_elems + k_stage_elems) * 2

    total_smem_elems = v_smem_elems + k_stage_elems
    # Q is staged into the same dynamic SMEM before the K/V mainloop reuses it,
    # so the (transient) Q tile must fit inside the V+K staging arena.
    assert BM * head_dim <= total_smem_elems, "Q tile does not fit in shared memory; reduce num_warps"
    TC_WORDS_PER_WARP = 512
    tc_total_bytes = NUM_WARPS * TC_WORDS_PER_WARP * 4
    assert tc_total_bytes <= total_smem_elems * 2

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def flash_attn_varlen_kernel(
        Q: fx.Pointer,
        KCache: fx.Pointer,
        VCache: fx.Pointer,
        CuSeqlensQ: fx.Pointer,
        SequsedK: fx.Pointer,
        BlockTable: fx.Tensor,
        Out: fx.Pointer,
    ):
        tid = fx.thread_idx.x
        if fx.const_expr(causal_balance == "head_parity"):
            # grid (max_q_tiles, batch, num_heads): q-block varies fastest.
            # Under causal, even heads reverse q-block order so adjacent
            # (odd, even) heads at the same block index do work
            # (i+1)+(N-i)=N+1 -- constant triangular load per head pair, while
            # each head still keeps its q-blocks consecutive (KV L2 locality).
            q_blk = fx.Int32(fx.block_idx.x)
            b = fx.Int32(fx.block_idx.y)
            hq = fx.Int32(fx.block_idx.z)
            if fx.const_expr(causal):
                rev = fx.Int32(fx.grid_dim.x) - fx.Int32(1) - q_blk
                is_even = (hq & fx.Int32(1)) == fx.Int32(0)
                q_tile = is_even.select(rev, q_blk)
            else:
                q_tile = q_blk
        else:
            # grid (num_heads, batch, max_q_tiles): head varies fastest so a
            # dispatch warp holds CTAs on the same q-tile (same causal KV span)
            # and, under GQA, adjacent heads share a KV head (L2 reuse).
            hq = fx.Int32(fx.block_idx.x)
            b = fx.Int32(fx.block_idx.y)
            if fx.const_expr(causal):
                # Largest q-tiles first; lighter tiles backfill as heavy CTAs retire.
                q_tile = fx.Int32(fx.grid_dim.z) - fx.Int32(1) - fx.Int32(fx.block_idx.z)
            else:
                q_tile = fx.Int32(fx.block_idx.z)
        warp_id = tid // WARP_SIZE
        # Prefer ``gpu.lane_id`` / ``fx.lane_id``.  Paged keeps the Iluvatar
        # ``llvm.bi.lane.id`` intrinsic for now: switching it to ``lane_id``
        # regresses this path.
        if fx.const_expr(paged):
            lane_id = fx.Int32(_llvm.call_intrinsic(T.i32, "llvm.bi.lane.id", [], [], []))
        else:
            lane_id = fx.Int32(gpu.lane_id)

        c_num_heads = fx.Int32(num_heads)
        c_repeat = fx.Int32(repeat)
        hkv = hq // c_repeat

        c_Hkv = fx.Int32(num_kv_heads)
        c_page = fx.Int32(page_block_size)
        c_D = fx.Int32(head_dim)
        c_BM = fx.Int32(BM)
        c_BN = fx.Int32(BN)

        # ---- CTA-uniform scalar loads (readfirstlane → SGPR) ----
        def _load_i32(base, idx):
            """Load one gmem i32 and broadcast to the warp as an SGPR."""
            ptr = fx.recast_iter(fx.PointerType.get(T.i32, fx.AddressSpace.Global), base)
            loaded = fx.ptr_load(fx.add_offset(ptr, fx.make_int_tuple(idx)), T.i32)
            return ixdl.readfirstlane(loaded)

        q_start = _load_i32(CuSeqlensQ, b)
        q_end = _load_i32(CuSeqlensQ, b + fx.Int32(1))
        seqlen_q_b = q_end - q_start
        if fx.const_expr(paged):
            k_start = fx.Int32(0)
            if fx.const_expr(seqused_is_cumulative):
                seqlen_k_b = _load_i32(SequsedK, b + fx.Int32(1)) - _load_i32(SequsedK, b)
            else:
                seqlen_k_b = _load_i32(SequsedK, b)

            # Hoist block-table row base once; ``b`` is CTA-invariant.
            table_row = fx.slice(BlockTable, (b, None))
            table_base = fx.inttoptr(
                fx.PointerType.get(T.i32, fx.AddressSpace.Global),
                ixdl.readfirstlane(fx.ptrtoint(fx.get_iter(table_row))),
            )

            def _load_block(logical_block):
                # Index may come from select/clamp; force uniform before addressing.
                logical_block = ixdl.readfirstlane(logical_block)
                loaded = fx.ptr_load(
                    fx.add_offset(table_base, fx.make_int_tuple(logical_block)), T.i32
                )
                return ixdl.readfirstlane(loaded)
        else:
            # Dense: BlockTable slot carries cu_seqlens_k [batch + 1].
            ck = fx.get_iter(BlockTable)
            k_start = _load_i32(ck, b)
            seqlen_k_b = _load_i32(ck, b + fx.Int32(1)) - k_start

        # Per-tile query row base (local within the sequence).
        tile_row0 = q_tile * c_BM
        tile_active = tile_row0 < seqlen_q_b

        # Global query-row base for this tile (packed [total_q, H, D]).  For
        # dead tiles we point at the sequence start (always valid memory); the
        # epilogue store guard prevents any write.
        q_row_base = tile_active.select(q_start + tile_row0, q_start)

        # ---- KV-tile loop bound (runtime) ----
        # non-causal: cover all keys of the sequence.
        # causal: cover only keys this tile can see:
        #   max_key = min(sk, tile_row0 + BM + sk - sq)
        if fx.const_expr(causal):
            delta = seqlen_k_b - seqlen_q_b
            cap = tile_row0 + c_BM + delta
            max_key = (cap < seqlen_k_b).select(cap, seqlen_k_b)
            # clamp negative (can happen for empty tiles / sq > sk edge)
            max_key = (max_key > fx.Int32(0)).select(max_key, fx.Int32(0))
        else:
            max_key = seqlen_k_b

        end_tile_full = (max_key + c_BN - fx.Int32(1)) // c_BN
        end_tile = tile_active.select(end_tile_full, fx.Int32(0))

        # Last valid (allocated) KV block; clamp paged lookups so we never read
        # past the sequence's allocation (extra keys are masked out anyway).
        last_block_raw = (seqlen_k_b - fx.Int32(1)) // c_page
        last_block = (last_block_raw > fx.Int32(0)).select(last_block_raw, fx.Int32(0))

        def _kv_group_base(pos_group, clamp=True):
            """Element base + row stride for a 16-key group at ``pos_group``.

            paged: vLLM NHD (upstream) layout [blocks, page, Hkv, D] (or HND).
            dense: packed [total_k, Hkv, D]; base = (k_start + pos) * Hkv * D.
            """
            if fx.const_expr(not paged):
                if fx.const_expr(clamp):
                    # Boundary/dead tiles need a safe source row.  Interior
                    # tiles are proven in range by their loop bound and skip
                    # these compares/selects on every KV iteration.
                    pos_c = (pos_group < seqlen_k_b).select(pos_group, seqlen_k_b - fx.Int32(1))
                    pos_c = (pos_c > fx.Int32(0)).select(pos_c, fx.Int32(0))
                else:
                    pos_c = fx.Int32(pos_group)
                gpos = k_start + pos_c
                base = (gpos * c_Hkv + hkv) * c_D
                stride = num_kv_heads * head_dim
                return base, stride
            if fx.const_expr(_page_pow2):
                logical_block = pos_group >> fx.Int32(_page_log2)
                block_off = pos_group & fx.Int32(page_block_size - 1)
            else:
                logical_block = pos_group // c_page
                block_off = pos_group % c_page
            if fx.const_expr(clamp):
                logical_block = (logical_block < last_block).select(logical_block, last_block)
            phys = _load_block(logical_block)
            base = (
                phys * fx.Int32(cache_block_stride)
                + hkv * fx.Int32(cache_head_stride)
                + block_off * fx.Int32(cache_page_stride)
            )
            return base, cache_page_stride

        # ---- MMA / copy setup ----
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

        tile_sme = fx.make_tile(SME_ROWS, SME_BF16_PER_ROW)
        tile_atom_A = fx.make_tile(ATOM_M, ATOM_K)
        tile_atom_B = fx.make_tile(ATOM_N, ATOM_K)

        smem_ptr = fx.get_dyn_shared()

        def _make_bf16_gmem_tile(tensor, elem_base, shape, stride):
            base_ptr = fx.recast_iter(fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global), tensor)
            ptr = fx.add_offset(base_ptr, fx.make_int_tuple(elem_base))
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

        warp_q_start = warp_id * fx.Int32(q_per_warp)
        warp_k_start = warp_id * fx.Int32(k_per_warp)
        # This warp's KV 16-row group index within a BN tile (constant per warp).
        ni_warp = warp_k_start // fx.Int32(cta_atoms_k)

        def _sync_arrive():
            ixdl.cp_async_wait_group(0)
            gpu.barrier()

        def _make_k_atoms():
            k_atoms_s = []
            for jm in fx.range_constexpr(WARP_ATOMS_M):
                row = []
                for ki in fx.range_constexpr(cta_atoms_k):
                    off = fx.Int32(k_smem_offset + (jm * cta_atoms_k + ki) * BRICK_ELEMS)
                    row.append(fx.zipped_divide(_sme_view_dyn(smem_ptr, fx.BFloat16, off, False), tile_atom_A))
                k_atoms_s.append(row)
            return k_atoms_s

        k_atoms_list = _make_k_atoms()

        # ---- Load Q (BM rows) through SMEM into registers ----
        q_elem_base = (q_row_base * c_num_heads + hq) * c_D
        q_row_stride = num_heads * head_dim
        q_sme_tile = _make_bf16_gmem_tile(Q, q_elem_base, (BM, head_dim), (q_row_stride, 1))
        q_div = fx.zipped_divide(q_sme_tile, tile_sme)
        for t in fx.range_constexpr(q_per_warp):
            atom_idx = warp_q_start + fx.Int32(t)
            mi = atom_idx // fx.Int32(cta_atoms_k_q)
            ki_q = atom_idx % fx.Int32(cta_atoms_k_q)
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

        # Output view (i32: two packed bf16 per store).
        out_i32_ptr = fx.recast_iter(fx.PointerType.get(T.i32, fx.AddressSpace.Global), Out)
        out_i32_elem_base = (q_row_base * c_num_heads + hq) * fx.Int32(head_dim // 2)
        out_row_stride_i32 = num_heads * (head_dim // 2)
        out_i32_tile = fx.Tensor(
            fx.make_view(
                fx.add_offset(out_i32_ptr, fx.make_int_tuple(out_i32_elem_base)),
                fx.make_layout((BM, head_dim // 2), (out_row_stride_i32, 1)),
            )
        )

        # ---- Paged 16-row-group K/V gather (incremental base) ----
        # K runs one tile ahead of V.  Each iteration resolves one group base
        # (K prefetch of tile+1); that base becomes V for the next tile, so V
        # does not re-walk the block table.  Carrying the base across the loop
        # also keeps fewer paged-address scalars live than recomputing from the
        # tile index every time.
        # Row (token) stride of a 16-key group.  Dense packed [total_k, Hkv, D]
        # and NHD [blocks, page, Hkv, D] both step Hkv*D between rows; HND
        # [blocks, Hkv, page, D] steps D.
        kv_stride = cache_page_stride if paged else (num_kv_heads * head_dim)

        def _kv_base(pos_group, clamp=True):
            base, _ = _kv_group_base(pos_group, clamp)
            return base

        def issue_k_from_base(base):
            k_grp = _make_bf16_gmem_tile(KCache, base, (SME_ROWS, BK), (kv_stride, 1))
            issue_k_from_tensor(k_grp)

        def issue_k_from_tensor(k_grp):
            k_div = fx.zipped_divide(k_grp, tile_sme)
            for t in fx.range_constexpr(k_per_warp):
                atom_idx = warp_k_start + fx.Int32(t)
                ki = atom_idx % fx.Int32(cta_atoms_k)
                k_off = fx.Int32(k_smem_offset) + atom_idx * fx.Int32(BRICK_ELEMS)
                fx.copy_atom_call(
                    sme_atom_K, fx.slice(k_div, (None, (0, ki))), _sme_view_dyn(smem_ptr, fx.BFloat16, k_off, False)
                )
            ixdl.cp_async_commit_group()

        def issue_v_from_base(base):
            v_grp = _make_bf16_gmem_tile(VCache, base, (SME_ROWS, BK), (kv_stride, 1))
            issue_v_from_tensor(v_grp)

        def issue_v_from_tensor(v_grp):
            v_div = fx.zipped_divide(v_grp, tile_sme)
            for t in fx.range_constexpr(k_per_warp):
                atom_idx = warp_k_start + fx.Int32(t)
                ki = atom_idx % fx.Int32(cta_atoms_k)
                v_off = atom_idx * fx.Int32(BRICK_ELEMS)
                fx.copy_atom_call(
                    sme_atom_col, fx.slice(v_div, (None, (0, ki))), _sme_view_dyn(smem_ptr, fx.BFloat16, v_off, True)
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
                fx.copy(
                    copy_atom_s2r,
                    thr_copy_A.partition_S(k_tile_s),
                    thr_copy_A.retile(frag_K),
                    pred=None,
                )
                kFs.append(frag_K)
            return fQ, kFs

        def mma_frags(frag_Q, k_frags):
            for jm in fx.range_constexpr(WARP_ATOMS_M):
                fx.gemm(mma_atom, accs[jm], k_frags[jm], frag_Q, accs[jm])

        def compute_qk_stage(sK):
            for kk in fx.range_constexpr(k_rep):
                fQ_cur, kFs_cur = load_k_frags(sK, kk)
                mma_frags(fQ_cur, kFs_cur)

        # Query-row absolute position (local within sequence) for masking.
        score_lane_col = lane_id % fx.Int32(16)  # query row within atom
        score_lane_row = lane_id // fx.Int32(16)  # key sub-row within atom
        q_row_abs = q_tile * c_BM + warp_id * fx.Int32(WARP_M) + score_lane_col
        q_row_valid = q_row_abs < seqlen_q_b
        if fx.const_expr(causal):
            causal_delta = seqlen_k_b - seqlen_q_b

        # ---- Split the KV range into an unmasked interior + a masked boundary.
        # A kv_tile needs NO masking when (a) the whole q-tile's rows are valid
        # (q_full), (b) every key in the tile is < seqlen_k (k_full), and, for
        # causal, (c) the tile's largest key is still <= the tile's smallest
        # query row + (sk - sq) so the causal predicate holds for all elements.
        # These interior tiles run a branch-free fast path; only the diagonal /
        # tail tiles pay the per-element select cost.
        q_full = tile_row0 + c_BM <= seqlen_q_b
        kfull_count = seqlen_k_b // c_BN
        if fx.const_expr(causal):
            cs_limit = tile_row0 + causal_delta - fx.Int32(BN - 1)
            cs_count = (cs_limit >= fx.Int32(0)).select(cs_limit // c_BN + fx.Int32(1), fx.Int32(0))
            interior_count = (kfull_count < cs_count).select(kfull_count, cs_count)
        else:
            interior_count = kfull_count
        interior_count = q_full.select(interior_count, fx.Int32(0))
        interior_end = (interior_count <= end_tile).select(interior_count, end_tile)

        # Constants used by the in-register C(f32) -> B(bf16) pack.
        _c16_p = fx.Int32(16)
        _cmask_p = fx.Int32(-65536)

        def process_tile(kv_tile, m_running, l_running, kv_state, masked, kv_v_state=None):
            if fx.const_expr(not paged):
                # Dense path: carry SME fat pointers across tiles.  Rebuilding
                # them from KCache/VCache + kv_base each iteration materializes
                # ptrtoint/descriptor vectors that spill under the SGPR budget.
                k_ptr = fx.Pointer(kv_state)
                v_ptr = fx.Pointer(kv_v_state)
                issue_v_from_tensor(fx.Tensor(fx.make_view(v_ptr, fx.make_layout((SME_ROWS, BK), (kv_stride, 1)))))
            else:
                # Paged V reuses the base resolved by the previous K prefetch.
                issue_v_from_base(fx.Int32(kv_state))

            if fx.const_expr(paged):
                # Resolve the following tile while QK/softmax is running.  The
                # page-table load is independent of the current K/V fragments,
                # but delaying it until immediately before the K G2S issue
                # exposes its global-memory latency on every long-sequence tile.
                pos_next = (fx.Int32(kv_tile) + fx.Int32(1)) * c_BN + ni_warp * fx.Int32(SME_ROWS)
                kv_base_next = _kv_base(pos_next, clamp=masked)

            if fx.const_expr(masked and causal):
                warp_q0 = q_tile * c_BM + warp_id * fx.Int32(WARP_M)
                warp_q_last = warp_q0 + fx.Int32(WARP_M - 1)
                tile_k0 = fx.Int32(kv_tile) * c_BN
                warp_q_valid = warp_q0 < seqlen_q_b
                warp_key_visible = tile_k0 <= warp_q_last + causal_delta
                warp_has_scores = warp_q_valid & warp_key_visible

            def _qk_softmax(m_cur, l_cur):
                for jm in fx.range_constexpr(WARP_ATOMS_M):
                    accs[jm].fill(0)

                compute_qk_stage(k_atoms_list)

                all_vals = []
                for jm in fx.range_constexpr(WARP_ATOMS_M):
                    all_vals.append(accs[jm])

                def _score_valid(jm, v):
                    kpos = fx.Int32(kv_tile) * c_BN + fx.Int32(jm * ATOM_N + v * 4) + score_lane_row
                    k_ok = kpos < seqlen_k_b
                    valid = q_row_valid & k_ok
                    if fx.const_expr(causal):
                        lim = q_row_abs + causal_delta
                        valid = valid & (kpos <= lim)
                    return valid

                local_max = c_neg_inf
                for jm in fx.range_constexpr(WARP_ATOMS_M):
                    for v in fx.range_constexpr(FRAG_ELEMS):
                        if fx.const_expr(masked):
                            sv = _score_valid(jm, v)
                            score_val = sv.select(all_vals[jm][v], c_neg_inf)
                        else:
                            score_val = fx.Float32(all_vals[jm][v])
                        local_max = arith.maxnumf(fx.Float32(local_max), score_val)

                for mask in [16, 32]:
                    peer = fx.Float32(local_max).shuffle_xor(fx.Int32(mask), fx.Int32(WARP_SIZE))
                    local_max = arith.maxnumf(fx.Float32(local_max), fx.Float32(peer))

                local_max = fx.Float32(local_max) * c_scale_log2e

                m_new = arith.maxnumf(fx.Float32(m_cur), fx.Float32(local_max))
                corr_exp = fx.Float32(m_cur - m_new).exp2()
                is_first_kv = fx.Int32(kv_tile) == fx.Int32(0)
                corr = is_first_kv.select(c_zero, corr_exp)

                for nt in fx.range_constexpr(d_atoms):
                    pv_v = pv_accs[nt]
                    elems = []
                    for i in fx.range_constexpr(FRAG_ELEMS):
                        elems.append(pv_v[i] * corr)
                    pv_accs[nt].store(
                        vector.from_elements(T.vec(FRAG_ELEMS, T.f32), [_to_raw(elem) for elem in elems])
                    )

                local_sum = fx.Float32(0.0)
                neg_m_new = -m_new
                for jm in fx.range_constexpr(WARP_ATOMS_M):
                    p_elems = []
                    for v in fx.range_constexpr(FRAG_ELEMS):
                        p_raw = fx.fma(all_vals[jm][v], c_scale_log2e, neg_m_new).exp2()
                        if fx.const_expr(masked):
                            sv = _score_valid(jm, v)
                            p = sv.select(p_raw, c_zero)
                        else:
                            p = p_raw
                        p_elems.append(p)
                        local_sum = local_sum + p
                    accs[jm].store(
                        vector.from_elements(T.vec(FRAG_ELEMS, T.f32), [_to_raw(elem) for elem in p_elems])
                    )

                l_new = fx.fma(l_cur, corr, local_sum)
                return [m_new, l_new]

            def _pack_p_frags():
                # f32→i32, take high 16 bits of each as bf16 (IEEE upper half).
                packed = []
                for ki in fx.range_constexpr(k_steps_pv):
                    p_vals = accs[ki]
                    pk = []
                    for v in fx.range_constexpr(FRAG_ELEMS // 2):
                        a = p_vals[v * 2].bitcast(fx.Int32)
                        bb = p_vals[v * 2 + 1].bitcast(fx.Int32)
                        pk.append((bb & _cmask_p) | a.shrui(_c16_p))
                    packed.append(
                        Vec.from_elements(pk, fx.Int32).bitcast(fx.BFloat16)
                    )
                return packed

            if fx.const_expr(masked and causal):
                if warp_has_scores:
                    m_running, l_running = _qk_softmax(m_running, l_running)
            else:
                m_running, l_running = _qk_softmax(m_running, l_running)
                packed_p = _pack_p_frags()

            _sync_arrive()

            # Prefetch next-tile K; carry the resolved group base so the next
            # iteration's V reuses it.
            if fx.const_expr(not paged):
                step = fx.make_int_tuple(BN * num_kv_heads * head_dim)
                k_ptr_next = fx.add_offset(k_ptr, step)
                v_ptr_next = fx.add_offset(v_ptr, step)
                if fx.Int32(kv_tile) != end_tile - fx.Int32(1):
                    issue_k_from_tensor(
                        fx.Tensor(fx.make_view(k_ptr_next, fx.make_layout((SME_ROWS, BK), (kv_stride, 1))))
                    )
            else:
                if fx.Int32(kv_tile) != end_tile - fx.Int32(1):
                    issue_k_from_base(kv_base_next)

            gpu.barrier()

            def _pv_mmad(packed_p):
                # ---- PV MMAD: V as A, P as B -> O^T[d, m] ----
                for ki in fx.range_constexpr(k_steps_pv):
                    frag_P = fx.make_fragment_like(q_reg_frags[0])
                    frag_P.store(packed_p[ki])

                    for nt in fx.range_constexpr(d_atoms):
                        slb_tcu_idx = ki * d_atoms + nt
                        brick_off = fx.Int32((slb_tcu_idx // 2) * BRICK_ELEMS)
                        d_sub = slb_tcu_idx % 2
                        v_view = _sme_view_dyn(smem_ptr, fx.BFloat16, brick_off, True)
                        # SME Col brick is V[key, d].  Compose a (d, k)->(key, d)
                        # map so A-fragment lanes see the PV key order:
                        # key = pi(k) with pi([b3|b2|b1|b0]) = [b3|b0|b2|b1].
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

            if fx.const_expr(masked and causal):
                if warp_has_scores:
                    _pv_mmad(_pack_p_frags())
            else:
                _pv_mmad(packed_p)

            _sync_arrive()

            if fx.const_expr(not paged):
                return [m_running, l_running, _to_raw(k_ptr_next), _to_raw(v_ptr_next)]
            return [m_running, l_running, _to_raw(kv_base_next)]

        # Prologue: resolve + stage the first K tile.  Safe even for dead tiles
        # (end_tile == 0): the paged base is clamped to a valid block and the
        # loops below simply run zero iterations.  ``kv_base`` (the group base of
        # the tile V will consume) is then carried through the KV loops.
        if fx.const_expr(not paged):
            kv_base0 = _kv_base(ni_warp * fx.Int32(SME_ROWS))
            k_tensor0 = _make_bf16_gmem_tile(KCache, kv_base0, (SME_ROWS, BK), (kv_stride, 1))
            v_tensor0 = _make_bf16_gmem_tile(VCache, kv_base0, (SME_ROWS, BK), (kv_stride, 1))
            k_ptr0 = fx.get_iter(k_tensor0)
            v_ptr0 = fx.get_iter(v_tensor0)
            issue_k_from_tensor(k_tensor0)
        else:
            kv_base0 = _kv_base(ni_warp * fx.Int32(SME_ROWS))
            issue_k_from_base(kv_base0)
        _sync_arrive()

        if fx.const_expr(not paged):
            # Interior: fully-valid tiles, branch-free (no masking).
            k_ptr = _to_raw(k_ptr0)
            v_ptr = _to_raw(v_ptr0)
            for kv_tile in range(0, interior_end, 1):
                m_running, l_running, k_ptr, v_ptr = process_tile(
                    kv_tile, m_running, l_running, k_ptr, masked=False, kv_v_state=v_ptr
                )

            # Dense boundary tiles remain physically contiguous into the
            # zero-padded guard rows, so the same descriptor increment is safe.
            for kv_tile in range(interior_end, end_tile, 1):
                m_running, l_running, k_ptr, v_ptr = process_tile(
                    kv_tile, m_running, l_running, k_ptr, masked=True, kv_v_state=v_ptr
                )
        else:
            kv_base = _to_raw(kv_base0)
            for kv_tile in range(0, interior_end, 1):
                m_running, l_running, kv_base = process_tile(kv_tile, m_running, l_running, kv_base, masked=False)

            for kv_tile in range(interior_end, end_tile, 1):
                m_running, l_running, kv_base = process_tile(kv_tile, m_running, l_running, kv_base, masked=True)

        for mask in [16, 32]:
            ps = fx.Float32(l_running).shuffle_xor(fx.Int32(mask), fx.Int32(WARP_SIZE))
            l_running = l_running + ps

        # ---- Epilogue: normalise + TransposeCToB16 ----
        has_visible_key = l_running > fx.Float32(0.0)
        inv_l = has_visible_key.select(fx.Float32(1.0) / l_running, fx.Float32(0.0))

        i32_smem_ptr = fx.recast_iter(fx.PointerType.get(T.i32, fx.AddressSpace.Shared), smem_ptr)
        tc_warp_base = warp_id * fx.Int32(TC_WORDS_PER_WARP)
        tc_smem_iter = fx.add_offset(i32_smem_ptr, fx.make_int_tuple(tc_warp_base))

        seq_q_g_base = warp_id * fx.Int32(WARP_M)
        out_tile_iter = fx.get_iter(out_i32_tile)

        # TransposeCToB16: address maps as layouts; R→S / S→R / R→G via fx.copy.
        # Pack/repack of bf16 pairs stay arithmetic (byte_permute).
        copy_atom_i32 = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
        tc_write_lyt = fx.make_composed_layout(
            fx.static(fx.SwizzleType.get(3, 1, 5)),
            fx.make_layout((8, (2, 2, 2, 2, 2, 2)), (2, (16, 32, 64, 128, 256, 1))),
        )
        tc_read_lyt = fx.make_composed_layout(
            fx.static(fx.SwizzleType.get(3, 1, 5)),
            fx.make_layout((8, 64), (64, 1)),
        )
        # (ei, tid) → offset in warp-local rows with CTA out row stride.
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
                v0 = r_load[i]
                v1 = r_load[i + 4]
                # lo16(v0)|lo16(v1)<<16 and hi16(v0)|hi16(v1); same as CUDA byte_perm.
                out0_vals.append(ixdl.byte_permute(v0, v1, 0x5410))
                out1_vals.append(ixdl.byte_permute(v0, v1, 0x7632))

            col_base0 = fx.Int32(ni_quad * 4 * ATOM_N // 2)
            col_base1 = fx.Int32((ni_quad * 4 + 2) * ATOM_N // 2)
            warp_elem = seq_q_g_base * fx.Int32(out_row_stride_i32)
            g0 = fx.make_view(
                fx.add_offset(out_tile_iter, fx.make_int_tuple(warp_elem + col_base0)), out_gmem_lyt
            )
            g1 = fx.make_view(
                fx.add_offset(out_tile_iter, fx.make_int_tuple(warp_elem + col_base1)), out_gmem_lyt
            )

            # Per-value OOB pred: row_abs = q_tile*BM + seq_q_g_base + ei*4 + tid//16
            lane_row = lane_id // fx.Int32(16)
            for vals, g in ((out0_vals, g0), (out1_vals, g1)):
                thr_d = thr_out.partition_D(g)
                r_o = fx.make_fragment_like(thr_d)
                r_o.store(Vec.from_elements(vals, fx.Int32))
                pred = fx.make_fragment_like(thr_d, dtype=fx.Boolean)
                for ei in fx.range_constexpr(FRAG_ELEMS):
                    row_abs = q_tile * c_BM + seq_q_g_base + fx.Int32(ei * 4) + lane_row
                    pred[ei] = row_abs < seqlen_q_b
                fx.copy(copy_atom_i32, r_o, thr_d, pred=pred)

    return flash_attn_varlen_kernel, threads, smem_bytes, (BM, BN, BK)
