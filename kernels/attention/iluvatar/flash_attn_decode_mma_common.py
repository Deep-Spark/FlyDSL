# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Shared geometry and tile helpers for baseline / pipelined MMA decode.

Both decode builders own their KV-scheduling loops (single-buffer sync vs
double-buffer prefetch). Softmax updates and the PV MMAD are shared here so
the two files cannot drift on the numerically sensitive path.

Masking contract (aligned with ``flash_attn_varlen_mma``)
--------------------------------------------------------
* **GQA pad** (``lane%16 >= repeats``) is always masked -- decode packs the
  GQA group into a 16-row MMAD atom.
* **K length** (``kpos >= seqlen_k_b``) is only required on the last partial
  KV tile. Full tiles use the branch-free interior path; the boundary loop
  pays per-element selects. Unlike varlen there is no causal diagonal.
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl._mlir.dialects import vector
from flydsl.expr import arith
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw

from .flash_attn_mma_common import (
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

# Tile shape: one warp owns the whole GQA group (16 MMAD rows).
BM = ATOM_M
BN = 128
BK = HEAD_DIM
NUM_WARPS = 1
WARP_M = ATOM_M

_sme_view_dyn = sme_view


def decode_mma_smem_bytes(*, block_n: int = BN, num_buffers: int = 1) -> int:
    """Bytes of dynamic SMEM for one decode CTA (bf16 elements × 2)."""
    v_smem_elems = HEAD_DIM * block_n
    k_stage_elems = block_n * BK
    return num_buffers * (v_smem_elems + k_stage_elems) * 2


def pipelined_mma_decode_smem_bytes(*, block_n: int = BN) -> int:
    """Double-buffered SMEM: ``[V0][V1][K0][K1]``."""
    return decode_mma_smem_bytes(block_n=block_n, num_buffers=2)


def validate_mma_decode_config(
    *,
    head_dim: int,
    num_heads: int,
    num_kv_heads: int,
    block_n: int,
    max_seqlen_k: int,
    num_splits: int,
    paged: bool,
    page_block_size: int,
) -> int:
    """Shared shape asserts. Returns ``repeats = num_heads // num_kv_heads``."""
    assert head_dim == HEAD_DIM, "MMA decode kernel only supports head_dim == 128"
    assert num_heads % num_kv_heads == 0
    assert block_n % ATOM_M == 0, "block_n must be a multiple of 16"
    assert max_seqlen_k % block_n == 0, "max_seqlen_k must be a multiple of block_n for the MMA decode path"
    assert num_splits >= 1
    if paged:
        assert page_block_size % SME_ROWS == 0, "page_block_size must be a multiple of 16"
    repeat = num_heads // num_kv_heads
    assert 1 <= repeat <= ATOM_M, "repeats (num_heads/num_kv_heads) must be <= 16"
    return repeat


def bound_decode_tile_end(*, tile_start, tile_end, seqlen_k, block_n: int):
    """Clamp a split's tile end to the runtime-visible KV tile count."""
    active_tiles = (seqlen_k + fx.Int32(block_n - 1)) // fx.Int32(block_n)
    bounded_end = (tile_end < active_tiles).select(tile_end, active_tiles)
    # A late split can start beyond the runtime sequence.  Returning start
    # makes the range empty and gives callers a uniform ``start < end`` guard
    # for their DMA prologue.
    return (bounded_end > tile_start).select(bounded_end, tile_start)


def clamp_decode_group_start(*, pos_group, seqlen_k, group_rows: int = SME_ROWS):
    """Map a boundary-tile gather to its last runtime-valid row group.

    Decode SME transfers operate on 16 rows at a time.  Reusing the final valid
    group for masked tail groups both initializes the complete SMEM tile and
    avoids looking up padded block-table entries.
    """
    last_pos = seqlen_k - fx.Int32(1)
    last_group = (last_pos // fx.Int32(group_rows)) * fx.Int32(group_rows)
    return (pos_group < seqlen_k).select(pos_group, last_group)


def apply_decode_online_softmax(
    *,
    kv_tile,
    m_running,
    l_running,
    accs,
    pv_accs,
    all_vals,
    seqlen_k_b,
    tile_start,
    c_repeat,
    score_lane_col,
    score_lane_row,
    c_neg_inf,
    c_zero,
    c_scale_log2e,
    block_n: int,
    warp_atoms_m: int,
    d_atoms: int,
    masked: bool,
):
    """Online-softmax update for one KV tile; writes P into ``accs``.

    When ``masked`` is false the tile is known fully in-range for K length;
    only the GQA pad rows are suppressed. When true, also mask ``kpos >= seqlen_k_b``.
    """
    q_row_valid = score_lane_col < c_repeat

    def _score_valid(jm, v):
        valid = q_row_valid
        if fx.const_expr(masked):
            k_col_abs = fx.Int32(kv_tile) * fx.Int32(block_n) + fx.Int32(jm * ATOM_N + v * 4) + score_lane_row
            valid = valid & (k_col_abs < seqlen_k_b)
        return valid

    local_max = c_neg_inf
    for jm in fx.range_constexpr(warp_atoms_m):
        for v in fx.range_constexpr(FRAG_ELEMS):
            score_val = _score_valid(jm, v).select(all_vals[jm][v], c_neg_inf)
            local_max = arith.maxnumf(fx.Float32(local_max), score_val)

    for mask in [16, 32]:
        peer = fx.Float32(local_max).shuffle_xor(fx.Int32(mask), fx.Int32(WARP_SIZE))
        local_max = arith.maxnumf(fx.Float32(local_max), fx.Float32(peer))

    local_max = fx.Float32(local_max) * c_scale_log2e

    m_old = m_running
    m_new = arith.maxnumf(fx.Float32(m_old), fx.Float32(local_max))
    corr_exp = fx.Float32(m_old - m_new).exp2()
    is_first_kv = fx.Int32(kv_tile) == tile_start
    tile_is_empty = fx.Int32(kv_tile) * fx.Int32(block_n) >= seqlen_k_b
    reset_corr = is_first_kv | tile_is_empty
    corr = reset_corr.select(c_zero, corr_exp)

    for nt in fx.range_constexpr(d_atoms):
        pv_v = pv_accs[nt]
        elems = []
        for i in fx.range_constexpr(FRAG_ELEMS):
            elems.append(pv_v[i] * corr)
        pv_accs[nt].store(vector.from_elements(T.vec(FRAG_ELEMS, T.f32), [_to_raw(elem) for elem in elems]))

    l_running_old = l_running
    local_sum = fx.Float32(0.0)
    neg_m_new = -m_new
    for jm in fx.range_constexpr(warp_atoms_m):
        p_elems = []
        for v in fx.range_constexpr(FRAG_ELEMS):
            p_raw = fx.fma(all_vals[jm][v], c_scale_log2e, neg_m_new).exp2()
            p = _score_valid(jm, v).select(p_raw, c_zero)
            p_elems.append(p)
            local_sum = local_sum + p
        accs[jm].store(vector.from_elements(T.vec(FRAG_ELEMS, T.f32), [_to_raw(elem) for elem in p_elems]))

    l_running = fx.fma(l_running_old, corr, local_sum)
    return m_new, l_running


def apply_decode_pv_mmad(
    *,
    accs,
    pv_accs,
    q_reg_frags,
    thr_mma,
    thr_copy_A,
    copy_atom_s2r,
    mma_atom,
    smem_ptr,
    tile_atom_A,
    v_buf_base,
    k_steps_pv: int,
    d_atoms: int,
    c16_p,
    cmask_p,
):
    """PV MMAD: V as A, packed P as B → accumulate into ``pv_accs``."""
    packed_p = []
    for ki in fx.range_constexpr(k_steps_pv):
        p_vals = accs[ki]
        pk = []
        for v in fx.range_constexpr(FRAG_ELEMS // 2):
            a = p_vals[v * 2].bitcast(fx.Int32)
            bb = p_vals[v * 2 + 1].bitcast(fx.Int32)
            pk.append((bb & cmask_p) | a.shrui(c16_p))
        packed_p.append(Vec.from_elements(pk, fx.Int32).bitcast(fx.BFloat16))

    for ki in fx.range_constexpr(k_steps_pv):
        frag_P = fx.make_fragment_like(q_reg_frags[0])
        frag_P.store(packed_p[ki])
        for nt in fx.range_constexpr(d_atoms):
            slb_tcu_idx = ki * d_atoms + nt
            brick_off = v_buf_base + fx.Int32((slb_tcu_idx // 2) * BRICK_ELEMS)
            d_sub = slb_tcu_idx % 2
            v_view = _sme_view_dyn(smem_ptr, fx.BFloat16, brick_off, True)
            # SME Col stores this brick logically as V[key, d]. Compose with
            # (d, k)->(key, d) so the source layout carries the permutation.
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
