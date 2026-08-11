# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""CQ SmexMtx G2S helpers for pipelined HGEMM (ivcore30).

Issues warp-collective ``CQSmexCpMtx`` bricks into Bypass shared memory via
``fx.copy_atom_call``. Pairs with ``CQMtxLoadn`` S2R -- do **not** mix LegacySme
``CQAsyncCp`` on the same buffer.

tn (k-major A/B) only for this bring-up. Brick placement mirrors MR cta_lin
order so S2R EmPart addressing stays consistent with ixmma Loadn16 B16.
"""

import flydsl.expr as fx
import flydsl.expr.ixdl as ixdl
from kernels.gemm.iluvatar.common import SMEM_ROWS
from kernels.gemm.iluvatar.cq.common import CQ_GEMM_GEOM, CqOperandGeom


def cq_g2s_smex_atom(elem_dtype, *, rows: int = SMEM_ROWS):
    """SmexMtx G2S copy atom for ``fx.copy_atom_call``.

    ``rows`` is the SMEX tile height; ``SMEM_ROWS`` (16) is the height that
    pairs with a ``CQMtxLoadn(pattern="loadn16")`` S2R read.
    """
    return fx.make_copy_atom(ixdl.CQSmexCpMtx(rows=rows), elem_dtype)


def cq_smex_brick_tile(*, geom: CqOperandGeom = CQ_GEMM_GEOM):
    """``fx.make_tile`` splitting a k-major CTA tile into SmexMtx G2S bricks."""
    return fx.make_tile(SMEM_ROWS, geom.values_per_sme_row)


def cq_smex_shared_view(smem_base, elem_offset, *, geom: CqOperandGeom = CQ_GEMM_GEOM):
    """One SmexMtx brick destination view in shared memory.

    SMEX writes the matrix format itself, so unlike the MR SME shared views this
    composes no swizzle: the view only carries the brick base (``elem_offset``
    element slots from ``smem_base``) and its ``SMEM_ROWS x vpr`` extents.
    """
    vpr = geom.values_per_sme_row
    smem_ptr = fx.add_offset(smem_base, fx.make_int_tuple(fx.Int32(elem_offset)))
    return fx.make_view(smem_ptr, fx.make_layout((SMEM_ROWS, vpr), (vpr, 1)))


def cq_cta_brick_counts(*, bm: int, bn: int, bk: int, geom: CqOperandGeom = CQ_GEMM_GEOM):
    """G2S brick counts for k-major (tn) A/B on one bm x bn x bk tile."""
    vpr = geom.values_per_sme_row
    if bk % vpr:
        raise ValueError(f"bk={bk} must be a multiple of values_per_sme_row={vpr}")
    if bm % SMEM_ROWS or bn % SMEM_ROWS:
        raise ValueError(f"bm/bn must be multiples of SMEM_ROWS={SMEM_ROWS}")
    k_bricks = bk // vpr
    a_bricks = (bm // SMEM_ROWS) * k_bricks
    b_bricks = (bn // SMEM_ROWS) * k_bricks
    return a_bricks, b_bricks, k_bricks


def cq_gemm_g2s_issue_operands(
    *,
    warp_id,
    a_per_warp: int,
    b_per_warp: int,
    a_cta_gmem_view,
    b_cta_gmem_view,
    smex_atom,
    smem_a,
    smem_b,
    bm: int,
    bn: int,
    bk: int,
    geom: CqOperandGeom = CQ_GEMM_GEOM,
):
    """Issue this warp's SmexMtx G2S bricks for one pipeline stage (tn / k-major).

    Each warp issues ``a_per_warp`` / ``b_per_warp`` bricks with
    ``cta_lin = warp_id * per_warp + t``. Does not commit the async group -- the
    caller issues ``cp_async_commit_group``.

    Args:
        warp_id: Block-linear warp index (typically ``tid // WARP_SIZE``).
        a_per_warp: G2S A bricks this warp issues (= ``a_bricks // num_warps``).
        b_per_warp: G2S B bricks this warp issues (= ``b_bricks // num_warps``).
        a_cta_gmem_view: Current K-tile A after ``ixdl.make_sme_gmem_tensor`` +
            ``fx.zipped_divide(..., cq_smex_brick_tile())``; sliced per
            ``(cta_m, cta_k)`` brick.
        b_cta_gmem_view: Same for B, sliced per ``(cta_n, cta_k)``.
        smex_atom: Copy atom from :func:`cq_g2s_smex_atom`.
        smem_a: Shared A buffer for this pipeline stage.
        smem_b: Shared B buffer for this pipeline stage.
        bm: CTA A-tile M extent (one block M slice, not full problem M).
        bn: CTA B-tile N extent (one block N slice).
        bk: CTA K-tile extent for this K-step (not full problem K).
        geom: Operand geometry; supplies vpr and ``cta_chunk_elems``.
    """
    chunk = geom.cta_chunk_elems
    _a_bricks, _b_bricks, k_bricks = cq_cta_brick_counts(bm=bm, bn=bn, bk=bk, geom=geom)

    warp_a_start = warp_id * fx.Int32(a_per_warp)
    for t in fx.range_constexpr(a_per_warp):
        cta_lin = warp_a_start + fx.Int32(t)
        cta_m = cta_lin // fx.Int32(k_bricks)
        cta_k = cta_lin % fx.Int32(k_bricks)
        # m-outer / k-inner: the decode order already equals the shared placement,
        # so an MMA atom's K-bricks are contiguous without a remap.
        fx.copy_atom_call(
            smex_atom,
            fx.slice(a_cta_gmem_view, (None, (cta_m, cta_k))),
            cq_smex_shared_view(smem_a, cta_lin * fx.Int32(chunk), geom=geom),
        )

    warp_b_start = warp_id * fx.Int32(b_per_warp)
    for t in fx.range_constexpr(b_per_warp):
        cta_lin = warp_b_start + fx.Int32(t)
        cta_n = cta_lin // fx.Int32(k_bricks)
        cta_k = cta_lin % fx.Int32(k_bricks)
        # n-outer / k-inner so S2R EmPart pairing matches ixmma B Col indexing.
        b_linear = cta_n * fx.Int32(k_bricks) + cta_k
        fx.copy_atom_call(
            smex_atom,
            fx.slice(b_cta_gmem_view, (None, (cta_n, cta_k))),
            cq_smex_shared_view(smem_b, b_linear * fx.Int32(chunk), geom=geom),
        )


__all__ = [
    "cq_cta_brick_counts",
    "cq_g2s_smex_atom",
    "cq_gemm_g2s_issue_operands",
    "cq_smex_brick_tile",
    "cq_smex_shared_view",
]
