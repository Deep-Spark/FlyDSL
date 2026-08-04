# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""CQ SmexMtx G2S helpers for pipelined HGEMM (ivcore30).

Issues warp-collective ``llvm.bi.smex.loadn.16x1b64.mtx`` bricks into Bypass
shared memory. Pairs with ``CQMtxLoadn`` S2R — do **not** mix LegacySme
``CQAsyncCp`` on the same buffer.

tn (k-major A/B) only for this bring-up. Brick placement mirrors MR cta_lin
order so S2R EmPart addressing stays consistent with ixmma Loadn16 B16.
"""

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import arith as _arith
from flydsl.expr.primitive import ptrtoint
from flydsl.expr.typing import T
import flydsl.expr as fx
from kernels.gemm.iluvatar.common import SMEM_ROWS
from kernels.gemm.iluvatar.cq.common import CQ_GEMM_GEOM, CqOperandGeom


def _llvm_ptr(ptr, addrspace: int):
    addr = _arith.unwrap(ptrtoint(ptr))
    return _llvm.inttoptr(ir.Type.parse(f"!llvm.ptr<{int(addrspace)}>"), addr)


def _const_i32(v):
    return _arith.unwrap(_arith.constant(int(v), type=T.i32))


def _const_i64(v):
    return _arith.unwrap(_arith.constant(int(v), type=T.i64))


def cq_smex_g2s_mtx(dst_shared, src_global, *, row_stride_elems: int, elem_bytes: int = 2):
    """Issue one warp-collective SmexMtx G2S (``smex.loadn.16x1b64.mtx``)."""
    _llvm.call_intrinsic(
        None,
        "llvm.bi.smex.loadn.16x1b64.mtx",
        [
            _llvm_ptr(dst_shared, 3),
            _llvm_ptr(src_global, 1),
            _const_i32(int(row_stride_elems) * int(elem_bytes)),
            _const_i32(0),
            _const_i64(-1),
            _const_i32(-1),
            _const_i32(1),
            _const_i32(1),
        ],
        [],
        [],
    )


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
    gA_k,
    gB_k,
    smem_a,
    smem_b,
    a_leading: int,
    b_leading: int,
    bm: int,
    bn: int,
    bk: int,
    geom: CqOperandGeom = CQ_GEMM_GEOM,
    elem_bytes: int = 2,
):
    """Issue this warp's SmexMtx G2S bricks for one pipeline stage (tn / k-major).

    ``gA_k`` / ``gB_k`` are the current K-tile views with logical layouts
    ``(bm, bk):(a_leading, 1)`` and ``(bn, bk):(b_leading, 1)``. Does not commit
    the async group — caller issues ``cp_async_commit_group``.
    """
    vpr = geom.values_per_sme_row
    chunk = geom.cta_chunk_elems
    k_bricks = bk // vpr

    warp_a_start = warp_id * fx.Int32(a_per_warp)
    for t in fx.range_constexpr(a_per_warp):
        cta_lin = warp_a_start + fx.Int32(t)
        cta_m = cta_lin // fx.Int32(k_bricks)
        cta_k = cta_lin % fx.Int32(k_bricks)
        # Element offset into k-major A(m,k): row = cta_m*16, col = cta_k*vpr.
        src_off = cta_m * fx.Int32(SMEM_ROWS * a_leading) + cta_k * fx.Int32(vpr)
        src_ptr = fx.add_offset(fx.get_iter(gA_k), fx.make_int_tuple(src_off))
        dst_ptr = fx.add_offset(smem_a, fx.make_int_tuple(cta_lin * fx.Int32(chunk)))
        cq_smex_g2s_mtx(dst_ptr, src_ptr, row_stride_elems=a_leading, elem_bytes=elem_bytes)

    warp_b_start = warp_id * fx.Int32(b_per_warp)
    for t in fx.range_constexpr(b_per_warp):
        cta_lin = warp_b_start + fx.Int32(t)
        cta_n = cta_lin // fx.Int32(k_bricks)
        cta_k = cta_lin % fx.Int32(k_bricks)
        src_off = cta_n * fx.Int32(SMEM_ROWS * b_leading) + cta_k * fx.Int32(vpr)
        src_ptr = fx.add_offset(fx.get_iter(gB_k), fx.make_int_tuple(src_off))
        # n-outer / k-inner so S2R EmPart pairing matches ixmma B Col indexing.
        b_linear = cta_n * fx.Int32(k_bricks) + cta_k
        dst_ptr = fx.add_offset(smem_b, fx.make_int_tuple(b_linear * fx.Int32(chunk)))
        cq_smex_g2s_mtx(dst_ptr, src_ptr, row_stride_elems=b_leading, elem_bytes=elem_bytes)


__all__ = [
    "cq_cta_brick_counts",
    "cq_gemm_g2s_issue_operands",
    "cq_smex_g2s_mtx",
]
