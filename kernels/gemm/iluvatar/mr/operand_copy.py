# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Reusable Iluvatar MR A/B operand copy helpers (G2S via SME async copy).

G2S maps global memory to swem using SME copy atoms (not TiledCopy). cta_lin is the
linear chunk index; mr_cta_smem_grid supplies the cta_lin // and % divisors.

Exports:
  mr_g2s_sme_config, mr_gemm_g2s_issue_a_warp, mr_gemm_g2s_issue_b_warp,
  mr_gemm_g2s_issue_operands, mr_cta_smem_grid, mr_sme_shared_view,
  mr_sme_shared_view_k_spanning

S2R helpers are in kernels.gemm.iluvatar.mr.s2r.
"""

from typing import NamedTuple

import flydsl.expr as fx
import flydsl.expr.ixdl as ixdl

from kernels.gemm.iluvatar.common import GemmLayout
from kernels.gemm.iluvatar.mr.common import SMEM_ROWS, MrCtaSmemGrid, MrOperandGeom, sme_atom_counts


class SmeConfig(NamedTuple):
    """Per-operand SME G2S copy atoms and shared-layout metadata.

    Built by mr_g2s_sme_config. Row vs col SME path follows GemmLayout major mode.
    """
    sme_atom_a: object
    sme_atom_b: object
    a_sme_sw: object
    b_sme_sw: object
    a_smem_major: object
    b_smem_major: object


def mr_g2s_sme_config(
    *,
    a_mn_major: bool,
    b_mn_major: bool,
    elem_dtype,
    row_atom,
    row_swizzle,
    col_atom=ixdl.MRAsyncCpCol,
) -> SmeConfig:
    """Select row/col SME G2S atoms and shared swizzle metadata.

    SME row/col copy paths match each operand's global major mode:

    * A k-major: row-SME, ``SMEMajor.K``.
    * A mn-major (M): col-SME, ``SMEMajor.MN``.
    * B mn-major (N): row-SME, ``SMEMajor.MN``.
    * B k-major: col-SME, ``SMEMajor.K``.

    Args:
        a_mn_major: True when logical ``A(m,k)`` is M-major.
        b_mn_major: True when logical ``B(n,k)`` is N-major.
        elem_dtype: Element type for ``fx.make_copy_atom``.
        row_atom: Callable returning the row-SME copy op type.
        row_swizzle: Row-SME swizzle enum (A k-major, B mn-major).
        col_atom: Callable returning the col-SME copy op type.

    Returns:
        Copy atoms and per-operand shared swizzle/major metadata for G2S issue.
    """
    if fx.const_expr(a_mn_major):
        sme_atom_a = fx.make_copy_atom(col_atom(), elem_dtype)
        a_sme_sw = ixdl.SMESwizzle.Col
        a_smem_major = ixdl.SMEMajor.MN
    else:
        sme_atom_a = fx.make_copy_atom(row_atom(), elem_dtype)
        a_sme_sw = row_swizzle
        a_smem_major = ixdl.SMEMajor.K

    if fx.const_expr(b_mn_major):
        sme_atom_b = fx.make_copy_atom(row_atom(), elem_dtype)
        b_sme_sw = row_swizzle
        b_smem_major = ixdl.SMEMajor.MN
    else:
        sme_atom_b = fx.make_copy_atom(col_atom(), elem_dtype)
        b_sme_sw = ixdl.SMESwizzle.Col
        b_smem_major = ixdl.SMEMajor.K

    return SmeConfig(
        sme_atom_a=sme_atom_a,
        sme_atom_b=sme_atom_b,
        a_sme_sw=a_sme_sw,
        b_sme_sw=b_sme_sw,
        a_smem_major=a_smem_major,
        b_smem_major=b_smem_major,
    )


def mr_sme_shared_view(smem_base, elem_offset, swizzle, elem_dtype, *, major):
    """Build one SME chunk destination view in dynamic shared memory.

    Offsets ``smem_base`` by ``elem_offset`` element slots, then wraps the pointer
    with an SME swizzled layout from ``ixdl.make_sme_shared_layout``. Used as the
    destination operand of ``fx.copy_atom_call`` in the G2S helpers.

    Args:
        smem_base: Dynamic shared-memory base pointer (``fx.get_dyn_shared()``).
        elem_offset: Offset from ``smem_base`` in **elements** of ``smem_base``'s
            pointee type (e.g. f16 slots when ``smem_base`` is an f16 shared pointer).
        swizzle: ``ixdl.SMESwizzle.*`` matching the copy atom (from ``SmeConfig``).
        elem_dtype: Element type of the shared tile.
        major: ``ixdl.SMEMajor.K`` or ``.MN`` (from ``SmeConfig``).

    Returns:
        A layout view suitable as the destination of an SME async copy.
    """
    smem_ptr = fx.add_offset(smem_base, fx.make_int_tuple(fx.Int32(elem_offset)))
    layout = ixdl.make_sme_shared_layout(swizzle, elem_dtype, major=major)
    return fx.make_view(smem_ptr, layout)


def mr_sme_shared_view_k_spanning(smem_base, elem_offset, swizzle, elem_dtype, *, major, mn_extent, k_total):
    """SME shared view whose K mode spans several contiguous SME bricks.

    For 8-bit MN-major operands one brick only holds K = ``SMEM_ROWS`` (16), but an
    i8 MMA atom needs K = 32 = 2 bricks. ``ixdl.make_sme_shared_layout_k_spanning``
    appends the brick selector as a clean outer K sub-mode so the MMA TV K-decode
    stays ``(within_brick_K, brick)``. The spanned K-bricks must be contiguous in
    shared starting at ``elem_offset``. When ``k_total`` equals one brick's K this
    reduces to :func:`mr_sme_shared_view`.
    """
    smem_ptr = fx.add_offset(smem_base, fx.make_int_tuple(fx.Int32(elem_offset)))
    layout = ixdl.make_sme_shared_layout_k_spanning(
        swizzle, elem_dtype, major=major, mn_extent=mn_extent, k_total=k_total
    )
    return fx.make_view(smem_ptr, layout)


def mr_cta_smem_grid(
    *,
    a_mn_major: bool,
    b_mn_major: bool,
    bm: int,
    bn: int,
    bk: int,
    geom: MrOperandGeom,
) -> MrCtaSmemGrid:
    """Build MrCtaSmemGrid for G2S cta_lin decode on one CTA K-step tile.

    Args:
        a_mn_major: True when logical A(m,k) is M-major (see GemmLayout).
        b_mn_major: True when logical B(n,k) is N-major.
        bm: CTA A-tile M extent (one block's M slice, not full problem M).
        bn: CTA B-tile N extent (one block's N slice).
        bk: CTA K-tile extent (one outer K-loop step; atoms_total scales with bm/bn x bk).
        geom: Operand geometry; supplies vpr (values_per_sme_row) and cta_chunk_elems.
    """
    layout = GemmLayout(a_mn_major=a_mn_major, b_mn_major=b_mn_major)
    _, _, cta_a_k_cnt, cta_b_k_cnt = sme_atom_counts(
        layout,
        bm,
        bn,
        bk,
        values_per_sme_row=geom.values_per_sme_row,
    )
    return MrCtaSmemGrid(
        cta_a_k_cnt=cta_a_k_cnt,
        cta_b_k_cnt=cta_b_k_cnt,
        cta_a_k_cnt_k_major=bk // geom.values_per_sme_row,
        cta_b_n_cnt=bn // geom.values_per_sme_row,
        cta_chunk_elems=geom.cta_chunk_elems,
    )


def mr_gemm_g2s_issue_a_warp(
    *,
    a_mn_major: bool,
    b_mn_major: bool,
    warp_id,
    a_per_warp: int,
    a_cta_gmem_view,
    g2s_sme: SmeConfig,
    smem_a,
    elem_dtype,
    bm: int,
    bn: int,
    bk: int,
    geom: MrOperandGeom,
):
    """Issue this warp's A-tile SME async G2S copies for one pipeline stage.

    A k-major: cta_m = cta_lin // cta_a_k_cnt_k_major, cta_k = cta_lin % cta_a_k_cnt_k_major.
    A mn-major: cta_m = cta_lin // cta_a_k_cnt, cta_k = cta_lin % cta_a_k_cnt.
    Smem: cta_lin * cta_chunk_elems within ``smem_a``. Does not commit async.
    Parameters: see mr_gemm_g2s_issue_operands.
    """
    cta_grid = mr_cta_smem_grid(
        a_mn_major=a_mn_major,
        b_mn_major=b_mn_major,
        bm=bm,
        bn=bn,
        bk=bk,
        geom=geom,
    )
    warp_a_start = warp_id * fx.Int32(a_per_warp)
    for t in fx.range_constexpr(a_per_warp):
        cta_lin = warp_a_start + fx.Int32(t)
        if fx.const_expr(a_mn_major):
            cta_m = cta_lin // fx.Int32(cta_grid.cta_a_k_cnt)
            cta_k = cta_lin % fx.Int32(cta_grid.cta_a_k_cnt)
        else:
            cta_m = cta_lin // fx.Int32(cta_grid.cta_a_k_cnt_k_major)
            cta_k = cta_lin % fx.Int32(cta_grid.cta_a_k_cnt_k_major)
        a_src = fx.slice(a_cta_gmem_view, (None, (cta_m, cta_k)))
        # m-outer / k-inner: A's decode order already equals the smem placement, so
        # an MMA atom's K-bricks are contiguous without a remap (cta_lin == cta_m *
        # cta_a_k_cnt + cta_k). The mirror decode lives in mr_gemm_s2r_a_tile.
        a_off = cta_lin * fx.Int32(cta_grid.cta_chunk_elems)
        fx.copy_atom_call(
            g2s_sme.sme_atom_a,
            a_src,
            mr_sme_shared_view(
                smem_a,
                a_off,
                g2s_sme.a_sme_sw,
                elem_dtype,
                major=g2s_sme.a_smem_major,
            ),
        )


def mr_gemm_g2s_issue_b_warp(
    *,
    a_mn_major: bool,
    b_mn_major: bool,
    warp_id,
    b_per_warp: int,
    b_cta_gmem_view,
    g2s_sme: SmeConfig,
    smem_b,
    elem_dtype,
    bm: int,
    bn: int,
    bk: int,
    geom: MrOperandGeom,
):
    """Issue this warp's B-tile SME async G2S copies for one pipeline stage.

    B smem uses ``smem_b`` directly. B mn-major: cta_n = cta_lin % cta_b_n_cnt,
    cta_k = cta_lin // cta_b_n_cnt. B k-major: cta_n = cta_lin // cta_b_k_cnt,
    cta_k = cta_lin % cta_b_k_cnt. Does not commit async.
    Parameters: see mr_gemm_g2s_issue_operands.
    """
    cta_grid = mr_cta_smem_grid(
        a_mn_major=a_mn_major,
        b_mn_major=b_mn_major,
        bm=bm,
        bn=bn,
        bk=bk,
        geom=geom,
    )
    warp_b_start = warp_id * fx.Int32(b_per_warp)
    for t in fx.range_constexpr(b_per_warp):
        cta_lin = warp_b_start + fx.Int32(t)
        if fx.const_expr(b_mn_major):
            cta_n = cta_lin % fx.Int32(cta_grid.cta_b_n_cnt)
            cta_k = cta_lin // fx.Int32(cta_grid.cta_b_n_cnt)
        else:
            cta_n = cta_lin // fx.Int32(cta_grid.cta_b_k_cnt)
            cta_k = cta_lin % fx.Int32(cta_grid.cta_b_k_cnt)
        b_src = fx.slice(b_cta_gmem_view, (None, (cta_n, cta_k)))
        # n-outer / k-inner so an MMA atom's K-bricks land contiguous in shared
        # (required by the i8 k-spanning S2R view; f16 uses the same order). For
        # k-major this equals cta_lin; the mirror decode lives in mr_gemm_s2r_b_tile.
        b_linear = cta_n * fx.Int32(cta_grid.cta_b_k_cnt) + cta_k
        b_off = b_linear * fx.Int32(cta_grid.cta_chunk_elems)
        fx.copy_atom_call(
            g2s_sme.sme_atom_b,
            b_src,
            mr_sme_shared_view(
                smem_b,
                b_off,
                g2s_sme.b_sme_sw,
                elem_dtype,
                major=g2s_sme.b_smem_major,
            ),
        )


def mr_gemm_g2s_issue_operands(
    *,
    a_mn_major: bool,
    b_mn_major: bool,
    warp_id,
    a_per_warp: int,
    b_per_warp: int,
    a_cta_gmem_view,
    b_cta_gmem_view,
    g2s_sme: SmeConfig,
    smem_a,
    smem_b,
    elem_dtype,
    bm: int,
    bn: int,
    bk: int,
    geom: MrOperandGeom,
    commit: bool = True,
):
    """Issue this warp's A and B SME async G2S copies for one pipeline stage.

    Calls mr_gemm_g2s_issue_a_warp then mr_gemm_g2s_issue_b_warp. Each warp issues
    a_per_warp / b_per_warp chunks with cta_lin = warp_id * per_warp + t. When commit
    is True (default), calls ixdl.cp_async_commit_group after both operands.

    Args:
        a_mn_major: True when logical A(m,k) is M-major; selects A cta_lin decode and
            row vs col SME copy path (see mr_g2s_sme_config).
        b_mn_major: True when logical B(n,k) is N-major; selects B decode and SME path.
        warp_id: Block-linear warp index (typically tid // WARP_SIZE).
        a_per_warp: G2S A chunks this warp issues (= a_atoms_total // num_warps).
        b_per_warp: G2S B chunks this warp issues (= b_atoms_total // num_warps).
        a_cta_gmem_view: GMEM A after ixdl.make_sme_gmem_tensor + fx.zipped_divide(..., tile_smem_A);
            issue_a_warp slices (cta_m, cta_k) per chunk.
        b_cta_gmem_view: GMEM B after ixdl.make_sme_gmem_tensor + fx.zipped_divide(..., tile_smem_B);
            issue_b_warp slices (cta_n, cta_k) per chunk.
        g2s_sme: Copy atoms, swizzle, and smem major from mr_g2s_sme_config.
        smem_a: Shared A buffer for this pipeline stage (f16 shared pointer).
        smem_b: Shared B buffer for this pipeline stage (f16 shared pointer).
        elem_dtype: Operand element type for copy_atom_call and smem views.
        bm: CTA A-tile M extent (one block M slice, not full problem M).
        bn: CTA B-tile N extent (one block N slice).
        bk: CTA K-tile extent for this K-step (not full problem K).
        geom: MrOperandGeom; supplies vpr and cta_chunk_elems for chunk grid.
        commit: If True, commit the async copy group after A and B issues.
    """
    mr_gemm_g2s_issue_a_warp(
        a_mn_major=a_mn_major,
        b_mn_major=b_mn_major,
        warp_id=warp_id,
        a_per_warp=a_per_warp,
        a_cta_gmem_view=a_cta_gmem_view,
        g2s_sme=g2s_sme,
        smem_a=smem_a,
        elem_dtype=elem_dtype,
        bm=bm,
        bn=bn,
        bk=bk,
        geom=geom,
    )
    mr_gemm_g2s_issue_b_warp(
        a_mn_major=a_mn_major,
        b_mn_major=b_mn_major,
        warp_id=warp_id,
        b_per_warp=b_per_warp,
        b_cta_gmem_view=b_cta_gmem_view,
        g2s_sme=g2s_sme,
        smem_b=smem_b,
        elem_dtype=elem_dtype,
        bm=bm,
        bn=bn,
        bk=bk,
        geom=geom,
    )
    if fx.const_expr(commit):
        ixdl.cp_async_commit_group()
