# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Reusable Iluvatar MR A/B operand copy helpers.

G2S (SME async copy):

* ``mr_pattern_g2s_sme_config`` — pattern/dtype SME atom + swizzle + major
* ``mr_hgemm_g2s_issue_a_warp`` / ``mr_hgemm_g2s_issue_b_warp`` — per-warp atom issue
* ``mr_hgemm_g2s_issue_operands`` — A + B issue with optional async commit

S2R helpers live in ``kernels.iluvatar_mr_s2r``.
"""

from typing import NamedTuple

import flydsl.expr as fx
import flydsl.expr.ixdl as ixdl

from kernels.iluvatar_mr_common import SMEM_F16_PER_ROW, SMEM_ROWS, pattern_sme_atom_counts


class MrG2sSmeConfig(NamedTuple):
    sme_atom_a: object
    sme_atom_b: object
    a_sme_sw: object
    b_sme_sw: object
    a_smem_major: object
    b_smem_major: object


def mr_g2s_brick_layout(
    pattern_id: int,
    bm: int,
    bn: int,
    bk: int,
    *,
    values_per_sme_row: int = SMEM_F16_PER_ROW,
) -> tuple[int, int, int, int, int]:
    """Return (a_smem_k_bricks, b_smem_k_bricks, cta_atoms_k, b_n_chunks, brick_elems)."""
    _, _, a_smem_k_bricks, b_smem_k_bricks = pattern_sme_atom_counts(
        pattern_id,
        bm,
        bn,
        bk,
        values_per_sme_row=values_per_sme_row,
    )
    cta_atoms_k = bk // values_per_sme_row
    b_n_chunks = bn // values_per_sme_row
    brick_elems = SMEM_ROWS * values_per_sme_row
    return a_smem_k_bricks, b_smem_k_bricks, cta_atoms_k, b_n_chunks, brick_elems


def mr_pattern_g2s_sme_config(
    pattern_id: int,
    elem_dtype,
    *,
    row_atom,
    row_swizzle,
    col_atom=ixdl.MRAsyncCpCol,
) -> MrG2sSmeConfig:
    """Build SME G2S copy atoms and shared-layout metadata for ``pattern_id``."""
    if fx.const_expr(pattern_id == 1 or pattern_id == 3):
        sme_atom_a = fx.make_copy_atom(col_atom(), elem_dtype)
        a_sme_sw = ixdl.SMESwizzle.Col
        a_smem_major = ixdl.SMEMajor.MN
    else:
        sme_atom_a = fx.make_copy_atom(row_atom(), elem_dtype)
        a_sme_sw = row_swizzle
        a_smem_major = ixdl.SMEMajor.K

    if fx.const_expr(pattern_id == 0 or pattern_id == 1):
        sme_atom_b = fx.make_copy_atom(row_atom(), elem_dtype)
        b_sme_sw = row_swizzle
        b_smem_major = ixdl.SMEMajor.MN
    else:
        sme_atom_b = fx.make_copy_atom(col_atom(), elem_dtype)
        b_sme_sw = ixdl.SMESwizzle.Col
        b_smem_major = ixdl.SMEMajor.K

    return MrG2sSmeConfig(
        sme_atom_a=sme_atom_a,
        sme_atom_b=sme_atom_b,
        a_sme_sw=a_sme_sw,
        b_sme_sw=b_sme_sw,
        a_smem_major=a_smem_major,
        b_smem_major=b_smem_major,
    )


def mr_sme_shared_view(smem_base, elem_offset, swizzle, elem_dtype, *, major):
    """Build an SME shared-memory view at ``elem_offset`` elements from ``smem_base``."""
    smem_ptr = fx.add_offset(smem_base, fx.make_int_tuple(fx.Int32(elem_offset)))
    layout = ixdl.make_sme_shared_layout(swizzle, elem_dtype, major=major)
    return fx.make_view(smem_ptr, layout)


def mr_hgemm_g2s_issue_a_warp(
    *,
    pattern_id: int,
    warp_id,
    a_per_warp: int,
    g_A_div,
    g2s_sme: MrG2sSmeConfig,
    smem_base,
    elem_dtype,
    bm: int,
    bn: int,
    bk: int,
    stage_base,
    values_per_sme_row: int = SMEM_F16_PER_ROW,
):
    """Issue this warp's A SME async-copy atoms into shared memory."""
    a_smem_k_bricks, _, cta_atoms_k, _, brick_elems = mr_g2s_brick_layout(
        pattern_id,
        bm,
        bn,
        bk,
        values_per_sme_row=values_per_sme_row,
    )
    warp_a_start = warp_id * fx.Int32(a_per_warp)
    for t in fx.range_constexpr(a_per_warp):
        atom_idx = warp_a_start + fx.Int32(t)
        if fx.const_expr(pattern_id == 0 or pattern_id == 2):
            mi = atom_idx // fx.Int32(cta_atoms_k)
            ki = atom_idx % fx.Int32(cta_atoms_k)
        else:
            mi = atom_idx // fx.Int32(a_smem_k_bricks)
            ki = atom_idx % fx.Int32(a_smem_k_bricks)
        a_src = fx.slice(g_A_div, (None, (mi, ki)))
        a_off = stage_base + atom_idx * fx.Int32(brick_elems)
        fx.copy_atom_call(
            g2s_sme.sme_atom_a,
            a_src,
            mr_sme_shared_view(
                smem_base,
                a_off,
                g2s_sme.a_sme_sw,
                elem_dtype,
                major=g2s_sme.a_smem_major,
            ),
        )


def mr_hgemm_g2s_issue_b_warp(
    *,
    pattern_id: int,
    warp_id,
    b_per_warp: int,
    g_B_div,
    g2s_sme: MrG2sSmeConfig,
    smem_base,
    elem_dtype,
    bm: int,
    bn: int,
    bk: int,
    stage_base,
    values_per_sme_row: int = SMEM_F16_PER_ROW,
):
    """Issue this warp's B SME async-copy atoms into shared memory."""
    _, b_smem_k_bricks, _, b_n_chunks, brick_elems = mr_g2s_brick_layout(
        pattern_id,
        bm,
        bn,
        bk,
        values_per_sme_row=values_per_sme_row,
    )
    b_storage_base = bm * bk
    warp_b_start = warp_id * fx.Int32(b_per_warp)
    for t in fx.range_constexpr(b_per_warp):
        atom_idx = warp_b_start + fx.Int32(t)
        if fx.const_expr(pattern_id == 0 or pattern_id == 1):
            ni = atom_idx % fx.Int32(b_n_chunks)
            ki = atom_idx // fx.Int32(b_n_chunks)
            b_src = fx.slice(g_B_div, (None, (ni, ki)))
            b_linear = ki * fx.Int32(b_n_chunks) + ni
            b_off = stage_base + fx.Int32(b_storage_base) + b_linear * fx.Int32(brick_elems)
        else:
            ni = atom_idx // fx.Int32(b_smem_k_bricks)
            ki = atom_idx % fx.Int32(b_smem_k_bricks)
            b_src = fx.slice(g_B_div, (None, (ni, ki)))
            b_off = stage_base + fx.Int32(b_storage_base) + atom_idx * fx.Int32(brick_elems)
        fx.copy_atom_call(
            g2s_sme.sme_atom_b,
            b_src,
            mr_sme_shared_view(
                smem_base,
                b_off,
                g2s_sme.b_sme_sw,
                elem_dtype,
                major=g2s_sme.b_smem_major,
            ),
        )


def mr_hgemm_g2s_issue_operands(
    *,
    pattern_id: int,
    warp_id,
    a_per_warp: int,
    b_per_warp: int,
    g_A_div,
    g_B_div,
    g2s_sme: MrG2sSmeConfig,
    smem_base,
    elem_dtype,
    bm: int,
    bn: int,
    bk: int,
    stage_base,
    values_per_sme_row: int = SMEM_F16_PER_ROW,
    commit: bool = True,
):
    """Issue this warp's A and B SME async-copy atoms; optionally commit the group."""
    mr_hgemm_g2s_issue_a_warp(
        pattern_id=pattern_id,
        warp_id=warp_id,
        a_per_warp=a_per_warp,
        g_A_div=g_A_div,
        g2s_sme=g2s_sme,
        smem_base=smem_base,
        elem_dtype=elem_dtype,
        bm=bm,
        bn=bn,
        bk=bk,
        stage_base=stage_base,
        values_per_sme_row=values_per_sme_row,
    )
    mr_hgemm_g2s_issue_b_warp(
        pattern_id=pattern_id,
        warp_id=warp_id,
        b_per_warp=b_per_warp,
        g_B_div=g_B_div,
        g2s_sme=g2s_sme,
        smem_base=smem_base,
        elem_dtype=elem_dtype,
        bm=bm,
        bn=bn,
        bk=bk,
        stage_base=stage_base,
        values_per_sme_row=values_per_sme_row,
    )
    if fx.const_expr(commit):
        ixdl.cp_async_commit_group()
