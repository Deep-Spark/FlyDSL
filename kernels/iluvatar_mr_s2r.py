# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Reusable Iluvatar MR HGEMM S2R (shared -> MMA register) helpers.

* ``mr_hgemm_s2r_copy_a`` / ``mr_hgemm_s2r_copy_b`` — single-tile ``make_tiled_copy_A/B``
* ``mr_hgemm_s2r_a_tile`` / ``mr_hgemm_s2r_b_tile`` — pattern-aware SME operand tile views
* ``mr_hgemm_s2r_load_ki`` — one Ki slice: all warp A/B fragments for MMA
"""

import flydsl.expr as fx

from kernels.iluvatar_mr_common import ATOM_K, ATOM_M, ATOM_N, SMEM_F16_PER_ROW
from kernels.iluvatar_mr_operand_copy import MrG2sSmeConfig, mr_g2s_brick_layout, mr_sme_shared_view


def mr_hgemm_s2r_copy_a(*, copy_atom, thr_copy_a, thr_mma, smem_a_tile):
    """S2R: shared A tile -> MMA A register fragment via ``make_tiled_copy_A``."""
    frag_a = thr_mma.make_fragment_A(smem_a_tile)
    fx.copy(
        copy_atom,
        thr_copy_a.partition_S(smem_a_tile),
        thr_copy_a.retile(frag_a),
        pred=None,
    )
    return frag_a


def mr_hgemm_s2r_copy_b(*, copy_atom, thr_copy_b, thr_mma, smem_b_tile):
    """S2R: shared B tile -> MMA B register fragment via ``make_tiled_copy_B``."""
    frag_b = thr_mma.make_fragment_B(smem_b_tile)
    fx.copy(
        copy_atom,
        thr_copy_b.partition_S(smem_b_tile),
        thr_copy_b.retile(frag_b),
        pred=None,
    )
    return frag_b


def mr_hgemm_s2r_a_tile(
    *,
    pattern_id: int,
    im: int,
    ki: int,
    stage_base,
    g2s_sme: MrG2sSmeConfig,
    smem_base,
    elem_dtype,
    warp_m_id,
    warp_atoms_m: int,
    bm: int,
    bn: int,
    bk: int,
    atom_m: int = ATOM_M,
    atom_k: int = ATOM_K,
    values_per_sme_row: int = SMEM_F16_PER_ROW,
):
    """Build the shared A operand tile view for one warp atom at Ki ``ki``."""
    a_smem_k_bricks, _, _, _, brick_elems = mr_g2s_brick_layout(
        pattern_id,
        bm,
        bn,
        bk,
        values_per_sme_row=values_per_sme_row,
    )
    tile_atom_a = fx.make_tile(atom_m, atom_k)
    warp_a_base = fx.Int32(warp_m_id) * fx.Int32(warp_atoms_m * a_smem_k_bricks * brick_elems)

    if fx.const_expr(pattern_id == 0 or pattern_id == 2):
        ki_brick = ki // 2
        ki_in_tile = ki % 2
    else:
        ki_brick = ki
        ki_in_tile = 0

    if fx.const_expr(pattern_id == 1 or pattern_id == 3):
        w_mi = fx.Int32(warp_m_id) * fx.Int32(warp_atoms_m) + fx.Int32(im)
        g2s_mi = w_mi // fx.Int32(2)
        m_half = w_mi % fx.Int32(2)
        linear = g2s_mi * fx.Int32(a_smem_k_bricks) + fx.Int32(ki_brick)
        off = stage_base + linear * fx.Int32(brick_elems)
        smem_view = mr_sme_shared_view(
            smem_base,
            off,
            g2s_sme.a_sme_sw,
            elem_dtype,
            major=g2s_sme.a_smem_major,
        )
        return fx.slice(fx.zipped_divide(smem_view, tile_atom_a), (None, m_half))

    off = stage_base + warp_a_base + fx.Int32((im * a_smem_k_bricks + ki_brick) * brick_elems)
    smem_view = mr_sme_shared_view(
        smem_base,
        off,
        g2s_sme.a_sme_sw,
        elem_dtype,
        major=g2s_sme.a_smem_major,
    )
    return fx.slice(fx.zipped_divide(smem_view, tile_atom_a), (None, ki_in_tile))


def mr_hgemm_s2r_b_tile(
    *,
    pattern_id: int,
    jn: int,
    ki: int,
    stage_base,
    g2s_sme: MrG2sSmeConfig,
    smem_base,
    elem_dtype,
    warp_n_id,
    warp_atoms_n: int,
    bm: int,
    bn: int,
    bk: int,
    atom_n: int = ATOM_N,
    atom_k: int = ATOM_K,
    values_per_sme_row: int = SMEM_F16_PER_ROW,
):
    """Build the shared B operand tile view for one warp atom at Ki ``ki``."""
    _, b_smem_k_bricks, _, b_n_chunks, brick_elems = mr_g2s_brick_layout(
        pattern_id,
        bm,
        bn,
        bk,
        values_per_sme_row=values_per_sme_row,
    )
    tile_atom_b = fx.make_tile(atom_n, atom_k)
    warp_b_base = fx.Int32(warp_n_id) * fx.Int32(warp_atoms_n * b_smem_k_bricks * brick_elems)

    if fx.const_expr(pattern_id == 0 or pattern_id == 1):
        ki_brick = ki
        ki_in_tile = 0
    else:
        ki_brick = ki // 2
        ki_in_tile = ki % 2

    if fx.const_expr(pattern_id == 0 or pattern_id == 1):
        w_ni = fx.Int32(warp_n_id) * fx.Int32(warp_atoms_n) + fx.Int32(jn)
        g2s_ni = w_ni // fx.Int32(2)
        n_half = w_ni % fx.Int32(2)
        linear = fx.Int32(ki_brick) * fx.Int32(b_n_chunks) + g2s_ni
        off = stage_base + fx.Int32(bm * bk) + linear * fx.Int32(brick_elems)
        smem_view = mr_sme_shared_view(
            smem_base,
            off,
            g2s_sme.b_sme_sw,
            elem_dtype,
            major=g2s_sme.b_smem_major,
        )
        return fx.slice(fx.zipped_divide(smem_view, tile_atom_b), (None, n_half))

    off = stage_base + warp_b_base + fx.Int32(bm * bk + (jn * b_smem_k_bricks + ki_brick) * brick_elems)
    smem_view = mr_sme_shared_view(
        smem_base,
        off,
        g2s_sme.b_sme_sw,
        elem_dtype,
        major=g2s_sme.b_smem_major,
    )
    return fx.slice(fx.zipped_divide(smem_view, tile_atom_b), (None, ki_in_tile))


def mr_hgemm_s2r_load_ki(
    *,
    pattern_id: int,
    ki: int,
    stage_base,
    g2s_sme: MrG2sSmeConfig,
    smem_base,
    elem_dtype,
    warp_m_id,
    warp_n_id,
    warp_atoms_m: int,
    warp_atoms_n: int,
    copy_atom_a,
    copy_atom_b,
    thr_copy_a,
    thr_copy_b,
    thr_mma,
    bm: int,
    bn: int,
    bk: int,
    values_per_sme_row: int = SMEM_F16_PER_ROW,
):
    """Load all warp A/B MMA operand fragments for one Ki slice from shared memory."""
    a_frags = []
    for im in fx.range_constexpr(warp_atoms_m):
        a_frags.append(
            mr_hgemm_s2r_copy_a(
                copy_atom=copy_atom_a,
                thr_copy_a=thr_copy_a,
                thr_mma=thr_mma,
                smem_a_tile=mr_hgemm_s2r_a_tile(
                    pattern_id=pattern_id,
                    im=im,
                    ki=ki,
                    stage_base=stage_base,
                    g2s_sme=g2s_sme,
                    smem_base=smem_base,
                    elem_dtype=elem_dtype,
                    warp_m_id=warp_m_id,
                    warp_atoms_m=warp_atoms_m,
                    bm=bm,
                    bn=bn,
                    bk=bk,
                    values_per_sme_row=values_per_sme_row,
                ),
            )
        )
    b_frags = []
    for jn in fx.range_constexpr(warp_atoms_n):
        b_frags.append(
            mr_hgemm_s2r_copy_b(
                copy_atom=copy_atom_b,
                thr_copy_b=thr_copy_b,
                thr_mma=thr_mma,
                smem_b_tile=mr_hgemm_s2r_b_tile(
                    pattern_id=pattern_id,
                    jn=jn,
                    ki=ki,
                    stage_base=stage_base,
                    g2s_sme=g2s_sme,
                    smem_base=smem_base,
                    elem_dtype=elem_dtype,
                    warp_n_id=warp_n_id,
                    warp_atoms_n=warp_atoms_n,
                    bm=bm,
                    bn=bn,
                    bk=bk,
                    values_per_sme_row=values_per_sme_row,
                ),
            )
        )
    return a_frags, b_frags
