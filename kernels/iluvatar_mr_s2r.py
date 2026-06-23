# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Reusable Iluvatar MR GEMM S2R (shared -> MMA register) helpers.

S2R loads smem into MMA register fragments via make_tiled_copy_A/B.
mr_hgemm_s2r_*_tile builds the smem source view; mr_hgemm_s2r_copy_* runs the copy.
mr_hgemm_s2r_load_mma_k loads all warp A/B fragments for one mma_k step.
"""

import flydsl.expr as fx

from kernels.iluvatar_mr_common import MrOperandGeom
from kernels.iluvatar_mr_operand_copy import SmeConfig, mr_cta_smem_grid, mr_sme_shared_view


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
    a_mn_major: bool,
    b_mn_major: bool,
    mma_m: int,
    mma_k: int,
    stage_base,
    g2s_sme: SmeConfig,
    smem_base,
    elem_dtype,
    warp_m_id,
    warp_atoms_m: int,
    bm: int,
    bn: int,
    bk: int,
    geom: MrOperandGeom,
):
    """Build the shared A operand tile view for one warp atom (mma_m) at mma_k.

    Returns an atom_m x atom_k smem view for make_tiled_copy_A. Uses sme_row_* to
    pick the in-row sub-slice when k-major (sme_row_k_sub) or mn-major (sme_row_m_sub).
    Shared parameters: see mr_hgemm_s2r_load_mma_k. Operand-specific: mma_m, warp_m_id,
    warp_atoms_m.
    """
    cta_grid = mr_cta_smem_grid(
        a_mn_major=a_mn_major,
        b_mn_major=b_mn_major,
        bm=bm,
        bn=bn,
        bk=bk,
        geom=geom,
    )
    sme_row_k = geom.sme_row_k_slices
    sme_row_m = geom.sme_row_a_m_atoms
    tile_atom_a = fx.make_tile(geom.atom_m, geom.atom_k)
    warp_a_base = fx.Int32(warp_m_id) * fx.Int32(warp_atoms_m * cta_grid.cta_a_k_cnt * cta_grid.cta_chunk_elems)

    if fx.const_expr(a_mn_major):
        cta_k_blk = mma_k
        cta_m_atom = fx.Int32(warp_m_id) * fx.Int32(warp_atoms_m) + fx.Int32(mma_m)
        cta_m_blk = cta_m_atom // fx.Int32(sme_row_m)
        sme_row_sub = cta_m_atom % fx.Int32(sme_row_m)
        linear = cta_m_blk * fx.Int32(cta_grid.cta_a_k_cnt) + fx.Int32(cta_k_blk)
        off = stage_base + linear * fx.Int32(cta_grid.cta_chunk_elems)
    else:
        cta_k_blk = mma_k // sme_row_k
        sme_row_sub = mma_k % sme_row_k
        off = stage_base + warp_a_base + fx.Int32((mma_m * cta_grid.cta_a_k_cnt + cta_k_blk) * cta_grid.cta_chunk_elems)

    smem_view = mr_sme_shared_view(
        smem_base,
        off,
        g2s_sme.a_sme_sw,
        elem_dtype,
        major=g2s_sme.a_smem_major,
    )
    return fx.slice(fx.zipped_divide(smem_view, tile_atom_a), (None, sme_row_sub))


def mr_hgemm_s2r_b_tile(
    *,
    a_mn_major: bool,
    b_mn_major: bool,
    mma_n: int,
    mma_k: int,
    stage_base,
    g2s_sme: SmeConfig,
    smem_base,
    elem_dtype,
    warp_n_id,
    warp_atoms_n: int,
    bm: int,
    bn: int,
    bk: int,
    geom: MrOperandGeom,
):
    """Build the shared B operand tile view for one warp atom (mma_n) at mma_k.

    B smem region starts at stage_base + bm * bk. Same sme_row_* sub-slice rules as A.
    Shared parameters: see mr_hgemm_s2r_load_mma_k. Operand-specific: mma_n, warp_n_id,
    warp_atoms_n.
    """
    cta_grid = mr_cta_smem_grid(
        a_mn_major=a_mn_major,
        b_mn_major=b_mn_major,
        bm=bm,
        bn=bn,
        bk=bk,
        geom=geom,
    )
    sme_row_k = geom.sme_row_k_slices
    sme_row_n = geom.sme_row_b_n_atoms
    tile_atom_b = fx.make_tile(geom.atom_n, geom.atom_k)
    warp_b_base = fx.Int32(warp_n_id) * fx.Int32(warp_atoms_n * cta_grid.cta_b_k_cnt * cta_grid.cta_chunk_elems)

    if fx.const_expr(b_mn_major):
        cta_k_blk = mma_k
        cta_n_atom = fx.Int32(warp_n_id) * fx.Int32(warp_atoms_n) + fx.Int32(mma_n)
        cta_n_blk = cta_n_atom // fx.Int32(sme_row_n)
        sme_row_sub = cta_n_atom % fx.Int32(sme_row_n)
        linear = fx.Int32(cta_k_blk) * fx.Int32(cta_grid.cta_b_n_cnt) + cta_n_blk
        off = stage_base + fx.Int32(bm * bk) + linear * fx.Int32(cta_grid.cta_chunk_elems)
    else:
        cta_k_blk = mma_k // sme_row_k
        sme_row_sub = mma_k % sme_row_k
        off = stage_base + warp_b_base + fx.Int32(bm * bk + (mma_n * cta_grid.cta_b_k_cnt + cta_k_blk) * cta_grid.cta_chunk_elems)

    smem_view = mr_sme_shared_view(
        smem_base,
        off,
        g2s_sme.b_sme_sw,
        elem_dtype,
        major=g2s_sme.b_smem_major,
    )
    return fx.slice(fx.zipped_divide(smem_view, tile_atom_b), (None, sme_row_sub))


def mr_hgemm_s2r_load_mma_k(
    *,
    a_mn_major: bool,
    b_mn_major: bool,
    mma_k: int,
    stage_base,
    g2s_sme: SmeConfig,
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
    geom: MrOperandGeom,
):
    """Load all warp A/B MMA operand fragments for one mma_k slice from shared memory.

    Loops mma_m in [0, warp_atoms_m) and mma_n in [0, warp_atoms_n); for each pair calls
    mr_hgemm_s2r_a_tile / _b_tile then mr_hgemm_s2r_copy_a / _copy_b. Returns two lists
    indexed by mma_m and mma_n for fx.gemm.

    Args:
        a_mn_major: True when logical A(m,k) is M-major; selects A smem offset / sme_row path.
        b_mn_major: True when logical B(n,k) is N-major; selects B smem offset / sme_row path.
        mma_k: K atom index within this CTA bk tile (0 .. bk/atom_k - 1); not cta_k or
            the outer problem-K loop index.
        stage_base: Smem element offset for this pipeline stage's A tile (Int32); B at + bm*bk.
        g2s_sme: Swizzle and smem major from mr_g2s_sme_config (must match prior G2S).
        smem_base: Dynamic shared base pointer (recast to elem_dtype).
        elem_dtype: Operand element type for mr_sme_shared_view and S2R copy atoms.
        warp_m_id: This warp's row index in the CTA warp grid (typically warp_id // warps_n).
        warp_n_id: This warp's col index in the CTA warp grid (typically warp_id % warps_n).
        warp_atoms_m: Number of atom_m tiles along M owned by each warp (MMA loop bound).
        warp_atoms_n: Number of atom_n tiles along N owned by each warp (MMA loop bound).
        copy_atom_a: S2R copy atom from fx.make_copy_atom(..., elem_dtype) + make_tiled_copy_A.
        copy_atom_b: S2R copy atom for B (make_tiled_copy_B).
        thr_copy_a: Tiled copy A slice for this lane (tiled_copy_a.get_slice(lane_id)).
        thr_copy_b: Tiled copy B slice for this lane.
        thr_mma: Tiled MMA slice for this lane (tiled_mma.thr_slice(lane_id)).
        bm: CTA A-tile M extent (one block M slice).
        bn: CTA B-tile N extent (one block N slice).
        bk: CTA K-tile extent for this K-step.
        geom: MrOperandGeom; atom_m/n/k, vpr, sme_row_* for in-row sub-slices.
    """
    a_frags = []
    for mma_m in fx.range_constexpr(warp_atoms_m):
        a_frags.append(
            mr_hgemm_s2r_copy_a(
                copy_atom=copy_atom_a,
                thr_copy_a=thr_copy_a,
                thr_mma=thr_mma,
                smem_a_tile=mr_hgemm_s2r_a_tile(
                    a_mn_major=a_mn_major,
                    b_mn_major=b_mn_major,
                    mma_m=mma_m,
                    mma_k=mma_k,
                    stage_base=stage_base,
                    g2s_sme=g2s_sme,
                    smem_base=smem_base,
                    elem_dtype=elem_dtype,
                    warp_m_id=warp_m_id,
                    warp_atoms_m=warp_atoms_m,
                    bm=bm,
                    bn=bn,
                    bk=bk,
                    geom=geom,
                ),
            )
        )
    b_frags = []
    for mma_n in fx.range_constexpr(warp_atoms_n):
        b_frags.append(
            mr_hgemm_s2r_copy_b(
                copy_atom=copy_atom_b,
                thr_copy_b=thr_copy_b,
                thr_mma=thr_mma,
                smem_b_tile=mr_hgemm_s2r_b_tile(
                    a_mn_major=a_mn_major,
                    b_mn_major=b_mn_major,
                    mma_n=mma_n,
                    mma_k=mma_k,
                    stage_base=stage_base,
                    g2s_sme=g2s_sme,
                    smem_base=smem_base,
                    elem_dtype=elem_dtype,
                    warp_n_id=warp_n_id,
                    warp_atoms_n=warp_atoms_n,
                    bm=bm,
                    bn=bn,
                    bk=bk,
                    geom=geom,
                ),
            )
        )
    return a_frags, b_frags
