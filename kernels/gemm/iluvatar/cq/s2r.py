# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""CQ loadn S2R helpers for pipelined HGEMM (ivcore30).

Loads SmexMtx shared bricks into CQMma A/B fragments via ``CQMtxLoadn`` +
``make_tiled_copy_A/B``. Shared pointers encode ixmma EmPart in the low bits
(``bp + em_part_addend``) so one loadn x2 fills one base-tile K-slice.

With even ``k_atoms``, EmPart is ``(mma_k % 2) * 2`` (Python constexpr) and the
brick base is ``(global_mn * (k_atoms // 2) + mma_k // 2) * cta_chunk_elems``.
"""

import flydsl.expr as fx
from kernels.gemm.iluvatar.cq.common import CQ_GEMM_GEOM, CqOperandGeom


def _smex_loadn_view(smem_base, *, tile_base_elems, em_part_addend: int, shape_mn_k, elem_dtype):
    """Shared view at SmexMtx brick ``tile_base`` with EmPart in pointer low bits."""
    brick_ptr = fx.add_offset(smem_base, fx.make_int_tuple(tile_base_elems))
    if int(em_part_addend) == 0:
        elem_ptr = brick_ptr
    else:
        # EmPart is a byte addend on the shared base (ixmma Loadn16 B16 contract).
        byte_ptr = fx.recast_iter(fx.Int8, brick_ptr)
        byte_ptr_em = fx.add_offset(byte_ptr, fx.make_int_tuple(int(em_part_addend)))
        elem_ptr = fx.recast_iter(elem_dtype, byte_ptr_em)
    mn, k = shape_mn_k
    return fx.make_view(elem_ptr, fx.make_layout((mn, k), (1, mn)))


def cq_gemm_s2r_copy_a(*, copy_atom, thr_copy_a, thr_mma, smem_a_tile):
    """S2R: shared A tile -> MMA A fragment via ``make_tiled_copy_A``."""
    frag_a = thr_mma.make_fragment_A(smem_a_tile)
    fx.copy(
        copy_atom,
        thr_copy_a.partition_S(smem_a_tile),
        thr_copy_a.retile(frag_a),
        pred=None,
    )
    return frag_a


def cq_gemm_s2r_copy_b(*, copy_atom, thr_copy_b, thr_mma, smem_b_tile):
    """S2R: shared B tile -> MMA B fragment via ``make_tiled_copy_B``."""
    frag_b = thr_mma.make_fragment_B(smem_b_tile)
    fx.copy(
        copy_atom,
        thr_copy_b.partition_S(smem_b_tile),
        thr_copy_b.retile(frag_b),
        pred=None,
    )
    return frag_b


def cq_gemm_s2r_a_tile(
    *,
    smem_a,
    elem_dtype,
    mma_m: int,
    mma_k: int,
    warp_m_id,
    warp_atoms_m: int,
    k_atoms: int,
    geom: CqOperandGeom = CQ_GEMM_GEOM,
):
    """Build EmPart-aware shared A view for one warp MMA atom (tn / k-major)."""
    global_mn = fx.Int32(warp_m_id) * fx.Int32(warp_atoms_m) + fx.Int32(mma_m)
    tile_base_elems = (global_mn * fx.Int32(k_atoms // 2) + fx.Int32(mma_k // 2)) * fx.Int32(geom.cta_chunk_elems)
    em_part = (int(mma_k) % 2) * 2
    return _smex_loadn_view(
        smem_a,
        tile_base_elems=tile_base_elems,
        em_part_addend=em_part,
        shape_mn_k=(geom.atom_m, geom.atom_k),
        elem_dtype=elem_dtype,
    )


def cq_gemm_s2r_b_tile(
    *,
    smem_b,
    elem_dtype,
    mma_n: int,
    mma_k: int,
    warp_n_id,
    warp_atoms_n: int,
    k_atoms: int,
    geom: CqOperandGeom = CQ_GEMM_GEOM,
):
    """Build EmPart-aware shared B view for one warp MMA atom (tn / k-major)."""
    global_mn = fx.Int32(warp_n_id) * fx.Int32(warp_atoms_n) + fx.Int32(mma_n)
    tile_base_elems = (global_mn * fx.Int32(k_atoms // 2) + fx.Int32(mma_k // 2)) * fx.Int32(geom.cta_chunk_elems)
    em_part = (int(mma_k) % 2) * 2
    return _smex_loadn_view(
        smem_b,
        tile_base_elems=tile_base_elems,
        em_part_addend=em_part,
        shape_mn_k=(geom.atom_n, geom.atom_k),
        elem_dtype=elem_dtype,
    )


def cq_gemm_s2r_load_mma_k(
    *,
    mma_k: int,
    smem_a,
    smem_b,
    elem_dtype,
    warp_m_id,
    warp_n_id,
    warp_atoms_m: int,
    warp_atoms_n: int,
    k_atoms: int,
    copy_atom_a,
    copy_atom_b,
    thr_copy_a,
    thr_copy_b,
    thr_mma,
    geom: CqOperandGeom = CQ_GEMM_GEOM,
):
    """Load all warp A/B fragments for one ``mma_k`` step (tn)."""
    a_frags = []
    for mma_m in fx.range_constexpr(warp_atoms_m):
        smem_tile = cq_gemm_s2r_a_tile(
            smem_a=smem_a,
            elem_dtype=elem_dtype,
            mma_m=mma_m,
            mma_k=mma_k,
            warp_m_id=warp_m_id,
            warp_atoms_m=warp_atoms_m,
            k_atoms=k_atoms,
            geom=geom,
        )
        a_frags.append(
            cq_gemm_s2r_copy_a(
                copy_atom=copy_atom_a,
                thr_copy_a=thr_copy_a,
                thr_mma=thr_mma,
                smem_a_tile=smem_tile,
            )
        )

    b_frags = []
    for mma_n in fx.range_constexpr(warp_atoms_n):
        smem_tile = cq_gemm_s2r_b_tile(
            smem_b=smem_b,
            elem_dtype=elem_dtype,
            mma_n=mma_n,
            mma_k=mma_k,
            warp_n_id=warp_n_id,
            warp_atoms_n=warp_atoms_n,
            k_atoms=k_atoms,
            geom=geom,
        )
        b_frags.append(
            cq_gemm_s2r_copy_b(
                copy_atom=copy_atom_b,
                thr_copy_b=thr_copy_b,
                thr_mma=thr_mma,
                smem_b_tile=smem_tile,
            )
        )
    return a_frags, b_frags


__all__ = [
    "cq_gemm_s2r_a_tile",
    "cq_gemm_s2r_b_tile",
    "cq_gemm_s2r_copy_a",
    "cq_gemm_s2r_copy_b",
    "cq_gemm_s2r_load_mma_k",
]
