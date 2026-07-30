# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar MR small-M HGEMM aligned with ixblas async tiny + bm=16.

Targets ``1 <= m <= 16`` with N/K tile-aligned.

* ``bm=16`` -- one SME row-brick (no wasted MMA on cropped rows).
* A and B both SME async G2S, ``STAGES=2`` (same pipeline as ``mr/hgemm.py``).
* Short M: DontCheck G2S (rows ``[m,16)`` may be OOB garbage); store ``row < m``.
* CTA auto-pick grows BN for large N when divisible.

Default: ``16x64x64`` stage-2 (1x2 warps, 1x2 atoms, k_atoms=4).
"""

from contextlib import contextmanager

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.ixdl as ixdl
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from kernels.gemm.iluvatar.common import WARP_SIZE
from kernels.gemm.iluvatar.epilogue import mr_hgemm_epilogue_store_shfl
from kernels.gemm.iluvatar.mr.common import (
    ATOM_K_B16,
    ATOM_M,
    ATOM_N,
    DEFAULT_SMEM_CAP_BYTES,
    MR_GEMM_GEOM,
    SMEM_ROWS,
    mr_stage_smem_ab,
)
from kernels.gemm.iluvatar.mr.operand_copy import mr_g2s_sme_config, mr_gemm_g2s_issue_operands
from kernels.gemm.iluvatar.mr.s2r import mr_gemm_s2r_load_mma_k

SUPPORTED_ELEM_DTYPES = (fx.Float16, fx.BFloat16)
DEFAULT_ELEM_DTYPE = fx.Float16
SMALL_M_MAX = 16
STAGES = 2
K_LOOP_UNROLL = 1


@contextmanager
def _if_then(if_op):
    # Functionally equivalent to kernels_common._if_then, but reuse that one will import ROCDL
    # buffer_ops, so keep this local version.
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            blk = if_op.then_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


def _validate_elem_dtype(elem_dtype):
    if elem_dtype not in SUPPORTED_ELEM_DTYPES:
        names = ", ".join(t.__name__ for t in SUPPORTED_ELEM_DTYPES)
        raise ValueError(f"elem_dtype must be one of {{{names}}}, got {elem_dtype!r}")
    return elem_dtype


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _cta_ok(warps_m, warps_n, warp_atoms_m, warp_atoms_n, k_atoms, N, K):
    bm = ATOM_M * warp_atoms_m * warps_m
    bn = ATOM_N * warp_atoms_n * warps_n
    bk = ATOM_K_B16 * k_atoms
    vpr = MR_GEMM_GEOM.values_per_sme_row
    if bm != SMEM_ROWS or bk % vpr != 0 or N % bn or K % bk:
        return False
    num_warps = warps_m * warps_n
    a_atoms = (bm // SMEM_ROWS) * (bk // vpr)
    b_atoms = (bn // SMEM_ROWS) * (bk // vpr)
    if a_atoms % num_warps or b_atoms % num_warps:
        return False
    # AB stages + C scratch
    smem_elems = (bm + bn) * bk * STAGES + bm * bn
    return smem_elems * 2 <= DEFAULT_SMEM_CAP_BYTES  # f16/bf16 = 2B


def _pick_cta(N: int, K: int):
    # (warps_m, warps_n, warp_atoms_m, warp_atoms_n, k_atoms) -- always bm=16
    candidates = [
        (1, 4, 1, 4, 8),  # 16x256x128, 4 warps -- wide N
        (1, 4, 1, 2, 8),  # 16x128x128, 4 warps
        (1, 2, 1, 4, 4),  # 16x128x64, 2 warps
        (1, 2, 1, 2, 4),  # 16x64x64, 2 warps
        (1, 1, 1, 4, 2),  # 16x64x32, 1 warp
        (1, 2, 1, 1, 4),  # 16x32x64, 2 warps
    ]
    order = [0, 1, 2, 3, 4, 5] if N >= 1024 else [3, 2, 1, 0, 4, 5]
    for i in order:
        c = candidates[i]
        if _cta_ok(*c, N, K):
            return c
    raise ValueError(f"no bm=16 A/B-SME CTA fits N={N} K={K}")


def compile_iluvatar_mr_small_m_hgemm(
    *,
    m: int,
    N: int,
    K: int,
    warps_m=None,
    warps_n=None,
    warp_atoms_m=None,
    warp_atoms_n=None,
    k_atoms=None,
    elem_dtype=DEFAULT_ELEM_DTYPE,
):
    """Build launcher C(m,N) = A(m,K) @ B(N,K).T. CTA auto-picked unless overridden."""
    elem_dtype = _validate_elem_dtype(elem_dtype)
    if not (1 <= m <= SMALL_M_MAX):
        raise ValueError(f"m must be in [1, {SMALL_M_MAX}], got {m}")

    if None in (warps_m, warps_n, warp_atoms_m, warp_atoms_n, k_atoms):
        wm, wn, wam, wan, ka = _pick_cta(N, K)
        warps_m = wm if warps_m is None else warps_m
        warps_n = wn if warps_n is None else warps_n
        warp_atoms_m = wam if warp_atoms_m is None else warp_atoms_m
        warp_atoms_n = wan if warp_atoms_n is None else warp_atoms_n
        k_atoms = ka if k_atoms is None else k_atoms

    warp_m = ATOM_M * warp_atoms_m
    warp_n = ATOM_N * warp_atoms_n
    bm = warp_m * warps_m
    bn = warp_n * warps_n
    bk = ATOM_K_B16 * k_atoms
    num_warps = warps_m * warps_n
    threads = num_warps * WARP_SIZE
    vpr = MR_GEMM_GEOM.values_per_sme_row

    if bm != SMEM_ROWS:
        raise ValueError(f"bm must be {SMEM_ROWS}, got {bm}")
    if bk % vpr:
        raise ValueError(f"bk must be multiple of {vpr}, got {bk}")
    if N % bn:
        raise ValueError(f"N must be multiple of bn={bn}, got N={N}")
    if K % bk:
        raise ValueError(f"K must be multiple of bk={bk}, got K={K}")

    k_bricks = bk // vpr
    a_atoms_total = (bm // SMEM_ROWS) * k_bricks
    b_atoms_total = (bn // SMEM_ROWS) * k_bricks
    if a_atoms_total % num_warps or b_atoms_total % num_warps:
        raise ValueError(f"SME bricks A={a_atoms_total} B={b_atoms_total} must divide {num_warps} warps")
    a_per_warp = a_atoms_total // num_warps
    b_per_warp = b_atoms_total // num_warps

    stage_elems = (bm + bn) * bk
    stage_stride = stage_elems
    c_elems = bm * bn
    total_smem_elems = stage_elems * STAGES + c_elems
    smem_bytes = total_smem_elems * (elem_dtype.width // 8)
    if smem_bytes > DEFAULT_SMEM_CAP_BYTES:
        raise ValueError(f"CTA smem {smem_bytes} B exceeds cap {DEFAULT_SMEM_CAP_BYTES} B")

    c_store_elems = m * bn
    c_store_iters = _ceil_div(c_store_elems, threads)
    a_mn_major = False
    b_mn_major = False
    k_tiles_const = K // bk
    main_k_trip = max(0, k_tiles_const - 2)
    main_k_full = (main_k_trip // K_LOOP_UNROLL) * K_LOOP_UNROLL
    main_k_remainder = main_k_trip - main_k_full

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def small_m_hgemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        bid_y = fx.block_idx.y
        warp_id = tid // WARP_SIZE
        lane_id = fx.Int32(fx.lane_id)
        warp_m_id = fx.Int32(0)
        warp_n_id = warp_id % warps_n

        a_tile = fx.make_view(fx.get_iter(A), fx.make_layout((bm, K), (K, 1)))
        b_logical = fx.make_view(fx.get_iter(B), fx.make_layout((N, K), (K, 1)))
        c_logical = fx.make_view(fx.get_iter(C), fx.make_layout((m, N), (N, 1)))

        gA = fx.slice(fx.flat_divide(a_tile, (bm, bk)), (None, None, 0, None))
        gB = fx.slice(fx.flat_divide(b_logical, (bn, bk)), (None, None, bid_y, None))
        gC_n = fx.slice(fx.flat_divide(c_logical, (m, bn)), (None, None, 0, bid_y))

        @fx.struct
        class MrSmallMSmem:
            buf: fx.Array[elem_dtype, total_smem_elems]

        smem_base = fx.SharedAllocator(static=True).allocate(MrSmallMSmem).peek().buf.ptr
        smem_ab_base = smem_base
        smem_c = fx.add_offset(smem_base, fx.make_int_tuple(fx.Int32(stage_elems * STAGES)))
        smem_c_view = fx.make_view(smem_c, fx.make_layout((bm, bn), (bn, 1)))

        mma_atom = fx.make_mma_atom(ixdl.MRMma(ATOM_M, ATOM_N, ATOM_K_B16, elem_dtype, elem_dtype, fx.Float32))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (1, 1, 1)))
        thr_mma = tiled_mma.thr_slice(lane_id)

        c_ref = fx.slice(fx.flat_divide(smem_c_view, (ATOM_M, ATOM_N)), (None, None, 0, 0))
        accs = []
        for _mm in fx.range_constexpr(warp_atoms_m):
            row = []
            for _mn in fx.range_constexpr(warp_atoms_n):
                frag = thr_mma.make_fragment_C(c_ref)
                frag.fill(0)
                row.append(frag)
            accs.append(row)

        g2s_sme = mr_g2s_sme_config(
            a_mn_major=a_mn_major,
            b_mn_major=b_mn_major,
            elem_dtype=elem_dtype,
            row_atom=ixdl.MRAsyncCpRow16b,
            row_swizzle=ixdl.SMESwizzle.Row16b,
        )
        copy_a = fx.make_copy_atom(fx.UniversalCopy32b(), elem_dtype)
        copy_b = fx.make_copy_atom(fx.UniversalCopy32b(), elem_dtype)
        thr_copy_a = fx.make_tiled_copy_A(copy_a, tiled_mma).get_slice(lane_id)
        thr_copy_b = fx.make_tiled_copy_B(copy_b, tiled_mma).get_slice(lane_id)
        tile_smem = fx.make_tile(SMEM_ROWS, vpr)

        def issue_stage(k_tile, stage_base):
            k_A = gA[None, None, k_tile]
            k_B = gB[None, None, k_tile]
            sme_A = ixdl.make_sme_gmem_tensor(k_A, leading_stride=K)
            sme_B = ixdl.make_sme_gmem_tensor(k_B, leading_stride=K)
            smem_a, smem_b = mr_stage_smem_ab(smem_ab_base, stage_base, bm * bk)
            mr_gemm_g2s_issue_operands(
                a_mn_major=a_mn_major,
                b_mn_major=b_mn_major,
                warp_id=warp_id,
                a_per_warp=a_per_warp,
                b_per_warp=b_per_warp,
                a_cta_gmem_view=fx.zipped_divide(sme_A, tile_smem),
                b_cta_gmem_view=fx.zipped_divide(sme_B, tile_smem),
                g2s_sme=g2s_sme,
                smem_a=smem_a,
                smem_b=smem_b,
                elem_dtype=elem_dtype,
                bm=bm,
                bn=bn,
                bk=bk,
                geom=MR_GEMM_GEOM,
            )

        def _mma_k_load(stage_base, mma_k):
            smem_a, smem_b = mr_stage_smem_ab(smem_ab_base, stage_base, bm * bk)
            return mr_gemm_s2r_load_mma_k(
                a_mn_major=a_mn_major,
                b_mn_major=b_mn_major,
                mma_k=mma_k,
                g2s_sme=g2s_sme,
                smem_a=smem_a,
                smem_b=smem_b,
                elem_dtype=elem_dtype,
                warp_m_id=warp_m_id,
                warp_n_id=warp_n_id,
                warp_atoms_m=warp_atoms_m,
                warp_atoms_n=warp_atoms_n,
                copy_atom_a=copy_a,
                copy_atom_b=copy_b,
                thr_copy_a=thr_copy_a,
                thr_copy_b=thr_copy_b,
                thr_mma=thr_mma,
                bm=bm,
                bn=bn,
                bk=bk,
                geom=MR_GEMM_GEOM,
            )

        def _mma_frags(a_frags, b_frags):
            for mma_n in fx.range_constexpr(warp_atoms_n):
                for mma_m in fx.range_constexpr(warp_atoms_m):
                    fx.gemm(mma_atom, accs[mma_m][mma_n], a_frags[mma_m], b_frags[mma_n], accs[mma_m][mma_n])

        def _s2r_mma_defer_last_into(stage_base, a_def, b_def):
            for mma_k in fx.range_constexpr(k_atoms - 1):
                a_frags, b_frags = _mma_k_load(stage_base, mma_k)
                _mma_frags(a_frags, b_frags)
            a_last, b_last = _mma_k_load(stage_base, k_atoms - 1)
            for mma_m in fx.range_constexpr(warp_atoms_m):
                a_def[mma_m].store(a_last[mma_m].load())
            for mma_n in fx.range_constexpr(warp_atoms_n):
                b_def[mma_n].store(b_last[mma_n].load())

        def _s2r_mma_defer_last(stage_base):
            for mma_k in fx.range_constexpr(k_atoms - 1):
                a_frags, b_frags = _mma_k_load(stage_base, mma_k)
                _mma_frags(a_frags, b_frags)
            return _mma_k_load(stage_base, k_atoms - 1)

        def _s2r_mma_all(stage_base):
            a_frags, b_frags = _s2r_mma_defer_last(stage_base)
            _mma_frags(a_frags, b_frags)

        issue_stage(fx.Int32(0), fx.Int32(0))
        fx.gpu.barrier()
        if k_tiles_const >= 2:
            issue_stage(fx.Int32(1), fx.Int32(stage_stride))

        a_def, b_def = _s2r_mma_defer_last(fx.Int32(0))

        def _k_iter_body(k_idx):
            fx.gpu.barrier()
            _mma_frags(a_def, b_def)
            load_stage_base = fx.Int32(k_idx % 2) * fx.Int32(stage_stride)
            comp_stage_base = load_stage_base ^ fx.Int32(stage_stride)
            issue_stage(fx.Int32(k_idx + 2), load_stage_base)
            _s2r_mma_defer_last_into(comp_stage_base, a_def, b_def)

        if fx.const_expr(main_k_full > 0):
            for k_base in fx.range(0, main_k_full, K_LOOP_UNROLL):
                for u in fx.range_constexpr(K_LOOP_UNROLL):
                    _k_iter_body(k_base + u)
        if fx.const_expr(main_k_remainder > 0):
            for u in fx.range_constexpr(main_k_remainder):
                _k_iter_body(main_k_full + u)

        fx.gpu.barrier()
        _mma_frags(a_def, b_def)
        if k_tiles_const >= 2:
            if main_k_trip % 2 == 0:
                _s2r_mma_all(fx.Int32(stage_stride))
            else:
                _s2r_mma_all(fx.Int32(0))

        if fx.const_expr(m == bm):
            # Full brick: production shfl store straight to GMEM (no smem C).
            gC_full = fx.slice(fx.flat_divide(c_logical, (bm, bn)), (None, None, 0, bid_y))
            gC_warp = fx.slice(
                fx.flat_divide(gC_full, (warp_m, warp_n)),
                (None, None, 0, warp_n_id),
            )
            mr_hgemm_epilogue_store_shfl(
                lane_id=lane_id,
                accs=accs,
                gC_warp=gC_warp,
                c_global_n=N,
                warp_atoms_m=warp_atoms_m,
                warp_atoms_n=warp_atoms_n,
                out_dtype=elem_dtype,
            )
        else:
            gC_smem_warp = fx.slice(
                fx.flat_divide(smem_c_view, (warp_m, warp_n)),
                (None, None, 0, warp_n_id),
            )
            mr_hgemm_epilogue_store_shfl(
                lane_id=lane_id,
                accs=accs,
                gC_warp=gC_smem_warp,
                c_global_n=bn,
                warp_atoms_m=warp_atoms_m,
                warp_atoms_n=warp_atoms_n,
                out_dtype=elem_dtype,
            )
            fx.gpu.barrier()
            for i in fx.range_constexpr(c_store_iters):
                idx = fx.Int32(i) * fx.Int32(threads) + tid
                ok = fx.arith.cmpi(fx.arith.CmpIPredicate.ult, idx, fx.Int32(c_store_elems))
                with _if_then(scf.IfOp(ok)):
                    row = idx // fx.Int32(bn)
                    col = idx % fx.Int32(bn)
                    gC_n[row, col] = smem_c_view[row, col]

    grid = (1, N // bn, 1)
    block = (threads, 1, 1)

    @flyc.jit
    def launch_gemm(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        small_m_hgemm_kernel(A, B, C).launch(grid=grid, block=block, stream=stream)

    return launch_gemm


__all__ = [
    "SMALL_M_MAX",
    "SUPPORTED_ELEM_DTYPES",
    "DEFAULT_ELEM_DTYPE",
    "compile_iluvatar_mr_small_m_hgemm",
]
