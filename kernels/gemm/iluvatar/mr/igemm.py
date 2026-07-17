# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar MR (ivcore11) tiledMma pipeline int8 GEMM (B8 GEMM).

Multi-warp double-buffered ``D = A @ B.T`` with int8 inputs and an int32 or int8
output. Shares the SME G2S / S2R / MMA helpers with the f16 MR HGEMM
(``mr_g2s_sme_config``, ``mr_gemm_g2s_issue_operands``, ``mr_gemm_s2r_load_mma_k``);
the int8 path uses a 16x16x32 MMA atom (``ATOM_K_B8``) and 64-value SME rows
(``MrOperandGeom.b8()``) vs f16's K=16 / 32-value rows.

Entry point: ``compile_iluvatar_mr_igemm(M=..., N=..., K=..., ...)`` returns a
``@flyc.jit`` launch wrapper ``launch_gemm(A, B, C, stream=...)``.

``major_pattern`` -- CUTLASS BLAS layout tag on logical ``A(m,k)`` / ``B(n,k)``
(see ``kernels.gemm.iluvatar.common.GemmLayout``). ``tn`` (both operands k-major) is
the default fast path; the mn-major patterns (``nn`` / ``nt`` / ``tt``) use i8
k-spanning S2R, where one MMA K atom (32) spans two SME bricks (K=16 each).

Output epilogue: ``i32`` (default, direct int32 store) or ``i8`` (packed store,
truncating cast, no quant scale).
"""

# NOTE: do NOT add ``from __future__ import annotations`` (Constexpr introspection).

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.ixdl as ixdl
from kernels.gemm.iluvatar.common import (
    DEFAULT_MAJOR_PATTERN,
    WARP_SIZE,
    parse_major_pattern,
)
from kernels.gemm.iluvatar.mr.common import (
    ATOM_K_B8,
    ATOM_M,
    ATOM_N,
    DEFAULT_SMEM_CAP_BYTES,
    SMEM_ROWS,
    MrOperandGeom,
    mr_stage_smem_ab,
    sme_atom_counts,
)
from kernels.gemm.iluvatar.epilogue import (
    mr_igemm_epilogue_store_i8_packed,
    mr_igemm_epilogue_store_i32,
)
from kernels.gemm.iluvatar.mr.operand_copy import mr_g2s_sme_config, mr_gemm_g2s_issue_operands
from kernels.gemm.iluvatar.mr.s2r import mr_gemm_s2r_load_mma_k

# int8 operand geometry: 16x16x32 MMA atom, 64 int8 per 512-bit SME row.
MR_IGEMM_GEOM = MrOperandGeom.b8()
ATOM_K_I8 = ATOM_K_B8
I8_VPR = MR_IGEMM_GEOM.values_per_sme_row

DEFAULT_K_REP = 2  # CTA K-tile: ATOM_K_B8 * k_rep = 64 (one int8 SME row width)
STAGES = 2
K_LOOP_UNROLL = 2

# Threadblock rasterization swizzle: group this many M-tiles per N
# column in launch order to raise L2 reuse on large GEMMs.
# Effective group is the largest power-of-2 divisor of grid_m that is <= this.
BLOCK_SWIZZLE_GROUP_M = 4


def _block_swizzle_group_m(grid_m: int, cap: int = BLOCK_SWIZZLE_GROUP_M) -> int:
    g = 1
    while g * 2 <= cap and grid_m % (g * 2) == 0:
        g *= 2
    return g


# Multi-stage wins on latency-bound, small
# grids; large grids are compute/occupancy-bound and prefer the 65KB 2-stage
# double buffer. Empirical crossover on ivcore11: <=16 CTAs -> 3-stage.
AUTO_STAGES_BLOCK_THRESH = 16
# Large grids whose operands span the MMA K-atom across >1 SME brick (every
# pattern with an mn-major operand) peak at a deeper 4-stage pipeline: the extra
# buffer hides the non-contiguous S2R latency.
AUTO_STAGES_S4_BLOCK_THRESH = 192


def resolve_igemm_stages(
    stages,
    M: int,
    N: int,
    K: int,
    bm: int,
    bn: int,
    bk: int,
    major_pattern: str = DEFAULT_MAJOR_PATTERN,
    smem_cap: int = DEFAULT_SMEM_CAP_BYTES,
) -> int:
    """Resolve the SMEM pipeline depth. ``stages=None`` => auto:

    - small/latency-bound grids (<= AUTO_STAGES_BLOCK_THRESH CTAs, >= 3 K-tiles) => 3 stages
    - large k-spanning grids (>= AUTO_STAGES_S4_BLOCK_THRESH CTAs, >= 4 K-tiles,
      4-stage SMEM within ``smem_cap``, an mn-major operand) => 4 stages
    - otherwise the proven 2-stage double buffer.
    """
    if stages is not None:
        return stages
    blocks = (M // bm) * (N // bn)
    ktiles = K // bk
    if blocks <= AUTO_STAGES_BLOCK_THRESH and ktiles >= 3:
        return 3
    layout = parse_major_pattern(major_pattern)
    k_spanning = layout.a_mn_major or layout.b_mn_major
    if (
        k_spanning
        and blocks >= AUTO_STAGES_S4_BLOCK_THRESH
        and ktiles >= 4
        and 4 * (bm + bn) * bk <= smem_cap
    ):
        return 4
    return 2


# Output epilogue: i32 direct store (D = A @ B.T, int32) or i8 packed store
# (int8 output, pure truncating cast, no quant scale).
EPILOGUE_I32 = "i32"
EPILOGUE_I8 = "i8"
EPILOGUE_CHOICES = (EPILOGUE_I32, EPILOGUE_I8)
DEFAULT_EPILOGUE = EPILOGUE_I32


def _igemm_cta_shape(
    warps_m: int,
    warps_n: int,
    k_rep: int,
    *,
    warp_atoms_m: int,
    warp_atoms_n: int,
    stages: int = STAGES,
) -> tuple[int, int, int, int, int]:
    """Swizzle-mode CTA geometry. Returns (bm, bn, bk, threads, pipeline_smem_bytes).

    int8 = 1 byte/elem, ``stages``-buffered G2S staging smem (epilogue smem handled
    separately by the caller).
    """
    warp_m = ATOM_M * warp_atoms_m
    warp_n = ATOM_N * warp_atoms_n
    bm = warp_m * warps_m
    bn = warp_n * warps_n
    bk = ATOM_K_B8 * k_rep
    threads = warps_m * warps_n * WARP_SIZE
    pipeline_smem = (bm + bn) * bk * stages
    return bm, bn, bk, threads, pipeline_smem


def _build_igemm_kernel(
    m: int,
    n: int,
    k: int,
    warps_m: int,
    warps_n: int,
    k_rep: int,
    warp_atoms_m: int,
    warp_atoms_n: int,
    major_pattern: str = DEFAULT_MAJOR_PATTERN,
    epilogue: str = DEFAULT_EPILOGUE,
    stages: int = STAGES,
):
    layout = parse_major_pattern(major_pattern)
    a_mn_major = layout.a_mn_major
    b_mn_major = layout.b_mn_major
    # i8 + k-major B -> N_SWIZZLE=4 + PackOnly.
    b_n_swizzle = 4 if (epilogue == EPILOGUE_I8 and not b_mn_major) else 1
    use_pack_only = b_n_swizzle > 1
    geom = MR_IGEMM_GEOM
    vpr = geom.values_per_sme_row
    warp_m = ATOM_M * warp_atoms_m
    warp_n = ATOM_N * warp_atoms_n
    bm = warp_m * warps_m
    bn = warp_n * warps_n
    bk = ATOM_K_B8 * k_rep
    num_warps = warps_m * warps_n
    threads = num_warps * WARP_SIZE
    grid_m = m // bm
    grid_n = n // bn
    swizzle_group_m = _block_swizzle_group_m(grid_m)

    assert k % bk == 0
    assert m % bm == 0 and n % bn == 0
    assert bk % vpr == 0, f"bk={bk} must be a multiple of {vpr} for int8 SME"

    a_atoms_total, b_atoms_total, _, _ = sme_atom_counts(layout, bm, bn, bk, values_per_sme_row=vpr)
    assert a_atoms_total % num_warps == 0, "A SME chunks must divide across warps"
    assert b_atoms_total % num_warps == 0, "B SME chunks must divide across warps"
    a_per_warp = a_atoms_total // num_warps
    b_per_warp = b_atoms_total // num_warps
    k_tiles_const = k // bk
    stage_elems = (bm + bn) * bk
    stage_stride = stage_elems
    main_k_trip = max(0, k_tiles_const - 2)
    main_k_full = (main_k_trip // K_LOOP_UNROLL) * K_LOOP_UNROLL
    main_k_remainder = main_k_trip - main_k_full
    # multi-stage (>=3): pipebar + sl.waitcnt keeps stages-1 G2S in
    # flight. 2-stage keeps the proven full-barrier double-buffer path.
    use_multistage = stages >= 3 and k_tiles_const >= stages
    g2s_load_inst = a_per_warp + b_per_warp

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        bid_x, bid_y, _ = fx.block_idx
        # Threadblock rasterization swizzle: remap
        # (bid_x=M-tile, bid_y=N-tile) so consecutive scheduled blocks sweep
        # `swizzle_group_m` M-tiles per N column -> better A-row L2 reuse. Bijection
        # holds because swizzle_group_m divides grid_m. group_m==1 is the identity.
        if fx.const_expr(swizzle_group_m > 1):
            pid = bid_x + bid_y * fx.Int32(grid_m)
            num_in_group = swizzle_group_m * grid_n
            group_id = pid // fx.Int32(num_in_group)
            pid_in_group = pid % fx.Int32(num_in_group)
            m_tile = group_id * fx.Int32(swizzle_group_m) + (pid_in_group % fx.Int32(swizzle_group_m))
            n_tile = pid_in_group // fx.Int32(swizzle_group_m)
        else:
            m_tile = bid_x
            n_tile = bid_y
        warp_id = tid // WARP_SIZE
        lane_id = tid % WARP_SIZE
        warp_m_id = warp_id // warps_n
        warp_n_id = warp_id % warps_n

        if fx.const_expr(a_mn_major):
            a_logical_stride = (1, m)
        else:
            a_logical_stride = (k, 1)
        a_logical = fx.make_view(fx.get_iter(A), fx.make_layout((m, k), a_logical_stride))
        gA = fx.slice(fx.flat_divide(a_logical, (bm, bk)), (None, None, m_tile, None))

        if fx.const_expr(b_mn_major):
            b_logical_stride = (1, n)
        else:
            b_logical_stride = (k, 1)
        b_logical = fx.make_view(fx.get_iter(B), fx.make_layout((n, k), b_logical_stride))
        gB = fx.slice(fx.flat_divide(b_logical, (bn, bk)), (None, None, n_tile, None))

        gC = fx.slice(fx.flat_divide(C, (bm, bn)), (None, None, m_tile, n_tile))

        # Static contiguous shared memory (matches hgemm): the compiler sizes the
        # bank, so launch(smem=...) stays unset. PackOnly needs no C-tile scratch;
        # PackSlb reuses the same bank for SLB staging (bm * bn).
        smem_elems = (
            max(stage_elems * stages, bm * bn)
            if epilogue == EPILOGUE_I8 and not use_pack_only
            else stage_elems * stages
        )

        @fx.struct
        class MrIgemmSmem:
            buf: fx.Array[fx.Int8, smem_elems]

        smem_base = fx.SharedAllocator(static=True).allocate(MrIgemmSmem).peek().buf.ptr

        mma_atom = fx.make_mma_atom(ixdl.MRMma(ATOM_M, ATOM_N, ATOM_K_B8, fx.Int8, fx.Int8, fx.Int32))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (1, 1, 1)))
        thr_mma = tiled_mma.thr_slice(lane_id)

        gC_warp = fx.slice(
            fx.flat_divide(gC, (warp_m, warp_n)),
            (None, None, warp_m_id, warp_n_id),
        )
        gC_atoms = fx.flat_divide(gC_warp, (ATOM_M, ATOM_N))

        accs = []
        for mma_m in fx.range_constexpr(warp_atoms_m):
            row = []
            for mma_n in fx.range_constexpr(warp_atoms_n):
                c_tile = fx.slice(gC_atoms, (None, None, mma_m, mma_n))
                frag = thr_mma.make_fragment_C(c_tile)
                frag.fill(0)
                row.append(frag)
            accs.append(row)

        def _run_pipeline():
            g2s_sme = mr_g2s_sme_config(
                a_mn_major=a_mn_major,
                b_mn_major=b_mn_major,
                elem_dtype=fx.Int8,
                row_atom=ixdl.MRAsyncCpRow8b,
                row_swizzle=ixdl.SMESwizzle.Row8b,
            )

            copy_atom_s2r_a = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int8)
            copy_atom_s2r_b = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int8)
            tiled_copy_a = fx.make_tiled_copy_A(copy_atom_s2r_a, tiled_mma)
            tiled_copy_b = fx.make_tiled_copy_B(copy_atom_s2r_b, tiled_mma)
            thr_copy_a = tiled_copy_a.get_slice(lane_id)
            thr_copy_b = tiled_copy_b.get_slice(lane_id)

            tile_smem = fx.make_tile(SMEM_ROWS, vpr)
            tile_smem_A = fx.make_tile(vpr, SMEM_ROWS) if fx.const_expr(a_mn_major) else tile_smem
            tile_smem_B = fx.make_tile(vpr, SMEM_ROWS) if fx.const_expr(b_mn_major) else tile_smem

            if fx.const_expr(a_mn_major):
                a_leading = m
            else:
                a_leading = k
            if fx.const_expr(b_mn_major):
                b_leading = n
            else:
                b_leading = k

            def issue_stage(k_tile, stage_base, commit=True):
                k_A = gA[None, None, k_tile]
                k_B = gB[None, None, k_tile]
                sme_A = ixdl.make_sme_gmem_tensor(k_A, leading_stride=a_leading)
                # PackOnly N_SWIZZLE: SME Col desc walks N with stride
                # b_leading * b_n_swizzle (e.g. *4), so one load covers every
                # swizzle-th N row. Matches the GMEM row remap in issue_b_warp.
                sme_B = ixdl.make_sme_gmem_tensor(k_B, leading_stride=b_leading * b_n_swizzle)
                smem_a, smem_b = mr_stage_smem_ab(smem_base, stage_base, bm * bk)
                if fx.const_expr(b_n_swizzle > 1):
                    b_cta_view = sme_B
                else:
                    b_cta_view = fx.zipped_divide(sme_B, tile_smem_B)
                mr_gemm_g2s_issue_operands(
                    a_mn_major=a_mn_major,
                    b_mn_major=b_mn_major,
                    warp_id=warp_id,
                    a_per_warp=a_per_warp,
                    b_per_warp=b_per_warp,
                    a_cta_gmem_view=fx.zipped_divide(sme_A, tile_smem_A),
                    b_cta_gmem_view=b_cta_view,
                    g2s_sme=g2s_sme,
                    smem_a=smem_a,
                    smem_b=smem_b,
                    elem_dtype=fx.Int8,
                    bm=bm,
                    bn=bn,
                    bk=bk,
                    geom=geom,
                    commit=commit,
                    b_n_swizzle=b_n_swizzle,
                    b_leading=b_leading,
                )

            def _mma_k_load(stage_base, mma_k):
                smem_a, smem_b = mr_stage_smem_ab(smem_base, stage_base, bm * bk)
                return mr_gemm_s2r_load_mma_k(
                    a_mn_major=a_mn_major,
                    b_mn_major=b_mn_major,
                    mma_k=mma_k,
                    g2s_sme=g2s_sme,
                    smem_a=smem_a,
                    smem_b=smem_b,
                    elem_dtype=fx.Int8,
                    warp_m_id=warp_m_id,
                    warp_n_id=warp_n_id,
                    warp_atoms_m=warp_atoms_m,
                    warp_atoms_n=warp_atoms_n,
                    copy_atom_a=copy_atom_s2r_a,
                    copy_atom_b=copy_atom_s2r_b,
                    thr_copy_a=thr_copy_a,
                    thr_copy_b=thr_copy_b,
                    thr_mma=thr_mma,
                    bm=bm,
                    bn=bn,
                    bk=bk,
                    geom=geom,
                )

            def _mma_frags(a_frags, b_frags):
                for mma_n in fx.range_constexpr(warp_atoms_n):
                    for mma_m in fx.range_constexpr(warp_atoms_m):
                        fx.gemm(mma_atom, accs[mma_m][mma_n], a_frags[mma_m], b_frags[mma_n], accs[mma_m][mma_n])

            def _copy_frag(dst, src):
                dst.store(src.load())

            def _copy_a_frags(dst, src):
                for mma_m in fx.range_constexpr(warp_atoms_m):
                    _copy_frag(dst[mma_m], src[mma_m])

            def _copy_b_frags(dst, src):
                for mma_n in fx.range_constexpr(warp_atoms_n):
                    _copy_frag(dst[mma_n], src[mma_n])

            def _s2r_mma_defer_last_into(stage_base, a_def, b_def):
                for mma_k in fx.range_constexpr(k_rep - 1):
                    a_frags, b_frags = _mma_k_load(stage_base, mma_k)
                    _mma_frags(a_frags, b_frags)
                a_last, b_last = _mma_k_load(stage_base, k_rep - 1)
                _copy_a_frags(a_def, a_last)
                _copy_b_frags(b_def, b_last)

            def _s2r_mma_rest_defer_last(stage_base, a0, b0):
                # Prefetch-aware variant for the multistage pipebar path: consume
                # prefetched mma_k=0 frags, stream mma_k 1..k_rep-2, and return the
                # last frags (register-resident) to defer their MMA to the next tile.
                _mma_frags(a0, b0)
                for mma_k in fx.range_constexpr(1, k_rep - 1):
                    a_frags, b_frags = _mma_k_load(stage_base, mma_k)
                    _mma_frags(a_frags, b_frags)
                return _mma_k_load(stage_base, k_rep - 1)

            def _s2r_mma_defer_last(stage_base):
                for mma_k in fx.range_constexpr(k_rep - 1):
                    a_frags, b_frags = _mma_k_load(stage_base, mma_k)
                    _mma_frags(a_frags, b_frags)
                return _mma_k_load(stage_base, k_rep - 1)

            def _s2r_mma_all(stage_base):
                a_frags, b_frags = _s2r_mma_defer_last(stage_base)
                _mma_frags(a_frags, b_frags)

            def _sync_arrive(g2s_cnt):
                # SyncArrive: drain own G2S to <= g2s_cnt (and shared-mem to 0),
                # then signal the split pipeline barrier.
                ixdl.sl_waitmem(g2s=g2s_cnt, lm=0)
                ixdl.sl_pipebar_arrive(0)

            def _sync_wait():
                ixdl.sl_pipebar_wait(0)

            if fx.const_expr(use_multistage):
                # N-stage: keep stages-1 G2S tiles in flight via pipebar
                # (split barrier) + sl.waitcnt; NO full barrier here (the pipebar
                # protocol forbids mixing sl_barrier with pipebar reqs). Main loop is
                # constexpr-unrolled so the mma_k-deferred MMA fragments thread through
                # as registers (only used for small/latency-bound shapes).
                full_cnt = (stages - 2) * g2s_load_inst

                def _calc_blk(stage_base, first, last, a_def, b_def):
                    # Prefetch mma_k=0 first to overlap its (strided) S2R with the
                    # deferred MMA of the previous tile, then defer this tile's last.
                    a0, b0 = _mma_k_load(stage_base, 0)
                    if fx.const_expr(first):
                        _mma_frags(a_def, b_def)
                    na, nb = _s2r_mma_rest_defer_last(stage_base, a0, b0)
                    if fx.const_expr(last):
                        _mma_frags(na, nb)
                    return na, nb

                for s in fx.range_constexpr(stages - 1):
                    issue_stage(fx.Int32(s), fx.Int32(s * stage_stride), commit=False)
                _sync_arrive(full_cnt)

                main_count = k_tiles_const - stages + 1
                a_def = b_def = None
                for i in fx.range_constexpr(main_count):
                    _sync_wait()
                    nxt = i + (stages - 1)
                    issue_stage(fx.Int32(nxt), fx.Int32((nxt % stages) * stage_stride), commit=False)
                    a_def, b_def = _calc_blk(
                        fx.Int32((i % stages) * stage_stride), i > 0, False, a_def, b_def
                    )
                    _sync_arrive(full_cnt)

                for j in fx.range_constexpr(stages - 1):
                    _sync_wait()
                    t = main_count + j
                    a_def, b_def = _calc_blk(
                        fx.Int32((t % stages) * stage_stride), True, j == stages - 2, a_def, b_def
                    )
                    if fx.const_expr(j < stages - 2):
                        _sync_arrive((stages - 2 - (j + 1)) * g2s_load_inst)
            else:
                # Prologue (match hgemm double-buffer peel):
                #   issue0 → barrier (IXDL drains g2scnt) → issue1 (no wait) →
                #   peel stage0 so S2R/MMA overlaps tile1 G2S.
                issue_stage(fx.Int32(0), fx.Int32(0))
                fx.gpu.barrier()

                if k_tiles_const >= 2:
                    issue_stage(fx.Int32(1), fx.Int32(stage_stride))

                a_def, b_def = _s2r_mma_defer_last(fx.Int32(0))

                def _k_iter_body(k_idx):
                    # Match hgemm: after barrier the compute stage is ready and
                    # the other stage is free — issue next G2S before deferred MMA so
                    # copies overlap that MMA. Stage pick is branchless via %2 + XOR.
                    fx.gpu.barrier()
                    k_tile = k_idx + 2
                    load_stage_base = fx.Int32(k_idx % 2) * fx.Int32(stage_stride)
                    comp_stage_base = load_stage_base ^ fx.Int32(stage_stride)
                    issue_stage(fx.Int32(k_tile), load_stage_base)
                    _mma_frags(a_def, b_def)
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

        _run_pipeline()

        if fx.const_expr(epilogue == EPILOGUE_I8):
            mr_igemm_epilogue_store_i8_packed(
                lane_id=lane_id,
                warp_id=warp_id,
                accs=accs,
                gC_warp=gC_warp,
                smem_base=smem_base,
                warp_atoms_m=warp_atoms_m,
                warp_atoms_n=warp_atoms_n,
                c_global_n=n,
                pack_only=use_pack_only,
            )
        else:
            mr_igemm_epilogue_store_i32(
                lane_id=lane_id,
                accs=accs,
                gC_warp=gC_warp,
                c_global_n=n,
                warp_atoms_m=warp_atoms_m,
                warp_atoms_n=warp_atoms_n,
            )

    smem_bytes = (
        max(stage_elems * stages, bm * bn)
        if epilogue == EPILOGUE_I8 and not use_pack_only
        else stage_elems * stages
    )
    return gemm_kernel, threads, smem_bytes, bm, bn, bk


def compile_iluvatar_mr_igemm(
    *,
    M: int,
    N: int,
    K: int,
    warps_m: int = 4,
    warps_n: int = 4,
    k_rep: int = DEFAULT_K_REP,
    warp_atoms_m: int = 4,
    warp_atoms_n: int = 4,
    major_pattern: str = DEFAULT_MAJOR_PATTERN,
    epilogue: str = DEFAULT_EPILOGUE,
    stages: int | None = None,
):
    """Build and return a JIT launch wrapper for the Iluvatar MR int8 GEMM.

    ``D = A @ B.T`` with int8 ``A(M, K)`` / ``B(N, K)``. ``epilogue`` selects the
    output: ``i32`` (default; direct int32 store) or ``i8`` (packed store, pure
    truncating cast, int8 ``C``). See the module docstring for ``major_pattern``
    and CTA shape semantics.
    """
    layout = parse_major_pattern(major_pattern)
    if epilogue not in EPILOGUE_CHOICES:
        raise ValueError(f"unknown epilogue: {epilogue}")

    bm, bn, bk, threads, _ = _igemm_cta_shape(
        warps_m,
        warps_n,
        k_rep,
        warp_atoms_m=warp_atoms_m,
        warp_atoms_n=warp_atoms_n,
        stages=2,
    )
    stages = resolve_igemm_stages(stages, M, N, K, bm, bn, bk, major_pattern=major_pattern)
    bm, bn, bk, threads, pipeline_smem = _igemm_cta_shape(
        warps_m,
        warps_n,
        k_rep,
        warp_atoms_m=warp_atoms_m,
        warp_atoms_n=warp_atoms_n,
        stages=stages,
    )
    # PackOnly (k-major B i8) needs no C-tile scratch beyond the pipeline buffers.
    use_pack_only = epilogue == EPILOGUE_I8 and not layout.b_mn_major
    smem_bytes = max(pipeline_smem, bm * bn) if (epilogue == EPILOGUE_I8 and not use_pack_only) else pipeline_smem
    if K % bk:
        raise ValueError(f"K must be a multiple of {bk} (ATOM_K_B8 * k_rep)")
    if M % bm or N % bn:
        raise ValueError(f"M,N must be multiples of {bm}/{bn}")
    if bk % I8_VPR:
        raise ValueError(f"BK={bk} must be a multiple of {I8_VPR}; use even k_rep")
    num_warps = warps_m * warps_n
    a_atoms_total, b_atoms_total, _, _ = sme_atom_counts(layout, bm, bn, bk, values_per_sme_row=I8_VPR)
    if a_atoms_total % num_warps or b_atoms_total % num_warps:
        raise ValueError(
            f"SME chunk count must divide evenly across {warps_m}x{warps_n} warps; "
            f"try larger k_rep (current BK={bk})"
        )
    if smem_bytes > DEFAULT_SMEM_CAP_BYTES:
        raise ValueError(
            f"CTA smem {smem_bytes} B exceeds device cap {DEFAULT_SMEM_CAP_BYTES} B "
            f"({bm}x{bn}x{bk}, {threads} threads); use smaller tile or k_rep"
        )

    gemm_kernel, threads, smem_bytes, bm, bn, _bk = _build_igemm_kernel(
        M,
        N,
        K,
        warps_m,
        warps_n,
        k_rep,
        warp_atoms_m,
        warp_atoms_n,
        major_pattern,
        epilogue,
        stages,
    )
    grid = (M // bm, N // bn, 1)
    block = (threads, 1, 1)

    @flyc.jit
    def launch_gemm(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        # Static SharedAllocator banks are sized by the compiler; leave launch smem unset.
        gemm_kernel(A, B, C).launch(grid=grid, block=block, stream=stream)

    return launch_gemm


__all__ = [
    "ATOM_K_I8",
    "DEFAULT_EPILOGUE",
    "DEFAULT_K_REP",
    "EPILOGUE_CHOICES",
    "EPILOGUE_I8",
    "EPILOGUE_I32",
    "I8_VPR",
    "compile_iluvatar_mr_igemm",
    "resolve_igemm_stages",
]
