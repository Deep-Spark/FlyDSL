# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar MR (ivcore11) tiledMma int8 GEMM (B8 GEMM).

Multi-warp single-buffer ``D = A @ B.T`` with int8 inputs and an int32
accumulator/output. Shares the SME G2S (``MRAsyncCpRow8b`` / ``MRAsyncCpCol``,
``make_sme_shared_layout``) and Ki S2R/MMA helpers with the f16 HGEMM; the int8
path uses a 16x16x32 MMA atom and 64-value SME rows (vs f16's K=16 / 32-value).

Entry point: ``compile_iluvatar_mr_igemm(M=..., N=..., K=..., ...)`` returns a
``@flyc.jit`` launch wrapper ``launch_gemm(A, B, C, stream=...)``.

``major_pattern`` — G2S global layout for A/B: ``nn``, ``tn``, ``nt`` (default),
``tt`` (two letters: A then B, ``n``=NoTrans/row SME, ``t``=Trans/col SME). The
kernel tensors are always logical ``A(m, k)``, ``B(n, k)``.

This is the stage-2 (correctness-first) kernel: single shared-memory buffer, no
software pipeline. The double-buffered pipeline lives in the f16 HGEMM and is a
follow-up for the int8 path.
"""

# NOTE: do NOT add ``from __future__ import annotations`` (Constexpr introspection).

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.ixdl as ixdl
from kernels.iluvatar_mr_common import (
    ATOM_M,
    ATOM_N,
    DEFAULT_MAJOR_PATTERN,
    DEFAULT_SMEM_CAP_BYTES,
    PATTERN_ID,
    SMEM_ROWS,
    WARP_SIZE,
    mr_swizzle_cta_shape,
    pattern_sme_atom_counts,
    sme_values_per_row,
)
from kernels.iluvatar_mr_epilogue import (
    mr_igemm_epilogue_store_i8_packed,
    mr_igemm_epilogue_store_i32,
)
from kernels.iluvatar_mr_operand_copy import mr_hgemm_g2s_issue_operands, mr_pattern_g2s_sme_config
from kernels.iluvatar_mr_s2r import mr_hgemm_s2r_load_ki

# int8 MMA atom K and SME row width (64 int8 per 512-bit SME row).
ATOM_K_I8 = 32
I8_VPR = sme_values_per_row(8)

DEFAULT_K_REP = 2  # CTA K-tile: ATOM_K_I8 * k_rep = 64 (one int8 SME brick width)
STAGES = 2
K_LOOP_UNROLL = 2

# Threadblock rasterization swizzle : group this many M-tiles per N 
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
# pattern except ``nt``) peak at a deeper 4-stage pipeline: the extra buffer hides
# the non-contiguous S2R latency.
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
    - large k_spanning grids (>= AUTO_STAGES_S4_BLOCK_THRESH CTAs, >= 4 K-tiles,
      4-stage SMEM within ``smem_cap``, pattern != ``nt``) => 4 stages
    - otherwise the proven 2-stage double buffer.
    """
    if stages is not None:
        return stages
    blocks = (M // bm) * (N // bn)
    ktiles = K // bk
    if blocks <= AUTO_STAGES_BLOCK_THRESH and ktiles >= 3:
        return 3
    k_spanning = major_pattern != "nt"
    if (
        k_spanning
        and blocks >= AUTO_STAGES_S4_BLOCK_THRESH
        and ktiles >= 4
        and 4 * (bm + bn) * bk <= smem_cap
    ):
        return 4
    return 2


def _sl_wait_count(g2s_cnt: int) -> int:
    """Pack an ivcore11 ``sl.waitcnt`` value (``union WaitCount``).
    Enable the G2S (global->shared) and LM (shared-mem) counters, wait until pending 
    G2S instructions are ``<= g2s_cnt`` and all shared-mem ops drained (LM_CNT=0).
    """
    return (1 << 3) | (1 << 2) | (int(g2s_cnt) << 23)

# Output epilogue: i32 direct store (D = A @ B.T, int32) or i8 packed CShuffle store
# (int8 output, pure saturating cast, no quant scale).
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
    # int8 = 1 byte/elem, ``stages``-buffered; returns pipeline (G2S staging) smem.
    return mr_swizzle_cta_shape(
        warps_m,
        warps_n,
        k_rep,
        warp_atoms_m=warp_atoms_m,
        warp_atoms_n=warp_atoms_n,
        atom_k=ATOM_K_I8,
        elem_bytes=1,
        stages=stages,
    )


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
    pattern_id = PATTERN_ID[major_pattern]
    warp_m = ATOM_M * warp_atoms_m
    warp_n = ATOM_N * warp_atoms_n
    bm = warp_m * warps_m
    bn = warp_n * warps_n
    bk = ATOM_K_I8 * k_rep
    num_warps = warps_m * warps_n
    threads = num_warps * WARP_SIZE
    grid_m = m // bm
    grid_n = n // bn
    swizzle_group_m = _block_swizzle_group_m(grid_m)

    assert k % bk == 0
    assert m % bm == 0 and n % bn == 0
    assert bk % I8_VPR == 0, f"bk={bk} must be a multiple of {I8_VPR} for int8 SME"

    a_atoms_total, b_atoms_total, _, _ = pattern_sme_atom_counts(
        pattern_id, bm, bn, bk, values_per_sme_row=I8_VPR
    )
    assert a_atoms_total % num_warps == 0, "A SME bricks must divide across warps"
    assert b_atoms_total % num_warps == 0, "B SME bricks must divide across warps"
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

        if fx.const_expr(pattern_id == 1 or pattern_id == 3):
            a_log_stride = (1, m)
        else:
            a_log_stride = (k, 1)
        a_logical = fx.make_view(fx.get_iter(A), fx.make_layout((m, k), a_log_stride))
        gA = fx.slice(fx.flat_divide(a_logical, (bm, bk)), (None, None, m_tile, None))

        if fx.const_expr(pattern_id == 0 or pattern_id == 1):
            b_log_stride = (1, n)
        else:
            b_log_stride = (k, 1)
        b_logical = fx.make_view(fx.get_iter(B), fx.make_layout((n, k), b_log_stride))
        gB = fx.slice(fx.flat_divide(b_logical, (bn, bk)), (None, None, n_tile, None))

        gC = fx.slice(fx.flat_divide(C, (bm, bn)), (None, None, m_tile, n_tile))

        smem_ptr = fx.get_dyn_shared()
        smem_base = fx.recast_iter(
            fx.PointerType.get(fx.Int8.ir_type, fx.AddressSpace.Shared),
            smem_ptr,
        )

        mma_atom = fx.make_mma_atom(ixdl.MRMma(ATOM_M, ATOM_N, ATOM_K_I8, fx.Int8, fx.Int8, fx.Int32))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (1, 1, 1)))
        thr_mma = tiled_mma.thr_slice(lane_id)

        gC_warp = fx.slice(
            fx.flat_divide(gC, (warp_m, warp_n)),
            (None, None, warp_m_id, warp_n_id),
        )
        gC_atoms = fx.flat_divide(gC_warp, (ATOM_M, ATOM_N))

        accs = []
        for im in fx.range_constexpr(warp_atoms_m):
            row = []
            for jn in fx.range_constexpr(warp_atoms_n):
                c_tile = fx.slice(gC_atoms, (None, None, im, jn))
                frag = thr_mma.make_fragment_C(c_tile)
                frag.fill(0)
                row.append(frag)
            accs.append(row)

        def _run_pipeline():
            g2s_sme = mr_pattern_g2s_sme_config(
                pattern_id,
                fx.Int8,
                row_atom=ixdl.MRAsyncCpRow8b,
                row_swizzle=ixdl.SMESwizzle.Row8b,
            )

            copy_atom_s2r_a = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int8)
            copy_atom_s2r_b = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int8)
            tiled_copy_a = fx.make_tiled_copy_A(copy_atom_s2r_a, tiled_mma)
            tiled_copy_b = fx.make_tiled_copy_B(copy_atom_s2r_b, tiled_mma)
            thr_copy_a = tiled_copy_a.get_slice(lane_id)
            thr_copy_b = tiled_copy_b.get_slice(lane_id)

            tile_smem = fx.make_tile(SMEM_ROWS, I8_VPR)
            tile_smem_A = (
                fx.make_tile(I8_VPR, SMEM_ROWS)
                if fx.const_expr(pattern_id == 1 or pattern_id == 3)
                else tile_smem
            )
            tile_smem_B = (
                fx.make_tile(I8_VPR, SMEM_ROWS)
                if fx.const_expr(pattern_id == 0 or pattern_id == 1)
                else tile_smem
            )

            if fx.const_expr(pattern_id == 1 or pattern_id == 3):
                a_leading = m
            else:
                a_leading = k
            if fx.const_expr(pattern_id == 0 or pattern_id == 1):
                b_leading = n
            else:
                b_leading = k

            def issue_stage(k_tile, stage_base, commit=True):
                k_A = gA[None, None, k_tile]
                k_B = gB[None, None, k_tile]
                sme_A = ixdl.make_sme_gmem_tensor(k_A, leading_stride=a_leading)
                sme_B = ixdl.make_sme_gmem_tensor(k_B, leading_stride=b_leading)
                mr_hgemm_g2s_issue_operands(
                    pattern_id=pattern_id,
                    warp_id=warp_id,
                    a_per_warp=a_per_warp,
                    b_per_warp=b_per_warp,
                    g_A_div=fx.zipped_divide(sme_A, tile_smem_A),
                    g_B_div=fx.zipped_divide(sme_B, tile_smem_B),
                    g2s_sme=g2s_sme,
                    smem_base=smem_base,
                    elem_dtype=fx.Int8,
                    bm=bm,
                    bn=bn,
                    bk=bk,
                    stage_base=stage_base,
                    values_per_sme_row=I8_VPR,
                    commit=commit,
                )

            def _ki_load(stage_base, ki):
                return mr_hgemm_s2r_load_ki(
                    pattern_id=pattern_id,
                    ki=ki,
                    stage_base=stage_base,
                    g2s_sme=g2s_sme,
                    smem_base=smem_base,
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
                    atom_k=ATOM_K_I8,
                    values_per_sme_row=I8_VPR,
                )

            def _mma_frags(a_frags, b_frags):
                for jn in fx.range_constexpr(warp_atoms_n):
                    for im in fx.range_constexpr(warp_atoms_m):
                        fx.gemm(mma_atom, accs[im][jn], a_frags[im], b_frags[jn], accs[im][jn])

            def _copy_frag(dst, src):
                dst.store(src.load())

            def _copy_a_frags(dst, src):
                for im in fx.range_constexpr(warp_atoms_m):
                    _copy_frag(dst[im], src[im])

            def _copy_b_frags(dst, src):
                for jn in fx.range_constexpr(warp_atoms_n):
                    _copy_frag(dst[jn], src[jn])

            def _s2r_mma_rest_defer_last_into(stage_base, a0, b0, a_def, b_def):
                # ki=0 fragments (a0/b0) were prefetched before the top-of-iter MMA
                # to overlap their (strided, for MN-major B) S2R latency; consume
                # them here, then stream the remaining Ki and defer the last.
                _mma_frags(a0, b0)
                for ki in fx.range_constexpr(1, k_rep - 1):
                    a_frags, b_frags = _ki_load(stage_base, ki)
                    _mma_frags(a_frags, b_frags)
                a_last, b_last = _ki_load(stage_base, k_rep - 1)
                _copy_a_frags(a_def, a_last)
                _copy_b_frags(b_def, b_last)

            def _s2r_mma_rest_defer_last(stage_base, a0, b0):
                # Returning variant of ``_s2r_mma_rest_defer_last_into``: consume the
                # prefetched ki=0 frags, stream ki 1..k_rep-2, and *return* the last
                # ki frags (register-resident) to defer their MMA to the next tile.
                _mma_frags(a0, b0)
                for ki in fx.range_constexpr(1, k_rep - 1):
                    a_frags, b_frags = _ki_load(stage_base, ki)
                    _mma_frags(a_frags, b_frags)
                return _ki_load(stage_base, k_rep - 1)

            def _s2r_mma_defer_last(stage_base):
                for ki in fx.range_constexpr(k_rep - 1):
                    a_frags, b_frags = _ki_load(stage_base, ki)
                    _mma_frags(a_frags, b_frags)
                return _ki_load(stage_base, k_rep - 1)

            def _s2r_mma_all(stage_base):
                a_frags, b_frags = _s2r_mma_defer_last(stage_base)
                _mma_frags(a_frags, b_frags)

            def _wait_stage():
                ixdl.cp_async_wait_group(0)

            def _sync_arrive(g2s_cnt):
                # SyncArrive: drain own G2S to <= g2s_cnt (and shared-mem to 0),
                # then signal the split pipeline barrier.
                ixdl.sl_waitmem(_sl_wait_count(g2s_cnt))
                ixdl.sl_pipebar_arrive(0)

            def _sync_wait():
                ixdl.sl_pipebar_wait(0)

            if fx.const_expr(use_multistage):
                # N-stage: keep stages-1 G2S tiles in flight via pipebar
                # (split barrier) + sl.waitcnt; NO full barrier here (the pipebar
                # protocol forbids mixing sl_barrier with pipebar reqs). Main loop is
                # constexpr-unrolled so the ki-deferred MMA fragments thread through
                # as registers (only used for small/latency-bound shapes).
                full_cnt = (stages - 2) * g2s_load_inst

                def _calc_blk(stage_base, first, last, a_def, b_def):
                    # Prefetch ki=0 first to overlap its (strided) S2R with the
                    # deferred MMA of the previous tile, then defer this tile's last.
                    a0, b0 = _ki_load(stage_base, 0)
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
                # Prologue prefetch + Ki-deferred S2R/MMA + pipelined K-loop.
                issue_stage(fx.Int32(0), fx.Int32(0))
                fx.gpu.barrier()
                _wait_stage()

                if k_tiles_const >= 2:
                    issue_stage(fx.Int32(1), fx.Int32(stage_stride))
                    fx.gpu.barrier()
                    _wait_stage()

                a_def, b_def = _s2r_mma_defer_last(fx.Int32(0))

                def _k_iter_body(k_idx):
                    fx.gpu.barrier()
                    # Prefetch the read-stage ki=0 S2R (strided for MN-major B) *before*
                    # the deferred MMA so its load latency overlaps that MMA, restoring
                    # double-buffer overlap. Stage bases live inside each parity branch
                    # (runtime `if`: values must not escape the branch).
                    k_tile = k_idx + 2
                    if k_idx % 2 == 0:
                        a0, b0 = _ki_load(fx.Int32(stage_stride), 0)
                        _mma_frags(a_def, b_def)
                        issue_stage(fx.Int32(k_tile), fx.Int32(0))
                        _s2r_mma_rest_defer_last_into(fx.Int32(stage_stride), a0, b0, a_def, b_def)
                    else:
                        a0, b0 = _ki_load(fx.Int32(0), 0)
                        _mma_frags(a_def, b_def)
                        issue_stage(fx.Int32(k_tile), fx.Int32(stage_stride))
                        _s2r_mma_rest_defer_last_into(fx.Int32(0), a0, b0, a_def, b_def)

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
                tiled_mma=tiled_mma,
                warp_atoms_m=warp_atoms_m,
                warp_atoms_n=warp_atoms_n,
                c_global_n=n,
            )
        else:
            mr_igemm_epilogue_store_i32(
                lane_id=lane_id,
                accs=accs,
                gC_warp=gC_warp,
                tiled_mma=tiled_mma,
                warp_atoms_m=warp_atoms_m,
                warp_atoms_n=warp_atoms_n,
            )

    smem_bytes = max(stage_elems * stages, bm * bn) if epilogue == EPILOGUE_I8 else stage_elems * stages
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
    output: ``i32`` (default; direct int32 store) or ``i8`` (packed CShuffle store,
    pure saturating cast, int8 ``C``). See the module docstring for ``major_pattern``
    and CTA shape semantics.
    """
    if major_pattern not in PATTERN_ID:
        raise ValueError(f"unknown major pattern: {major_pattern}")
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
    smem_bytes = max(pipeline_smem, bm * bn) if epilogue == EPILOGUE_I8 else pipeline_smem
    if K % bk:
        raise ValueError(f"K must be a multiple of {bk} (ATOM_K_I8 * k_rep)")
    if M % bm or N % bn:
        raise ValueError(f"M,N must be multiples of {bm}/{bn}")
    if bk % I8_VPR:
        raise ValueError(f"BK={bk} must be a multiple of {I8_VPR}; use even k_rep")
    num_warps = warps_m * warps_n
    a_atoms_total, b_atoms_total, _, _ = pattern_sme_atom_counts(
        PATTERN_ID[major_pattern], bm, bn, bk, values_per_sme_row=I8_VPR
    )
    if a_atoms_total % num_warps or b_atoms_total % num_warps:
        raise ValueError(
            f"SME brick count must divide evenly across {warps_m}x{warps_n} warps; "
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
        gemm_kernel(A, B, C).launch(grid=grid, block=block, smem=smem_bytes, stream=stream)

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
