# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar MR (ivcore11) tiledMma pipeline int8 GEMM (B8 GEMM).

Multi-warp double-buffered ``D = A @ B.T`` with int8 inputs. Shares the SME
G2S / S2R / MMA helpers with the f16 MR HGEMM (``mr_g2s_sme_config``,
``mr_gemm_g2s_issue_a_warp`` / ``mr_gemm_g2s_issue_b_warp``, ``mr_gemm_s2r_load_mma_k``);
the int8 path uses
a 16x16x32 MMA atom (``ATOM_K_B8``) and 64-value SME rows
(``MrOperandGeom.b8()``) vs f16's K=16 / 32-value rows.

Entry point: ``compile_iluvatar_mr_igemm(M=..., N=..., K=..., ...)`` returns a
``@flyc.jit`` launch wrapper. Plain epilogues use ``launch(A, B, C)``; scaled
W8A8 epilogues use ``launch(A, B, scale_a, scale_b, C[, bias])``.

``major_pattern`` -- CUTLASS BLAS layout tag on logical ``A(m,k)`` / ``B(n,k)``
(see ``kernels.gemm.iluvatar.common.GemmLayout``). ``tn`` (both operands k-major) is
the default fast path; the mn-major patterns (``nn`` / ``nt`` / ``tt``) use i8
k-spanning S2R, where one MMA K atom (32) spans two SME bricks (K=16 each).

Output epilogue:
  * ``i32`` (default) -- direct int32 store
  * ``i8`` -- packed store, truncating cast, no quant scale
  * ``scaled_bf16`` / ``scaled_fp16`` --
    ``D = acc * scale_a[m] * scale_b[n] [+ bias[n]]`` via PackSlb b16 store
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.ixdl as ixdl
from flydsl.compiler.ast_rewriter import ASTRewriter
from kernels.gemm.iluvatar.common import (
    DEFAULT_MAJOR_PATTERN,
    WARP_SIZE,
    parse_major_pattern,
)
from kernels.gemm.iluvatar.epilogue import (
    B16_PACK_SLB_BYTES_PER_WARP,
    mr_igemm_epilogue_store_i8_packed,
    mr_igemm_epilogue_store_i32,
    mr_igemm_epilogue_store_scaled,
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
from kernels.gemm.iluvatar.mr.operand_copy import (
    mr_g2s_sme_config,
    mr_gemm_g2s_issue_a_warp,
    mr_gemm_g2s_issue_b_warp,
)
from kernels.gemm.iluvatar.mr.s2r import (
    mr_gemm_s2r_a_tile,
    mr_gemm_s2r_b_tile,
    mr_gemm_s2r_copy_a,
    mr_gemm_s2r_copy_b,
    mr_gemm_s2r_load_mma_k,
)

# int8 operand geometry: 16x16x32 MMA atom, 64 int8 per 512-bit SME row.
MR_IGEMM_GEOM = MrOperandGeom.b8()
ATOM_K_I8 = ATOM_K_B8
I8_VPR = MR_IGEMM_GEOM.values_per_sme_row

DEFAULT_K_REP = 2  # CTA K-tile: ATOM_K_B8 * k_rep = 64 (one int8 SME row width)
STAGES = 2
K_LOOP_UNROLL = 2
# Multistage pipebar main: full ``range_constexpr`` only for small trip counts.
# Large mains use ``scf.for`` so IR/compile stay bounded; recover issue density
# with the partial unroll below.
MULTISTAGE_CONSTEXPR_MAIN_MAX = 16
# Partial unroll for the scf multistage main. Prefer a multiple of common stage
# depths (2/4) so ``i % stages`` folds per-lane in the unrolled body.
MULTISTAGE_K_LOOP_UNROLL = 4

# Threadblock rasterization swizzle: group this many M-tiles per N
# column in launch order to raise L2 reuse on large GEMMs.
# Effective group is the largest power-of-2 divisor of grid_m that is <= this.
# 256x256 launches M on grid.x so consecutive CTAs reuse one N-slice of B
# (prefill with several M-tiles). 64x128 / 128x256 stay N-on-X (fat X).
BLOCK_SWIZZLE_GROUP_M = 4


def _block_swizzle_group_m(grid_m: int, cap: int = BLOCK_SWIZZLE_GROUP_M) -> int:
    g = 1
    while g * 2 <= cap and grid_m % (g * 2) == 0:
        g *= 2
    return g


def _dynamic_grid(grid_m, grid_n):
    """Launch grid for dynamic-M IGEMM.

    One M-tile -> (grid_n, 1) so X stays fat (64x128 decode). Several M-tiles
    -> (grid_m, grid_n) so consecutive CTAs reuse one N-slice of B (prefill).
    """
    one = fx.Int32(1)
    gn = fx.Int32(grid_n)
    is_single = fx.arith.cmpi(fx.arith.CmpIPredicate.eq, grid_m, one)
    gx = fx.Int32(fx.arith.select(is_single, gn, grid_m))
    gy = fx.Int32(fx.arith.select(is_single, one, gn))
    return (gx, gy, 1)


# Multi-stage wins on latency-bound, small
# grids; large grids are compute/occupancy-bound and prefer the 65KB 2-stage
# double buffer -- except deep S4 when S2R is k-spanning, or i32 square/short-K.
# Empirical crossover on ivcore11: <=16 CTAs -> 3-stage.
AUTO_STAGES_BLOCK_THRESH = 16
# Large grids (>= this many CTAs, >= 4 K-tiles, 4-stage SMEM within cap):
#   - mn-major / k-spanning operands -> S4 (hides non-contiguous S2R)
#   - i32 pure k-major (tn): S4 only when K <= min(M, N) (square/short-K);
#     tall-K (including K == max(M,N) on rectangular shapes) stays on S2.
#     ``max(M,N)`` wrongly pushed those tall cases onto S4.
#   - i8 pure k-major (tn) -> keep S2 (deeper pipe costs occupancy)
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
    epilogue: str | None = None,
) -> int:
    """Resolve the SMEM pipeline depth. ``stages=None`` => auto:

    - small/latency-bound grids (<= AUTO_STAGES_BLOCK_THRESH CTAs, >= 3 K-tiles) => 3 stages
    - large grids (>= AUTO_STAGES_S4_BLOCK_THRESH CTAs, >= 4 K-tiles, 4-stage
      SMEM within ``smem_cap``): S4 when an operand is mn-major (k-spanning S2R),
      or when ``epilogue`` is i32 **and** ``K <= min(M, N)`` (square/short-K i32
      k-major). i8 pure k-major and tall-K i32 k-major stay on S2.
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
    if blocks >= AUTO_STAGES_S4_BLOCK_THRESH and ktiles >= 4 and 4 * (bm + bn) * bk <= smem_cap:
        # i8 k-major: deeper pipe costs occupancy / issue for little latency hide.
        if epilogue == EPILOGUE_I8 and not k_spanning:
            return 2
        if k_spanning:
            return 4
        # i32 pure k-major: S4 helps square/short-K; tall K prefers S2 occupancy.
        # min(M, N) keeps K==max(M,N) rectangular tall cases on S2 (max() did not).
        if epilogue in (EPILOGUE_I32, EPILOGUE_SCALED_BF16, EPILOGUE_SCALED_FP16) and K <= min(M, N):
            return 4
    return 2


# Output epilogue:
#   i32 / i8 -- raw stores; scaled_* -- dequant (+ optional bias) to f16/bf16
#   (acc * scale_a[m] * scale_b[n] [+ bias[n]]).
EPILOGUE_I32 = "i32"
EPILOGUE_I8 = "i8"
EPILOGUE_SCALED_BF16 = "scaled_bf16"
EPILOGUE_SCALED_FP16 = "scaled_fp16"
EPILOGUE_CHOICES = (
    EPILOGUE_I32,
    EPILOGUE_I8,
    EPILOGUE_SCALED_BF16,
    EPILOGUE_SCALED_FP16,
)
DEFAULT_EPILOGUE = EPILOGUE_I32

_SCALED_EPILOGUES = (EPILOGUE_SCALED_BF16, EPILOGUE_SCALED_FP16)
_SCALED_OUT_DTYPE = {
    EPILOGUE_SCALED_BF16: fx.BFloat16,
    EPILOGUE_SCALED_FP16: fx.Float16,
}


def _is_scaled_epilogue(epilogue: str) -> bool:
    return epilogue in _SCALED_EPILOGUES


def _scaled_out_dtype(epilogue: str):
    try:
        return _SCALED_OUT_DTYPE[epilogue]
    except KeyError as e:
        raise ValueError(f"not a scaled epilogue: {epilogue}") from e


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
    apply_bias: bool = False,
    m_valid: int | None = None,
    dynamic_m: bool = False,
):
    layout = parse_major_pattern(major_pattern)
    a_mn_major = layout.a_mn_major
    b_mn_major = layout.b_mn_major
    scaled = _is_scaled_epilogue(epilogue)
    # ``m`` is the CTA-grid M extent (ceil to BM). ``m_valid`` is the live row
    # count written by scaled stores (short-M / dynamic-M edge).
    # Under ``dynamic_m`` the row count arrives as a kernel argument instead and
    # ``m`` only bounds the A/C tile grid, so the compile key drops M.
    if dynamic_m:
        if not scaled:
            raise ValueError("dynamic_m requires a scaled_* epilogue")
        if m_valid is not None:
            raise ValueError("dynamic_m takes the row count at launch; leave m_valid unset")
    else:
        if m_valid is None:
            m_valid = m
        if not (1 <= m_valid <= m):
            raise ValueError(f"m_valid={m_valid} must be in [1, m={m}]")
    out_dtype_fx = _scaled_out_dtype(epilogue) if scaled else None
    if scaled and apply_bias and out_dtype_fx is None:
        raise AssertionError("unreachable")
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
    # 64x128 / 128x256 serving grids are a single M-tile: put N on X (fat grid).
    # 256x256 prefill has several M-tiles; M-on-X plus the swizzle keeps one
    # N-slice of B live across them. A single 256x256 M-tile still launches
    # (grid_n, 1) via _dynamic_grid.
    n_major_grid = bm < 256
    if n_major_grid:
        swizzle_group_m = 1
    elif dynamic_m:
        # grid_m is unknown until launch, so keep the nominal group and let the
        # kernel check divisibility at runtime.
        swizzle_group_m = BLOCK_SWIZZLE_GROUP_M
    else:
        swizzle_group_m = _block_swizzle_group_m(grid_m)
        # Block swizzle helps multi-wave L2 reuse; on single-wave grids (<=16 CTAs
        # on ivcore11) it only adds prologue div/mod and does not improve L2.
        if grid_m * grid_n <= AUTO_STAGES_BLOCK_THRESH:
            swizzle_group_m = 1

    assert k % bk == 0
    assert m % bm == 0 and n % bn == 0
    assert bk % vpr == 0, f"bk={bk} must be a multiple of {vpr} for int8 SME"
    # CTA presets keep warp_atoms_m even (>=2); the S2R/MMA loop no longer
    # splits on that, but the tile table and PackSlb path still assume it.
    assert warp_atoms_m >= 2 and warp_atoms_m % 2 == 0, f"warp_atoms_m={warp_atoms_m} must be even and >= 2"

    a_atoms_total, b_atoms_total, _, _ = sme_atom_counts(layout, bm, bn, bk, values_per_sme_row=vpr)
    # Chunk counts need neither divide nor reach the warp count: surplus warps
    # run one extra iteration that _clamp_chunk keeps addressable and _chunk_guard
    # predicates off. Passing None when they do divide keeps that path free of
    # the clamp entirely.
    a_per_warp = -(-a_atoms_total // num_warps)
    b_per_warp = -(-b_atoms_total // num_warps)
    a_atoms_arg = None if a_atoms_total % num_warps == 0 else a_atoms_total
    b_atoms_arg = None if b_atoms_total % num_warps == 0 else b_atoms_total
    k_tiles_const = k // bk
    stage_elems = (bm + bn) * bk
    stage_stride = stage_elems
    main_k_trip = max(0, k_tiles_const - 2)
    main_k_full = (main_k_trip // K_LOOP_UNROLL) * K_LOOP_UNROLL
    main_k_remainder = main_k_trip - main_k_full
    # pipebar + sl.waitcnt for STAGE>=2 (CTA barrier peel is the legacy
    # fallback when stages==1 or k_tiles < stages).
    use_multistage = stages >= 2 and k_tiles_const >= stages
    g2s_load_inst = a_per_warp + b_per_warp
    # Sized here (not inside the body): after ASTRewriter.transform, bare Python
    # ``if`` on host constants can be rewritten to scf.if and ``smem_elems``
    # would no longer be a Python local for the @fx.struct annotation.
    pack_slb_bytes = num_warps * B16_PACK_SLB_BYTES_PER_WARP
    if epilogue == EPILOGUE_I8 and not use_pack_only:
        smem_elems = max(stage_elems * stages, bm * bn)
    elif scaled:
        smem_elems = max(stage_elems * stages, pack_slb_bytes)
    else:
        smem_elems = stage_elems * stages

    def _igemm_body(A, B, C, scale_a, scale_b, bias, m_rows=None):
        tid = fx.thread_idx.x
        bid_x, bid_y, _ = fx.block_idx
        # Tile mapping. n_major_grid (bm < 256): N on X, M on Y.
        # Else M on X (256x256 prefill); dynamic-M may still launch a single
        # M-tile as (grid_n, 1) so X stays fat.
        if fx.const_expr(n_major_grid):
            m_tile = bid_y
            n_tile = bid_x
        elif fx.const_expr(dynamic_m):
            # grid_m is per-launch. A single M-tile is launched as (grid_n, 1)
            # so X stays fat; otherwise (grid_m, grid_n) with the M-group
            # swizzle so consecutive CTAs reuse one N-slice of B.
            gx = fx.Int32(fx.grid_dim.x)
            gy = fx.Int32(fx.grid_dim.y)
            one = fx.Int32(1)
            n_on_x = fx.arith.andi(
                fx.arith.cmpi(fx.arith.CmpIPredicate.eq, gy, one),
                fx.arith.cmpi(fx.arith.CmpIPredicate.eq, gx, fx.Int32(grid_n)),
            )
            group = fx.Int32(swizzle_group_m)
            num_in_group = group * fx.Int32(grid_n)
            pid = bid_x + bid_y * gx
            group_id = pid // num_in_group
            pid_in_group = pid % num_in_group
            swz_m = group_id * group + (pid_in_group % group)
            swz_n = pid_in_group // group
            aligned = fx.arith.cmpi(fx.arith.CmpIPredicate.eq, gx % group, fx.Int32(0))
            m_tile_m = fx.Int32(fx.arith.select(aligned, swz_m, bid_x))
            n_tile_m = fx.Int32(fx.arith.select(aligned, swz_n, bid_y))
            m_tile = fx.Int32(fx.arith.select(n_on_x, fx.Int32(0), m_tile_m))
            n_tile = fx.Int32(fx.arith.select(n_on_x, bid_x, n_tile_m))
        elif fx.const_expr(swizzle_group_m > 1):
            pid = bid_x + bid_y * fx.Int32(grid_m)
            num_in_group = swizzle_group_m * grid_n
            group_id = pid // fx.Int32(num_in_group)
            pid_in_group = pid % fx.Int32(num_in_group)
            m_tile = group_id * fx.Int32(swizzle_group_m) + (pid_in_group % fx.Int32(swizzle_group_m))
            n_tile = pid_in_group // fx.Int32(swizzle_group_m)
        else:
            m_tile = bid_x
            n_tile = bid_y
        warp_id = tid.shrui(fx.Int32(6))  # WARP_SIZE==64; avoid signed floordivsi
        # Prefer hardware lane_id over tid%64 so Row S2R keeps a natural
        # base+4*lane form (lane&63 never lowers to hardware lane id).
        lane_id = fx.Int32(fx.lane_id)
        if fx.const_expr((warps_n & (warps_n - 1)) == 0):
            warp_n_id = warp_id & fx.Int32(warps_n - 1)
            _wn_bits = int(warps_n).bit_length() - 1
            warp_m_id = warp_id.shrui(fx.Int32(_wn_bits))
        else:
            warp_m_id = warp_id // warps_n
            warp_n_id = warp_id % warps_n

        if fx.const_expr(a_mn_major):
            a_logical_stride = (1, m)
        else:
            a_logical_stride = (k, 1)
        a_logical = fx.make_view(fx.get_iter(A), fx.make_layout((m, k), a_logical_stride))
        gA = fx.slice(fx.flat_divide(a_logical, (bm, bk)), (None, None, m_tile, None))

        # G2S skips A chunks that start past the row count, so A only has to be
        # allocated for the real rows rather than padded up to the CTA grid. A
        # chunk straddling the boundary still reads its full SMEM_ROWS rows.
        if fx.const_expr(a_mn_major):
            g2s_m_valid = None
        elif fx.const_expr(dynamic_m):
            g2s_m_valid = m_rows
        elif fx.const_expr(m_valid < m):
            g2s_m_valid = fx.Int32(m_valid)
        else:
            g2s_m_valid = None

        if fx.const_expr(b_mn_major):
            b_logical_stride = (1, n)
        else:
            b_logical_stride = (k, 1)
        b_logical = fx.make_view(fx.get_iter(B), fx.make_layout((n, k), b_logical_stride))
        gB = fx.slice(fx.flat_divide(b_logical, (bn, bk)), (None, None, n_tile, None))

        gC = fx.slice(fx.flat_divide(C, (bm, bn)), (None, None, m_tile, n_tile))

        # Static contiguous shared memory (matches hgemm): the compiler sizes the
        # bank, so launch(smem=...) stays unset. Size is captured from the outer
        # ``smem_elems`` (see note above the body).
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

            # Wrap the full K-stack once so the SME descriptor is loop-invariant
            # and each k_tile only advances the fat-ptr byte_offset (gOffset).
            # Building the descriptor on a per-tile slice would bake K into the
            # 64-bit base and rebuild the vector<4xi32> every issue.
            sme_gA = ixdl.make_sme_gmem_tensor(gA, leading_stride=a_leading)
            # PackOnly N_SWIZZLE: Col desc walks N with stride b_leading * swizzle
            # so one load covers every swizzle-th N row (matches issue_b_warp).
            sme_gB = ixdl.make_sme_gmem_tensor(gB, leading_stride=b_leading * b_n_swizzle)

            # Loop-invariant G2S row base. Recomputing it inside issue_stage
            # kept a signed mul live across every K-tile.
            a_row_base = m_tile * fx.Int32(bm)

            def issue_stage(k_tile, stage_base, commit=True):
                k_A = sme_gA[None, None, k_tile]
                k_B = sme_gB[None, None, k_tile]
                smem_a, smem_b = mr_stage_smem_ab(smem_base, stage_base, bm * bk)
                if fx.const_expr(b_n_swizzle > 1):
                    b_cta_view = k_B
                else:
                    b_cta_view = fx.zipped_divide(k_B, tile_smem_B)
                a_cta_view = fx.zipped_divide(k_A, tile_smem_A)

                def issue_a():
                    mr_gemm_g2s_issue_a_warp(
                        a_mn_major=a_mn_major,
                        b_mn_major=b_mn_major,
                        warp_id=warp_id,
                        a_per_warp=a_per_warp,
                        a_cta_gmem_view=a_cta_view,
                        g2s_sme=g2s_sme,
                        smem_a=smem_a,
                        elem_dtype=fx.Int8,
                        bm=bm,
                        bn=bn,
                        bk=bk,
                        geom=geom,
                        a_atoms_total=a_atoms_arg,
                        a_row_base=a_row_base,
                        m_valid=g2s_m_valid,
                        num_warps=num_warps,
                    )

                def issue_b():
                    mr_gemm_g2s_issue_b_warp(
                        a_mn_major=a_mn_major,
                        b_mn_major=b_mn_major,
                        warp_id=warp_id,
                        b_per_warp=b_per_warp,
                        b_cta_gmem_view=b_cta_view,
                        g2s_sme=g2s_sme,
                        smem_b=smem_b,
                        elem_dtype=fx.Int8,
                        bm=bm,
                        bn=bn,
                        b_n_swizzle=b_n_swizzle,
                        b_leading=b_leading,
                        bk=bk,
                        geom=geom,
                        b_atoms_total=b_atoms_arg,
                        num_warps=num_warps,
                    )

                # TT/NT: B is N-contiguous (the larger burst). Issue it first so
                # it is in flight while A copies issue. k-major B keeps A then B.
                if fx.const_expr(b_mn_major):
                    issue_b()
                    issue_a()
                else:
                    issue_a()
                    issue_b()
                if fx.const_expr(commit):
                    ixdl.cp_async_commit_group()

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

            def _a_tile(stage_base, mma_k, mma_m):
                smem_a, _smem_b = mr_stage_smem_ab(smem_base, stage_base, bm * bk)
                return mr_gemm_s2r_a_tile(
                    a_mn_major=a_mn_major,
                    b_mn_major=b_mn_major,
                    mma_m=mma_m,
                    mma_k=mma_k,
                    g2s_sme=g2s_sme,
                    smem_a=smem_a,
                    elem_dtype=fx.Int8,
                    warp_m_id=warp_m_id,
                    warp_atoms_m=warp_atoms_m,
                    bm=bm,
                    bn=bn,
                    bk=bk,
                    geom=geom,
                )

            def _b_tile(stage_base, mma_k, mma_n):
                _smem_a, smem_b = mr_stage_smem_ab(smem_base, stage_base, bm * bk)
                return mr_gemm_s2r_b_tile(
                    a_mn_major=a_mn_major,
                    b_mn_major=b_mn_major,
                    mma_n=mma_n,
                    mma_k=mma_k,
                    g2s_sme=g2s_sme,
                    smem_b=smem_b,
                    elem_dtype=fx.Int8,
                    warp_n_id=warp_n_id,
                    warp_atoms_n=warp_atoms_n,
                    bm=bm,
                    bn=bn,
                    bk=bk,
                    geom=geom,
                )

            def _load_a_frag(stage_base, mma_k, mma_m):
                return mr_gemm_s2r_copy_a(
                    copy_atom=copy_atom_s2r_a,
                    thr_copy_a=thr_copy_a,
                    thr_mma=thr_mma,
                    smem_a_tile=_a_tile(stage_base, mma_k, mma_m),
                )

            def _load_b_frag(stage_base, mma_k, mma_n):
                return mr_gemm_s2r_copy_b(
                    copy_atom=copy_atom_s2r_b,
                    thr_copy_b=thr_copy_b,
                    thr_mma=thr_mma,
                    smem_b_tile=_b_tile(stage_base, mma_k, mma_n),
                )

            def _s2r_a_into(stage_base, mma_k, mma_m, dst):
                tile = _a_tile(stage_base, mma_k, mma_m)
                fx.copy(
                    copy_atom_s2r_a,
                    thr_copy_a.partition_S(tile),
                    thr_copy_a.retile(dst),
                    pred=None,
                )

            def _s2r_b_into(stage_base, mma_k, mma_n, dst):
                tile = _b_tile(stage_base, mma_k, mma_n)
                fx.copy(
                    copy_atom_s2r_b,
                    thr_copy_b.partition_S(tile),
                    thr_copy_b.retile(dst),
                    pred=None,
                )

            def _mma_frags(a_frags, b_frags):
                for mma_m in fx.range_constexpr(warp_atoms_m):
                    for mma_n in fx.range_constexpr(warp_atoms_n):
                        fx.gemm(
                            mma_atom,
                            accs[mma_m][mma_n],
                            a_frags[mma_m],
                            b_frags[mma_n],
                            accs[mma_m][mma_n],
                        )

            def _s2r_mma_k(stage_base, mma_k):
                # Load all A, then software-pipeline B along N: prefetch B[n+1]
                # before MMA of B[n] so the wider operand's S2R hides behind MMA.
                a_frags = []
                for mma_m in fx.range_constexpr(warp_atoms_m):
                    a_frags.append(_load_a_frag(stage_base, mma_k, mma_m))
                b_frag = _load_b_frag(stage_base, mma_k, 0)
                for mma_n in fx.range_constexpr(warp_atoms_n):
                    if fx.const_expr(mma_n + 1 < warp_atoms_n):
                        b_next = _load_b_frag(stage_base, mma_k, mma_n + 1)
                    for mma_m in fx.range_constexpr(warp_atoms_m):
                        fx.gemm(
                            mma_atom,
                            accs[mma_m][mma_n],
                            a_frags[mma_m],
                            b_frag,
                            accs[mma_m][mma_n],
                        )
                    if fx.const_expr(mma_n + 1 < warp_atoms_n):
                        b_frag = b_next

            def _s2r_last_into(stage_base, a_def, b_def):
                last_k = k_rep - 1
                for mma_m in fx.range_constexpr(warp_atoms_m):
                    _s2r_a_into(stage_base, last_k, mma_m, a_def[mma_m])
                for mma_n in fx.range_constexpr(warp_atoms_n):
                    _s2r_b_into(stage_base, last_k, mma_n, b_def[mma_n])

            def _s2r_mma_defer_last_into(stage_base, a_def, b_def):
                for mma_k in fx.range_constexpr(k_rep - 1):
                    _s2r_mma_k(stage_base, mma_k)
                _s2r_last_into(stage_base, a_def, b_def)

            def _s2r_mma_defer_last(stage_base):
                for mma_k in fx.range_constexpr(k_rep - 1):
                    _s2r_mma_k(stage_base, mma_k)
                return _mma_k_load(stage_base, k_rep - 1)

            def _s2r_mma_all(stage_base):
                for mma_k in fx.range_constexpr(k_rep):
                    _s2r_mma_k(stage_base, mma_k)

            def _mma_prev(a_def, b_def):
                # Drain last-k S2R issued before the previous arrive. That load
                # stays in flight across wait + G2S; MMA of those frags needs LM=0.
                ixdl.sl_waitmem(lm=0)
                _mma_frags(a_def, b_def)

            def _sync_arrive(g2s_cnt):
                # Drain G2S only. Last-k S2R stays in flight so it can overlap
                # the next wait + G2S; _mma_prev waits LM before using those frags.
                ixdl.sl_waitmem(g2s=g2s_cnt)
                ixdl.sl_pipebar_arrive(0)

            def _sync_wait():
                ixdl.sl_pipebar_wait(0)

            if fx.const_expr(use_multistage):
                # N-stage: keep stages-1 G2S tiles in flight via pipebar
                # (split barrier) + sl.waitcnt; NO full barrier here (the pipebar
                # protocol forbids mixing sl_barrier with pipebar reqs).
                # Prologue/epilogue stay constexpr (trip = stages-1, tiny).
                # Main loop: constexpr only when main_count is small; large-K uses
                # scf.for so IR/compile stay bounded (deferred frags mutate in place,
                # same as the 2-stage barrier path).
                full_cnt = (stages - 2) * g2s_load_inst

                for s in fx.range_constexpr(stages - 1):
                    issue_stage(fx.Int32(s), fx.Int32(s * stage_stride), commit=False)
                _sync_arrive(full_cnt)

                main_count = k_tiles_const - stages + 1
                # First tile: allocate deferred frags; later tiles reuse via ``into``.
                _sync_wait()
                issue_stage(
                    fx.Int32(stages - 1),
                    fx.Int32(((stages - 1) % stages) * stage_stride),
                    commit=False,
                )
                a_def, b_def = _s2r_mma_defer_last(fx.Int32(0))
                _sync_arrive(full_cnt)

                def _multistage_main_body(i, load_slot, comp_slot):
                    # load_slot / comp_slot are Python ints (the ring index).
                    # Passing them in avoids ``i % stages`` as a signed rem in
                    # the scf.for body -- that rem does not fold from the
                    # induction variable, even when the unroll is a multiple
                    # of stages.
                    _sync_wait()
                    nxt = i + (stages - 1)
                    issue_stage(fx.Int32(nxt), fx.Int32(load_slot * stage_stride), commit=False)
                    _mma_prev(a_def, b_def)
                    _s2r_mma_defer_last_into(fx.Int32(comp_slot * stage_stride), a_def, b_def)
                    _sync_arrive(full_cnt)

                # i runs [1, main_count); tile 0 was peeled above.
                rest = max(0, main_count - 1)
                if fx.const_expr(main_count <= MULTISTAGE_CONSTEXPR_MAIN_MAX):
                    for i in fx.range_constexpr(1, main_count):
                        _multistage_main_body(i, (i + stages - 1) % stages, i % stages)
                elif fx.const_expr(rest > 0):
                    # Real scf.for (body is AST-rewritten below). Prefer
                    # MULTISTAGE_K_LOOP_UNROLL so stage-phase patterns stay
                    # visible without full-unrolling a long main trip count.
                    # Unroll must be a multiple of ``stages`` so the ring slot
                    # is a function of the constexpr lane ``u`` only
                    # (start=1, step=u_factor):
                    #   i % stages == (1 + u) % stages
                    #   (i + stages - 1) % stages == u % stages
                    u_factor = MULTISTAGE_K_LOOP_UNROLL
                    if u_factor % stages != 0:
                        u_factor = stages
                    rest_full = (rest // u_factor) * u_factor
                    rest_rem = rest - rest_full
                    if fx.const_expr(rest_full > 0):
                        for i_base in fx.range(1, 1 + rest_full, u_factor):
                            for u in fx.range_constexpr(u_factor):
                                _multistage_main_body(
                                    i_base + u,
                                    u % stages,
                                    (1 + u) % stages,
                                )
                    if fx.const_expr(rest_rem > 0):
                        for u in fx.range_constexpr(rest_rem):
                            i = 1 + rest_full + u
                            _multistage_main_body(i, (i + stages - 1) % stages, i % stages)

                for j in fx.range_constexpr(stages - 1):
                    _sync_wait()
                    t = main_count + j
                    _mma_prev(a_def, b_def)
                    if fx.const_expr(j == stages - 2):
                        _s2r_mma_all(fx.Int32((t % stages) * stage_stride))
                    else:
                        _s2r_mma_defer_last_into(fx.Int32((t % stages) * stage_stride), a_def, b_def)
                    if fx.const_expr(j < stages - 2):
                        _sync_arrive((stages - 2 - (j + 1)) * g2s_load_inst)
            else:
                # Prologue (match hgemm double-buffer peel):
                #   issue0 -> barrier (IXDL drains g2scnt) -> issue1 (no wait) ->
                #   peel stage0 so S2R/MMA overlaps tile1 G2S.
                issue_stage(fx.Int32(0), fx.Int32(0))
                fx.gpu.barrier()

                if fx.const_expr(k_tiles_const >= 2):
                    issue_stage(fx.Int32(1), fx.Int32(stage_stride))

                a_def, b_def = _s2r_mma_defer_last(fx.Int32(0))

                def _k_iter_body(k_idx, k_parity):
                    # k_parity is a Python 0/1. The scf.for starts at 0 with
                    # step K_LOOP_UNROLL (even), so k_idx % 2 == u % 2.
                    fx.gpu.barrier()
                    k_tile = fx.Int32(k_idx) + 2
                    load_stage_base = fx.Int32(k_parity * stage_stride)
                    comp_stage_base = load_stage_base ^ fx.Int32(stage_stride)
                    issue_stage(k_tile, load_stage_base)
                    _mma_frags(a_def, b_def)
                    _s2r_mma_defer_last_into(comp_stage_base, a_def, b_def)

                if fx.const_expr(main_k_full > 0):
                    for k_base in fx.range(0, main_k_full, K_LOOP_UNROLL):
                        for u in fx.range_constexpr(K_LOOP_UNROLL):
                            _k_iter_body(k_base + u, u % 2)

                if fx.const_expr(main_k_remainder > 0):
                    for u in fx.range_constexpr(main_k_remainder):
                        _k_iter_body(main_k_full + u, u % 2)

                fx.gpu.barrier()
                if fx.const_expr(k_tiles_const >= 2):
                    last_base = fx.Int32(stage_stride) if fx.const_expr(main_k_trip % 2 == 0) else fx.Int32(0)
                    _mma_frags(a_def, b_def)
                    _s2r_mma_all(last_base)
                else:
                    _mma_frags(a_def, b_def)

        _run_pipeline()

        # PackSlb reuses a pipeline stage the final S2R is not reading, so the
        # CTA barrier between MMA and pack can drop. Last S2R consumed
        # (K_tiles - 1) % stages; the next ring slot is free. Depth 2 is the
        # (k_tiles % 2) case; the same ring rule covers depth 3/4.
        _last_s2r_stage = (k_tiles_const - 1) % stages
        _free_pack_stage = (_last_s2r_stage + 1) % stages
        _free_pack_base = smem_base
        if fx.const_expr(_free_pack_stage != 0):
            _free_pack_base = fx.add_offset(smem_base, fx.make_int_tuple(fx.Int32(_free_pack_stage * stage_stride)))

        if fx.const_expr(epilogue == EPILOGUE_I8):
            # PackOnly (N_SWIZZLE path): no SLB. PackSlb: reuse a free pipeline
            # stage when one stage holds WarpCnt*1024 B.
            epi_smem = smem_base
            skip_bar = False
            if fx.const_expr(not use_pack_only and stages >= 2 and stage_elems >= num_warps * 1024):
                skip_bar = True
                epi_smem = _free_pack_base
            mr_igemm_epilogue_store_i8_packed(
                lane_id=lane_id,
                warp_id=warp_id,
                accs=accs,
                gC_warp=gC_warp,
                smem_base=epi_smem,
                warp_atoms_m=warp_atoms_m,
                warp_atoms_n=warp_atoms_n,
                c_global_n=n,
                skip_cta_barrier=skip_bar,
                pack_only=use_pack_only,
            )
        elif fx.const_expr(scaled):
            # Scaled PackSlb (512B/warp). Prefer a free pipeline stage when
            # one stage holds WarpCnt*512 B.
            epi_smem = smem_base
            skip_bar = False
            if fx.const_expr(stages >= 2 and stage_elems >= num_warps * B16_PACK_SLB_BYTES_PER_WARP):
                skip_bar = True
                epi_smem = _free_pack_base
            mr_igemm_epilogue_store_scaled(
                lane_id=lane_id,
                warp_id=warp_id,
                warp_m_id=warp_m_id,
                warp_n_id=warp_n_id,
                m_tile=m_tile,
                n_tile=n_tile,
                accs=accs,
                scale_a=scale_a,
                scale_b=scale_b,
                bias=bias,
                gC_warp=gC_warp,
                smem_base=epi_smem,
                c_global_n=n,
                bm=bm,
                bn=bn,
                warp_m=warp_m,
                warp_n=warp_n,
                warp_atoms_m=warp_atoms_m,
                warp_atoms_n=warp_atoms_n,
                out_dtype=out_dtype_fx,
                apply_bias=apply_bias,
                skip_cta_barrier=skip_bar,
                m_valid=m_rows if dynamic_m else (m_valid if m_valid < m else None),
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

    # ``_igemm_body`` is called from @flyc.kernel but defined out here, so the
    # kernel decorator never rewrites it. Without this, ``fx.range`` stays a
    # plain Python range and the K main is fully unrolled -- compile time then
    # scales with K (tens of seconds on Qwen prefill shapes).
    _igemm_body = ASTRewriter.transform(_igemm_body)

    if scaled and dynamic_m:
        if apply_bias:

            @flyc.kernel(known_block_size=[threads, 1, 1])
            def mr_igemm(
                A: fx.Tensor,
                B: fx.Tensor,
                scale_a: fx.Tensor,
                scale_b: fx.Tensor,
                C: fx.Tensor,
                bias: fx.Tensor,
                m_rows: fx.Int32,
            ):
                _igemm_body(A, B, C, scale_a, scale_b, bias, m_rows)

        else:

            @flyc.kernel(known_block_size=[threads, 1, 1])
            def mr_igemm(
                A: fx.Tensor,
                B: fx.Tensor,
                scale_a: fx.Tensor,
                scale_b: fx.Tensor,
                C: fx.Tensor,
                m_rows: fx.Int32,
            ):
                _igemm_body(A, B, C, scale_a, scale_b, None, m_rows)

    elif scaled:
        if apply_bias:

            @flyc.kernel(known_block_size=[threads, 1, 1])
            def mr_igemm(
                A: fx.Tensor,
                B: fx.Tensor,
                scale_a: fx.Tensor,
                scale_b: fx.Tensor,
                C: fx.Tensor,
                bias: fx.Tensor,
            ):
                _igemm_body(A, B, C, scale_a, scale_b, bias)

        else:

            @flyc.kernel(known_block_size=[threads, 1, 1])
            def mr_igemm(
                A: fx.Tensor,
                B: fx.Tensor,
                scale_a: fx.Tensor,
                scale_b: fx.Tensor,
                C: fx.Tensor,
            ):
                _igemm_body(A, B, C, scale_a, scale_b, None)

    else:

        @flyc.kernel(known_block_size=[threads, 1, 1])
        def mr_igemm(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
            _igemm_body(A, B, C, None, None, None)

    return mr_igemm, threads, smem_elems, bm, bn, bk


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
    apply_bias: bool = False,
    allow_dynamic_m: bool = False,
    dynamic_m: bool = False,
    m_hint: int | None = None,
):
    """Build and return a JIT launch wrapper for the Iluvatar MR int8 GEMM.

    ``D = A @ B.T`` with int8 ``A(M, K)`` / ``B(N, K)``.

    Epilogues:
      * ``i32`` / ``i8`` -- ``launch(A, B, C)``
      * ``scaled_bf16`` / ``scaled_fp16`` --
        ``launch(A, B, scale_a, scale_b, C)`` or with ``bias`` when
        ``apply_bias=True`` (``acc * sa[m] * sb[n] [+ bias[n]]``).

    ``scale_a`` is ``[M]`` fp32, ``scale_b`` / ``bias`` are ``[N]`` fp32 (bias
    may be ``out_dtype``; cast to fp32 in epilogue).

    When ``allow_dynamic_m=True`` (scaled only), ``M`` need not be a multiple of
    BM: the kernel is built for ``m_ceil = ceil(M / BM) * BM`` and stores only
    the first ``M`` rows. A and scale_a only need ``M`` rows -- with k-major A,
    G2S predicates off the chunks starting past ``M``, and the epilogue clamps
    its scale_a row. C must be allocated with at least ``m_ceil`` rows; live
    output is ``C[:M]``.

    When ``dynamic_m=True`` (scaled only) ``M`` is instead an **upper bound** on
    rows and drops out of the compile key: the row count and M-grid extent are
    launch arguments, so one kernel serves every M under the bound. Use
    ``m_hint`` to steer the pipeline-depth heuristic toward the expected M.
    Launch is ``launch(A, B, scale_a, scale_b, C, [bias,] m_rows, grid_m)``.
    """
    layout = parse_major_pattern(major_pattern)
    if epilogue not in EPILOGUE_CHOICES:
        raise ValueError(f"unknown epilogue: {epilogue}")
    if apply_bias and not _is_scaled_epilogue(epilogue):
        raise ValueError("apply_bias requires a scaled_* epilogue")
    if allow_dynamic_m and not _is_scaled_epilogue(epilogue):
        raise ValueError("allow_dynamic_m requires a scaled_* epilogue")
    if dynamic_m:
        if not _is_scaled_epilogue(epilogue):
            raise ValueError("dynamic_m requires a scaled_* epilogue")
        if allow_dynamic_m:
            raise ValueError("dynamic_m and allow_dynamic_m are mutually exclusive")
    if warp_atoms_m < 2 or warp_atoms_m % 2 != 0:
        raise ValueError(f"warp_atoms_m={warp_atoms_m} must be even and >= 2")

    bm, bn, bk, threads, _ = _igemm_cta_shape(
        warps_m,
        warps_n,
        k_rep,
        warp_atoms_m=warp_atoms_m,
        warp_atoms_n=warp_atoms_n,
        stages=2,
    )
    m_ceil = ((M + bm - 1) // bm) * bm if (allow_dynamic_m or dynamic_m) else M
    # Under dynamic_m the real M is unknown here, so steer the depth heuristic
    # with the caller's expected M rather than the upper bound.
    stages_m = m_ceil
    if dynamic_m and m_hint is not None:
        stages_m = ((m_hint + bm - 1) // bm) * bm
    stages = resolve_igemm_stages(stages, stages_m, N, K, bm, bn, bk, major_pattern=major_pattern, epilogue=epilogue)
    bm, bn, bk, threads, pipeline_smem = _igemm_cta_shape(
        warps_m,
        warps_n,
        k_rep,
        warp_atoms_m=warp_atoms_m,
        warp_atoms_n=warp_atoms_n,
        stages=stages,
    )
    m_ceil = ((M + bm - 1) // bm) * bm if (allow_dynamic_m or dynamic_m) else M
    # PackOnly (k-major B i8) needs no C-tile scratch beyond the pipeline buffers.
    use_pack_only = epilogue == EPILOGUE_I8 and not layout.b_mn_major
    num_warps = warps_m * warps_n
    if epilogue == EPILOGUE_I8 and not use_pack_only:
        smem_bytes = max(pipeline_smem, bm * bn)
    elif _is_scaled_epilogue(epilogue):
        if warp_atoms_n % 2 != 0:
            raise ValueError(f"scaled PackSlb requires even warp_atoms_n, got {warp_atoms_n}")
        smem_bytes = max(pipeline_smem, num_warps * B16_PACK_SLB_BYTES_PER_WARP)
    else:
        smem_bytes = pipeline_smem
    if K % bk:
        raise ValueError(f"K must be a multiple of {bk} (ATOM_K_B8 * k_rep)")
    if (not (allow_dynamic_m or dynamic_m)) and (M % bm or N % bn):
        raise ValueError(f"M,N must be multiples of {bm}/{bn}")
    if (allow_dynamic_m or dynamic_m) and N % bn:
        raise ValueError(f"N must be a multiple of {bn}, got N={N}")
    if bk % I8_VPR:
        raise ValueError(f"BK={bk} must be a multiple of {I8_VPR}; use even k_rep")
    a_atoms_total, b_atoms_total, _, _ = sme_atom_counts(layout, bm, bn, bk, values_per_sme_row=I8_VPR)
    # One operand may have fewer chunks than warps -- the surplus warps are
    # predicated off in G2S -- but if neither reaches the warp count the tile is
    # simply too small for this many warps.
    if max(a_atoms_total, b_atoms_total) < num_warps:
        raise ValueError(
            f"SME chunk counts ({a_atoms_total}/{b_atoms_total}) must reach the "
            f"{warps_m}x{warps_n}={num_warps} warp count; use a larger tile or k_rep"
        )
    if smem_bytes > DEFAULT_SMEM_CAP_BYTES:
        raise ValueError(
            f"CTA smem {smem_bytes} B exceeds device cap {DEFAULT_SMEM_CAP_BYTES} B "
            f"({bm}x{bn}x{bk}, {threads} threads); use smaller tile or k_rep"
        )

    gemm_kernel, threads, smem_bytes, bm, bn, _bk = _build_igemm_kernel(
        m_ceil,
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
        apply_bias=apply_bias,
        m_valid=None if dynamic_m else M,
        dynamic_m=dynamic_m,
    )
    grid_n = N // bn
    grid_m_static = m_ceil // bm
    n_major_grid = bm < 256
    grid = (grid_n, grid_m_static, 1) if n_major_grid else (grid_m_static, grid_n, 1)
    block = (threads, 1, 1)

    if dynamic_m:
        if apply_bias:
            if n_major_grid:

                @flyc.jit
                def launch_gemm(
                    A: fx.Tensor,
                    B: fx.Tensor,
                    scale_a: fx.Tensor,
                    scale_b: fx.Tensor,
                    C: fx.Tensor,
                    bias: fx.Tensor,
                    m_rows: fx.Int32,
                    grid_m: fx.Int32,
                    stream: fx.Stream = fx.Stream(None),
                ):
                    gemm_kernel(A, B, scale_a, scale_b, C, bias, m_rows).launch(
                        grid=(grid_n, grid_m, 1), block=block, stream=stream
                    )

            else:

                @flyc.jit
                def launch_gemm(
                    A: fx.Tensor,
                    B: fx.Tensor,
                    scale_a: fx.Tensor,
                    scale_b: fx.Tensor,
                    C: fx.Tensor,
                    bias: fx.Tensor,
                    m_rows: fx.Int32,
                    grid_m: fx.Int32,
                    stream: fx.Stream = fx.Stream(None),
                ):
                    gemm_kernel(A, B, scale_a, scale_b, C, bias, m_rows).launch(
                        grid=_dynamic_grid(grid_m, grid_n), block=block, stream=stream
                    )

        else:
            if n_major_grid:

                @flyc.jit
                def launch_gemm(
                    A: fx.Tensor,
                    B: fx.Tensor,
                    scale_a: fx.Tensor,
                    scale_b: fx.Tensor,
                    C: fx.Tensor,
                    m_rows: fx.Int32,
                    grid_m: fx.Int32,
                    stream: fx.Stream = fx.Stream(None),
                ):
                    gemm_kernel(A, B, scale_a, scale_b, C, m_rows).launch(
                        grid=(grid_n, grid_m, 1), block=block, stream=stream
                    )

            else:

                @flyc.jit
                def launch_gemm(
                    A: fx.Tensor,
                    B: fx.Tensor,
                    scale_a: fx.Tensor,
                    scale_b: fx.Tensor,
                    C: fx.Tensor,
                    m_rows: fx.Int32,
                    grid_m: fx.Int32,
                    stream: fx.Stream = fx.Stream(None),
                ):
                    gemm_kernel(A, B, scale_a, scale_b, C, m_rows).launch(
                        grid=_dynamic_grid(grid_m, grid_n), block=block, stream=stream
                    )

    elif _is_scaled_epilogue(epilogue):
        if apply_bias:

            @flyc.jit
            def launch_gemm(
                A: fx.Tensor,
                B: fx.Tensor,
                scale_a: fx.Tensor,
                scale_b: fx.Tensor,
                C: fx.Tensor,
                bias: fx.Tensor,
                stream: fx.Stream = fx.Stream(None),
            ):
                gemm_kernel(A, B, scale_a, scale_b, C, bias).launch(grid=grid, block=block, stream=stream)

        else:

            @flyc.jit
            def launch_gemm(
                A: fx.Tensor,
                B: fx.Tensor,
                scale_a: fx.Tensor,
                scale_b: fx.Tensor,
                C: fx.Tensor,
                stream: fx.Stream = fx.Stream(None),
            ):
                gemm_kernel(A, B, scale_a, scale_b, C).launch(grid=grid, block=block, stream=stream)

    else:

        @flyc.jit
        def launch_gemm(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
            # Static SharedAllocator banks are sized by the compiler; leave launch smem unset.
            gemm_kernel(A, B, C).launch(grid=grid, block=block, stream=stream)

    launch_gemm.grid = grid
    launch_gemm.grid_n = grid_n
    launch_gemm.n_major_grid = n_major_grid
    launch_gemm.block = block
    launch_gemm.bm = bm
    launch_gemm.bn = bn
    launch_gemm.bk = bk
    launch_gemm.m_ceil = m_ceil
    launch_gemm.dynamic_m = dynamic_m
    launch_gemm.smem_bytes = smem_bytes
    return launch_gemm


__all__ = [
    "ATOM_K_I8",
    "DEFAULT_EPILOGUE",
    "DEFAULT_K_REP",
    "EPILOGUE_CHOICES",
    "EPILOGUE_I8",
    "EPILOGUE_I32",
    "EPILOGUE_SCALED_BF16",
    "EPILOGUE_SCALED_FP16",
    "I8_VPR",
    "compile_iluvatar_mr_igemm",
    "resolve_igemm_stages",
    "_is_scaled_epilogue",
]
