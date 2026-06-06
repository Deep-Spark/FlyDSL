# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar MR (ivcore11) tiledMma pipeline HGEMM.

Double-buffered shared-memory pipeline with async SME G2S
(``MRAsyncCpRow16b`` / ``MRAsyncCpCol``), ``make_sme_shared_layout``, Ki-deferred
S2R/MMA mainloop, and ``UniversalCopy32b`` for S2R.

Entry point: ``compile_iluvatar_mr_hgemm(M=..., N=..., K=..., ...)`` returns a
``@flyc.jit`` launch wrapper ``launch_gemm(A, B, C, stream=...)``.

Tuning parameters
-----------------

**epilogue**

* ``no_c_read`` (default) — ``D = A @ B.T``, fp16 output, accumulator zeroed (no
  global C read). Store path selected by ``epilogue_store``. Not to be confused
  with ``major_pattern`` ``nn`` (a G2S layout tag).
* ``read_c_accum`` — ``C = A @ B.T + C``, fp32 output, load existing C into the
  accumulator before MMA.

**epilogue_store** (``no_c_read`` only)

* ``shfl`` (default) — f32 acc → fp16 via warp ``shuffle_idx`` + packed i32 store.
* ``tiled`` — vector ``trunc_f`` + ``make_tiled_copy_C`` / ``UniversalCopy16b``.

**major_pattern** — G2S global layout for A/B: ``nn``, ``tn``, ``nt`` (default),
``tt`` (two letters: A then B, ``n``=NoTrans/row SME, ``t``=Trans/col SME).
Kernel tensors are always logical ``A(m,k)``, ``B(n,k)``; the pattern selects
how SME views map to those layouts.

**CTA shape** — ``warps_m``, ``warps_n``, ``warp_atoms_m``, ``warp_atoms_n``,
``k_rep`` (``BK = 16 * k_rep``). ``SWIZZLE_CTA_PRESETS`` lists common presets:

* ``1024`` — ``4 x 4`` warps, ``4 x 4`` atoms/warp → ``256 x 256`` CTA tile,
  ``64 x 64``/warp; preset ``default_k_rep=4`` → ``BK=64``.
* ``2048`` — ``4 x 8`` warps, ``4 x 2`` atoms/warp → same ``256 x 256`` tile,
  ``64 x 32``/warp; typically needs ``k_rep >= 4`` so SME brick work divides
  across warps and smem stays within 128 KiB/device.

Default ``compile_iluvatar_mr_hgemm`` kwargs: ``4 x 4`` warps, ``4 x 4`` atoms/warp,
``k_rep=2`` (``BK=32``, lower smem / peak-tuned). Override per preset or problem
size; ``M``, ``N``, ``K`` must align with the resulting ``bm``/``bn``/``bk``.
"""

# NOTE: do NOT add ``from __future__ import annotations`` (Constexpr introspection).

from typing import NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.ixdl as ixdl
from kernels.iluvatar_mr_common import (
    ATOM_K,
    ATOM_M,
    ATOM_N,
    DEFAULT_MAJOR_PATTERN,
    DEFAULT_SMEM_CAP_BYTES,
    MAJOR_PATTERN_CHOICES,
    PATTERN_ID,
    SMEM_F16_PER_ROW,
    SMEM_ROWS,
    WARP_SIZE,
)
from kernels.iluvatar_mr_epilogue import (
    mr_hgemm_epilogue_store_read_c_accum,
    mr_hgemm_epilogue_store_shfl,
    mr_hgemm_epilogue_store_tiled,
)
from kernels.iluvatar_mr_operand_copy import mr_hgemm_g2s_issue_operands, mr_pattern_g2s_sme_config
from kernels.iluvatar_mr_s2r import mr_hgemm_s2r_load_ki

DEFAULT_K_REP = 2  # CTA K-tile: ATOM_K * k_rep = 32
STAGES = 2
K_LOOP_UNROLL = 2

EPILOGUE_NO_C_READ = "no_c_read"
EPILOGUE_READ_C_ACCUM = "read_c_accum"
DEFAULT_EPILOGUE = EPILOGUE_NO_C_READ

EPILOGUE_STORE_TILED = "tiled"
EPILOGUE_STORE_SHFL = "shfl"
DEFAULT_EPILOGUE_STORE = EPILOGUE_STORE_SHFL

_PATTERN_ID = PATTERN_ID


class SwizzleCtaPreset(NamedTuple):
    """Swizzle-mode CTA: (warps_m x warps_n) warps, each (warp_atoms_m x warp_atoms_n) MMA atoms."""

    name: str
    warps_m: int
    warps_n: int
    warp_atoms_m: int
    warp_atoms_n: int
    default_k_rep: int


SWIZZLE_CTA_PRESETS: dict[str, SwizzleCtaPreset] = {
    # 16 warps x 64 lanes; warp tile 64x64; CTA 256x256; smem ~64 KiB @ k_rep=4.
    "1024": SwizzleCtaPreset("1024", 4, 4, 4, 4, 4),
    # 32 warps x 64 lanes; warp tile 64x32; CTA still 256x256; smem ~128 KiB @ k_rep=4.
    "2048": SwizzleCtaPreset("2048", 4, 8, 4, 2, 4),
}
DEFAULT_SWIZZLE_CTA = "1024"


def _swizzle_cta_shape(
    warps_m: int,
    warps_n: int,
    k_rep: int,
    *,
    warp_atoms_m: int,
    warp_atoms_n: int,
) -> tuple[int, int, int, int, int]:
    warp_m = ATOM_M * warp_atoms_m
    warp_n = ATOM_N * warp_atoms_n
    bm = warp_m * warps_m
    bn = warp_n * warps_n
    bk = ATOM_K * k_rep
    threads = warps_m * warps_n * WARP_SIZE
    smem_bytes = (bm + bn) * bk * 2 * STAGES
    return bm, bn, bk, threads, smem_bytes


def _swizzle_atom_work_ok(bm: int, bn: int, bk: int, warps_m: int, warps_n: int) -> bool:
    num_warps = warps_m * warps_n
    cta_atoms_k = bk // SMEM_F16_PER_ROW
    a_atoms_total = (bm // SMEM_ROWS) * cta_atoms_k
    b_atoms_total = (bn // SMEM_ROWS) * cta_atoms_k
    return a_atoms_total % num_warps == 0 and b_atoms_total % num_warps == 0


def _build_swizzle_kernel(
    m: int,
    n: int,
    k: int,
    warps_m: int,
    warps_n: int,
    k_rep: int,
    warp_atoms_m: int,
    warp_atoms_n: int,
    epilogue: str,
    epilogue_store: str = DEFAULT_EPILOGUE_STORE,
    major_pattern: str = DEFAULT_MAJOR_PATTERN,
):
    pattern_id = _PATTERN_ID[major_pattern]
    load_c = epilogue == EPILOGUE_READ_C_ACCUM
    out_fp16 = epilogue == EPILOGUE_NO_C_READ
    no_c_read_shfl_store = out_fp16 and epilogue_store == EPILOGUE_STORE_SHFL
    no_c_read_tiled_store = out_fp16 and not no_c_read_shfl_store
    warp_m = ATOM_M * warp_atoms_m
    warp_n = ATOM_N * warp_atoms_n
    bm = warp_m * warps_m
    bn = warp_n * warps_n
    bk = ATOM_K * k_rep
    num_warps = warps_m * warps_n
    threads = num_warps * WARP_SIZE

    assert k % bk == 0
    assert m % bm == 0 and n % bn == 0
    assert bk % SMEM_F16_PER_ROW == 0

    cta_atoms_m = bm // SMEM_ROWS
    cta_atoms_n = bn // SMEM_ROWS
    cta_atoms_k = bk // SMEM_F16_PER_ROW
    a_atoms_total = cta_atoms_m * cta_atoms_k
    b_atoms_total = cta_atoms_n * cta_atoms_k
    a_per_warp = a_atoms_total // num_warps
    b_per_warp = b_atoms_total // num_warps
    assert a_atoms_total % num_warps == 0
    assert b_atoms_total % num_warps == 0
    stage_elems = (bm + bn) * bk
    stage_stride = stage_elems
    k_tiles_const = k // bk
    main_k_trip = max(0, k_tiles_const - 2)
    main_k_full = (main_k_trip // K_LOOP_UNROLL) * K_LOOP_UNROLL
    main_k_remainder = main_k_trip - main_k_full

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        bid_x, bid_y, _ = fx.block_idx
        warp_id = tid // WARP_SIZE
        lane_id = tid % WARP_SIZE
        warp_m_id = warp_id // warps_n
        warp_n_id = warp_id % warps_n

        if fx.const_expr(pattern_id == 1 or pattern_id == 3):
            a_log_stride = (1, m)
        else:
            a_log_stride = (k, 1)
        a_logical = fx.make_view(fx.get_iter(A), fx.make_layout((m, k), a_log_stride))
        gA = fx.slice(fx.flat_divide(a_logical, (bm, bk)), (None, None, bid_x, None))

        if fx.const_expr(pattern_id == 0 or pattern_id == 1):
            b_log_stride = (1, n)
        else:
            b_log_stride = (k, 1)
        b_logical = fx.make_view(fx.get_iter(B), fx.make_layout((n, k), b_log_stride))
        gB = fx.slice(fx.flat_divide(b_logical, (bn, bk)), (None, None, bid_y, None))

        gC = fx.slice(fx.flat_divide(C, (bm, bn)), (None, None, bid_x, bid_y))

        smem_ptr = fx.get_dyn_shared()

        mma_atom = fx.make_mma_atom(ixdl.MRMma(ATOM_M, ATOM_N, ATOM_K, fx.Float16, fx.Float16, fx.Float32))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (1, 1, 1)))
        thr_mma = tiled_mma.thr_slice(lane_id)

        if fx.const_expr(load_c):
            copy_atom_c_f32 = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
            tiled_copy_c_f32 = fx.make_tiled_copy_C(copy_atom_c_f32, tiled_mma)
            thr_copy_c_f32 = tiled_copy_c_f32.get_slice(lane_id)

        smem_f16_base = fx.recast_iter(
            fx.PointerType.get(fx.Float16.ir_type, fx.AddressSpace.Shared),
            smem_ptr,
        )

        gC_atoms = fx.flat_divide(
            fx.slice(
                fx.flat_divide(gC, (warp_m, warp_n)),
                (None, None, warp_m_id, warp_n_id),
            ),
            (ATOM_M, ATOM_N),
        )

        accs = []
        for im in fx.range_constexpr(warp_atoms_m):
            row = []
            for jn in fx.range_constexpr(warp_atoms_n):
                c_tile = fx.slice(gC_atoms, (None, None, im, jn))
                frag = thr_mma.make_fragment_C(c_tile)
                if load_c:
                    fx.copy(
                        copy_atom_c_f32,
                        thr_copy_c_f32.partition_S(c_tile),
                        thr_copy_c_f32.retile(frag),
                        pred=None,
                    )
                else:
                    frag.fill(0)
                row.append(frag)
            accs.append(row)

        def _run_pipeline():
            g2s_sme = mr_pattern_g2s_sme_config(
                pattern_id,
                fx.Float16,
                row_atom=ixdl.MRAsyncCpRow16b,
                row_swizzle=ixdl.SMESwizzle.Row16b,
            )

            copy_atom_s2r_a = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float16)
            copy_atom_s2r_b = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float16)
            tiled_copy_a = fx.make_tiled_copy_A(copy_atom_s2r_a, tiled_mma)
            tiled_copy_b = fx.make_tiled_copy_B(copy_atom_s2r_b, tiled_mma)
            thr_copy_a = tiled_copy_a.get_slice(lane_id)
            thr_copy_b = tiled_copy_b.get_slice(lane_id)

            tile_smem = fx.make_tile(SMEM_ROWS, SMEM_F16_PER_ROW)
            tile_smem_A = (
                fx.make_tile(SMEM_F16_PER_ROW, SMEM_ROWS)
                if fx.const_expr(pattern_id == 1 or pattern_id == 3)
                else tile_smem
            )
            tile_smem_B = (
                fx.make_tile(SMEM_F16_PER_ROW, SMEM_ROWS)
                if fx.const_expr(pattern_id == 0 or pattern_id == 1)
                else tile_smem
            )

            def issue_stage(k_tile, stage_base):
                k_A = gA[None, None, k_tile]
                k_B = gB[None, None, k_tile]
                if fx.const_expr(pattern_id == 1 or pattern_id == 3):
                    a_leading = m
                else:
                    a_leading = k
                if fx.const_expr(pattern_id == 0 or pattern_id == 1):
                    b_leading = n
                else:
                    b_leading = k
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
                    smem_base=smem_f16_base,
                    elem_dtype=fx.Float16,
                    bm=bm,
                    bn=bn,
                    bk=bk,
                    stage_base=stage_base,
                )

            def _ki_load(stage_base, ki):
                return mr_hgemm_s2r_load_ki(
                    pattern_id=pattern_id,
                    ki=ki,
                    stage_base=stage_base,
                    g2s_sme=g2s_sme,
                    smem_base=smem_f16_base,
                    elem_dtype=fx.Float16,
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

            def _s2r_mma_defer_last_into(stage_base, a_def, b_def):
                for ki in fx.range_constexpr(k_rep - 1):
                    a_frags, b_frags = _ki_load(stage_base, ki)
                    _mma_frags(a_frags, b_frags)
                a_last, b_last = _ki_load(stage_base, k_rep - 1)
                _copy_a_frags(a_def, a_last)
                _copy_b_frags(b_def, b_last)

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
                _mma_frags(a_def, b_def)
                k_tile = k_idx + 2
                if k_idx % 2 == 0:
                    issue_stage(fx.Int32(k_tile), fx.Int32(0))
                    _s2r_mma_defer_last_into(fx.Int32(stage_stride), a_def, b_def)
                else:
                    issue_stage(fx.Int32(k_tile), fx.Int32(stage_stride))
                    _s2r_mma_defer_last_into(fx.Int32(0), a_def, b_def)

            # ROCm-style K-loop: outer scf.for + inner range_constexpr partial unroll.
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

        gC_warp = fx.slice(
            fx.flat_divide(gC, (warp_m, warp_n)),
            (None, None, warp_m_id, warp_n_id),
        )
        if fx.const_expr(no_c_read_shfl_store):
            mr_hgemm_epilogue_store_shfl(
                lane_id=lane_id,
                accs=accs,
                gC_warp=gC_warp,
                c_global_n=n,
                warp_atoms_m=warp_atoms_m,
                warp_atoms_n=warp_atoms_n,
            )
        elif fx.const_expr(no_c_read_tiled_store):
            mr_hgemm_epilogue_store_tiled(
                lane_id=lane_id,
                accs=accs,
                gC_warp=gC_warp,
                tiled_mma=tiled_mma,
                warp_atoms_m=warp_atoms_m,
                warp_atoms_n=warp_atoms_n,
            )
        else:
            mr_hgemm_epilogue_store_read_c_accum(
                lane_id=lane_id,
                accs=accs,
                gC_warp=gC_warp,
                tiled_mma=tiled_mma,
                warp_atoms_m=warp_atoms_m,
                warp_atoms_n=warp_atoms_n,
            )

    smem_bytes = stage_elems * 2 * STAGES
    return gemm_kernel, threads, smem_bytes, bm, bn, bk


def compile_iluvatar_mr_hgemm(
    *,
    M: int,
    N: int,
    K: int,
    warps_m: int = 4,
    warps_n: int = 4,
    k_rep: int = DEFAULT_K_REP,
    warp_atoms_m: int = 4,
    warp_atoms_n: int = 4,
    epilogue: str = DEFAULT_EPILOGUE,
    epilogue_store: str = DEFAULT_EPILOGUE_STORE,
    major_pattern: str = DEFAULT_MAJOR_PATTERN,
):
    """Build and return a JIT launch wrapper for the Iluvatar MR HGEMM.

    See the module docstring for ``epilogue``, ``epilogue_store``, ``major_pattern``,
    and CTA preset semantics.
    """
    if major_pattern not in _PATTERN_ID:
        raise ValueError(f"unknown major pattern: {major_pattern}")
    if epilogue not in (EPILOGUE_NO_C_READ, EPILOGUE_READ_C_ACCUM):
        raise ValueError(f"unknown epilogue: {epilogue}")

    bm, bn, bk, threads, smem_bytes = _swizzle_cta_shape(
        warps_m,
        warps_n,
        k_rep,
        warp_atoms_m=warp_atoms_m,
        warp_atoms_n=warp_atoms_n,
    )
    if K % bk:
        raise ValueError(f"K must be a multiple of {bk} (16 * k_rep)")
    if M % bm or N % bn:
        raise ValueError(f"M,N must be multiples of {bm}/{bn} for swizzle CTA")
    if not _swizzle_atom_work_ok(bm, bn, bk, warps_m, warps_n):
        raise ValueError(
            f"SME brick count must divide evenly across {warps_m}x{warps_n} warps; "
            f"try larger k_rep (current BK={bk})"
        )
    if smem_bytes > DEFAULT_SMEM_CAP_BYTES:
        raise ValueError(
            f"CTA smem {smem_bytes} B exceeds device cap {DEFAULT_SMEM_CAP_BYTES} B "
            f"({bm}x{bn}x{bk}, {threads} threads); use smaller tile or k_rep"
        )

    gemm_kernel, threads, smem_bytes, bm, bn, _bk = _build_swizzle_kernel(
        M,
        N,
        K,
        warps_m,
        warps_n,
        k_rep,
        warp_atoms_m,
        warp_atoms_n,
        epilogue,
        epilogue_store,
        major_pattern,
    )
    grid = (M // bm, N // bn, 1)
    block = (threads, 1, 1)

    @flyc.jit
    def launch_gemm(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        gemm_kernel(A, B, C).launch(grid=grid, block=block, smem=smem_bytes, stream=stream)

    return launch_gemm


__all__ = [
    "DEFAULT_EPILOGUE",
    "DEFAULT_EPILOGUE_STORE",
    "DEFAULT_K_REP",
    "DEFAULT_MAJOR_PATTERN",
    "DEFAULT_SWIZZLE_CTA",
    "EPILOGUE_READ_C_ACCUM",
    "EPILOGUE_NO_C_READ",
    "EPILOGUE_STORE_SHFL",
    "EPILOGUE_STORE_TILED",
    "MAJOR_PATTERN_CHOICES",
    "SWIZZLE_CTA_PRESETS",
    "SwizzleCtaPreset",
    "compile_iluvatar_mr_hgemm",
]
