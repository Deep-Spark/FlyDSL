# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar CQ (ivcore30) tiledMma pipeline HGEMM (f16 / bf16).

Double-buffered smem pipeline: SmexMtx G2S (``smex.loadn.16x1b64.mtx``),
``CQMtxLoadn`` S2R, ``CQMma``, and CQ Scheme C sync (``nbarrier_sync``).
**No pipebar** (MR-only).

Entry: ``compile_iluvatar_cq_hgemm(M=..., N=..., K=..., elem_dtype=fx.Float16|fx.BFloat16)``
  -> ``launch_gemm(A, B, C, stream=...)``.

Bring-up scope:
  * fp16 / bf16 -> f16/bf16 out (f32 accum); base CQMma 16x16x16
  * ``major_pattern="tn"`` only (k-major A/B)
  * no s8, no long-mtx, no split-K
"""

from typing import NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.ixdl as ixdl
from kernels.gemm.iluvatar.common import (
    DEFAULT_MAJOR_PATTERN,
    MAJOR_PATTERN_CHOICES,
    WARP_SIZE,
    parse_major_pattern,
)
from kernels.gemm.iluvatar.cq.common import (
    ATOM_K_B16,
    ATOM_M,
    ATOM_N,
    CQ_GEMM_GEOM,
    DEFAULT_SMEM_CAP_BYTES,
    cq_stage_smem_ab,
)
from kernels.gemm.iluvatar.cq.operand_copy import (
    cq_cta_brick_counts,
    cq_g2s_smex_atom,
    cq_gemm_g2s_issue_operands,
    cq_smex_brick_tile,
)
from kernels.gemm.iluvatar.cq.s2r import cq_gemm_s2r_load_mma_k
from kernels.gemm.iluvatar.epilogue import (
    mr_hgemm_epilogue_store_read_c_accum,
    mr_hgemm_epilogue_store_shfl,
    mr_hgemm_epilogue_store_tiled,
)

DEFAULT_K_ATOMS = 2  # CTA K-tile: ATOM_K_B16 * k_atoms = 32
STAGES = 2
K_LOOP_UNROLL = 1

EPILOGUE_NO_C_READ = "no_c_read"
EPILOGUE_READ_C_ACCUM = "read_c_accum"
DEFAULT_EPILOGUE = EPILOGUE_NO_C_READ

EPILOGUE_STORE_TILED = "tiled"
EPILOGUE_STORE_SHFL = "shfl"
DEFAULT_EPILOGUE_STORE = EPILOGUE_STORE_SHFL

SUPPORTED_ELEM_DTYPES = (fx.Float16, fx.BFloat16)
DEFAULT_ELEM_DTYPE = fx.Float16

# Pipelined CQ HGEMM currently supports tn (k-major A/B) only.
SUPPORTED_MAJOR_PATTERNS = (DEFAULT_MAJOR_PATTERN,)


def _validate_elem_dtype(elem_dtype):
    if elem_dtype not in SUPPORTED_ELEM_DTYPES:
        names = ", ".join(t.__name__ for t in SUPPORTED_ELEM_DTYPES)
        raise ValueError(f"elem_dtype must be one of {{{names}}}, got {elem_dtype!r}")
    return elem_dtype


class SwizzleCtaPreset(NamedTuple):
    """CTA: (warps_m x warps_n) warps, each (warp_atoms_m x warp_atoms_n) MMA atoms."""

    name: str
    warps_m: int
    warps_n: int
    warp_atoms_m: int
    warp_atoms_n: int
    default_k_atoms: int


SWIZZLE_CTA_PRESETS: dict[str, SwizzleCtaPreset] = {
    # 4 warps; warp tile 32x32; CTA 64x64; ~16 KiB smem @ k_atoms=2, stages=2.
    "256": SwizzleCtaPreset("256", 2, 2, 2, 2, 2),
    # 16 warps; warp tile 64x64; CTA 256x256.
    # Prefer k_atoms=2 (BK=32, ~64 KiB @ stages=2) so ~3 CTA/CU on 192 KiB SLB.
    # k_atoms=4 (BK=64, ~128 KiB) still fits the 192 KiB cap but drops to ~1 CTA/CU.
    "1024": SwizzleCtaPreset("1024", 4, 4, 4, 4, 2),
    # 32 warps (2048 threads); warp tile 64x64; CTA 512x256, k_atoms=4 (BK=64).
    # ~192 KiB smem (SLB cap, ~1 CTA/CU) but the wide tile amortizes epilogue/sync:
    # fastest measured at >=3k shapes (reaches torch parity at 4k^3). BK=64 also
    # makes the SmexMtx brick count divide evenly across 32 warps.
    "2048": SwizzleCtaPreset("2048", 8, 4, 4, 4, 4),
}
DEFAULT_SWIZZLE_CTA = "256"
SWIZZLE_CTA_AUTO = "auto"
# Prefer the 2048-thread wide tile once both M and N are at least this large
# (measured: beats 1024-thread at 2k^3 and clearly at >=3k^3).
_SWIZZLE_CTA_2048_MIN_MN = 2048


def _swizzle_cta_shape(
    warps_m: int,
    warps_n: int,
    k_atoms: int,
    *,
    warp_atoms_m: int,
    warp_atoms_n: int,
) -> tuple[int, int, int, int, int]:
    warp_m = ATOM_M * warp_atoms_m
    warp_n = ATOM_N * warp_atoms_n
    bm = warp_m * warps_m
    bn = warp_n * warps_n
    bk = ATOM_K_B16 * k_atoms
    threads = warps_m * warps_n * WARP_SIZE
    smem_bytes = (bm + bn) * bk * 2 * STAGES
    return bm, bn, bk, threads, smem_bytes


def _swizzle_atom_work_ok(bm: int, bn: int, bk: int, warps_m: int, warps_n: int) -> bool:
    num_warps = warps_m * warps_n
    a_bricks, b_bricks, _ = cq_cta_brick_counts(bm=bm, bn=bn, bk=bk, geom=CQ_GEMM_GEOM)
    return a_bricks % num_warps == 0 and b_bricks % num_warps == 0


def _preset_cta_shape(preset: SwizzleCtaPreset) -> tuple[int, int, int, int, int]:
    """Return ``(bm, bn, bk, threads, smem_bytes)`` for a CTA preset."""
    return _swizzle_cta_shape(
        preset.warps_m,
        preset.warps_n,
        preset.default_k_atoms,
        warp_atoms_m=preset.warp_atoms_m,
        warp_atoms_n=preset.warp_atoms_n,
    )


def _shape_fits_preset(m: int, n: int, k: int, preset: SwizzleCtaPreset) -> bool:
    bm, bn, bk, _threads, smem_bytes = _preset_cta_shape(preset)
    if smem_bytes > DEFAULT_SMEM_CAP_BYTES:
        return False
    if m % bm or n % bn or k % bk:
        return False
    return _swizzle_atom_work_ok(bm, bn, bk, preset.warps_m, preset.warps_n)


def select_swizzle_cta(m: int, n: int, k: int) -> SwizzleCtaPreset:
    """Pick a CTA preset for ``(M, N, K)`` from measured CQ HGEMM trade-offs.

    Preference order (first that fits the shape and size band):

    * ``\"2048\"`` (512x256, k_atoms=4): large GEMMs with ``min(M,N) >= 2048``;
      uses the full 192 KiB SLB but amortizes sync/epilogue best.
    * ``\"1024\"`` (256x256, k_atoms=2): mid-size / occupancy-friendly default.
    * ``\"256\"`` (64x64, k_atoms=2): small / bring-up shapes.

    Raises ``ValueError`` if no preset divides ``(M, N, K)``.
    """
    p2048 = SWIZZLE_CTA_PRESETS["2048"]
    p1024 = SWIZZLE_CTA_PRESETS["1024"]
    p256 = SWIZZLE_CTA_PRESETS["256"]

    if min(m, n) >= _SWIZZLE_CTA_2048_MIN_MN and _shape_fits_preset(m, n, k, p2048):
        return p2048
    if _shape_fits_preset(m, n, k, p1024):
        return p1024
    if _shape_fits_preset(m, n, k, p256):
        return p256

    # Last resort: any remaining preset that divides the shape (largest first).
    for name in ("2048", "1024", "256"):
        preset = SWIZZLE_CTA_PRESETS[name]
        if _shape_fits_preset(m, n, k, preset):
            return preset
    raise ValueError(
        f"no CQ HGEMM CTA preset divides M={m} N={n} K={k}; "
        f"try multiples of 64x64x32 (256), 256x256x32 (1024), or 512x256x64 (2048)"
    )


def _build_pipeline_kernel(
    m: int,
    n: int,
    k: int,
    warps_m: int,
    warps_n: int,
    k_atoms: int,
    warp_atoms_m: int,
    warp_atoms_n: int,
    epilogue: str,
    epilogue_store: str = DEFAULT_EPILOGUE_STORE,
    major_pattern: str = DEFAULT_MAJOR_PATTERN,
    elem_dtype=DEFAULT_ELEM_DTYPE,
):
    elem_dtype = _validate_elem_dtype(elem_dtype)
    gemm_layout = parse_major_pattern(major_pattern)
    if major_pattern not in SUPPORTED_MAJOR_PATTERNS:
        raise ValueError(
            f"CQ HGEMM bring-up supports major_pattern in {SUPPORTED_MAJOR_PATTERNS}, got {major_pattern!r}"
        )
    a_mn_major = gemm_layout.a_mn_major
    b_mn_major = gemm_layout.b_mn_major
    assert not a_mn_major and not b_mn_major

    load_c = epilogue == EPILOGUE_READ_C_ACCUM
    out_b16 = epilogue == EPILOGUE_NO_C_READ
    no_c_read_shfl_store = out_b16 and epilogue_store == EPILOGUE_STORE_SHFL
    no_c_read_tiled_store = out_b16 and not no_c_read_shfl_store

    warp_m = ATOM_M * warp_atoms_m
    warp_n = ATOM_N * warp_atoms_n
    bm = warp_m * warps_m
    bn = warp_n * warps_n
    bk = ATOM_K_B16 * k_atoms
    num_warps = warps_m * warps_n
    threads = num_warps * WARP_SIZE
    vpr = CQ_GEMM_GEOM.values_per_sme_row

    assert k % bk == 0
    assert m % bm == 0 and n % bn == 0
    assert bk % vpr == 0
    assert k_atoms % CQ_GEMM_GEOM.sme_row_k_slices == 0

    a_bricks, b_bricks, _k_bricks = cq_cta_brick_counts(bm=bm, bn=bn, bk=bk, geom=CQ_GEMM_GEOM)
    a_per_warp = a_bricks // num_warps
    b_per_warp = b_bricks // num_warps
    assert a_bricks % num_warps == 0
    assert b_bricks % num_warps == 0

    stage_elems = (bm + bn) * bk
    stage_stride = stage_elems
    k_tiles_const = k // bk
    main_k_trip = max(0, k_tiles_const - 2)
    main_k_full = (main_k_trip // K_LOOP_UNROLL) * K_LOOP_UNROLL
    main_k_remainder = main_k_trip - main_k_full

    def _cq_stage_sync():
        """Drain async G2S then CQ named-barrier + CTA barrier (no pipebar)."""
        ixdl.cp_async_commit_group()
        ixdl.cp_async_wait_group(0)
        ixdl.nbarrier_sync(0, num_warps)
        fx.gpu.barrier()

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        bid_x, bid_y, _ = fx.block_idx
        warp_id = tid // WARP_SIZE
        lane_id = fx.Int32(fx.lane_id)
        warp_m_id = warp_id // warps_n
        warp_n_id = warp_id % warps_n

        a_logical = fx.make_view(fx.get_iter(A), fx.make_layout((m, k), (k, 1)))
        gA = fx.slice(fx.flat_divide(a_logical, (bm, bk)), (None, None, bid_x, None))

        b_logical = fx.make_view(fx.get_iter(B), fx.make_layout((n, k), (k, 1)))
        gB = fx.slice(fx.flat_divide(b_logical, (bn, bk)), (None, None, bid_y, None))

        gC = fx.slice(fx.flat_divide(C, (bm, bn)), (None, None, bid_x, bid_y))

        @fx.struct
        class CqPipelineSmem:
            buf: fx.Array[elem_dtype, stage_elems * STAGES]

        smem_ab_base = fx.SharedAllocator(static=True).allocate(CqPipelineSmem).peek().buf.ptr

        mma_atom = fx.make_mma_atom(ixdl.CQMma(ATOM_M, ATOM_N, ATOM_K_B16, elem_dtype, elem_dtype, fx.Float32))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (1, 1, 1)))
        thr_mma = tiled_mma.thr_slice(lane_id)

        if fx.const_expr(load_c):
            copy_atom_c_f32 = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
            tiled_copy_c_f32 = fx.make_tiled_copy_C(copy_atom_c_f32, tiled_mma)
            thr_copy_c_f32 = tiled_copy_c_f32.get_slice(lane_id)

        gC_atoms = fx.flat_divide(
            fx.slice(
                fx.flat_divide(gC, (warp_m, warp_n)),
                (None, None, warp_m_id, warp_n_id),
            ),
            (ATOM_M, ATOM_N),
        )

        accs = []
        for mma_m in fx.range_constexpr(warp_atoms_m):
            row = []
            for mma_n in fx.range_constexpr(warp_atoms_n):
                c_tile = fx.slice(gC_atoms, (None, None, mma_m, mma_n))
                frag = thr_mma.make_fragment_C(c_tile)
                if fx.const_expr(load_c):
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
            loadn_a = fx.make_copy_atom(
                ixdl.CQMtxLoadn(elem_dtype, pattern="loadn16", direction="row"),
                elem_dtype,
            )
            loadn_b = fx.make_copy_atom(
                ixdl.CQMtxLoadn(elem_dtype, pattern="loadn16", direction="col"),
                elem_dtype,
            )
            thr_copy_a = fx.make_tiled_copy_A(loadn_a, tiled_mma).get_slice(lane_id)
            thr_copy_b = fx.make_tiled_copy_B(loadn_b, tiled_mma).get_slice(lane_id)

            smex_atom = cq_g2s_smex_atom(elem_dtype)
            tile_smex = cq_smex_brick_tile(geom=CQ_GEMM_GEOM)

            def issue_stage(k_tile, stage_base):
                k_A = gA[None, None, k_tile]
                k_B = gB[None, None, k_tile]
                sme_A = ixdl.make_sme_gmem_tensor(k_A, leading_stride=k)
                sme_B = ixdl.make_sme_gmem_tensor(k_B, leading_stride=k)
                smem_a, smem_b = cq_stage_smem_ab(smem_ab_base, stage_base, bm * bk)
                cq_gemm_g2s_issue_operands(
                    warp_id=warp_id,
                    a_per_warp=a_per_warp,
                    b_per_warp=b_per_warp,
                    a_cta_gmem_view=fx.zipped_divide(sme_A, tile_smex),
                    b_cta_gmem_view=fx.zipped_divide(sme_B, tile_smex),
                    smex_atom=smex_atom,
                    smem_a=smem_a,
                    smem_b=smem_b,
                    bm=bm,
                    bn=bn,
                    bk=bk,
                    geom=CQ_GEMM_GEOM,
                )

            def _mma_k_load(stage_base, mma_k):
                smem_a, smem_b = cq_stage_smem_ab(smem_ab_base, stage_base, bm * bk)
                return cq_gemm_s2r_load_mma_k(
                    mma_k=mma_k,
                    smem_a=smem_a,
                    smem_b=smem_b,
                    elem_dtype=elem_dtype,
                    warp_m_id=warp_m_id,
                    warp_n_id=warp_n_id,
                    warp_atoms_m=warp_atoms_m,
                    warp_atoms_n=warp_atoms_n,
                    k_atoms=k_atoms,
                    copy_atom_a=loadn_a,
                    copy_atom_b=loadn_b,
                    thr_copy_a=thr_copy_a,
                    thr_copy_b=thr_copy_b,
                    thr_mma=thr_mma,
                    geom=CQ_GEMM_GEOM,
                )

            def _mma_frags(a_frags, b_frags):
                for mma_n in fx.range_constexpr(warp_atoms_n):
                    for mma_m in fx.range_constexpr(warp_atoms_m):
                        fx.gemm(
                            mma_atom,
                            accs[mma_m][mma_n],
                            a_frags[mma_m],
                            b_frags[mma_n],
                            accs[mma_m][mma_n],
                        )

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

            # Prologue: tile0 G2S -> sync; optional tile1 issue; peel S2R on stage0.
            issue_stage(fx.Int32(0), fx.Int32(0))
            _cq_stage_sync()

            if k_tiles_const >= 2:
                issue_stage(fx.Int32(1), fx.Int32(stage_stride))

            a_def, b_def = _s2r_mma_defer_last(fx.Int32(0))

            def _k_iter_body(k_idx):
                # Finish prior G2S (tile k_idx+1 issued last iter / prologue) before MMA.
                _cq_stage_sync()
                _mma_frags(a_def, b_def)
                k_tile = k_idx + 2
                load_stage_base = fx.Int32(k_idx % 2) * fx.Int32(stage_stride)
                comp_stage_base = load_stage_base ^ fx.Int32(stage_stride)
                issue_stage(fx.Int32(k_tile), load_stage_base)
                _s2r_mma_defer_last_into(comp_stage_base, a_def, b_def)

            if fx.const_expr(main_k_full > 0):
                for k_base in fx.range(0, main_k_full, K_LOOP_UNROLL):
                    for u in fx.range_constexpr(K_LOOP_UNROLL):
                        _k_iter_body(k_base + u)

            if fx.const_expr(main_k_remainder > 0):
                for u in fx.range_constexpr(main_k_remainder):
                    _k_iter_body(main_k_full + u)

            _cq_stage_sync()
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
                out_dtype=elem_dtype,
            )
        elif fx.const_expr(no_c_read_tiled_store):
            mr_hgemm_epilogue_store_tiled(
                lane_id=lane_id,
                accs=accs,
                gC_warp=gC_warp,
                tiled_mma=tiled_mma,
                warp_atoms_m=warp_atoms_m,
                warp_atoms_n=warp_atoms_n,
                out_dtype=elem_dtype,
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


def compile_iluvatar_cq_hgemm(
    *,
    M: int,
    N: int,
    K: int,
    warps_m: int = 2,
    warps_n: int = 2,
    k_atoms: int = DEFAULT_K_ATOMS,
    warp_atoms_m: int = 2,
    warp_atoms_n: int = 2,
    epilogue: str = DEFAULT_EPILOGUE,
    epilogue_store: str = DEFAULT_EPILOGUE_STORE,
    major_pattern: str = DEFAULT_MAJOR_PATTERN,
    elem_dtype=DEFAULT_ELEM_DTYPE,
):
    """Build and return a JIT launch wrapper for the Iluvatar CQ HGEMM.

    Computes D(M,N) = A(M,K) @ B(N,K).T. ``elem_dtype`` is Float16 (default) or
    BFloat16. M/N/K must be multiples of derived bm/bn/bk. Bring-up supports
    ``major_pattern="tn"`` only.
    """
    elem_dtype = _validate_elem_dtype(elem_dtype)
    parse_major_pattern(major_pattern)
    if major_pattern not in SUPPORTED_MAJOR_PATTERNS:
        raise ValueError(
            f"CQ HGEMM bring-up supports major_pattern in {SUPPORTED_MAJOR_PATTERNS}, got {major_pattern!r}"
        )
    if epilogue not in (EPILOGUE_NO_C_READ, EPILOGUE_READ_C_ACCUM):
        raise ValueError(f"unknown epilogue: {epilogue}")
    if k_atoms < 2 or k_atoms % 2:
        raise ValueError(f"CQ HGEMM k_atoms must be a positive even integer (SmexMtx EmPart pairs), got {k_atoms}")

    bm, bn, bk, threads, smem_bytes = _swizzle_cta_shape(
        warps_m,
        warps_n,
        k_atoms,
        warp_atoms_m=warp_atoms_m,
        warp_atoms_n=warp_atoms_n,
    )
    if K % bk:
        raise ValueError(f"K must be a multiple of {bk} (ATOM_K_B16 * k_atoms)")
    if M % bm or N % bn:
        raise ValueError(f"M,N must be multiples of {bm}/{bn} for CQ CTA")
    if not _swizzle_atom_work_ok(bm, bn, bk, warps_m, warps_n):
        raise ValueError(
            f"SmexMtx brick count must divide evenly across {warps_m}x{warps_n} warps; "
            f"try different k_atoms (current BK={bk})"
        )
    if smem_bytes > DEFAULT_SMEM_CAP_BYTES:
        raise ValueError(
            f"CTA smem {smem_bytes} B exceeds device cap {DEFAULT_SMEM_CAP_BYTES} B "
            f"({bm}x{bn}x{bk}, {threads} threads); use smaller tile or k_atoms"
        )

    gemm_kernel, threads, smem_bytes, bm, bn, _bk = _build_pipeline_kernel(
        M,
        N,
        K,
        warps_m,
        warps_n,
        k_atoms,
        warp_atoms_m,
        warp_atoms_n,
        epilogue,
        epilogue_store,
        major_pattern,
        elem_dtype,
    )
    grid = (M // bm, N // bn, 1)
    block = (threads, 1, 1)

    @flyc.jit
    def launch_gemm(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        gemm_kernel(A, B, C).launch(grid=grid, block=block, stream=stream)

    return launch_gemm


__all__ = [
    "ATOM_K_B16",
    "DEFAULT_ELEM_DTYPE",
    "DEFAULT_EPILOGUE",
    "DEFAULT_EPILOGUE_STORE",
    "DEFAULT_K_ATOMS",
    "DEFAULT_MAJOR_PATTERN",
    "DEFAULT_SWIZZLE_CTA",
    "EPILOGUE_READ_C_ACCUM",
    "EPILOGUE_NO_C_READ",
    "EPILOGUE_STORE_SHFL",
    "EPILOGUE_STORE_TILED",
    "MAJOR_PATTERN_CHOICES",
    "SUPPORTED_ELEM_DTYPES",
    "SUPPORTED_MAJOR_PATTERNS",
    "SWIZZLE_CTA_AUTO",
    "SWIZZLE_CTA_PRESETS",
    "SwizzleCtaPreset",
    "_swizzle_atom_work_ok",
    "_swizzle_cta_shape",
    "compile_iluvatar_cq_hgemm",
    "select_swizzle_cta",
]
