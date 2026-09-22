# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar MR (ivcore11) tiledMma pipeline HGEMM (f16 / bf16).

Double-buffered smem pipeline: async SME G2S (MRAsyncCpRow16b / MRAsyncCpCol),
make_sme_shared_layout, Ki-deferred S2R/MMA mainloop, UniversalCopy32b S2R.

Entry: compile_iluvatar_mr_hgemm(M=..., N=..., K=..., elem_dtype=fx.Float16|fx.BFloat16)
  -> launch_gemm(A, B, C, stream=...).

Tuning:
  elem_dtype -- A/B (and no_c_read C) element type: Float16 (default) or BFloat16.
    Geometry / SME / S2R are shared; only MRMma multiplicand type differs.
  epilogue no_c_read (default) -- D = A @ B.T, f16/bf16 out, acc zeroed. epilogue_store: shfl (default) or tiled.
  epilogue read_c_accum -- C = A @ B.T + C, fp32 out, load C before MMA.
  epilogue write_c_fp32 -- D = A @ B.T, fp32 out, acc zeroed, no C load.
  crop_m -- when set, A stays M rows (pad) and C is crop_m x N; the store
    writes only those rows. Requires M == bm and no_c_read or write_c_fp32.

  major_pattern -- BLAS layout tags nn/nt/tn (default)/tt on logical A(m,k)/B(n,k); see GemmLayout.
    Default tn: both k-major, PyTorch (m,k)/(n,k) need no host transpose.

  CTA shape -- warps_m/n, warp_atoms_m/n, k_atoms (BK = ATOM_K_B16 * k_atoms).
  SWIZZLE_CTA_PRESETS: 1024 (4x4 warps, 4x4 atoms/warp, 256x256 tile), 2048 (4x8 warps, 4x2 atoms).
  Default compile kwargs: 4x4 warps, 4x4 atoms, k_atoms=2 (BK=32).
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
from kernels.gemm.iluvatar.epilogue import (
    mr_hgemm_epilogue_store_fp32_stp,
    mr_hgemm_epilogue_store_read_c_accum,
    mr_hgemm_epilogue_store_shfl,
    mr_hgemm_epilogue_store_tiled,
)
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

DEFAULT_K_ATOMS = 2  # CTA K-tile: ATOM_K_B16 * k_atoms = 32
STAGES = 2
K_LOOP_UNROLL = 1

EPILOGUE_NO_C_READ = "no_c_read"
EPILOGUE_READ_C_ACCUM = "read_c_accum"
EPILOGUE_WRITE_C_FP32 = "write_c_fp32"
DEFAULT_EPILOGUE = EPILOGUE_NO_C_READ

EPILOGUE_STORE_TILED = "tiled"
EPILOGUE_STORE_SHFL = "shfl"
DEFAULT_EPILOGUE_STORE = EPILOGUE_STORE_SHFL

# A/B (and no_c_read C) element types; both use ATOM_K_B16 / Row16b geometry.
SUPPORTED_ELEM_DTYPES = (fx.Float16, fx.BFloat16)
DEFAULT_ELEM_DTYPE = fx.Float16


def _validate_elem_dtype(elem_dtype):
    if elem_dtype not in SUPPORTED_ELEM_DTYPES:
        names = ", ".join(t.__name__ for t in SUPPORTED_ELEM_DTYPES)
        raise ValueError(f"elem_dtype must be one of {{{names}}}, got {elem_dtype!r}")
    return elem_dtype


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


class SwizzleCtaPreset(NamedTuple):
    """Swizzle-mode CTA: (warps_m x warps_n) warps, each (warp_atoms_m x warp_atoms_n) MMA atoms."""

    name: str
    warps_m: int
    warps_n: int
    warp_atoms_m: int
    warp_atoms_n: int
    default_k_atoms: int


SWIZZLE_CTA_PRESETS: dict[str, SwizzleCtaPreset] = {
    # 16 warps x 64 lanes; warp tile 64x64; CTA 256x256; smem ~64 KiB @ k_atoms=4.
    "1024": SwizzleCtaPreset("1024", 4, 4, 4, 4, 4),
    # 32 warps x 64 lanes; warp tile 64x32; CTA still 256x256; smem ~128 KiB @ k_atoms=4.
    "2048": SwizzleCtaPreset("2048", 4, 8, 4, 2, 4),
}
DEFAULT_SWIZZLE_CTA = "1024"


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
    vpr = MR_GEMM_GEOM.values_per_sme_row
    k_bricks_row = bk // vpr
    a_atoms_total = (bm // SMEM_ROWS) * k_bricks_row
    b_atoms_total = (bn // SMEM_ROWS) * k_bricks_row
    return a_atoms_total % num_warps == 0 and b_atoms_total % num_warps == 0


def _build_swizzle_kernel(
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
    crop_m=None,
    k_loop_unroll: int = K_LOOP_UNROLL,
    n_first: bool = False,
    lda=None,
    ldc=None,
    batch: int = 1,
    a_batch_stride: int = 0,
    b_batch_stride: int = 0,
    c_batch_stride: int = 0,
):
    elem_dtype = _validate_elem_dtype(elem_dtype)
    gemm_layout = parse_major_pattern(major_pattern)
    crop_m = int(crop_m) if crop_m else None
    batch = int(batch)
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    a_mn_major = gemm_layout.a_mn_major
    b_mn_major = gemm_layout.b_mn_major
    load_c = epilogue == EPILOGUE_READ_C_ACCUM
    out_b16 = epilogue == EPILOGUE_NO_C_READ
    no_c_read_shfl_store = out_b16 and epilogue_store == EPILOGUE_STORE_SHFL and crop_m is None
    no_c_read_tiled_store = out_b16 and not no_c_read_shfl_store and crop_m is None
    warp_m = ATOM_M * warp_atoms_m
    warp_n = ATOM_N * warp_atoms_n
    bm = warp_m * warps_m
    bn = warp_n * warps_n
    bk = ATOM_K_B16 * k_atoms
    num_warps = warps_m * warps_n
    threads = num_warps * WARP_SIZE
    vpr = MR_GEMM_GEOM.values_per_sme_row

    assert k % bk == 0
    assert m % bm == 0 and n % bn == 0
    assert bk % vpr == 0
    a_ld = int(lda) if lda is not None else k
    c_ld = int(ldc) if ldc is not None else n
    if a_ld < k:
        raise ValueError(f"lda must be >= K ({k}), got {a_ld}")
    if c_ld < n:
        raise ValueError(f"ldc must be >= N ({n}), got {c_ld}")
    if a_mn_major and a_ld != k:
        raise ValueError("lda != K is only valid for k-major A")
    if (not a_mn_major) and a_ld != k and a_ld % 32 != 0:
        raise ValueError(f"lda must be a multiple of 32 for SME G2S, got {a_ld}")
    if batch > 1:
        if a_batch_stride <= 0 or b_batch_stride <= 0 or c_batch_stride <= 0:
            raise ValueError("batch>1 requires positive A/B/C batch strides")
    use_crop = crop_m is not None
    if use_crop:
        if m != bm:
            raise ValueError(f"crop_m requires M == bm ({bm}), got M={m}")
        if not (1 <= crop_m < m):
            raise ValueError(f"crop_m must be in [1, {m}), got {crop_m}")
        if epilogue not in (EPILOGUE_WRITE_C_FP32, EPILOGUE_NO_C_READ):
            raise ValueError("crop_m requires epilogue=no_c_read or write_c_fp32")
    c_store_elems = crop_m * bn if use_crop else 0
    c_store_iters = _ceil_div(c_store_elems, threads) if use_crop else 0
    c_smem_dtype = fx.Float32 if epilogue == EPILOGUE_WRITE_C_FP32 else elem_dtype

    cta_atoms_m = bm // SMEM_ROWS
    cta_atoms_n = bn // SMEM_ROWS
    k_bricks_row = bk // vpr
    a_atoms_total = cta_atoms_m * k_bricks_row
    b_atoms_total = cta_atoms_n * k_bricks_row
    a_per_warp = a_atoms_total // num_warps
    b_per_warp = b_atoms_total // num_warps
    assert a_atoms_total % num_warps == 0
    assert b_atoms_total % num_warps == 0
    stage_elems = (bm + bn) * bk
    stage_stride = stage_elems
    k_tiles_const = k // bk
    main_k_trip = max(0, k_tiles_const - 2)
    unroll = int(k_loop_unroll)
    if unroll < 1:
        raise ValueError(f"k_loop_unroll must be >= 1, got {unroll}")
    main_k_full = (main_k_trip // unroll) * unroll
    main_k_remainder = main_k_trip - main_k_full

    elem_tag = "bf16" if elem_dtype is fx.BFloat16 else "fp16"
    store_tag = ""
    if epilogue == EPILOGUE_NO_C_READ and epilogue_store != DEFAULT_EPILOGUE_STORE:
        store_tag = f"_{epilogue_store}"
    kname = (
        f"mr_hgemm_{elem_tag}_t{warps_m}x{warps_n}"
        f"_a{warp_atoms_m}x{warp_atoms_n}_k{k_atoms}_{epilogue}{store_tag}"
        + (f"_crop{crop_m}" if crop_m else "")
        + ("_nfirst" if n_first else "")
        + (f"_b{batch}" if batch > 1 else "")
        + (f"_lda{a_ld}" if a_ld != k else "")
        + (f"_ldc{c_ld}" if c_ld != n else "")
    )

    @flyc.kernel(name=kname, known_block_size=[threads, 1, 1])
    def mr_hgemm(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        bid_x, bid_y, bid_z = fx.block_idx
        tile_m = bid_y if fx.const_expr(n_first) else bid_x
        tile_n = bid_x if fx.const_expr(n_first) else bid_y
        warp_id = tid // WARP_SIZE
        lane_id = fx.Int32(fx.lane_id)
        warp_m_id = warp_id // warps_n
        warp_n_id = warp_id % warps_n

        a_ptr = fx.get_iter(A)
        b_ptr = fx.get_iter(B)
        c_ptr = fx.get_iter(C)
        if fx.const_expr(batch > 1):
            a_ptr = fx.add_offset(a_ptr, fx.make_int_tuple(bid_z * fx.Int32(a_batch_stride)))
            b_ptr = fx.add_offset(b_ptr, fx.make_int_tuple(bid_z * fx.Int32(b_batch_stride)))
            c_ptr = fx.add_offset(c_ptr, fx.make_int_tuple(bid_z * fx.Int32(c_batch_stride)))

        if fx.const_expr(a_mn_major):
            a_logical_stride = (1, m)
        else:
            a_logical_stride = (a_ld, 1)
        a_logical = fx.make_view(a_ptr, fx.make_layout((m, k), a_logical_stride))
        gA = fx.slice(fx.flat_divide(a_logical, (bm, bk)), (None, None, tile_m, None))

        if fx.const_expr(b_mn_major):
            b_logical_stride = (1, n)
        else:
            b_logical_stride = (k, 1)
        b_logical = fx.make_view(b_ptr, fx.make_layout((n, k), b_logical_stride))
        gB = fx.slice(fx.flat_divide(b_logical, (bn, bk)), (None, None, tile_n, None))

        # Contiguous static shared memory so stage pick can stay branchless XOR.
        # Split s0/s1 Array symbols cannot XOR element offsets across banks.
        if fx.const_expr(use_crop):

            @fx.struct
            class MrPipelineSmem:
                buf: fx.Array[elem_dtype, stage_elems * STAGES]
                c: fx.Array[c_smem_dtype, bm * bn]

            smem = fx.SharedAllocator(static=True).allocate(MrPipelineSmem).peek()
            smem_ab_base = smem.buf.ptr
            smem_c_view = smem.c.view(fx.make_layout((bm, bn), (bn, 1)))
            gC = smem_c_view
        else:

            @fx.struct
            class MrPipelineSmem:
                buf: fx.Array[elem_dtype, stage_elems * STAGES]

            smem_ab_base = fx.SharedAllocator(static=True).allocate(MrPipelineSmem).peek().buf.ptr
            c_logical = fx.make_view(c_ptr, fx.make_layout((m, n), (c_ld, 1)))
            gC = fx.slice(fx.flat_divide(c_logical, (bm, bn)), (None, None, tile_m, tile_n))
            smem_c_view = None

        mma_atom = fx.make_mma_atom(ixdl.MRMma(ATOM_M, ATOM_N, ATOM_K_B16, elem_dtype, elem_dtype, fx.Float32))
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
            g2s_sme = mr_g2s_sme_config(
                a_mn_major=a_mn_major,
                b_mn_major=b_mn_major,
                elem_dtype=elem_dtype,
                row_atom=ixdl.MRAsyncCpRow16b,
                row_swizzle=ixdl.SMESwizzle.Row16b,
            )

            copy_atom_s2r_a = fx.make_copy_atom(fx.UniversalCopy32b(), elem_dtype)
            copy_atom_s2r_b = fx.make_copy_atom(fx.UniversalCopy32b(), elem_dtype)
            tiled_copy_a = fx.make_tiled_copy_A(copy_atom_s2r_a, tiled_mma)
            tiled_copy_b = fx.make_tiled_copy_B(copy_atom_s2r_b, tiled_mma)
            thr_copy_a = tiled_copy_a.get_slice(lane_id)
            thr_copy_b = tiled_copy_b.get_slice(lane_id)

            tile_smem = fx.make_tile(SMEM_ROWS, vpr)
            tile_smem_A = fx.make_tile(vpr, SMEM_ROWS) if fx.const_expr(a_mn_major) else tile_smem
            tile_smem_B = fx.make_tile(vpr, SMEM_ROWS) if fx.const_expr(b_mn_major) else tile_smem

            def issue_stage(k_tile, stage_base):
                k_A = gA[None, None, k_tile]
                k_B = gB[None, None, k_tile]
                if fx.const_expr(a_mn_major):
                    a_leading = m
                else:
                    a_leading = a_ld
                if fx.const_expr(b_mn_major):
                    b_leading = n
                else:
                    b_leading = k
                sme_A = ixdl.make_sme_gmem_tensor(k_A, leading_stride=a_leading)
                sme_B = ixdl.make_sme_gmem_tensor(k_B, leading_stride=b_leading)
                smem_a, smem_b = mr_stage_smem_ab(smem_ab_base, stage_base, bm * bk)
                mr_gemm_g2s_issue_operands(
                    a_mn_major=a_mn_major,
                    b_mn_major=b_mn_major,
                    warp_id=warp_id,
                    a_per_warp=a_per_warp,
                    b_per_warp=b_per_warp,
                    a_cta_gmem_view=fx.zipped_divide(sme_A, tile_smem_A),
                    b_cta_gmem_view=fx.zipped_divide(sme_B, tile_smem_B),
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
                    copy_atom_a=copy_atom_s2r_a,
                    copy_atom_b=copy_atom_s2r_b,
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

            # Prologue: tile0 G2S -> barrier (IXDL drains g2scnt before barrier);
            # tile1 issue only so Peel S2R on stage0 overlaps tile1 G2S.
            issue_stage(fx.Int32(0), fx.Int32(0))
            fx.gpu.barrier()

            if k_tiles_const >= 2:
                issue_stage(fx.Int32(1), fx.Int32(stage_stride))

            a_def, b_def = _s2r_mma_defer_last(fx.Int32(0))

            def _k_iter_body(k_idx):
                fx.gpu.barrier()
                _mma_frags(a_def, b_def)
                k_tile = k_idx + 2
                load_stage_base = fx.Int32(k_idx % 2) * fx.Int32(stage_stride)
                comp_stage_base = load_stage_base ^ fx.Int32(stage_stride)
                issue_stage(fx.Int32(k_tile), load_stage_base)
                _s2r_mma_defer_last_into(comp_stage_base, a_def, b_def)

            # ROCm-style K-loop: outer scf.for + inner range_constexpr partial unroll.
            if fx.const_expr(main_k_full > 0):
                for k_base in fx.range(0, main_k_full, unroll):
                    for u in fx.range_constexpr(unroll):
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
        if fx.const_expr(use_crop and out_b16):
            mr_hgemm_epilogue_store_tiled(
                lane_id=lane_id,
                accs=accs,
                gC_warp=gC_warp,
                tiled_mma=tiled_mma,
                warp_atoms_m=warp_atoms_m,
                warp_atoms_n=warp_atoms_n,
                out_dtype=elem_dtype,
            )
            fx.gpu.barrier()
            c_out = fx.make_view(c_ptr, fx.make_layout((crop_m, n), (c_ld, 1)))
            gC_n = fx.slice(fx.flat_divide(c_out, (crop_m, bn)), (None, None, 0, tile_n))
            for i in fx.range_constexpr(c_store_iters):
                idx = fx.Int32(i) * fx.Int32(threads) + tid
                if idx < fx.Int32(c_store_elems):
                    row = idx // fx.Int32(bn)
                    col = idx % fx.Int32(bn)
                    gC_n[row, col] = smem_c_view[row, col]
        elif fx.const_expr(no_c_read_shfl_store):
            mr_hgemm_epilogue_store_shfl(
                lane_id=lane_id,
                accs=accs,
                gC_warp=gC_warp,
                c_global_n=c_ld,
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
        elif fx.const_expr(epilogue == EPILOGUE_WRITE_C_FP32):
            if fx.const_expr(use_crop):
                mr_hgemm_epilogue_store_read_c_accum(
                    lane_id=lane_id,
                    accs=accs,
                    gC_warp=gC_warp,
                    tiled_mma=tiled_mma,
                    warp_atoms_m=warp_atoms_m,
                    warp_atoms_n=warp_atoms_n,
                )
                fx.gpu.barrier()
                c_out = fx.make_view(c_ptr, fx.make_layout((crop_m, n), (c_ld, 1)))
                gC_n = fx.slice(
                    fx.flat_divide(c_out, (crop_m, bn)),
                    (None, None, 0, tile_n),
                )
                for i in fx.range_constexpr(c_store_iters):
                    idx = fx.Int32(i) * fx.Int32(threads) + tid
                    if idx < fx.Int32(c_store_elems):
                        row = idx // fx.Int32(bn)
                        col = idx % fx.Int32(bn)
                        gC_n[row, col] = smem_c_view[row, col]
            else:
                mr_hgemm_epilogue_store_fp32_stp(
                    lane_id=lane_id,
                    accs=accs,
                    gC_warp=gC_warp,
                    c_global_n=c_ld,
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
    return mr_hgemm, threads, smem_bytes, bm, bn, bk


def compile_iluvatar_mr_hgemm(
    *,
    M: int,
    N: int,
    K: int,
    warps_m: int = 4,
    warps_n: int = 4,
    k_atoms: int = DEFAULT_K_ATOMS,
    warp_atoms_m: int = 4,
    warp_atoms_n: int = 4,
    epilogue: str = DEFAULT_EPILOGUE,
    epilogue_store: str = DEFAULT_EPILOGUE_STORE,
    major_pattern: str = DEFAULT_MAJOR_PATTERN,
    elem_dtype=DEFAULT_ELEM_DTYPE,
    crop_m=None,
    k_loop_unroll: int = K_LOOP_UNROLL,
    n_first: bool = False,
    lda=None,
    ldc=None,
    batch: int = 1,
    a_batch_stride: int = 0,
    b_batch_stride: int = 0,
    c_batch_stride: int = 0,
):
    """Build and return a JIT launch wrapper for the Iluvatar MR HGEMM.

    Computes D(M,N) = A(M,K) @ B(N,K).T. epilogue selects output dtype and accumulate mode.
    elem_dtype is Float16 (default) or BFloat16 for A/B and no_c_read C.
    M/N/K must be multiples of derived bm/bn/bk or ValueError is raised.
    bm = ATOM_M * warp_atoms_m * warps_m, bn = ATOM_N * warp_atoms_n * warps_n,
    bk = ATOM_K_B16 * k_atoms. See module doc for epilogue and major_pattern.
    lda/ldc are element row strides of k-major A and C (default K / N).
    batch>1 walks A/B/C by the given element batch strides on grid.z.
    """
    elem_dtype = _validate_elem_dtype(elem_dtype)
    parse_major_pattern(major_pattern)
    if epilogue not in (EPILOGUE_NO_C_READ, EPILOGUE_READ_C_ACCUM, EPILOGUE_WRITE_C_FP32):
        raise ValueError(f"unknown epilogue: {epilogue}")

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
        raise ValueError(f"M,N must be multiples of {bm}/{bn} for swizzle CTA")
    if not _swizzle_atom_work_ok(bm, bn, bk, warps_m, warps_n):
        raise ValueError(
            f"SME brick count must divide evenly across {warps_m}x{warps_n} warps; "
            f"try larger k_atoms (current BK={bk})"
        )
    if smem_bytes > DEFAULT_SMEM_CAP_BYTES:
        raise ValueError(
            f"CTA smem {smem_bytes} B exceeds device cap {DEFAULT_SMEM_CAP_BYTES} B "
            f"({bm}x{bn}x{bk}, {threads} threads); use smaller tile or k_atoms"
        )

    gemm_kernel, threads, smem_bytes, bm, bn, _bk = _build_swizzle_kernel(
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
        crop_m=crop_m,
        k_loop_unroll=k_loop_unroll,
        n_first=n_first,
        lda=lda,
        ldc=ldc,
        batch=batch,
        a_batch_stride=a_batch_stride,
        b_batch_stride=b_batch_stride,
        c_batch_stride=c_batch_stride,
    )
    grid_z = int(batch)
    grid = (N // bn, M // bm, grid_z) if n_first else (M // bm, N // bn, grid_z)
    block = (threads, 1, 1)

    @flyc.jit
    def launch_gemm(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        # Static SharedAllocator banks are sized by the compiler; leave launch smem unset.
        gemm_kernel(A, B, C).launch(grid=grid, block=block, stream=stream)

    return launch_gemm


__all__ = [
    "DEFAULT_ELEM_DTYPE",
    "DEFAULT_EPILOGUE",
    "DEFAULT_EPILOGUE_STORE",
    "DEFAULT_K_ATOMS",
    "DEFAULT_MAJOR_PATTERN",
    "DEFAULT_SWIZZLE_CTA",
    "EPILOGUE_READ_C_ACCUM",
    "EPILOGUE_NO_C_READ",
    "EPILOGUE_WRITE_C_FP32",
    "EPILOGUE_STORE_SHFL",
    "EPILOGUE_STORE_TILED",
    "MAJOR_PATTERN_CHOICES",
    "SUPPORTED_ELEM_DTYPES",
    "SWIZZLE_CTA_PRESETS",
    "SwizzleCtaPreset",
    "compile_iluvatar_mr_hgemm",
]
