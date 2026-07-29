# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar MR Global Split-K HGEMM (separate from the non-split-K golden path).

Mirrors the ROCm layout: ``hgemm.py`` stays split-K-free; this module owns the
Global Split-K kernel and launch wrappers.

Entry: ``compile_iluvatar_mr_hgemm_splitk(..., split_k, split_k_mode=...)``
  with ``split_k > 1`` and ``epilogue=no_c_read``.

Modes (see ``splitk_utils``):

* **serial** — ordered load-add-store; per-tile GMEM locks + cmpxchg turnstile.
* **parallel** — fp32 workspace[split_k,M,N] + reduce kernel.
* **atomic** — zero C then scalar UniversalAtomicAdd.
"""

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
    mr_hgemm_epilogue_store_atomic_splitk,
    mr_hgemm_epilogue_store_read_c_accum,
    mr_hgemm_epilogue_store_serial_splitk,
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
from kernels.gemm.iluvatar.mr.hgemm import (
    DEFAULT_ELEM_DTYPE,
    DEFAULT_EPILOGUE,
    DEFAULT_EPILOGUE_STORE,
    DEFAULT_K_ATOMS,
    DEFAULT_SWIZZLE_CTA,
    EPILOGUE_NO_C_READ,
    EPILOGUE_READ_C_ACCUM,
    EPILOGUE_STORE_SHFL,
    EPILOGUE_STORE_TILED,
    K_LOOP_UNROLL,
    STAGES,
    SUPPORTED_ELEM_DTYPES,
    SWIZZLE_CTA_PRESETS,
    SwizzleCtaPreset,
    _swizzle_atom_work_ok,
    _swizzle_cta_shape,
    _validate_elem_dtype,
)
from kernels.gemm.iluvatar.mr.operand_copy import mr_g2s_sme_config, mr_gemm_g2s_issue_operands
from kernels.gemm.iluvatar.mr.s2r import mr_gemm_s2r_load_mma_k
from kernels.gemm.iluvatar.mr.splitk_utils import (
    DEFAULT_SPLIT_K_MODE,
    SPLIT_K_MODE_ATOMIC,
    SPLIT_K_MODE_CHOICES,
    SPLIT_K_MODE_PARALLEL,
    SPLIT_K_MODE_SERIAL,
    build_c_zero_kernel,
    build_splitk_reduce_kernel,
    make_splitk_locks,
    make_splitk_workspace,
    resolve_device_from_tensor,
    serial_turnstile_arrive,
    serial_turnstile_wait,
)


def _build_splitk_kernel(
    m: int,
    n: int,
    k: int,
    warps_m: int,
    warps_n: int,
    k_atoms: int,
    warp_atoms_m: int,
    warp_atoms_n: int,
    *,
    split_k: int,
    split_k_mode: str,
    major_pattern: str = DEFAULT_MAJOR_PATTERN,
    elem_dtype=DEFAULT_ELEM_DTYPE,
):
    """Build the Global Split-K MR HGEMM device kernel (``split_k > 1`` only)."""
    elem_dtype = _validate_elem_dtype(elem_dtype)
    gemm_layout = parse_major_pattern(major_pattern)
    a_mn_major = gemm_layout.a_mn_major
    b_mn_major = gemm_layout.b_mn_major
    is_serial = split_k_mode == SPLIT_K_MODE_SERIAL
    is_parallel = split_k_mode == SPLIT_K_MODE_PARALLEL
    warp_m = ATOM_M * warp_atoms_m
    warp_n = ATOM_N * warp_atoms_n
    bm = warp_m * warps_m
    bn = warp_n * warps_n
    bk = ATOM_K_B16 * k_atoms
    num_warps = warps_m * warps_n
    threads = num_warps * WARP_SIZE
    vpr = MR_GEMM_GEOM.values_per_sme_row
    mn_elems = m * n

    assert split_k > 1
    assert k % bk == 0
    assert m % bm == 0 and n % bn == 0
    assert bk % vpr == 0
    assert k % (split_k * bk) == 0

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
    k_tiles_const = (k // split_k) // bk
    main_k_trip = max(0, k_tiles_const - 2)
    main_k_full = (main_k_trip // K_LOOP_UNROLL) * K_LOOP_UNROLL
    main_k_remainder = main_k_trip - main_k_full
    tiles_n = n // bn

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, aux: fx.Tensor):
        tid = fx.thread_idx.x
        bid_x, bid_y, bid_z = fx.block_idx
        warp_id = tid // WARP_SIZE
        lane_id = fx.Int32(fx.lane_id)
        warp_m_id = warp_id // warps_n
        warp_n_id = warp_id % warps_n

        if fx.const_expr(a_mn_major):
            a_logical_stride = (1, m)
        else:
            a_logical_stride = (k, 1)
        a_logical = fx.make_view(fx.get_iter(A), fx.make_layout((m, k), a_logical_stride))
        gA = fx.slice(fx.flat_divide(a_logical, (bm, bk)), (None, None, bid_x, None))

        if fx.const_expr(b_mn_major):
            b_logical_stride = (1, n)
        else:
            b_logical_stride = (k, 1)
        b_logical = fx.make_view(fx.get_iter(B), fx.make_layout((n, k), b_logical_stride))
        gB = fx.slice(fx.flat_divide(b_logical, (bn, bk)), (None, None, bid_y, None))

        if fx.const_expr(is_parallel):
            # C is fp32 workspace[split_k, M, N]; address the bid_z slice.
            w_ptr = fx.add_offset(fx.get_iter(C), fx.make_int_tuple(fx.Int32(bid_z) * fx.Int32(mn_elems)))
            w_view = fx.make_view(w_ptr, fx.make_layout((m, n), (n, 1)))
            gC = fx.slice(fx.flat_divide(w_view, (bm, bn)), (None, None, bid_x, bid_y))
        else:
            gC = fx.slice(fx.flat_divide(C, (bm, bn)), (None, None, bid_x, bid_y))

        @fx.struct
        class MrPipelineSmem:
            buf: fx.Array[elem_dtype, stage_elems * STAGES]

        smem_ab_base = fx.SharedAllocator(static=True).allocate(MrPipelineSmem).peek().buf.ptr

        mma_atom = fx.make_mma_atom(ixdl.MRMma(ATOM_M, ATOM_N, ATOM_K_B16, elem_dtype, elem_dtype, fx.Float32))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (1, 1, 1)))
        thr_mma = tiled_mma.thr_slice(lane_id)

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
                gmem_k_tile = k_tile + fx.Int32(bid_z) * fx.Int32(k_tiles_const)
                k_A = gA[None, None, gmem_k_tile]
                k_B = gB[None, None, gmem_k_tile]
                if fx.const_expr(a_mn_major):
                    a_leading = m
                else:
                    a_leading = k
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

        if fx.const_expr(is_serial):
            # Per-tile cmpxchg turnstile: compute may overlap across K-slices;
            # only the ordered load-add-store into C is serialized.
            locks_ptr = fx.recast_iter(
                fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global),
                fx.get_iter(aux),
            )
            lock_idx = bid_x * fx.Int32(tiles_n) + bid_y
            lock_ptr = fx.add_offset(locks_ptr, fx.make_int_tuple(lock_idx))
            expected = fx.Int32(bid_z)
            is_t0 = tid == fx.Int32(0)
            if is_t0:
                serial_turnstile_wait(lock_ptr, expected)
            fx.gpu.barrier()
            mr_hgemm_epilogue_store_serial_splitk(
                lane_id=lane_id,
                accs=accs,
                gC_warp=gC_warp,
                tiled_mma=tiled_mma,
                warp_atoms_m=warp_atoms_m,
                warp_atoms_n=warp_atoms_n,
                out_dtype=elem_dtype,
            )
            fx.gpu.barrier()
            if is_t0:
                serial_turnstile_arrive(lock_ptr, expected)
        elif fx.const_expr(is_parallel):
            mr_hgemm_epilogue_store_read_c_accum(
                lane_id=lane_id,
                accs=accs,
                gC_warp=gC_warp,
                tiled_mma=tiled_mma,
                warp_atoms_m=warp_atoms_m,
                warp_atoms_n=warp_atoms_n,
            )
        else:
            # atomic
            mr_hgemm_epilogue_store_atomic_splitk(
                lane_id=lane_id,
                accs=accs,
                gC_warp=gC_warp,
                tiled_mma=tiled_mma,
                warp_atoms_m=warp_atoms_m,
                warp_atoms_n=warp_atoms_n,
                out_dtype=elem_dtype,
            )

    smem_bytes = stage_elems * 2 * STAGES
    return gemm_kernel, threads, smem_bytes, bm, bn, bk


def compile_iluvatar_mr_hgemm_splitk(
    *,
    M: int,
    N: int,
    K: int,
    warps_m: int = 4,
    warps_n: int = 4,
    k_atoms: int = DEFAULT_K_ATOMS,
    warp_atoms_m: int = 4,
    warp_atoms_n: int = 4,
    epilogue: str = EPILOGUE_NO_C_READ,
    major_pattern: str = DEFAULT_MAJOR_PATTERN,
    elem_dtype=DEFAULT_ELEM_DTYPE,
    split_k: int,
    split_k_mode: str = DEFAULT_SPLIT_K_MODE,
):
    """Build a JIT launch wrapper for Iluvatar MR HGEMM Global Split-K.

    Requires ``split_k > 1`` and ``epilogue=no_c_read``. For the non-split-K path
    use ``compile_iluvatar_mr_hgemm`` in ``hgemm.py``.

    Launch signature stays ``(A, B, C, stream=...)``; turnstile locks /
    workspace are allocated and cached inside the wrapper.
    """
    elem_dtype = _validate_elem_dtype(elem_dtype)
    parse_major_pattern(major_pattern)
    if split_k <= 1:
        raise ValueError(f"split_k must be > 1 for the split-K entry point, got {split_k}")
    if split_k_mode not in SPLIT_K_MODE_CHOICES:
        raise ValueError(f"split_k_mode must be one of {SPLIT_K_MODE_CHOICES}, got {split_k_mode!r}")
    if epilogue != EPILOGUE_NO_C_READ:
        raise ValueError(f"split_k requires epilogue={EPILOGUE_NO_C_READ!r}, got {epilogue!r}")

    bm, bn, bk, threads, smem_bytes = _swizzle_cta_shape(
        warps_m,
        warps_n,
        k_atoms,
        warp_atoms_m=warp_atoms_m,
        warp_atoms_n=warp_atoms_n,
    )
    if K % bk:
        raise ValueError(f"K must be a multiple of {bk} (ATOM_K_B16 * k_atoms)")
    if K % (split_k * bk):
        raise ValueError(f"K must be a multiple of split_k * bk = {split_k * bk} for split_k={split_k}")
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

    gemm_kernel, threads, smem_bytes, bm, bn, _bk = _build_splitk_kernel(
        M,
        N,
        K,
        warps_m,
        warps_n,
        k_atoms,
        warp_atoms_m,
        warp_atoms_n,
        split_k=split_k,
        split_k_mode=split_k_mode,
        major_pattern=major_pattern,
        elem_dtype=elem_dtype,
    )
    grid = (M // bm, N // bn, split_k)
    block = (threads, 1, 1)
    tiles_m, tiles_n = M // bm, N // bn
    _cache = {"locks": None, "ws": None, "dummy_aux": None}

    if split_k_mode == SPLIT_K_MODE_SERIAL:
        serial_zero_c = build_c_zero_kernel(bm=bm, bn=bn, n=N, threads=threads, elem_dtype=elem_dtype)
        serial_zero_grid = (tiles_m, tiles_n, 1)

        @flyc.jit
        def _launch_serial(A, B, C, locks, stream: fx.Stream = fx.Stream(None)):
            serial_zero_c(C).launch(grid=serial_zero_grid, block=block, stream=stream)
            gemm_kernel(A, B, C, locks).launch(grid=grid, block=block, stream=stream)

        def launch_gemm_serial(A, B, C, stream=fx.Stream(None)):
            import torch

            device = resolve_device_from_tensor(C)
            if _cache["locks"] is None or _cache["locks"].device != device:
                _cache["locks"] = make_splitk_locks(tiles_m, tiles_n, device)
            fxs = stream if isinstance(stream, fx.Stream) else fx.Stream(stream)
            cuda_stream = fxs.value
            if cuda_stream is None:
                _cache["locks"].zero_()
            else:
                with torch.cuda.stream(cuda_stream):
                    _cache["locks"].zero_()
            return _launch_serial(A, B, C, _cache["locks"], fxs)

        return launch_gemm_serial

    if split_k_mode == SPLIT_K_MODE_PARALLEL:
        reduce_kernel = build_splitk_reduce_kernel(
            bm=bm, bn=bn, m=M, n=N, split_k=split_k, threads=threads, elem_dtype=elem_dtype
        )
        reduce_grid = (tiles_m, tiles_n, 1)

        @flyc.jit
        def _launch_parallel(A, B, workspace, C, aux, stream: fx.Stream = fx.Stream(None)):
            gemm_kernel(A, B, workspace, aux).launch(grid=grid, block=block, stream=stream)
            reduce_kernel(workspace, C).launch(grid=reduce_grid, block=block, stream=stream)

        def launch_gemm_parallel(A, B, C, stream=fx.Stream(None)):
            import torch

            device = resolve_device_from_tensor(C)
            if _cache["ws"] is None or _cache["ws"].device != device:
                _cache["ws"] = make_splitk_workspace(split_k, M, N, device)
            if _cache["dummy_aux"] is None or _cache["dummy_aux"].device != device:
                _cache["dummy_aux"] = torch.zeros((1,), dtype=torch.int32, device=device)
            return _launch_parallel(A, B, _cache["ws"], C, _cache["dummy_aux"], stream)

        return launch_gemm_parallel

    zero_c_kernel = build_c_zero_kernel(bm=bm, bn=bn, n=N, threads=threads, elem_dtype=elem_dtype)
    zero_grid = (tiles_m, tiles_n, 1)

    @flyc.jit
    def _launch_atomic(A, B, C, aux, stream: fx.Stream = fx.Stream(None)):
        zero_c_kernel(C).launch(grid=zero_grid, block=block, stream=stream)
        gemm_kernel(A, B, C, aux).launch(grid=grid, block=block, stream=stream)

    def launch_gemm_atomic(A, B, C, stream=fx.Stream(None)):
        import torch

        device = resolve_device_from_tensor(C)
        if _cache["dummy_aux"] is None or _cache["dummy_aux"].device != device:
            _cache["dummy_aux"] = torch.zeros((1,), dtype=torch.int32, device=device)
        return _launch_atomic(A, B, C, _cache["dummy_aux"], stream)

    return launch_gemm_atomic


__all__ = [
    "DEFAULT_ELEM_DTYPE",
    "DEFAULT_EPILOGUE",
    "DEFAULT_EPILOGUE_STORE",
    "DEFAULT_K_ATOMS",
    "DEFAULT_MAJOR_PATTERN",
    "DEFAULT_SPLIT_K_MODE",
    "DEFAULT_SWIZZLE_CTA",
    "EPILOGUE_NO_C_READ",
    "EPILOGUE_READ_C_ACCUM",
    "EPILOGUE_STORE_SHFL",
    "EPILOGUE_STORE_TILED",
    "MAJOR_PATTERN_CHOICES",
    "SPLIT_K_MODE_ATOMIC",
    "SPLIT_K_MODE_CHOICES",
    "SPLIT_K_MODE_PARALLEL",
    "SPLIT_K_MODE_SERIAL",
    "SUPPORTED_ELEM_DTYPES",
    "SWIZZLE_CTA_PRESETS",
    "SwizzleCtaPreset",
    "compile_iluvatar_mr_hgemm_splitk",
]
