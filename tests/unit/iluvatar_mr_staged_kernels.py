# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Staged Iluvatar MR test kernels (G2S -> S2R and G2S -> S2R -> MMA).

These are test-only JIT kernel builders; production HGEMM lives in ``kernels/``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.ixdl as ixdl
from kernels.gemm.iluvatar.common import WARP_SIZE
from kernels.gemm.iluvatar.mr.common import ATOM_M, ATOM_N, SMEM_ROWS
from kernels.gemm.iluvatar.mr.operand_copy import (
    mr_g2s_sme_config,
    mr_gemm_g2s_issue_operands,
)
from kernels.gemm.iluvatar.mr.s2r import (
    mr_gemm_s2r_a_tile,
    mr_gemm_s2r_b_tile,
    mr_gemm_s2r_copy_a,
    mr_gemm_s2r_copy_b,
)
from tests.unit.iluvatar_mr_hgemm_test_common import (
    STAGED_WARP_ATOMS_M,
    STAGED_WARP_ATOMS_N,
    STAGED_WARPS_N,
    staged_k_atoms_config,
)


def build_mr_g2s_s2r_ki_dump_launch(*, major_pattern: str, k_atoms: int, operand: str):
    """Return (launch, brick_k, ki_slices, dump_elems) for G2S -> S2R fragment dump.

    Uses production ``mr_gemm_s2r_copy_*`` + ``make_tiled_copy_A/B`` readback (not
    scalar smem unpack) so swizzled SME layouts match the MMA path.

    ``operand`` is ``"A"`` or ``"B"``. A/B use separate kernels so the JIT cache
    cannot mix up readback destinations.
    """
    if operand == "A":
        return _build_mr_g2s_s2r_a_dump_launch(major_pattern=major_pattern, k_atoms=k_atoms)
    if operand == "B":
        return _build_mr_g2s_s2r_b_dump_launch(major_pattern=major_pattern, k_atoms=k_atoms)
    raise ValueError(f"operand must be 'A' or 'B', got {operand!r}")


def _build_mr_g2s_s2r_a_dump_launch(*, major_pattern: str, k_atoms: int):
    cfg = staged_k_atoms_config(major_pattern=major_pattern, k_atoms=k_atoms)
    a_mn_major = cfg["a_mn_major"]
    b_mn_major = cfg["b_mn_major"]
    brick_m = cfg["brick_m"]
    brick_n = cfg["brick_n"]
    brick_k = cfg["brick_k"]
    geom = cfg["geom"]
    vpr = geom.values_per_sme_row
    threads = cfg["threads"]
    smem_elems = cfg["smem_elems"]
    ki_slices = cfg["mma_k_slices"]
    a_logical_stride = cfg["a_logical_stride"]
    b_logical_stride = cfg["b_logical_stride"]
    a_per_warp = cfg["a_per_warp"]
    b_per_warp = cfg["b_per_warp"]
    fx_dtype = fx.Float16
    dump_elems = ki_slices * ATOM_M * geom.atom_k

    kernel_name = f"g2s_s2r_ki_dump_a_{major_pattern}_k{k_atoms}"

    @flyc.kernel(known_block_size=[threads, 1, 1], name=kernel_name)
    def g2s_s2r_ki_dump_a_kernel(A: fx.Tensor, B: fx.Tensor, Out: fx.Tensor):
        tid = fx.thread_idx.x
        warp_id = tid // WARP_SIZE
        lane_id = tid % WARP_SIZE
        warp_m_id = warp_id // fx.Int32(STAGED_WARPS_N)

        a_logical_view = fx.make_view(
            fx.get_iter(A),
            fx.make_layout((brick_m, brick_k), a_logical_stride),
        )
        b_logical_view = fx.make_view(
            fx.get_iter(B),
            fx.make_layout((brick_n, brick_k), b_logical_stride),
        )
        g_A = fx.slice(fx.flat_divide(a_logical_view, (brick_m, brick_k)), (None, None, 0, None))
        g_B = fx.slice(fx.flat_divide(b_logical_view, (brick_n, brick_k)), (None, None, 0, None))

        smem_elem_base = fx.recast_iter(
            fx.PointerType.get(fx_dtype.ir_type, fx.AddressSpace.Shared),
            fx.get_dyn_shared(),
        )
        smem_b_base = fx.add_offset(
            smem_elem_base,
            fx.make_int_tuple(fx.Int32(brick_m * brick_k)),
        )
        g2s_sme = mr_g2s_sme_config(
            a_mn_major=a_mn_major,
            b_mn_major=b_mn_major,
            elem_dtype=fx_dtype,
            row_atom=ixdl.MRAsyncCpRow16b,
            row_swizzle=ixdl.SMESwizzle.Row16b,
        )
        if fx.const_expr(a_mn_major):
            a_leading = brick_m
        else:
            a_leading = brick_k
        if fx.const_expr(b_mn_major):
            b_leading = brick_n
        else:
            b_leading = brick_k

        tile_smem = fx.make_tile(SMEM_ROWS, vpr)
        tile_smem_A = fx.make_tile(vpr, SMEM_ROWS) if fx.const_expr(a_mn_major) else tile_smem
        tile_smem_B = fx.make_tile(vpr, SMEM_ROWS) if fx.const_expr(b_mn_major) else tile_smem
        sme_A = ixdl.make_sme_gmem_tensor(g_A[None, None, 0], leading_stride=a_leading)
        sme_B = ixdl.make_sme_gmem_tensor(g_B[None, None, 0], leading_stride=b_leading)

        mr_gemm_g2s_issue_operands(
            a_mn_major=a_mn_major,
            b_mn_major=b_mn_major,
            warp_id=warp_id,
            a_per_warp=a_per_warp,
            b_per_warp=b_per_warp,
            a_cta_gmem_view=fx.zipped_divide(sme_A, tile_smem_A),
            b_cta_gmem_view=fx.zipped_divide(sme_B, tile_smem_B),
            g2s_sme=g2s_sme,
            smem_a=smem_elem_base,
            smem_b=smem_b_base,
            elem_dtype=fx_dtype,
            bm=brick_m,
            bn=brick_n,
            bk=brick_k,
            geom=geom,
        )
        ixdl.cp_async_wait_group(0)
        fx.gpu.barrier()

        if warp_id == fx.Int32(0):
            mma_atom = fx.make_mma_atom(
                ixdl.MRMma(geom.atom_m, geom.atom_n, geom.atom_k, fx_dtype, fx_dtype, fx.Float32)
            )
            tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (1, 1, 1)))
            thr_mma = tiled_mma.thr_slice(lane_id)
            copy_atom_s2r_a = fx.make_copy_atom(fx.UniversalCopy32b(), fx_dtype)
            tiled_copy_a = fx.make_tiled_copy_A(copy_atom_s2r_a, tiled_mma)
            thr_copy_a = tiled_copy_a.get_slice(lane_id)

            for mma_k in fx.range_constexpr(ki_slices):
                smem_a_tile = mr_gemm_s2r_a_tile(
                    a_mn_major=a_mn_major,
                    b_mn_major=b_mn_major,
                    mma_m=0,
                    mma_k=mma_k,
                    g2s_sme=g2s_sme,
                    smem_a=smem_elem_base,
                    elem_dtype=fx_dtype,
                    warp_m_id=warp_m_id,
                    warp_atoms_m=STAGED_WARP_ATOMS_M,
                    bm=brick_m,
                    bn=brick_n,
                    bk=brick_k,
                    geom=geom,
                )
                frag_a = mr_gemm_s2r_copy_a(
                    copy_atom=copy_atom_s2r_a,
                    thr_copy_a=thr_copy_a,
                    thr_mma=thr_mma,
                    smem_a_tile=smem_a_tile,
                )
                # make_tiled_copy_A partition_D lands K-major; host compares after .T
                dst = fx.make_view(
                    fx.add_offset(fx.get_iter(Out), fx.Int32(mma_k * ATOM_M * geom.atom_k)),
                    fx.make_layout((geom.atom_k, ATOM_M), (1, geom.atom_k)),
                )
                fx.copy(
                    copy_atom_s2r_a,
                    thr_copy_a.retile(frag_a),
                    thr_copy_a.partition_D(dst),
                    pred=None,
                )

    @flyc.jit
    def launch_g2s_s2r_ki_dump_a(
        A: fx.Tensor,
        B: fx.Tensor,
        Out: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        g2s_s2r_ki_dump_a_kernel(A, B, Out).launch(
            grid=(1, 1, 1),
            block=(threads, 1, 1),
            smem=smem_elems * 2,
            stream=stream,
        )

    return launch_g2s_s2r_ki_dump_a, brick_k, ki_slices, dump_elems


def _build_mr_g2s_s2r_b_dump_launch(*, major_pattern: str, k_atoms: int):
    cfg = staged_k_atoms_config(major_pattern=major_pattern, k_atoms=k_atoms)
    a_mn_major = cfg["a_mn_major"]
    b_mn_major = cfg["b_mn_major"]
    brick_m = cfg["brick_m"]
    brick_n = cfg["brick_n"]
    brick_k = cfg["brick_k"]
    geom = cfg["geom"]
    vpr = geom.values_per_sme_row
    threads = cfg["threads"]
    smem_elems = cfg["smem_elems"]
    ki_slices = cfg["mma_k_slices"]
    a_logical_stride = cfg["a_logical_stride"]
    b_logical_stride = cfg["b_logical_stride"]
    a_per_warp = cfg["a_per_warp"]
    b_per_warp = cfg["b_per_warp"]
    fx_dtype = fx.Float16
    dump_elems = ki_slices * ATOM_N * geom.atom_k

    kernel_name = f"g2s_s2r_ki_dump_b_{major_pattern}_k{k_atoms}"

    @flyc.kernel(known_block_size=[threads, 1, 1], name=kernel_name)
    def g2s_s2r_ki_dump_b_kernel(A: fx.Tensor, B: fx.Tensor, Out: fx.Tensor):
        tid = fx.thread_idx.x
        warp_id = tid // WARP_SIZE
        lane_id = tid % WARP_SIZE
        warp_n_id = warp_id % fx.Int32(STAGED_WARPS_N)

        a_logical_view = fx.make_view(
            fx.get_iter(A),
            fx.make_layout((brick_m, brick_k), a_logical_stride),
        )
        b_logical_view = fx.make_view(
            fx.get_iter(B),
            fx.make_layout((brick_n, brick_k), b_logical_stride),
        )
        g_A = fx.slice(fx.flat_divide(a_logical_view, (brick_m, brick_k)), (None, None, 0, None))
        g_B = fx.slice(fx.flat_divide(b_logical_view, (brick_n, brick_k)), (None, None, 0, None))

        smem_elem_base = fx.recast_iter(
            fx.PointerType.get(fx_dtype.ir_type, fx.AddressSpace.Shared),
            fx.get_dyn_shared(),
        )
        smem_b_base = fx.add_offset(
            smem_elem_base,
            fx.make_int_tuple(fx.Int32(brick_m * brick_k)),
        )
        g2s_sme = mr_g2s_sme_config(
            a_mn_major=a_mn_major,
            b_mn_major=b_mn_major,
            elem_dtype=fx_dtype,
            row_atom=ixdl.MRAsyncCpRow16b,
            row_swizzle=ixdl.SMESwizzle.Row16b,
        )
        if fx.const_expr(a_mn_major):
            a_leading = brick_m
        else:
            a_leading = brick_k
        if fx.const_expr(b_mn_major):
            b_leading = brick_n
        else:
            b_leading = brick_k

        tile_smem = fx.make_tile(SMEM_ROWS, vpr)
        tile_smem_A = fx.make_tile(vpr, SMEM_ROWS) if fx.const_expr(a_mn_major) else tile_smem
        tile_smem_B = fx.make_tile(vpr, SMEM_ROWS) if fx.const_expr(b_mn_major) else tile_smem
        sme_A = ixdl.make_sme_gmem_tensor(g_A[None, None, 0], leading_stride=a_leading)
        sme_B = ixdl.make_sme_gmem_tensor(g_B[None, None, 0], leading_stride=b_leading)

        mr_gemm_g2s_issue_operands(
            a_mn_major=a_mn_major,
            b_mn_major=b_mn_major,
            warp_id=warp_id,
            a_per_warp=a_per_warp,
            b_per_warp=b_per_warp,
            a_cta_gmem_view=fx.zipped_divide(sme_A, tile_smem_A),
            b_cta_gmem_view=fx.zipped_divide(sme_B, tile_smem_B),
            g2s_sme=g2s_sme,
            smem_a=smem_elem_base,
            smem_b=smem_b_base,
            elem_dtype=fx_dtype,
            bm=brick_m,
            bn=brick_n,
            bk=brick_k,
            geom=geom,
        )
        ixdl.cp_async_wait_group(0)
        fx.gpu.barrier()

        if warp_id == fx.Int32(0):
            mma_atom = fx.make_mma_atom(
                ixdl.MRMma(geom.atom_m, geom.atom_n, geom.atom_k, fx_dtype, fx_dtype, fx.Float32)
            )
            tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (1, 1, 1)))
            thr_mma = tiled_mma.thr_slice(lane_id)
            copy_atom_s2r_b = fx.make_copy_atom(fx.UniversalCopy32b(), fx_dtype)
            tiled_copy_b = fx.make_tiled_copy_B(copy_atom_s2r_b, tiled_mma)
            thr_copy_b = tiled_copy_b.get_slice(lane_id)

            for mma_k in fx.range_constexpr(ki_slices):
                smem_b_tile = mr_gemm_s2r_b_tile(
                    a_mn_major=a_mn_major,
                    b_mn_major=b_mn_major,
                    mma_n=0,
                    mma_k=mma_k,
                    g2s_sme=g2s_sme,
                    smem_b=smem_b_base,
                    elem_dtype=fx_dtype,
                    warp_n_id=warp_n_id,
                    warp_atoms_n=STAGED_WARP_ATOMS_N,
                    bm=brick_m,
                    bn=brick_n,
                    bk=brick_k,
                    geom=geom,
                )
                frag_b = mr_gemm_s2r_copy_b(
                    copy_atom=copy_atom_s2r_b,
                    thr_copy_b=thr_copy_b,
                    thr_mma=thr_mma,
                    smem_b_tile=smem_b_tile,
                )
                dst = fx.make_view(
                    fx.add_offset(fx.get_iter(Out), fx.Int32(mma_k * ATOM_N * geom.atom_k)),
                    fx.make_layout((ATOM_N, geom.atom_k), (geom.atom_k, 1)),
                )
                fx.copy(
                    copy_atom_s2r_b,
                    thr_copy_b.retile(frag_b),
                    thr_copy_b.partition_D(dst),
                    pred=None,
                )

    @flyc.jit
    def launch_g2s_s2r_ki_dump_b(
        A: fx.Tensor,
        B: fx.Tensor,
        Out: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        g2s_s2r_ki_dump_b_kernel(A, B, Out).launch(
            grid=(1, 1, 1),
            block=(threads, 1, 1),
            smem=smem_elems * 2,
            stream=stream,
        )

    return launch_g2s_s2r_ki_dump_b, brick_k, ki_slices, dump_elems


def build_mr_g2s_s2r_mma_warp00_launch(*, major_pattern: str, k_atoms: int):
    """Return (launch, brick_k) for warp-00 atom G2S -> S2R -> MMA (no epilogue)."""
    cfg = staged_k_atoms_config(major_pattern=major_pattern, k_atoms=k_atoms)
    a_mn_major = cfg["a_mn_major"]
    b_mn_major = cfg["b_mn_major"]
    brick_m = cfg["brick_m"]
    brick_n = cfg["brick_n"]
    brick_k = cfg["brick_k"]
    geom = cfg["geom"]
    vpr = geom.values_per_sme_row
    threads = cfg["threads"]
    smem_elems = cfg["smem_elems"]
    ki_slices = cfg["mma_k_slices"]
    a_logical_stride = cfg["a_logical_stride"]
    b_logical_stride = cfg["b_logical_stride"]
    a_per_warp = cfg["a_per_warp"]
    b_per_warp = cfg["b_per_warp"]
    fx_dtype = fx.Float16
    kernel_name = f"g2s_s2r_mma_warp00_{major_pattern}_k{k_atoms}"

    @flyc.kernel(known_block_size=[threads, 1, 1], name=kernel_name)
    def g2s_s2r_mma_warp00_kernel(A: fx.Tensor, B: fx.Tensor, C_out: fx.Tensor):
        tid = fx.thread_idx.x
        warp_id = tid // WARP_SIZE
        lane_id = tid % WARP_SIZE
        warp_m_id = warp_id // fx.Int32(STAGED_WARPS_N)
        warp_n_id = warp_id % fx.Int32(STAGED_WARPS_N)

        a_logical_view = fx.make_view(
            fx.get_iter(A),
            fx.make_layout((brick_m, brick_k), a_logical_stride),
        )
        b_logical_view = fx.make_view(
            fx.get_iter(B),
            fx.make_layout((brick_n, brick_k), b_logical_stride),
        )
        g_A = fx.slice(fx.flat_divide(a_logical_view, (brick_m, brick_k)), (None, None, 0, None))
        g_B = fx.slice(fx.flat_divide(b_logical_view, (brick_n, brick_k)), (None, None, 0, None))

        smem_elem_base = fx.recast_iter(
            fx.PointerType.get(fx_dtype.ir_type, fx.AddressSpace.Shared),
            fx.get_dyn_shared(),
        )
        smem_b_base = fx.add_offset(
            smem_elem_base,
            fx.make_int_tuple(fx.Int32(brick_m * brick_k)),
        )
        g2s_sme = mr_g2s_sme_config(
            a_mn_major=a_mn_major,
            b_mn_major=b_mn_major,
            elem_dtype=fx_dtype,
            row_atom=ixdl.MRAsyncCpRow16b,
            row_swizzle=ixdl.SMESwizzle.Row16b,
        )
        if fx.const_expr(a_mn_major):
            a_leading = brick_m
        else:
            a_leading = brick_k
        if fx.const_expr(b_mn_major):
            b_leading = brick_n
        else:
            b_leading = brick_k

        tile_smem = fx.make_tile(SMEM_ROWS, vpr)
        tile_smem_A = fx.make_tile(vpr, SMEM_ROWS) if fx.const_expr(a_mn_major) else tile_smem
        tile_smem_B = fx.make_tile(vpr, SMEM_ROWS) if fx.const_expr(b_mn_major) else tile_smem
        sme_A = ixdl.make_sme_gmem_tensor(g_A[None, None, 0], leading_stride=a_leading)
        sme_B = ixdl.make_sme_gmem_tensor(g_B[None, None, 0], leading_stride=b_leading)

        mr_gemm_g2s_issue_operands(
            a_mn_major=a_mn_major,
            b_mn_major=b_mn_major,
            warp_id=warp_id,
            a_per_warp=a_per_warp,
            b_per_warp=b_per_warp,
            a_cta_gmem_view=fx.zipped_divide(sme_A, tile_smem_A),
            b_cta_gmem_view=fx.zipped_divide(sme_B, tile_smem_B),
            g2s_sme=g2s_sme,
            smem_a=smem_elem_base,
            smem_b=smem_b_base,
            elem_dtype=fx_dtype,
            bm=brick_m,
            bn=brick_n,
            bk=brick_k,
            geom=geom,
        )
        ixdl.cp_async_wait_group(0)
        fx.gpu.barrier()

        if warp_id == fx.Int32(0):
            mma_atom = fx.make_mma_atom(
                ixdl.MRMma(geom.atom_m, geom.atom_n, geom.atom_k, fx_dtype, fx_dtype, fx.Float32)
            )
            tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (1, 1, 1)))
            thr_mma = tiled_mma.thr_slice(lane_id)

            copy_atom_s2r_a = fx.make_copy_atom(fx.UniversalCopy32b(), fx_dtype)
            copy_atom_s2r_b = fx.make_copy_atom(fx.UniversalCopy32b(), fx_dtype)
            tiled_copy_a = fx.make_tiled_copy_A(copy_atom_s2r_a, tiled_mma)
            tiled_copy_b = fx.make_tiled_copy_B(copy_atom_s2r_b, tiled_mma)
            thr_copy_a = tiled_copy_a.get_slice(lane_id)
            thr_copy_b = tiled_copy_b.get_slice(lane_id)

            copy_atom_c = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
            tiled_copy_c = fx.make_tiled_copy_C(copy_atom_c, tiled_mma)
            thr_copy_c = tiled_copy_c.get_slice(lane_id)

            c_dst = fx.make_view(
                fx.get_iter(C_out),
                fx.make_layout((ATOM_M, ATOM_N), (ATOM_N, 1)),
            )
            acc = thr_mma.make_fragment_C(c_dst)
            acc.fill(0)

            for mma_k in fx.range_constexpr(ki_slices):
                smem_a_tile = mr_gemm_s2r_a_tile(
                    a_mn_major=a_mn_major,
                    b_mn_major=b_mn_major,
                    mma_m=0,
                    mma_k=mma_k,
                    g2s_sme=g2s_sme,
                    smem_a=smem_elem_base,
                    elem_dtype=fx_dtype,
                    warp_m_id=warp_m_id,
                    warp_atoms_m=STAGED_WARP_ATOMS_M,
                    bm=brick_m,
                    bn=brick_n,
                    bk=brick_k,
                    geom=geom,
                )
                smem_b_tile = mr_gemm_s2r_b_tile(
                    a_mn_major=a_mn_major,
                    b_mn_major=b_mn_major,
                    mma_n=0,
                    mma_k=mma_k,
                    g2s_sme=g2s_sme,
                    smem_b=smem_b_base,
                    elem_dtype=fx_dtype,
                    warp_n_id=warp_n_id,
                    warp_atoms_n=STAGED_WARP_ATOMS_N,
                    bm=brick_m,
                    bn=brick_n,
                    bk=brick_k,
                    geom=geom,
                )
                frag_a = mr_gemm_s2r_copy_a(
                    copy_atom=copy_atom_s2r_a,
                    thr_copy_a=thr_copy_a,
                    thr_mma=thr_mma,
                    smem_a_tile=smem_a_tile,
                )
                frag_b = mr_gemm_s2r_copy_b(
                    copy_atom=copy_atom_s2r_b,
                    thr_copy_b=thr_copy_b,
                    thr_mma=thr_mma,
                    smem_b_tile=smem_b_tile,
                )
                fx.gemm(mma_atom, acc, frag_a, frag_b, acc)

            fx.copy(
                copy_atom_c,
                thr_copy_c.retile(acc),
                thr_copy_c.partition_D(c_dst),
                pred=None,
            )

    @flyc.jit
    def launch_g2s_s2r_mma_warp00(
        A: fx.Tensor,
        B: fx.Tensor,
        C_out: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        g2s_s2r_mma_warp00_kernel(A, B, C_out).launch(
            grid=(1, 1, 1),
            block=(threads, 1, 1),
            smem=smem_elems * 2,
            stream=stream,
        )

    return launch_g2s_s2r_mma_warp00, brick_k
