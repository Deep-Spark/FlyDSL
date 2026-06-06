# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Staged Iluvatar MR test kernels (G2S -> S2R and G2S -> S2R -> MMA).

These are test-only JIT kernel builders; production HGEMM lives in ``kernels/``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.ixdl as ixdl

from kernels.iluvatar_mr_common import ATOM_K, ATOM_M, ATOM_N, SMEM_ROWS, WARP_SIZE
from kernels.iluvatar_mr_operand_copy import (
    mr_hgemm_g2s_issue_operands,
    mr_pattern_g2s_sme_config,
)
from kernels.iluvatar_mr_s2r import (
    mr_hgemm_s2r_a_tile,
    mr_hgemm_s2r_b_tile,
    mr_hgemm_s2r_copy_a,
    mr_hgemm_s2r_copy_b,
)
from tests.unit.iluvatar_mr_hgemm_test_common import (
    STAGED_WARP_ATOMS_M,
    STAGED_WARP_ATOMS_N,
    STAGED_WARPS_N,
    staged_k_rep_config,
)


def build_mr_g2s_s2r_ki_dump_launch(*, major_pattern: str, k_rep: int, operand: str):
    """Return (launch, brick_k, ki_slices, dump_elems) for G2S -> S2R tile dump.

    ``operand`` is ``"A"`` or ``"B"``. A/B use separate kernels so the JIT cache
    cannot mix up scalar readback destinations.
    """
    if operand == "A":
        return _build_mr_g2s_s2r_a_dump_launch(major_pattern=major_pattern, k_rep=k_rep)
    if operand == "B":
        return _build_mr_g2s_s2r_b_dump_launch(major_pattern=major_pattern, k_rep=k_rep)
    raise ValueError(f"operand must be 'A' or 'B', got {operand!r}")


def _build_mr_g2s_s2r_a_dump_launch(*, major_pattern: str, k_rep: int):
    cfg = staged_k_rep_config(major_pattern=major_pattern, k_rep=k_rep)
    pattern_id = cfg["pattern_id"]
    brick_m = cfg["brick_m"]
    brick_n = cfg["brick_n"]
    brick_k = cfg["brick_k"]
    values_per_sme_row = cfg["values_per_sme_row"]
    threads = cfg["threads"]
    smem_elems = cfg["smem_elems"]
    ki_slices = cfg["ki_slices"]
    a_logical_stride = cfg["a_logical_stride"]
    b_logical_stride = cfg["b_logical_stride"]
    a_per_warp = cfg["a_per_warp"]
    b_per_warp = cfg["b_per_warp"]
    fx_dtype = fx.Float16
    dump_elems = ki_slices * ATOM_M * ATOM_K

    kernel_name = f"g2s_s2r_ki_dump_a_{major_pattern}_k{k_rep}"

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
        g2s_sme = mr_pattern_g2s_sme_config(
            pattern_id,
            fx_dtype,
            row_atom=ixdl.MRAsyncCpRow16b,
            row_swizzle=ixdl.SMESwizzle.Row16b,
        )
        if fx.const_expr(pattern_id == 1 or pattern_id == 3):
            a_leading = brick_m
        else:
            a_leading = brick_k
        if fx.const_expr(pattern_id == 0 or pattern_id == 1):
            b_leading = brick_n
        else:
            b_leading = brick_k

        tile_smem = fx.make_tile(SMEM_ROWS, values_per_sme_row)
        tile_smem_A = (
            fx.make_tile(values_per_sme_row, SMEM_ROWS)
            if fx.const_expr(pattern_id == 1 or pattern_id == 3)
            else tile_smem
        )
        tile_smem_B = (
            fx.make_tile(values_per_sme_row, SMEM_ROWS)
            if fx.const_expr(pattern_id == 0 or pattern_id == 1)
            else tile_smem
        )
        sme_A = ixdl.make_sme_gmem_tensor(g_A[None, None, 0], leading_stride=a_leading)
        sme_B = ixdl.make_sme_gmem_tensor(g_B[None, None, 0], leading_stride=b_leading)

        mr_hgemm_g2s_issue_operands(
            pattern_id=pattern_id,
            warp_id=warp_id,
            a_per_warp=a_per_warp,
            b_per_warp=b_per_warp,
            g_A_div=fx.zipped_divide(sme_A, tile_smem_A),
            g_B_div=fx.zipped_divide(sme_B, tile_smem_B),
            g2s_sme=g2s_sme,
            smem_base=smem_elem_base,
            elem_dtype=fx_dtype,
            bm=brick_m,
            bn=brick_n,
            bk=brick_k,
            stage_base=fx.Int32(0),
            values_per_sme_row=values_per_sme_row,
        )
        ixdl.cp_async_wait_group(0)
        fx.gpu.barrier()

        scalar_atom = fx.make_copy_atom(fx.UniversalCopy16b(), fx_dtype)
        tiled_st_k = fx.make_tiled_copy_tv(
            scalar_atom,
            fx.make_layout((SMEM_ROWS, WARP_SIZE // SMEM_ROWS), (1, SMEM_ROWS)),
            fx.make_layout((1, values_per_sme_row // (WARP_SIZE // SMEM_ROWS)), (1, 1)),
        )
        tiled_st_mn = fx.make_tiled_copy_tv(
            scalar_atom,
            fx.make_layout(
                (values_per_sme_row, WARP_SIZE // values_per_sme_row),
                (1, values_per_sme_row),
            ),
            fx.make_layout((1, SMEM_ROWS // (WARP_SIZE // values_per_sme_row)), (1, 1)),
        )

        for ki in fx.range_constexpr(ki_slices):
            smem_a_tile = mr_hgemm_s2r_a_tile(
                pattern_id=pattern_id,
                im=0,
                ki=ki,
                stage_base=fx.Int32(0),
                g2s_sme=g2s_sme,
                smem_base=smem_elem_base,
                elem_dtype=fx_dtype,
                warp_m_id=warp_m_id,
                warp_atoms_m=STAGED_WARP_ATOMS_M,
                bm=brick_m,
                bn=brick_n,
                bk=brick_k,
                values_per_sme_row=values_per_sme_row,
            )
            if warp_id == fx.Int32(0):
                st_k = tiled_st_k.get_slice(lane_id)
                st_mn = tiled_st_mn.get_slice(lane_id)
                dst = fx.make_view(
                    fx.add_offset(fx.get_iter(Out), fx.Int32(ki * ATOM_M * ATOM_K)),
                    fx.make_layout((ATOM_M, ATOM_K), (ATOM_K, 1)),
                )
                if fx.const_expr(pattern_id == 1 or pattern_id == 3):
                    frag = fx.make_fragment_like(st_mn.partition_S(smem_a_tile))
                    fx.copy(scalar_atom, st_mn.partition_S(smem_a_tile), frag)
                    fx.copy(scalar_atom, frag, st_mn.partition_D(dst))
                else:
                    frag = fx.make_fragment_like(st_k.partition_S(smem_a_tile))
                    fx.copy(scalar_atom, st_k.partition_S(smem_a_tile), frag)
                    fx.copy(scalar_atom, frag, st_k.partition_D(dst))

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


def _build_mr_g2s_s2r_b_dump_launch(*, major_pattern: str, k_rep: int):
    cfg = staged_k_rep_config(major_pattern=major_pattern, k_rep=k_rep)
    pattern_id = cfg["pattern_id"]
    brick_m = cfg["brick_m"]
    brick_n = cfg["brick_n"]
    brick_k = cfg["brick_k"]
    values_per_sme_row = cfg["values_per_sme_row"]
    threads = cfg["threads"]
    smem_elems = cfg["smem_elems"]
    ki_slices = cfg["ki_slices"]
    a_logical_stride = cfg["a_logical_stride"]
    b_logical_stride = cfg["b_logical_stride"]
    a_per_warp = cfg["a_per_warp"]
    b_per_warp = cfg["b_per_warp"]
    fx_dtype = fx.Float16
    dump_elems = ki_slices * ATOM_N * ATOM_K

    kernel_name = f"g2s_s2r_ki_dump_b_{major_pattern}_k{k_rep}"

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
        g2s_sme = mr_pattern_g2s_sme_config(
            pattern_id,
            fx_dtype,
            row_atom=ixdl.MRAsyncCpRow16b,
            row_swizzle=ixdl.SMESwizzle.Row16b,
        )
        if fx.const_expr(pattern_id == 1 or pattern_id == 3):
            a_leading = brick_m
        else:
            a_leading = brick_k
        if fx.const_expr(pattern_id == 0 or pattern_id == 1):
            b_leading = brick_n
        else:
            b_leading = brick_k

        tile_smem = fx.make_tile(SMEM_ROWS, values_per_sme_row)
        tile_smem_A = (
            fx.make_tile(values_per_sme_row, SMEM_ROWS)
            if fx.const_expr(pattern_id == 1 or pattern_id == 3)
            else tile_smem
        )
        tile_smem_B = (
            fx.make_tile(values_per_sme_row, SMEM_ROWS)
            if fx.const_expr(pattern_id == 0 or pattern_id == 1)
            else tile_smem
        )
        sme_A = ixdl.make_sme_gmem_tensor(g_A[None, None, 0], leading_stride=a_leading)
        sme_B = ixdl.make_sme_gmem_tensor(g_B[None, None, 0], leading_stride=b_leading)

        mr_hgemm_g2s_issue_operands(
            pattern_id=pattern_id,
            warp_id=warp_id,
            a_per_warp=a_per_warp,
            b_per_warp=b_per_warp,
            g_A_div=fx.zipped_divide(sme_A, tile_smem_A),
            g_B_div=fx.zipped_divide(sme_B, tile_smem_B),
            g2s_sme=g2s_sme,
            smem_base=smem_elem_base,
            elem_dtype=fx_dtype,
            bm=brick_m,
            bn=brick_n,
            bk=brick_k,
            stage_base=fx.Int32(0),
            values_per_sme_row=values_per_sme_row,
        )
        ixdl.cp_async_wait_group(0)
        fx.gpu.barrier()

        scalar_atom = fx.make_copy_atom(fx.UniversalCopy16b(), fx_dtype)
        tiled_st_k = fx.make_tiled_copy_tv(
            scalar_atom,
            fx.make_layout((SMEM_ROWS, WARP_SIZE // SMEM_ROWS), (1, SMEM_ROWS)),
            fx.make_layout((1, values_per_sme_row // (WARP_SIZE // SMEM_ROWS)), (1, 1)),
        )
        tiled_st_mn = fx.make_tiled_copy_tv(
            scalar_atom,
            fx.make_layout(
                (values_per_sme_row, WARP_SIZE // values_per_sme_row),
                (1, values_per_sme_row),
            ),
            fx.make_layout((1, SMEM_ROWS // (WARP_SIZE // values_per_sme_row)), (1, 1)),
        )

        for ki in fx.range_constexpr(ki_slices):
            smem_b_tile = mr_hgemm_s2r_b_tile(
                pattern_id=pattern_id,
                jn=0,
                ki=ki,
                stage_base=fx.Int32(0),
                g2s_sme=g2s_sme,
                smem_base=smem_elem_base,
                elem_dtype=fx_dtype,
                warp_n_id=warp_n_id,
                warp_atoms_n=STAGED_WARP_ATOMS_N,
                bm=brick_m,
                bn=brick_n,
                bk=brick_k,
                values_per_sme_row=values_per_sme_row,
            )
            if warp_id == fx.Int32(0):
                st_k = tiled_st_k.get_slice(lane_id)
                st_mn = tiled_st_mn.get_slice(lane_id)
                dst = fx.make_view(
                    fx.add_offset(fx.get_iter(Out), fx.Int32(ki * ATOM_N * ATOM_K)),
                    fx.make_layout((ATOM_N, ATOM_K), (ATOM_K, 1)),
                )
                if fx.const_expr(pattern_id == 0 or pattern_id == 1):
                    frag = fx.make_fragment_like(st_mn.partition_S(smem_b_tile))
                    fx.copy(scalar_atom, st_mn.partition_S(smem_b_tile), frag)
                    fx.copy(scalar_atom, frag, st_mn.partition_D(dst))
                else:
                    frag = fx.make_fragment_like(st_k.partition_S(smem_b_tile))
                    fx.copy(scalar_atom, st_k.partition_S(smem_b_tile), frag)
                    fx.copy(scalar_atom, frag, st_k.partition_D(dst))

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


def build_mr_g2s_s2r_mma_warp00_launch(*, major_pattern: str, k_rep: int):
    """Return (launch, brick_k) for warp-00 atom G2S -> S2R -> MMA (no epilogue)."""
    cfg = staged_k_rep_config(major_pattern=major_pattern, k_rep=k_rep)
    pattern_id = cfg["pattern_id"]
    brick_m = cfg["brick_m"]
    brick_n = cfg["brick_n"]
    brick_k = cfg["brick_k"]
    values_per_sme_row = cfg["values_per_sme_row"]
    threads = cfg["threads"]
    smem_elems = cfg["smem_elems"]
    ki_slices = cfg["ki_slices"]
    a_logical_stride = cfg["a_logical_stride"]
    b_logical_stride = cfg["b_logical_stride"]
    a_per_warp = cfg["a_per_warp"]
    b_per_warp = cfg["b_per_warp"]
    fx_dtype = fx.Float16
    kernel_name = f"g2s_s2r_mma_warp00_{major_pattern}_k{k_rep}"

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
        g2s_sme = mr_pattern_g2s_sme_config(
            pattern_id,
            fx_dtype,
            row_atom=ixdl.MRAsyncCpRow16b,
            row_swizzle=ixdl.SMESwizzle.Row16b,
        )
        if fx.const_expr(pattern_id == 1 or pattern_id == 3):
            a_leading = brick_m
        else:
            a_leading = brick_k
        if fx.const_expr(pattern_id == 0 or pattern_id == 1):
            b_leading = brick_n
        else:
            b_leading = brick_k

        tile_smem = fx.make_tile(SMEM_ROWS, values_per_sme_row)
        tile_smem_A = (
            fx.make_tile(values_per_sme_row, SMEM_ROWS)
            if fx.const_expr(pattern_id == 1 or pattern_id == 3)
            else tile_smem
        )
        tile_smem_B = (
            fx.make_tile(values_per_sme_row, SMEM_ROWS)
            if fx.const_expr(pattern_id == 0 or pattern_id == 1)
            else tile_smem
        )
        sme_A = ixdl.make_sme_gmem_tensor(g_A[None, None, 0], leading_stride=a_leading)
        sme_B = ixdl.make_sme_gmem_tensor(g_B[None, None, 0], leading_stride=b_leading)

        mr_hgemm_g2s_issue_operands(
            pattern_id=pattern_id,
            warp_id=warp_id,
            a_per_warp=a_per_warp,
            b_per_warp=b_per_warp,
            g_A_div=fx.zipped_divide(sme_A, tile_smem_A),
            g_B_div=fx.zipped_divide(sme_B, tile_smem_B),
            g2s_sme=g2s_sme,
            smem_base=smem_elem_base,
            elem_dtype=fx_dtype,
            bm=brick_m,
            bn=brick_n,
            bk=brick_k,
            stage_base=fx.Int32(0),
            values_per_sme_row=values_per_sme_row,
        )
        ixdl.cp_async_wait_group(0)
        fx.gpu.barrier()

        if warp_id == fx.Int32(0):
            mma_atom = fx.make_mma_atom(ixdl.MRMma(ATOM_M, ATOM_N, ATOM_K, fx_dtype, fx_dtype, fx.Float32))
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

            for ki in fx.range_constexpr(ki_slices):
                smem_a_tile = mr_hgemm_s2r_a_tile(
                    pattern_id=pattern_id,
                    im=0,
                    ki=ki,
                    stage_base=fx.Int32(0),
                    g2s_sme=g2s_sme,
                    smem_base=smem_elem_base,
                    elem_dtype=fx_dtype,
                    warp_m_id=warp_m_id,
                    warp_atoms_m=STAGED_WARP_ATOMS_M,
                    bm=brick_m,
                    bn=brick_n,
                    bk=brick_k,
                    values_per_sme_row=values_per_sme_row,
                )
                smem_b_tile = mr_hgemm_s2r_b_tile(
                    pattern_id=pattern_id,
                    jn=0,
                    ki=ki,
                    stage_base=fx.Int32(0),
                    g2s_sme=g2s_sme,
                    smem_base=smem_elem_base,
                    elem_dtype=fx_dtype,
                    warp_n_id=warp_n_id,
                    warp_atoms_n=STAGED_WARP_ATOMS_N,
                    bm=brick_m,
                    bn=brick_n,
                    bk=brick_k,
                    values_per_sme_row=values_per_sme_row,
                )
                frag_a = mr_hgemm_s2r_copy_a(
                    copy_atom=copy_atom_s2r_a,
                    thr_copy_a=thr_copy_a,
                    thr_mma=thr_mma,
                    smem_a_tile=smem_a_tile,
                )
                frag_b = mr_hgemm_s2r_copy_b(
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
