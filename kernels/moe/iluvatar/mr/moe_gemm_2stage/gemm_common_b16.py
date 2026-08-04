# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Shared routed b16 MR-MMA projection for the two MoE stages."""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.ixdl as ixdl
from flydsl.expr.typing import Vector as Vec
from kernels.gemm.iluvatar.common import WARP_SIZE, parse_major_pattern
from kernels.gemm.iluvatar.mr.common import (
    ATOM_K_B16,
    ATOM_M,
    ATOM_N,
    DEFAULT_SMEM_CAP_BYTES,
    MR_GEMM_GEOM,
    SMEM_ROWS,
    TCU_LANE_COLS,
    sme_atom_counts,
)
from kernels.gemm.iluvatar.mr.operand_copy import (
    mr_cta_smem_grid,
    mr_g2s_sme_config,
    mr_gemm_g2s_issue_b_warp,
    mr_sme_shared_view,
)
from kernels.gemm.iluvatar.mr.s2r import mr_gemm_s2r_load_mma_k

B16_DTYPES = ("f16", "bf16")
_DTYPE_FX = {"f16": fx.Float16, "bf16": fx.BFloat16}


@dataclass(frozen=True)
class GroupedB16Config:
    bm: int
    bn: int
    bk: int
    threads: int
    n_tiles: int
    smem_bytes: int


def validate_grouped_b16_config(
    *,
    N: int,
    K: int,
    dtype: str,
    topk: int,
    experts: int,
    warps_m: int,
    warps_n: int,
    warp_atoms_m: int,
    warp_atoms_n: int,
    k_atoms: int,
    stages: int,
) -> GroupedB16Config:
    """Validate the public b16 contract and derive the routing CTA geometry."""
    if dtype not in B16_DTYPES:
        raise ValueError(f"dtype must be one of {B16_DTYPES}, got {dtype!r}")
    if min(N, K, topk, experts) <= 0:
        raise ValueError(
            f"N, K, topk and experts must be positive, got {N}, {K}, {topk}, {experts}"
        )
    if topk > 255:
        raise ValueError(
            f"packed routing slot is 8-bit; topk must be <=255, got {topk}"
        )
    if stages != 2:
        raise ValueError(
            f"only a two-stage MR-compatible pipeline is supported, got stages={stages}"
        )
    if min(warps_m, warps_n, warp_atoms_m, warp_atoms_n, k_atoms) <= 0:
        raise ValueError("warp counts, atom counts and k_atoms must be positive")
    if warp_atoms_n % 2:
        raise ValueError(
            "warp_atoms_n must be even for the paired MR shuffle epilogue"
        )

    bm = ATOM_M * warp_atoms_m * warps_m
    bn = ATOM_N * warp_atoms_n * warps_n
    bk = ATOM_K_B16 * k_atoms
    threads = warps_m * warps_n * WARP_SIZE
    if threads > 1024:
        raise ValueError(f"CTA has {threads} threads; Iluvatar limit is 1024")
    if K % bk:
        raise ValueError(f"K={K} must be divisible by bk={bk}")
    if N % ATOM_N:
        raise ValueError(
            f"N={N} must be divisible by the MR output atom width {ATOM_N}"
        )
    if bk % MR_GEMM_GEOM.values_per_sme_row:
        raise ValueError(
            f"bk={bk} must be divisible by SME row width "
            f"{MR_GEMM_GEOM.values_per_sme_row}; use an even k_atoms"
        )

    # SME B bricks are distributed uniformly across all CTA warps.
    _, b_atoms, _, _ = sme_atom_counts(
        parse_major_pattern("tn"),
        bm,
        bn,
        bk,
        values_per_sme_row=MR_GEMM_GEOM.values_per_sme_row,
    )
    num_warps = warps_m * warps_n
    if b_atoms % num_warps:
        raise ValueError(
            f"B SME chunk count {b_atoms} must divide across {num_warps} warps"
        )
    smem_bytes = stages * (bm + bn) * bk * 2
    if smem_bytes > DEFAULT_SMEM_CAP_BYTES:
        raise ValueError(
            f"CTA shared memory {smem_bytes} B exceeds {DEFAULT_SMEM_CAP_BYTES} B"
        )
    return GroupedB16Config(
        bm=bm,
        bn=bn,
        bk=bk,
        threads=threads,
        n_tiles=(N + bn - 1) // bn,
        smem_bytes=smem_bytes,
    )


def build_grouped_b16_kernel(
    *,
    N: int,
    K: int,
    topk: int,
    dtype: str,
    input_has_slots: bool,
    apply_route_weight: bool,
    accumulate: bool,
    warps_m: int,
    warps_n: int,
    warp_atoms_m: int,
    warp_atoms_n: int,
    k_atoms: int,
    stages: int,
    config: GroupedB16Config,
):
    """Build a dense ``W[E,N,K]`` routed MR projection with FP32 accumulation."""
    if accumulate:
        raise ValueError(
            "Iluvatar b16 atomic accumulation is not supported; "
            "use per-slot output followed by reduction"
        )
    elem_dtype = _DTYPE_FX[dtype]
    bm, bn, bk, threads = config.bm, config.bn, config.bk, config.threads
    num_warps = warps_m * warps_n
    warp_m = ATOM_M * warp_atoms_m
    warp_n = ATOM_N * warp_atoms_n
    vpr = MR_GEMM_GEOM.values_per_sme_row
    k_tiles_const = K // bk
    main_k_trip = max(0, k_tiles_const - 2)
    a_stage_elems = bm * bk
    b_stage_elems = bn * bk
    stage_elems = a_stage_elems + b_stage_elems
    smem_elems = stages * stage_elems
    gather_iters = (a_stage_elems + threads - 1) // threads
    _, b_atoms_total, _, _ = sme_atom_counts(
        parse_major_pattern("tn"),
        bm,
        bn,
        bk,
        values_per_sme_row=vpr,
    )
    b_per_warp = b_atoms_total // num_warps
    cta_grid = mr_cta_smem_grid(
        a_mn_major=False,
        b_mn_major=False,
        bm=bm,
        bn=bn,
        bk=bk,
        geom=MR_GEMM_GEOM,
    )

    # Kernel tensor contract (all layouts are dense row-major):
    #   Out:
    #     - shape [tokens, topk, N], dtype = elem_dtype (FP16 or BF16)
    #     - strides [topk*N, N, 1]
    #     - Stage1 specialization: N = 2*inter_dim, storing [gate | up]
    #     - Stage2 specialization: N = model_dim, storing one down-projection
    #       contribution for each (token, top-k slot)
    #   X:
    #     - input_has_slots=False: [tokens, K], strides [K, 1]
    #     - input_has_slots=True:  [tokens, topk, K], strides [topk*K, K, 1]
    #     - dtype = elem_dtype
    #   W:
    #     - shape [experts, N, K], strides [N*K, K, 1]
    #     - dtype = elem_dtype; W[e, n, :] is one Expert output row
    #   sorted_token_ids:
    #     - contiguous int32 [num_sorted_rows]
    #     - low 24 bits = token, high 8 bits = top-k slot
    #   sorted_expert_ids:
    #     - contiguous int32 [num_expert_blocks]
    #     - one Expert ID for every bm consecutive sorted rows
    #   sorted_weights:
    #     - contiguous FP32 [num_sorted_rows], aligned with sorted_token_ids
    #   num_valid_ids:
    #     - contiguous int32 [>=1]; element 0 is the padded valid-row count
    #   tokens_in:
    #     - scalar int32 containing the original token count
    #
    # Grid layout:
    #   blockIdx.x selects a bn-wide output-column tile;
    #   blockIdx.y selects one bm-row Expert group.
    #
    # Main kernel steps:
    #   Step 1: Decode the CTA's Expert, sorted-token row tile, and output-column
    #           tile from blockIdx.y and blockIdx.x.
    #   Step 2: Gather the routed X rows selected by sorted_token_ids into the
    #           Row16b/K-major SME shared-memory layout; zero-fill padded rows.
    #   Step 3: Asynchronously copy the matching W[expert] tile into the B-side
    #           SME shared-memory layout, including output-column tail handling.
    #   Step 4: Double-buffer A and B across K tiles so the next shared-memory
    #           stage is prepared while the current stage is consumed.
    #   Step 5: Load shared-memory fragments into registers and execute
    #           16x16x16 MR MMA operations with FP32 accumulators.
    #   Step 6: Convert the MRMma accumulator lane layout into logical matrix
    #           rows and columns using the HGEMM pairwise shuffle mapping.
    #   Step 7: Optionally apply each route weight, convert to FP16/BF16, and
    #           scatter the result to Out[token, topk_slot, output_column].
    @flyc.kernel(known_block_size=[threads, 1, 1])
    def grouped_b16_kernel(
        Out: fx.Tensor,
        X: fx.Tensor,
        W: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        num_valid_ids: fx.Tensor,
        tokens_in: fx.Int32,
    ):
        # Step 1: Map this CTA and its warps to one Expert row group and one
        # output-column tile.
        tid = fx.thread_idx.x
        warp_id = tid // WARP_SIZE
        lane_id = fx.Int32(fx.lane_id)
        warp_m_id = warp_id // warps_n
        warp_n_id = warp_id % warps_n
        n_base = fx.Int32(fx.block_idx.x) * fx.Int32(bn)
        expert_block = fx.Int32(fx.block_idx.y)
        m_base = expert_block * fx.Int32(bm)
        expert = fx.Int32(sorted_expert_ids[expert_block])
        tokens = fx.Int32(tokens_in)

        @fx.struct
        class MoEPipelineSmem:
            buf: fx.Array[elem_dtype, smem_elems]

        smem_base = fx.SharedAllocator(static=True).allocate(MoEPipelineSmem).peek().buf.ptr

        def _stage_a_ptr(stage_idx):
            return fx.add_offset(
                smem_base,
                fx.make_int_tuple(fx.Int32(stage_idx) * fx.Int32(stage_elems)),
            )

        def _stage_b_ptr(stage_idx):
            return fx.add_offset(
                _stage_a_ptr(stage_idx),
                fx.make_int_tuple(fx.Int32(a_stage_elems)),
            )

        # Tail B bricks are never issued, so initialize both stage banks once.
        b_zero_iters = (stages * b_stage_elems + threads - 1) // threads
        for zi in fx.range_constexpr(b_zero_iters):
            zlin = tid + fx.Int32(zi * threads)
            if zlin < fx.Int32(stages * b_stage_elems):
                stage_idx = zlin // fx.Int32(b_stage_elems)
                stage_off = zlin % fx.Int32(b_stage_elems)
                fx.ptr_store(
                    elem_dtype(0),
                    fx.add_offset(_stage_b_ptr(stage_idx), fx.make_int_tuple(stage_off)),
                )
        fx.gpu.barrier()

        w_expert = fx.add_offset(
            fx.get_iter(W),
            fx.make_int_tuple(expert * fx.Int32(N * K)),
        )
        mma_atom = fx.make_mma_atom(
            ixdl.MRMma(16, 16, 16, elem_dtype, elem_dtype, fx.Float32)
        )
        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout((1, 1, 1), (1, 1, 1)),
        )
        thr_mma = tiled_mma.thr_slice(lane_id)

        # The output is scattered, so this tile is only an allocation shape.
        c_ref = fx.make_view(
            smem_base,
            fx.make_layout((ATOM_M, ATOM_N), (ATOM_N, 1)),
        )
        accs = []
        for _mm in fx.range_constexpr(warp_atoms_m):
            row = []
            for _mn in fx.range_constexpr(warp_atoms_n):
                frag = thr_mma.make_fragment_C(c_ref)
                frag.fill(0)
                row.append(frag)
            accs.append(row)

        g2s_sme = mr_g2s_sme_config(
            a_mn_major=False,
            b_mn_major=False,
            elem_dtype=elem_dtype,
            row_atom=ixdl.MRAsyncCpRow16b,
            row_swizzle=ixdl.SMESwizzle.Row16b,
        )
        copy_atom_s2r_a = fx.make_copy_atom(fx.UniversalCopy32b(), elem_dtype)
        copy_atom_s2r_b = fx.make_copy_atom(fx.UniversalCopy32b(), elem_dtype)
        thr_copy_a = fx.make_tiled_copy_A(copy_atom_s2r_a, tiled_mma).get_slice(lane_id)
        thr_copy_b = fx.make_tiled_copy_B(copy_atom_s2r_b, tiled_mma).get_slice(lane_id)
        tile_smem_b = fx.make_tile(SMEM_ROWS, vpr)

        # Step 2: Gather routed X rows into the A-side SME layout. Invalid
        # padding rows are represented by zeros so every CTA keeps a full tile.
        def gather_a(k_tile, stage_idx):
            smem_a = _stage_a_ptr(stage_idx)
            k_base = fx.Int32(k_tile) * fx.Int32(bk)
            for gi in fx.range_constexpr(gather_iters):
                lin = tid + fx.Int32(gi * threads)
                if lin < fx.Int32(a_stage_elems):
                    row = lin // fx.Int32(bk)
                    col = lin % fx.Int32(bk)
                    sorted_row = m_base + row
                    fused = fx.Int32(sorted_token_ids[sorted_row])
                    token = fused & fx.Int32(0xFFFFFF)
                    slot = fused.shrui(fx.Int32(24))
                    valid = (
                        (sorted_row < fx.Int32(num_valid_ids[0]))
                        & (token < tokens)
                        & (slot < fx.Int32(topk))
                    )
                    token_safe = fx.arith.select(valid, token, fx.Int32(0))
                    slot_safe = fx.arith.select(valid, slot, fx.Int32(0))
                    if fx.const_expr(input_has_slots):
                        x_val = elem_dtype(X[token_safe, slot_safe, k_base + col])
                    else:
                        x_val = elem_dtype(X[token_safe, k_base + col])
                    store_val = fx.arith.select(valid, x_val, elem_dtype(0))

                    # Gather into the same Row16b/K-major SME brick layout consumed
                    # by mr_gemm_s2r_a_tile; a plain [bm,bk] view is not equivalent.
                    brick_m = row // fx.Int32(SMEM_ROWS)
                    brick_k = col // fx.Int32(vpr)
                    within_m = row % fx.Int32(SMEM_ROWS)
                    within_k = col % fx.Int32(vpr)
                    linear_brick = brick_m * fx.Int32(bk // vpr) + brick_k
                    a_brick = mr_sme_shared_view(
                        smem_a,
                        linear_brick * fx.Int32(SMEM_ROWS * vpr),
                        g2s_sme.a_sme_sw,
                        elem_dtype,
                        major=g2s_sme.a_smem_major,
                    )
                    a_brick[within_m, within_k] = store_val

        # Step 3: Issue asynchronous copies for the selected Expert's W tile
        # into the B-side SME layout, skipping pre-zeroed N-tail bricks.
        def issue_b(k_tile, stage_idx):
            smem_b = _stage_b_ptr(stage_idx)
            k_base = fx.Int32(k_tile) * fx.Int32(bk)
            if fx.const_expr(N % bn == 0):
                b_tile = fx.make_view(
                    fx.add_offset(
                        w_expert,
                        fx.make_int_tuple(n_base * fx.Int32(K) + k_base),
                    ),
                    fx.make_layout((bn, bk), (K, 1)),
                )
                sme_b = ixdl.make_sme_gmem_tensor(b_tile, leading_stride=K)
                mr_gemm_g2s_issue_b_warp(
                    a_mn_major=False,
                    b_mn_major=False,
                    warp_id=warp_id,
                    b_per_warp=b_per_warp,
                    b_cta_gmem_view=fx.zipped_divide(sme_b, tile_smem_b),
                    g2s_sme=g2s_sme,
                    smem_b=smem_b,
                    elem_dtype=elem_dtype,
                    bm=bm,
                    bn=bn,
                    bk=bk,
                    geom=MR_GEMM_GEOM,
                )
            else:
                warp_b_start = warp_id * fx.Int32(b_per_warp)
                for bi in fx.range_constexpr(b_per_warp):
                    cta_lin = warp_b_start + fx.Int32(bi)
                    cta_n = cta_lin // fx.Int32(cta_grid.cta_b_k_cnt)
                    cta_k = cta_lin % fx.Int32(cta_grid.cta_b_k_cnt)
                    global_n0 = n_base + cta_n * fx.Int32(SMEM_ROWS)
                    brick_ok = global_n0 + fx.Int32(SMEM_ROWS) <= fx.Int32(N)
                    if brick_ok:
                        b_linear = cta_n * fx.Int32(cta_grid.cta_b_k_cnt) + cta_k
                        b_off = b_linear * fx.Int32(cta_grid.cta_chunk_elems)
                        elem_off = (
                            global_n0 * fx.Int32(K)
                            + k_base
                            + cta_k * fx.Int32(vpr)
                        )
                        b_brick = fx.make_view(
                            fx.add_offset(w_expert, fx.make_int_tuple(elem_off)),
                            fx.make_layout((SMEM_ROWS, vpr), (K, 1)),
                        )
                        sme_brick = ixdl.make_sme_gmem_tensor(
                            b_brick,
                            leading_stride=K,
                        )
                        fx.copy_atom_call(
                            g2s_sme.sme_atom_b,
                            sme_brick,
                            mr_sme_shared_view(
                                smem_b,
                                b_off,
                                g2s_sme.b_sme_sw,
                                elem_dtype,
                                major=g2s_sme.b_smem_major,
                            ),
                        )
            ixdl.cp_async_commit_group()

        # Step 5: Move one shared-memory K slice into register fragments and
        # execute 16x16x16 MR MMA operations with FP32 accumulation.
        def _mma_k_load(stage_idx, mma_k):
            return mr_gemm_s2r_load_mma_k(
                a_mn_major=False,
                b_mn_major=False,
                mma_k=mma_k,
                g2s_sme=g2s_sme,
                smem_a=_stage_a_ptr(stage_idx),
                smem_b=_stage_b_ptr(stage_idx),
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
                    fx.gemm(
                        mma_atom,
                        accs[mma_m][mma_n],
                        a_frags[mma_m],
                        b_frags[mma_n],
                        accs[mma_m][mma_n],
                    )

        def _s2r_mma_defer_last_into(stage_idx, a_def, b_def):
            for mma_k in fx.range_constexpr(k_atoms - 1):
                a_frags, b_frags = _mma_k_load(stage_idx, mma_k)
                _mma_frags(a_frags, b_frags)
            a_last, b_last = _mma_k_load(stage_idx, k_atoms - 1)
            for mma_m in fx.range_constexpr(warp_atoms_m):
                a_def[mma_m].store(a_last[mma_m].load())
            for mma_n in fx.range_constexpr(warp_atoms_n):
                b_def[mma_n].store(b_last[mma_n].load())

        def _s2r_mma_defer_last(stage_idx):
            for mma_k in fx.range_constexpr(k_atoms - 1):
                a_frags, b_frags = _mma_k_load(stage_idx, mma_k)
                _mma_frags(a_frags, b_frags)
            return _mma_k_load(stage_idx, k_atoms - 1)

        def _s2r_mma_all(stage_idx):
            a_frags, b_frags = _s2r_mma_defer_last(stage_idx)
            _mma_frags(a_frags, b_frags)

        # Step 4: Drive the two-stage K pipeline. While one stage is consumed
        # by Step 5, gather/copy prepares the other stage for the next K tile.
        def _issue_stage(k_tile, stage_idx):
            gather_a(k_tile, stage_idx)
            issue_b(k_tile, stage_idx)

        _issue_stage(fx.Int32(0), fx.Int32(0))
        fx.gpu.barrier()
        if k_tiles_const >= 2:
            _issue_stage(fx.Int32(1), fx.Int32(1))

        a_def, b_def = _s2r_mma_defer_last(fx.Int32(0))

        def _k_iter_body(k_idx):
            fx.gpu.barrier()
            _mma_frags(a_def, b_def)
            load_stage = fx.Int32(k_idx % 2)
            comp_stage = load_stage ^ fx.Int32(1)
            _issue_stage(fx.Int32(k_idx + 2), load_stage)
            _s2r_mma_defer_last_into(comp_stage, a_def, b_def)

        for k_idx in fx.range(0, main_k_trip, 1):
            _k_iter_body(k_idx)

        fx.gpu.barrier()
        _mma_frags(a_def, b_def)
        if k_tiles_const >= 2:
            if main_k_trip % 2 == 0:
                _s2r_mma_all(fx.Int32(1))
            else:
                _s2r_mma_all(fx.Int32(0))

        # Step 6: Build the lane selectors used to transform the native MRMma
        # accumulator layout into logical matrix row/column coordinates.
        lane_col = lane_id % fx.Int32(TCU_LANE_COLS)
        lane_row = lane_id // fx.Int32(TCU_LANE_COLS)
        lane_select0 = (
            lane_row * fx.Int32(TCU_LANE_COLS)
            + (lane_col * fx.Int32(2)) % fx.Int32(TCU_LANE_COLS)
        )
        lane_select1 = lane_select0 + fx.Int32(1)
        lane_em = lane_col // fx.Int32(8)
        width = fx.Int32(WARP_SIZE)
        warp_m_base = fx.Int32(warp_m_id) * fx.Int32(warp_m)
        warp_n_base = fx.Int32(warp_n_id) * fx.Int32(warp_n)

        # MRMma C lanes are not row-major. Reproduce hgemm's pairwise shuffle
        # mapping in FP32, then weight, convert, and scatter routed rows.
        for mma_m in fx.range_constexpr(warp_atoms_m):
            for ei in fx.range_constexpr(4):
                local_m = (
                    warp_m_base
                    + fx.Int32(mma_m * ATOM_M + ei * 4)
                    + lane_row
                )
                sorted_row = m_base + local_m
                fused = fx.Int32(sorted_token_ids[sorted_row])
                token = fused & fx.Int32(0xFFFFFF)
                slot = fused.shrui(fx.Int32(24))
                row_valid = (
                    (sorted_row < fx.Int32(num_valid_ids[0]))
                    & (token < tokens)
                    & (slot < fx.Int32(topk))
                )
                token_safe = fx.arith.select(row_valid, token, fx.Int32(0))
                slot_safe = fx.arith.select(row_valid, slot, fx.Int32(0))

                # Step 7: Apply the optional routing weight in FP32. The stores
                # below then convert and scatter to Out[token, slot, column].
                if fx.const_expr(apply_route_weight):
                    route_weight = fx.Float32(sorted_weights[sorted_row])
                else:
                    route_weight = fx.Float32(1.0)

                for phys_n in fx.range_constexpr(
                    0,
                    warp_n,
                    TCU_LANE_COLS * 2,
                ):
                    mma_n0 = phys_n // TCU_LANE_COLS
                    mma_n1 = mma_n0 + 1
                    raw0 = Vec(accs[mma_m][mma_n0].load())[ei]
                    raw1 = Vec(accs[mma_m][mma_n1].load())[ei]
                    raw0_s0 = raw0.shuffle_idx(lane_select0, width)
                    raw1_s0 = raw1.shuffle_idx(lane_select0, width)
                    raw0_s1 = raw0.shuffle_idx(lane_select1, width)
                    raw1_s1 = raw1.shuffle_idx(lane_select1, width)
                    value0 = (lane_em == fx.Int32(0)).select(
                        raw0_s0, raw1_s0
                    )
                    value1 = (lane_em == fx.Int32(0)).select(
                        raw0_s1, raw1_s1
                    )
                    global_n0 = (
                        n_base
                        + warp_n_base
                        + fx.Int32(phys_n)
                        + lane_col * fx.Int32(2)
                    )
                    global_n1 = global_n0 + fx.Int32(1)
                    if row_valid & (global_n0 < fx.Int32(N)):
                        Out[token_safe, slot_safe, global_n0] = (
                            value0 * route_weight
                        ).to(elem_dtype)
                    if row_valid & (global_n1 < fx.Int32(N)):
                        Out[token_safe, slot_safe, global_n1] = (
                            value1 * route_weight
                        ).to(elem_dtype)

    return grouped_b16_kernel


__all__ = [
    "B16_DTYPES",
    "GroupedB16Config",
    "build_grouped_b16_kernel",
    "validate_grouped_b16_config",
]
