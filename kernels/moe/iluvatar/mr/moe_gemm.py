# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar MR MoE grouped GEMM (int8 / int8smooth).

Single sorted grouped projection: ``Out[t,s,:] = dequant(X_row @ W[e].T) [* route_w]``.

A is vector-gathered into row-major SMEM ``[bm, bk]`` (not SME). B uses the
dense MR G2S helpers (``tn`` -> Col SME for k-major B; row atom/swizzle args are
``MRAsyncCpRow8b`` / ``SMESwizzle.Row8b`` for the shared config). MMA is
``MRMma(16,16,32, Int8, Int8, Int32)`` with a 2-stage software pipeline.

Entry: ``compile_iluvatar_mr_moe_gemm(...)`` -> ``@flyc.jit`` launcher.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.ixdl as ixdl
from flydsl.expr.typing import Vector as Vec
from kernels.gemm.iluvatar.common import (
    DEFAULT_MAJOR_PATTERN,
    MAJOR_PATTERN_TN,
    WARP_SIZE,
    parse_major_pattern,
)
from kernels.gemm.iluvatar.mr.common import (
    ATOM_K_B8,
    ATOM_M,
    ATOM_N,
    DEFAULT_SMEM_CAP_BYTES,
    SMEM_ROWS,
    TCU_LANE_COLS,
    MrOperandGeom,
    sme_atom_counts,
)
from kernels.gemm.iluvatar.mr.operand_copy import (
    mr_cta_smem_grid,
    mr_g2s_sme_config,
    mr_gemm_g2s_issue_b_warp,
    mr_sme_shared_view,
)
from kernels.gemm.iluvatar.mr.s2r import mr_gemm_s2r_copy_a, mr_gemm_s2r_copy_b, mr_gemm_s2r_b_tile

MR_GEOM = MrOperandGeom.b8()
I8_VPR = MR_GEOM.values_per_sme_row

QUANT_INT8 = "int8"
QUANT_INT8SMOOTH = "int8smooth"
QUANT_CHOICES = (QUANT_INT8, QUANT_INT8SMOOTH)

OUT_F16 = "f16"
OUT_BF16 = "bf16"
OUT_F32 = "f32"
OUT_CHOICES = (OUT_F16, OUT_BF16, OUT_F32)

_OUT_FX = {
    OUT_F16: fx.Float16,
    OUT_BF16: fx.BFloat16,
    OUT_F32: fx.Float32,
}

DEFAULT_WARPS_M = 2
DEFAULT_WARPS_N = 4
DEFAULT_WARP_ATOMS_M = 1
DEFAULT_WARP_ATOMS_N = 2
DEFAULT_K_REP = 2
DEFAULT_STAGES = 2
K_LOOP_UNROLL = 2


def _cta_shape(warps_m, warps_n, k_rep, warp_atoms_m, warp_atoms_n, stages):
    warp_m = ATOM_M * warp_atoms_m
    warp_n = ATOM_N * warp_atoms_n
    bm = warp_m * warps_m
    bn = warp_n * warps_n
    bk = ATOM_K_B8 * k_rep
    threads = warps_m * warps_n * WARP_SIZE
    a_elems = stages * bm * bk
    b_elems = stages * bn * bk
    return bm, bn, bk, threads, a_elems, b_elems


def _build_moe_gemm_kernel(
    *,
    N: int,
    K: int,
    topk: int,
    quant_mode: str,
    out_dtype_fx,
    apply_route_weight: bool,
    warps_m: int,
    warps_n: int,
    warp_atoms_m: int,
    warp_atoms_n: int,
    k_rep: int,
    stages: int,
):
    is_smooth = quant_mode == QUANT_INT8SMOOTH
    geom = MR_GEOM
    vpr = geom.values_per_sme_row
    warp_m = ATOM_M * warp_atoms_m
    warp_n = ATOM_N * warp_atoms_n
    bm = warp_m * warps_m
    bn = warp_n * warps_n
    bk = ATOM_K_B8 * k_rep
    num_warps = warps_m * warps_n
    threads = num_warps * WARP_SIZE
    n_tiles = (N + bn - 1) // bn
    k_tiles_const = K // bk
    a_stage_elems = bm * bk
    b_stage_elems = bn * bk
    a_elems = stages * a_stage_elems
    b_elems = stages * b_stage_elems
    smem_elems = a_elems + b_elems

    layout = parse_major_pattern(MAJOR_PATTERN_TN)
    _, b_atoms_total, _, _ = sme_atom_counts(layout, bm, bn, bk, values_per_sme_row=vpr)
    b_per_warp = b_atoms_total // num_warps
    main_k_trip = max(0, k_tiles_const - 2)
    main_k_full = (main_k_trip // K_LOOP_UNROLL) * K_LOOP_UNROLL
    main_k_remainder = main_k_trip - main_k_full
    gather_iters = (a_stage_elems + threads - 1) // threads

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def moe_gemm_kernel(
        Out: fx.Tensor,
        X: fx.Tensor,
        W: fx.Tensor,
        scale_x: fx.Tensor,
        scale_w: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        tokens_in: fx.Int32,
    ):
        tid = fx.thread_idx.x
        n_tile = fx.block_idx.x
        expert_block = fx.block_idx.y
        warp_id = tid // WARP_SIZE
        lane_id = fx.Int32(fx.lane_id)
        warp_m_id = warp_id // warps_n
        warp_n_id = warp_id % warps_n

        tokens_i32 = fx.Int32(tokens_in)
        n_base = n_tile * fx.Int32(bn)
        m_base = expert_block * fx.Int32(bm)

        @fx.struct
        class MoESmem:
            buf: fx.Array[fx.Int8, smem_elems]

        smem_base = fx.SharedAllocator(static=True).allocate(MoESmem).peek().buf.ptr
        a_base = smem_base
        b_base = fx.add_offset(smem_base, fx.make_int_tuple(fx.Int32(a_elems)))

        # Zero B SMEM once so N-tail skipped SME chunks stay zero.
        b_zero_iters = (b_elems + threads - 1) // threads
        for zi in fx.range_constexpr(b_zero_iters):
            zidx = tid + fx.Int32(zi * threads)
            if zidx < fx.Int32(b_elems):
                fx.ptr_store(fx.Int8(0), fx.add_offset(b_base, fx.make_int_tuple(zidx)))
        fx.gpu.barrier()

        expert_id = fx.Int32(sorted_expert_ids[expert_block])
        w_expert = fx.add_offset(fx.get_iter(W), fx.make_int_tuple(expert_id * fx.Int32(N * K)))

        mma_atom = fx.make_mma_atom(ixdl.MRMma(ATOM_M, ATOM_N, ATOM_K_B8, fx.Int8, fx.Int8, fx.Int32))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (1, 1, 1)))
        thr_mma = tiled_mma.thr_slice(lane_id)

        # Dummy C tile shape for fragment allocation (scatter epilogue, not tiled store).
        c_ref = fx.make_view(
            fx.add_offset(a_base, fx.make_int_tuple(fx.Int32(0))),
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
            elem_dtype=fx.Int8,
            row_atom=ixdl.MRAsyncCpRow8b,
            row_swizzle=ixdl.SMESwizzle.Row8b,
        )
        copy_atom_s2r_a = fx.make_copy_atom(fx.UniversalCopy8b(), fx.Int8)
        copy_atom_s2r_b = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int8)
        thr_copy_a = fx.make_tiled_copy_A(copy_atom_s2r_a, tiled_mma).get_slice(lane_id)
        thr_copy_b = fx.make_tiled_copy_B(copy_atom_s2r_b, tiled_mma).get_slice(lane_id)
        tile_smem_B = fx.make_tile(SMEM_ROWS, vpr)
        cta_grid = mr_cta_smem_grid(
            a_mn_major=False,
            b_mn_major=False,
            bm=bm,
            bn=bn,
            bk=bk,
            geom=geom,
        )

        def _stage_a_ptr(stage_idx):
            return fx.add_offset(a_base, fx.make_int_tuple(fx.Int32(stage_idx) * fx.Int32(a_stage_elems)))

        def _stage_b_ptr(stage_idx):
            return fx.add_offset(b_base, fx.make_int_tuple(fx.Int32(stage_idx) * fx.Int32(b_stage_elems)))

        def _x_row_index(token, slot):
            if fx.const_expr(is_smooth):
                # Slot-major [topk*tokens, K]: row = slot * tokens + token
                return slot * tokens_i32 + token
            return token

        def gather_a(k_tile, stage_idx):
            smem_a = _stage_a_ptr(stage_idx)
            k_base = fx.Int32(k_tile) * fx.Int32(bk)
            for gi in fx.range_constexpr(gather_iters):
                lin = tid + fx.Int32(gi * threads)
                if lin < fx.Int32(a_stage_elems):
                    row = lin // fx.Int32(bk)
                    col = lin % fx.Int32(bk)
                    fused = fx.Int32(sorted_token_ids[m_base + row])
                    token = fused & fx.Int32((1 << 24) - 1)
                    slot = fused.shrui(fx.Int32(24))
                    valid = fx.arith.cmpi(fx.arith.CmpIPredicate.ult, token, tokens_i32)
                    token_safe = fx.arith.select(valid, token, fx.Int32(0))
                    slot_safe = fx.arith.select(valid, slot, fx.Int32(0))
                    x_row = _x_row_index(token_safe, slot_safe)
                    k_idx = k_base + col
                    x_val = fx.Int8(X[x_row, k_idx])
                    store_val = fx.arith.select(valid, x_val, fx.Int8(0))
                    fx.ptr_store(
                        store_val,
                        fx.add_offset(smem_a, fx.make_int_tuple(row * fx.Int32(bk) + col)),
                    )

        def issue_b(k_tile, stage_idx, commit=True):
            smem_b = _stage_b_ptr(stage_idx)
            k_base = fx.Int32(k_tile) * fx.Int32(bk)
            if fx.const_expr(N % bn == 0):
                b_tile = fx.make_view(
                    fx.add_offset(w_expert, fx.make_int_tuple(n_base * fx.Int32(K) + k_base)),
                    fx.make_layout((bn, bk), (K, 1)),
                )
                sme_B = ixdl.make_sme_gmem_tensor(b_tile, leading_stride=K)
                mr_gemm_g2s_issue_b_warp(
                    a_mn_major=False,
                    b_mn_major=False,
                    warp_id=warp_id,
                    b_per_warp=b_per_warp,
                    b_cta_gmem_view=fx.zipped_divide(sme_B, tile_smem_B),
                    g2s_sme=g2s_sme,
                    smem_b=smem_b,
                    elem_dtype=fx.Int8,
                    bm=bm,
                    bn=bn,
                    bk=bk,
                    geom=geom,
                )
            else:
                # N-tail: issue only SME bricks fully inside N; rest stay zero.
                warp_b_start = warp_id * fx.Int32(b_per_warp)
                for t in fx.range_constexpr(b_per_warp):
                    cta_lin = warp_b_start + fx.Int32(t)
                    cta_n = cta_lin // fx.Int32(cta_grid.cta_b_k_cnt)
                    cta_k = cta_lin % fx.Int32(cta_grid.cta_b_k_cnt)
                    global_n0 = n_base + cta_n * fx.Int32(SMEM_ROWS)
                    brick_ok = fx.arith.cmpi(
                        fx.arith.CmpIPredicate.sle,
                        global_n0 + fx.Int32(SMEM_ROWS),
                        fx.Int32(N),
                    )
                    if brick_ok:
                        b_linear = cta_n * fx.Int32(cta_grid.cta_b_k_cnt) + cta_k
                        b_off = b_linear * fx.Int32(cta_grid.cta_chunk_elems)
                        elem_off = global_n0 * fx.Int32(K) + k_base + cta_k * fx.Int32(vpr)
                        b_brick = fx.make_view(
                            fx.add_offset(w_expert, fx.make_int_tuple(elem_off)),
                            fx.make_layout((SMEM_ROWS, vpr), (K, 1)),
                        )
                        sme_brick = ixdl.make_sme_gmem_tensor(b_brick, leading_stride=K)
                        fx.copy_atom_call(
                            g2s_sme.sme_atom_b,
                            sme_brick,
                            mr_sme_shared_view(
                                smem_b,
                                b_off,
                                g2s_sme.b_sme_sw,
                                fx.Int8,
                                major=g2s_sme.b_smem_major,
                            ),
                        )
            if fx.const_expr(commit):
                ixdl.cp_async_commit_group()

        def _load_a_frags(stage_idx, mma_k):
            smem_a = _stage_a_ptr(stage_idx)
            smem_a_view = fx.make_view(smem_a, fx.make_layout((bm, bk), (bk, 1)))
            a_frags = []
            for mma_m in fx.range_constexpr(warp_atoms_m):
                m_atom = fx.Int32(warp_m_id) * fx.Int32(warp_atoms_m) + fx.Int32(mma_m)
                a_tile = fx.slice(
                    fx.flat_divide(smem_a_view, (ATOM_M, ATOM_K_B8)),
                    (None, None, m_atom, fx.Int32(mma_k)),
                )
                a_frags.append(
                    mr_gemm_s2r_copy_a(
                        copy_atom=copy_atom_s2r_a,
                        thr_copy_a=thr_copy_a,
                        thr_mma=thr_mma,
                        smem_a_tile=a_tile,
                    )
                )
            return a_frags

        def _load_b_frags(stage_idx, mma_k):
            smem_b = _stage_b_ptr(stage_idx)
            b_frags = []
            for mma_n in fx.range_constexpr(warp_atoms_n):
                b_frags.append(
                    mr_gemm_s2r_copy_b(
                        copy_atom=copy_atom_s2r_b,
                        thr_copy_b=thr_copy_b,
                        thr_mma=thr_mma,
                        smem_b_tile=mr_gemm_s2r_b_tile(
                            a_mn_major=False,
                            b_mn_major=False,
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
                        ),
                    )
                )
            return b_frags

        def _mma_k_load(stage_idx, mma_k):
            return _load_a_frags(stage_idx, mma_k), _load_b_frags(stage_idx, mma_k)

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

        def _copy_frag(dst, src):
            dst.store(src.load())

        def _copy_a_frags(dst, src):
            for mma_m in fx.range_constexpr(warp_atoms_m):
                _copy_frag(dst[mma_m], src[mma_m])

        def _copy_b_frags(dst, src):
            for mma_n in fx.range_constexpr(warp_atoms_n):
                _copy_frag(dst[mma_n], src[mma_n])

        def _s2r_mma_defer_last_into(stage_idx, a_def, b_def):
            for mma_k in fx.range_constexpr(k_rep - 1):
                a_frags, b_frags = _mma_k_load(stage_idx, mma_k)
                _mma_frags(a_frags, b_frags)
            a_last, b_last = _mma_k_load(stage_idx, k_rep - 1)
            _copy_a_frags(a_def, a_last)
            _copy_b_frags(b_def, b_last)

        def _s2r_mma_defer_last(stage_idx):
            for mma_k in fx.range_constexpr(k_rep - 1):
                a_frags, b_frags = _mma_k_load(stage_idx, mma_k)
                _mma_frags(a_frags, b_frags)
            return _mma_k_load(stage_idx, k_rep - 1)

        def _s2r_mma_all(stage_idx):
            a_frags, b_frags = _s2r_mma_defer_last(stage_idx)
            _mma_frags(a_frags, b_frags)

        def _issue_stage(k_tile, stage_idx):
            gather_a(k_tile, stage_idx)
            issue_b(k_tile, stage_idx, commit=True)

        # 2-stage double buffer: stage_idx in {0,1} indexes both A and B banks.
        _issue_stage(fx.Int32(0), fx.Int32(0))
        fx.gpu.barrier()

        if k_tiles_const >= 2:
            _issue_stage(fx.Int32(1), fx.Int32(1))
            a_def, b_def = _s2r_mma_defer_last(fx.Int32(0))

            def _k_iter_body(k_idx):
                fx.gpu.barrier()
                k_tile = k_idx + 2
                load_stage = fx.Int32(k_idx % 2)
                comp_stage = load_stage ^ fx.Int32(1)
                _issue_stage(fx.Int32(k_tile), load_stage)
                _mma_frags(a_def, b_def)
                _s2r_mma_defer_last_into(comp_stage, a_def, b_def)

            if fx.const_expr(main_k_full > 0):
                for k_base in fx.range(0, main_k_full, K_LOOP_UNROLL):
                    for u in fx.range_constexpr(K_LOOP_UNROLL):
                        _k_iter_body(k_base + u)

            if fx.const_expr(main_k_remainder > 0):
                for u in fx.range_constexpr(main_k_remainder):
                    _k_iter_body(main_k_full + u)

            fx.gpu.barrier()
            _mma_frags(a_def, b_def)

            if main_k_trip % 2 == 0:
                _s2r_mma_all(fx.Int32(1))
            else:
                _s2r_mma_all(fx.Int32(0))
        else:
            _s2r_mma_all(fx.Int32(0))

        # Epilogue: dequant + optional route weight + scatter to [tokens,topk,N].
        lane_row = lane_id.shrui(fx.Int32(4))
        lane_col = lane_id & fx.Int32(TCU_LANE_COLS - 1)
        warp_m_base = fx.Int32(warp_m_id) * fx.Int32(warp_m)
        warp_n_base = fx.Int32(warp_n_id) * fx.Int32(warp_n)

        for im in fx.range_constexpr(warp_atoms_m):
            loaded = [Vec(accs[im][jn].load()) for jn in range(warp_atoms_n)]
            for ei in fx.range_constexpr(4):
                local_m = warp_m_base + fx.Int32(im * ATOM_M + ei * 4) + lane_row
                fused = fx.Int32(sorted_token_ids[m_base + local_m])
                token = fused & fx.Int32((1 << 24) - 1)
                slot = fused.shrui(fx.Int32(24))
                row_valid = fx.arith.cmpi(fx.arith.CmpIPredicate.ult, token, tokens_i32)
                token_safe = fx.arith.select(row_valid, token, fx.Int32(0))
                slot_safe = fx.arith.select(row_valid, slot, fx.Int32(0))
                x_row = _x_row_index(token_safe, slot_safe)
                sx = fx.Float32(scale_x[x_row])
                if fx.const_expr(apply_route_weight):
                    rw = fx.Float32(sorted_weights[m_base + local_m])
                else:
                    rw = fx.Float32(1.0)
                for jn in fx.range_constexpr(warp_atoms_n):
                    local_n = warp_n_base + fx.Int32(jn * ATOM_N) + lane_col
                    global_n = n_base + local_n
                    n_valid = fx.arith.cmpi(fx.arith.CmpIPredicate.ult, global_n, fx.Int32(N))
                    do_store = row_valid & n_valid
                    global_n_safe = fx.arith.select(n_valid, global_n, fx.Int32(0))
                    sw = fx.Float32(scale_w[expert_id, global_n_safe])
                    acc_f = loaded[jn][ei].to(fx.Float32)
                    out_f = acc_f * sx * sw * rw
                    out_v = out_f.to(out_dtype_fx)
                    if do_store:
                        Out[token_safe, slot_safe, global_n_safe] = out_v

    return moe_gemm_kernel, threads, bm, bn, bk, n_tiles


def compile_iluvatar_mr_moe_gemm(
    *,
    N: int,
    K: int,
    topk: int,
    quant_mode: str = QUANT_INT8,
    out_dtype: str = OUT_F16,
    apply_route_weight: bool = False,
    warps_m: int = DEFAULT_WARPS_M,
    warps_n: int = DEFAULT_WARPS_N,
    warp_atoms_m: int = DEFAULT_WARP_ATOMS_M,
    warp_atoms_n: int = DEFAULT_WARP_ATOMS_N,
    k_rep: int = DEFAULT_K_REP,
    stages: int = DEFAULT_STAGES,
    major_pattern: str = DEFAULT_MAJOR_PATTERN,
    bm: int | None = None,
    bn: int | None = None,
    bk: int | None = None,
):
    """Compile Iluvatar MR MoE grouped GEMM (int8 / int8smooth).

    Returns a ``@flyc.jit`` launcher::

        launch(Out, X, W, scale_x, scale_w,
               sorted_token_ids, sorted_expert_ids, sorted_weights,
               tokens, num_valid_expert_blocks, stream=None)

    Tensor contract:
      - ``quant_mode="int8"``: ``X[tokens,K]``, ``scale_x[tokens]`` (or ``[tokens,1]``)
      - ``quant_mode="int8smooth"``: slot-major ``X[topk*tokens,K]``,
        ``scale_x[topk*tokens]`` with row ``slot*tokens+token``
      - ``W[E,N,K]`` contiguous int8; ``scale_w[E,N]`` or ``[E,N,1]``
      - ``Out[tokens,topk,N]`` in ``out_dtype``
      - routing: packed ``sorted_token_ids`` ``(slot<<24)|token``, sentinel token>=tokens
    """
    if major_pattern != MAJOR_PATTERN_TN:
        raise ValueError(f"only major_pattern 'tn' is supported, got {major_pattern!r}")
    if quant_mode not in QUANT_CHOICES:
        raise ValueError(f"quant_mode must be one of {QUANT_CHOICES}, got {quant_mode!r}")
    if out_dtype not in OUT_CHOICES:
        raise ValueError(f"out_dtype must be one of {OUT_CHOICES}, got {out_dtype!r}")
    if stages != 2:
        raise ValueError(f"V1 only supports stages=2, got {stages}")
    if topk <= 0:
        raise ValueError(f"topk must be > 0, got {topk}")
    if N <= 0 or K <= 0:
        raise ValueError(f"N and K must be > 0, got N={N}, K={K}")

    bm_d, bn_d, bk_d, threads, a_elems, b_elems = _cta_shape(
        warps_m, warps_n, k_rep, warp_atoms_m, warp_atoms_n, stages
    )
    if bm is not None and bm != bm_d:
        raise ValueError(f"bm={bm} inconsistent with warps/atoms (expected {bm_d})")
    if bn is not None and bn != bn_d:
        raise ValueError(f"bn={bn} inconsistent with warps/atoms (expected {bn_d})")
    if bk is not None and bk != bk_d:
        raise ValueError(f"bk={bk} inconsistent with k_rep (expected {bk_d})")
    bm, bn, bk = bm_d, bn_d, bk_d

    if K % bk:
        raise ValueError(f"K must be a multiple of bk={bk} (ATOM_K_B8 * k_rep), got K={K}")
    if bk % I8_VPR:
        raise ValueError(f"bk={bk} must be a multiple of {I8_VPR}; use even k_rep")

    layout = parse_major_pattern(MAJOR_PATTERN_TN)
    num_warps = warps_m * warps_n
    _, b_atoms_total, _, _ = sme_atom_counts(layout, bm, bn, bk, values_per_sme_row=I8_VPR)
    if b_atoms_total % num_warps:
        raise ValueError(
            f"B SME chunk count ({b_atoms_total}) must divide across {warps_m}x{warps_n} warps"
        )

    smem_bytes = a_elems + b_elems  # int8
    if smem_bytes > DEFAULT_SMEM_CAP_BYTES:
        raise ValueError(
            f"CTA smem {smem_bytes} B exceeds device cap {DEFAULT_SMEM_CAP_BYTES} B "
            f"({bm}x{bn}x{bk}, stages={stages})"
        )

    out_dtype_fx = _OUT_FX[out_dtype]
    kernel, threads, bm, bn, bk, n_tiles = _build_moe_gemm_kernel(
        N=N,
        K=K,
        topk=topk,
        quant_mode=quant_mode,
        out_dtype_fx=out_dtype_fx,
        apply_route_weight=apply_route_weight,
        warps_m=warps_m,
        warps_n=warps_n,
        warp_atoms_m=warp_atoms_m,
        warp_atoms_n=warp_atoms_n,
        k_rep=k_rep,
        stages=stages,
    )

    @flyc.jit
    def launch_moe_gemm(
        Out: fx.Tensor,
        X: fx.Tensor,
        W: fx.Tensor,
        scale_x: fx.Tensor,
        scale_w: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        tokens_in: fx.Int32,
        num_valid_expert_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel(
            Out,
            X,
            W,
            scale_x,
            scale_w,
            sorted_token_ids,
            sorted_expert_ids,
            sorted_weights,
            tokens_in,
        ).launch(
            grid=(n_tiles, num_valid_expert_blocks, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    def launch(
        Out,
        X,
        W,
        scale_x,
        scale_w,
        sorted_token_ids,
        sorted_expert_ids,
        sorted_weights,
        tokens,
        num_valid_expert_blocks,
        stream=None,
    ):
        if apply_route_weight and sorted_weights is None:
            raise ValueError("sorted_weights is required when apply_route_weight=True")
        weights = sorted_weights if sorted_weights is not None else sorted_token_ids
        sw = scale_w.squeeze(-1) if hasattr(scale_w, "ndim") and scale_w.ndim == 3 else scale_w
        sx = scale_x.view(-1) if hasattr(scale_x, "view") else scale_x
        if stream is None:
            launch_moe_gemm(
                Out,
                X,
                W,
                sx,
                sw,
                sorted_token_ids,
                sorted_expert_ids,
                weights,
                int(tokens),
                int(num_valid_expert_blocks),
            )
        else:
            launch_moe_gemm(
                Out,
                X,
                W,
                sx,
                sw,
                sorted_token_ids,
                sorted_expert_ids,
                weights,
                int(tokens),
                int(num_valid_expert_blocks),
                stream=stream,
            )

    launch.bm = bm
    launch.bn = bn
    launch.bk = bk
    launch.threads = threads
    launch.n_tiles = n_tiles
    return launch


__all__ = [
    "QUANT_CHOICES",
    "QUANT_INT8",
    "QUANT_INT8SMOOTH",
    "OUT_CHOICES",
    "OUT_F16",
    "OUT_BF16",
    "OUT_F32",
    "compile_iluvatar_mr_moe_gemm",
]
