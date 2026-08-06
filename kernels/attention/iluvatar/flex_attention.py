# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar flex-attention forward kernel (V1 + V2-1 varlen).

This is a variant subset of PyTorch flex_attention, not a general ``score_mod``
compiler. It supports f16/bf16 inputs, head dimensions 64 and 128, MHA/GQA,
self/cross-attention, physical sequence padding, causal masking, sliding-window
attention, Gemma-2 softcap, and packed varlen self-attention via ``cu_seqlens``.
"""

import math
from typing import Callable

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.ixdl as ixdl
from flydsl.expr import arith
from flydsl.expr import math as fmath
from flydsl.expr.typing import Vector as Vec
from kernels.gemm.iluvatar.common import GemmLayout
from kernels.gemm.iluvatar.mr.common import (
    ATOM_K_B16,
    ATOM_M,
    ATOM_N,
    MR_GEMM_GEOM,
    SMEM_ROWS,
    TCU_LANE_COLS,
    sme_atom_counts,
)

# NB: kernels.gemm.iluvatar.epilogue is imported lazily inside the launcher build
# to avoid pulling ``byte_permute`` / ``stp_vs_b32`` (recent ixdl intrinsics) at
# module import time; that keeps the validation-only tests runnable against stale
# ``build-fly/python_packages`` Python bindings until a rebuild happens.

WARP_SIZE = 64

# --- Fixed tile parameters ----------------------------------------------------
BLOCK_M = 64
BLOCK_N = 64
WARPS_M = 2
WARPS_N = 2
WARP_ATOMS_M = 2  # per-warp M atom count -> warp M extent = ATOM_M * WARP_ATOMS_M = 32
WARP_ATOMS_N = 2  # per-warp N atom count -> warp N extent = ATOM_N * WARP_ATOMS_N = 32
NUM_WARPS = WARPS_M * WARPS_N  # 4 waves = 256 threads
BLOCK_THREADS = NUM_WARPS * WARP_SIZE

assert BLOCK_M == ATOM_M * WARP_ATOMS_M * WARPS_M
assert BLOCK_N == ATOM_N * WARP_ATOMS_N * WARPS_N
# The cross-warp softmax reduce is unconditional to sidestep AST-rewriter
# branch scoping, so this configuration requires more than one N warp.
assert WARPS_N > 1, "WARPS_N must be > 1 for the SMEM rowmax/rowsum reduce"


# --- Supported variants -------------------------------------------------------
_SUPPORTED_DTYPES = ("f16", "bf16")
_SUPPORTED_D = (64, 128)
_DTYPE_STR_TO_FX = {"f16": fx.Float16, "bf16": fx.BFloat16}


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _phys_seq(seq: int, block: int) -> int:
    """Pad logical sequence length up to a multiple of ``block`` for G2S tiles."""
    return _ceil_div(seq, block) * block


def _validate_scope(
    *,
    B: int,
    H: int,
    Hkv: int,
    Sq: int,
    Skv: int,
    D: int,
    dtype: str,
    is_causal: bool,
    window_size: int | None,
    softcap: float | None,
) -> None:
    """Validate compile-time inputs against the supported V1 envelope."""
    if B <= 0 or H <= 0 or Hkv <= 0 or Sq <= 0 or Skv <= 0 or D <= 0:
        raise ValueError(f"all shape dims must be positive; got B={B} H={H} Hkv={Hkv} Sq={Sq} Skv={Skv} D={D}")
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"dtype must be one of {_SUPPORTED_DTYPES}, got {dtype!r}")
    if D not in _SUPPORTED_D:
        raise ValueError(f"D must be one of {_SUPPORTED_D}, got {D}")
    if H % Hkv != 0:
        raise ValueError(f"H ({H}) must be divisible by Hkv ({Hkv})")
    if is_causal and Sq != Skv:
        raise ValueError(f"is_causal=True requires Sq == Skv (self-attention); got Sq={Sq}, Skv={Skv}")
    if softcap is not None and softcap <= 0:
        raise ValueError(f"softcap must be > 0 when set, got {softcap}")
    if window_size is not None and window_size <= 0:
        raise ValueError(f"window_size must be > 0 when set, got {window_size}")


def _validate_qk_dot_subset(*, H: int, Hkv: int, Sq: int, Skv: int) -> None:
    """Keep the qk_dot helper on MHA and block-aligned self-attention only."""
    if H != Hkv:
        raise NotImplementedError(
            f"qk_dot helper is MHA-only (Hkv=H); got H={H}, Hkv={Hkv}. " "Use compile_iluvatar_flex_attention for GQA."
        )
    if Sq != Skv:
        raise NotImplementedError(
            f"qk_dot helper requires Sq == Skv; got Sq={Sq}, Skv={Skv}. "
            "Use compile_iluvatar_flex_attention for cross-attention."
        )
    if Sq % BLOCK_M != 0 or Skv % BLOCK_N != 0:
        raise NotImplementedError(
            f"qk_dot helper requires Sq/Skv aligned to BLOCK_M/N ({BLOCK_M}/{BLOCK_N}); "
            f"got Sq={Sq}, Skv={Skv}. Use compile_iluvatar_flex_attention for tail."
        )


def _tensor_shape(t) -> tuple[int, ...]:
    return tuple(int(x) for x in t.shape)


def _build_qk_dot_launcher(*, B: int, H: int, Sq: int, Skv: int, D: int, dtype: str, sm_scale_f32: float) -> Callable:
    """Compile a plain ``S = (Q @ K^T) * sm_scale`` development launcher.

    Layouts:
      * Q, K : ``[B, H, S*, D]`` bf16 (or f16), k-major in D.
      * S    : ``[B, H, Sq, Skv]`` fp32 output.

    One CTA processes one (b, h, q-tile) triple and loops over kv-tiles
    internally. K is single-buffered without G2S/S2R pipelining.
    """
    from kernels.gemm.iluvatar.epilogue import mr_hgemm_epilogue_store_read_c_accum
    from kernels.gemm.iluvatar.mr.operand_copy import (
        mr_g2s_sme_config,
        mr_gemm_g2s_issue_a_warp,
        mr_gemm_g2s_issue_b_warp,
    )
    from kernels.gemm.iluvatar.mr.s2r import mr_gemm_s2r_load_mma_k

    elem_dtype = _DTYPE_STR_TO_FX[dtype]
    num_q_tiles = Sq // BLOCK_M
    num_kv_tiles = Skv // BLOCK_N
    bk = D  # Inner MMA K axis is the full head dimension; no D-axis pipeline.
    k_atoms = bk // ATOM_K_B16  # e.g. 128 / 16 = 8

    a_atoms_total, b_atoms_total, _, _ = sme_atom_counts(
        GemmLayout(a_mn_major=False, b_mn_major=False),
        BLOCK_M,
        BLOCK_N,
        bk,
        values_per_sme_row=MR_GEMM_GEOM.values_per_sme_row,
    )
    assert a_atoms_total % NUM_WARPS == 0, f"a_atoms_total={a_atoms_total} must divide NUM_WARPS={NUM_WARPS}"
    assert b_atoms_total % NUM_WARPS == 0, f"b_atoms_total={b_atoms_total} must divide NUM_WARPS={NUM_WARPS}"
    a_per_warp = a_atoms_total // NUM_WARPS
    b_per_warp = b_atoms_total // NUM_WARPS

    q_smem_elems = BLOCK_M * bk
    k_smem_elems = BLOCK_N * bk

    q_row_stride = D
    k_row_stride = D
    s_row_stride = Skv

    q_batch_stride = H * Sq * D
    q_head_stride = Sq * D
    k_batch_stride = H * Skv * D
    k_head_stride = Skv * D
    s_batch_stride = H * Sq * Skv
    s_head_stride = Sq * Skv

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def qk_dot_kernel(Q: fx.Tensor, K: fx.Tensor, S: fx.Tensor):
        # @fx.struct-annotated storage schemas must live inside the traced kernel
        # body so that @flyc.kernel's tracing scope resolves closure variables
        # (elem_dtype / q_smem_elems / k_smem_elems) at struct-definition time.
        # Declaring the schema at the outer launcher-builder scope stores the
        # annotations as unresolved string identifiers (PEP 563 / typing lazy
        # evaluation) and trips ``__dsl_size_of__`` with a "does not implement
        # the Storable protocol" TypeError. Matches HGEMM's ``MrPipelineSmem``
        # placement pattern.
        @fx.struct
        class QKSmem:
            q: fx.Array[elem_dtype, q_smem_elems]
            k: fx.Array[elem_dtype, k_smem_elems]

        bh_idx = fx.block_idx.x
        q_tile_idx = fx.block_idx.y
        tid = fx.thread_idx.x
        warp_id = tid // fx.Int32(WARP_SIZE)
        lane_id = fx.Int32(fx.lane_id)
        warp_m_id = warp_id // fx.Int32(WARPS_N)
        warp_n_id = warp_id % fx.Int32(WARPS_N)
        b_idx = bh_idx // fx.Int32(H)
        h_idx = bh_idx % fx.Int32(H)

        # 4D logical views (batch, head, seq, dim) with contiguous BHSD strides.
        Q_view = fx.make_view(
            fx.get_iter(Q),
            fx.make_layout((B, H, Sq, D), (q_batch_stride, q_head_stride, q_row_stride, 1)),
        )
        K_view = fx.make_view(
            fx.get_iter(K),
            fx.make_layout((B, H, Skv, D), (k_batch_stride, k_head_stride, k_row_stride, 1)),
        )
        S_view = fx.make_view(
            fx.get_iter(S),
            fx.make_layout((B, H, Sq, Skv), (s_batch_stride, s_head_stride, s_row_stride, 1)),
        )

        Q_bh = fx.slice(Q_view, (b_idx, h_idx, None, None))
        K_bh = fx.slice(K_view, (b_idx, h_idx, None, None))
        S_bh = fx.slice(S_view, (b_idx, h_idx, None, None))

        gQ = fx.slice(fx.flat_divide(Q_bh, (BLOCK_M, bk)), (None, None, q_tile_idx, 0))
        gK_all = fx.flat_divide(K_bh, (BLOCK_N, bk))  # (BLOCK_N, bk, num_kv_tiles, 1)
        gS_all = fx.flat_divide(S_bh, (BLOCK_M, BLOCK_N))  # (BM, BN, num_q_tiles, num_kv_tiles)

        smem = fx.SharedAllocator(static=True).allocate(QKSmem).peek()
        q_smem = smem.q.ptr
        k_smem = smem.k.ptr

        mma_atom = fx.make_mma_atom(ixdl.MRMma(ATOM_M, ATOM_N, ATOM_K_B16, elem_dtype, elem_dtype, fx.Float32))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (1, 1, 1)))
        thr_mma = tiled_mma.thr_slice(lane_id)

        copy_atom_s2r_a = fx.make_copy_atom(fx.UniversalCopy32b(), elem_dtype)
        copy_atom_s2r_b = fx.make_copy_atom(fx.UniversalCopy32b(), elem_dtype)
        tiled_copy_a = fx.make_tiled_copy_A(copy_atom_s2r_a, tiled_mma)
        tiled_copy_b = fx.make_tiled_copy_B(copy_atom_s2r_b, tiled_mma)
        thr_copy_a = tiled_copy_a.get_slice(lane_id)
        thr_copy_b = tiled_copy_b.get_slice(lane_id)

        g2s_sme = mr_g2s_sme_config(
            a_mn_major=False,
            b_mn_major=False,
            elem_dtype=elem_dtype,
            row_atom=ixdl.MRAsyncCpRow16b,
            row_swizzle=ixdl.SMESwizzle.Row16b,
        )

        vpr = MR_GEMM_GEOM.values_per_sme_row
        tile_smem_qk = fx.make_tile(SMEM_ROWS, vpr)

        # Template S tile bound to (q_tile_idx, 0) -- only its shape is used to size the
        # per-warp accumulator fragments. Actual gmem addresses come from a runtime
        # (q_tile_idx, kv_idx) slice inside the KV loop.
        gS_template = fx.slice(gS_all, (None, None, q_tile_idx, fx.Int32(0)))
        gS_warp_template = fx.slice(
            fx.flat_divide(gS_template, (ATOM_M * WARP_ATOMS_M, ATOM_N * WARP_ATOMS_N)),
            (None, None, warp_m_id, warp_n_id),
        )
        gS_atoms_template = fx.flat_divide(gS_warp_template, (ATOM_M, ATOM_N))

        S_frags = []
        for mma_m in fx.range_constexpr(WARP_ATOMS_M):
            row = []
            for mma_n in fx.range_constexpr(WARP_ATOMS_N):
                c_tile = fx.slice(gS_atoms_template, (None, None, mma_m, mma_n))
                row.append(thr_mma.make_fragment_C(c_tile))
            S_frags.append(row)

        # ---- G2S Q once (invariant across KV loop) ---------------------------
        sme_Q = ixdl.make_sme_gmem_tensor(gQ, leading_stride=q_row_stride)
        mr_gemm_g2s_issue_a_warp(
            a_mn_major=False,
            b_mn_major=False,
            warp_id=warp_id,
            a_per_warp=a_per_warp,
            a_cta_gmem_view=fx.zipped_divide(sme_Q, tile_smem_qk),
            g2s_sme=g2s_sme,
            smem_a=q_smem,
            elem_dtype=elem_dtype,
            bm=BLOCK_M,
            bn=BLOCK_N,
            bk=bk,
            geom=MR_GEMM_GEOM,
        )
        ixdl.cp_async_commit_group()
        fx.gpu.barrier()

        # ---- Main KV loop -----------------------------------------------------
        for kv_idx in fx.range(0, num_kv_tiles, 1):
            gK = fx.slice(gK_all, (None, None, kv_idx, 0))
            sme_K = ixdl.make_sme_gmem_tensor(gK, leading_stride=k_row_stride)

            mr_gemm_g2s_issue_b_warp(
                a_mn_major=False,
                b_mn_major=False,
                warp_id=warp_id,
                b_per_warp=b_per_warp,
                b_cta_gmem_view=fx.zipped_divide(sme_K, tile_smem_qk),
                g2s_sme=g2s_sme,
                smem_b=k_smem,
                elem_dtype=elem_dtype,
                bm=BLOCK_M,
                bn=BLOCK_N,
                bk=bk,
                geom=MR_GEMM_GEOM,
            )
            ixdl.cp_async_commit_group()
            fx.gpu.barrier()

            for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                for mma_n in fx.range_constexpr(WARP_ATOMS_N):
                    S_frags[mma_m][mma_n].fill(0)

            for mma_k in fx.range_constexpr(k_atoms):
                a_frags, b_frags = mr_gemm_s2r_load_mma_k(
                    a_mn_major=False,
                    b_mn_major=False,
                    mma_k=mma_k,
                    g2s_sme=g2s_sme,
                    smem_a=q_smem,
                    smem_b=k_smem,
                    elem_dtype=elem_dtype,
                    warp_m_id=warp_m_id,
                    warp_n_id=warp_n_id,
                    warp_atoms_m=WARP_ATOMS_M,
                    warp_atoms_n=WARP_ATOMS_N,
                    copy_atom_a=copy_atom_s2r_a,
                    copy_atom_b=copy_atom_s2r_b,
                    thr_copy_a=thr_copy_a,
                    thr_copy_b=thr_copy_b,
                    thr_mma=thr_mma,
                    bm=BLOCK_M,
                    bn=BLOCK_N,
                    bk=bk,
                    geom=MR_GEMM_GEOM,
                )
                for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                    for mma_n in fx.range_constexpr(WARP_ATOMS_N):
                        fx.gemm(
                            mma_atom,
                            S_frags[mma_m][mma_n],
                            a_frags[mma_m],
                            b_frags[mma_n],
                            S_frags[mma_m][mma_n],
                        )

            for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                for mma_n in fx.range_constexpr(WARP_ATOMS_N):
                    acc = S_frags[mma_m][mma_n]
                    acc.store(Vec(acc.load()) * fx.Float32(sm_scale_f32))

            gS_tile = fx.slice(gS_all, (None, None, q_tile_idx, kv_idx))
            gS_warp = fx.slice(
                fx.flat_divide(gS_tile, (ATOM_M * WARP_ATOMS_M, ATOM_N * WARP_ATOMS_N)),
                (None, None, warp_m_id, warp_n_id),
            )
            mr_hgemm_epilogue_store_read_c_accum(
                lane_id=lane_id,
                accs=S_frags,
                gC_warp=gS_warp,
                tiled_mma=tiled_mma,
                warp_atoms_m=WARP_ATOMS_M,
                warp_atoms_n=WARP_ATOMS_N,
            )

            fx.gpu.barrier()

    @flyc.jit
    def launch_qk_dot(
        Q: fx.Tensor,
        K: fx.Tensor,
        S: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        qk_dot_kernel(Q, K, S).launch(
            grid=(B * H, num_q_tiles, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_qk_dot


def _build_flex_attention_launcher(  # noqa: C901  (readability over cyclomatic score)
    *,
    B: int,
    H: int,
    Hkv: int,
    Sq: int,
    Skv: int,
    D: int,
    dtype: str,
    sm_scale_f32: float,
    is_causal: bool = False,
    window_size: int | None = None,
    softcap: float | None = None,
    varlen: bool = False,
) -> Callable:
    """Compile the flex-attention forward launcher.

    FA-1 chain: Q resident in SMEM; per-KV-tile MMA1 (Q @ K^T) -> online softmax
    with rowmax/rowsum cross-warp reduce -> P via SMEM (bf16, plain logical
    ``(BLOCK_M, BLOCK_N)`` layout -- NOT SME-swizzled; the write side uses C
    operand TV via ``make_tiled_copy_C.partition_S`` and the read side uses A
    operand TV via ``make_tiled_copy_A.partition_S``, so both agree on the
    logical MN coordinates without a permutation) -> MMA2 (P @ V, host-side
    transposed V so both operands are k-major "tn") -> divide by rowsum ->
    write bf16 O back to gmem.

    Score modifications after MMA1:
      * softcap set: ``S *= sm_scale`` -> ``softcap * tanh(S/softcap)`` -> ``S *= log2e``
      * softcap None: fused ``S *= (sm_scale * log2e)``
      * then optional causal (``kv > q``) and SWA (``q - kv > window_size``) masks
        in the log2 domain with ``NEG_LARGE_F``.
      * when ``Skv`` is not a multiple of ``BLOCK_N``, also mask
        ``kv_g >= Skv``. Callers must pass contiguous phys-padded Q/K/V/O.

    Loop-carried state per lane: ``ROWS_PER_LANE = WARP_ATOMS_M * 4`` fp32
    ``m_running`` values and the same count of ``l_running`` values, flat-packed
    on the ``fx.range(init=...)`` / ``yield`` boundary. ``m_running`` is
    initialised to ``NEG_LARGE_F`` so
    ``alpha = exp2(m_prev - m_new)`` stays finite under fastmath in the first
    KV iteration. Causal fully-masked KV tiles are skipped by shortening the
    trip count to ``kv_end``; K/V SMEM is 2-stage double-buffered.

    MRMma C-fragment TV (from ``lib/Dialect/FlyIXDL/MR/MmaAtom.cpp``
    ``getThrValLayoutC = FxLayout(FxShape(FxThr(16, 4), FxVal(4)),
    FxStride(FxThr(16, 1), FxVal(4)))``): with CuTe column-major codomain
    convention (M inner, N outer, ``flat = m + n * 16``), the mapping is
    ``(m, n) = (lane_row + 4 * ei, lane_col)`` -- i.e., 4 accums per lane occupy
    4 different rows within the SAME column ``n = lane_col``. So rowmax /
    rowsum reduce over ``mma_n`` (per-``ei`` pairing) and then over the 16
    ``lane_col`` values via ``shuffle_xor(8, 4, 2, 1)``. Each lane ends up
    owning ``WARP_ATOMS_M * 4`` rows (``m = warp_m_id * 32 + mma_m * 16 +
    ei * 4 + lane_row``); the SMEM cross-warp reduce stores from
    ``lane_col == 0`` only.
    """
    from kernels.gemm.iluvatar.epilogue import mr_hgemm_epilogue_store_tiled
    from kernels.gemm.iluvatar.mr.operand_copy import (
        mr_g2s_sme_config,
        mr_gemm_g2s_issue_a_warp,
        mr_gemm_g2s_issue_b_warp,
    )
    from kernels.gemm.iluvatar.mr.s2r import (
        mr_gemm_s2r_b_tile,
        mr_gemm_s2r_copy_a,
        mr_gemm_s2r_copy_b,
        mr_gemm_s2r_load_mma_k,
    )

    if varlen and Sq != Skv:
        raise ValueError(f"varlen self-attn requires Sq == Skv (max_seqlen); got Sq={Sq}, Skv={Skv}")

    elem_dtype = _DTYPE_STR_TO_FX[dtype]
    Sq_phys = _phys_seq(Sq, BLOCK_M)
    Skv_phys = _phys_seq(Skv, BLOCK_N)
    num_q_tiles = Sq_phys // BLOCK_M
    num_kv_tiles = Skv_phys // BLOCK_N
    # Varlen always applies a runtime kv>=seqlen mask; dense only when phys-padded.
    has_kv_tail = bool(varlen) or (Skv < Skv_phys)
    group_size = H // Hkv

    bk_qk = D  # MMA1 inner K = head dim
    bk_pv = BLOCK_N  # MMA2 inner K = kv-tile size
    k_atoms_qk = bk_qk // ATOM_K_B16
    k_atoms_pv = bk_pv // ATOM_K_B16

    # Warp partition:
    #   MMA1 (S = Q @ K^T): S extent per CTA = (BLOCK_M, BLOCK_N) = (64, 64);
    #     WARPS_M x WARPS_N warp grid; warp tile = (32, 32); warp_atoms = (2, 2).
    #   MMA2 (O += P @ V):  O extent per CTA = (BLOCK_M, D) = (64, 128);
    #     same warp grid; warp_m_id still splits M into halves (32 rows/warp),
    #     but warp_n_id now splits D=128 (not BLOCK_N=64), so each warp owns
    #     32 x (D / WARPS_N) = 32 x 64 of O -> warp_atoms_n_pv = 4.
    WARP_ATOMS_N_PV = D // (ATOM_N * WARPS_N)
    assert BLOCK_M * D == BLOCK_M * (ATOM_N * WARP_ATOMS_N_PV * WARPS_N)

    # G2S chunk counts
    a_atoms_qk, b_atoms_qk, _, _ = sme_atom_counts(
        GemmLayout(a_mn_major=False, b_mn_major=False),
        BLOCK_M,
        BLOCK_N,
        bk_qk,
        values_per_sme_row=MR_GEMM_GEOM.values_per_sme_row,
    )
    # V G2S: MMA2 B tile has shape (D, BLOCK_N) k-major, so pass bn=D, bk=BLOCK_N.
    _, b_atoms_v, _, _ = sme_atom_counts(
        GemmLayout(a_mn_major=False, b_mn_major=False),
        BLOCK_M,
        D,
        bk_pv,
        values_per_sme_row=MR_GEMM_GEOM.values_per_sme_row,
    )
    assert a_atoms_qk % NUM_WARPS == 0, f"a_atoms_qk={a_atoms_qk} must divide NUM_WARPS={NUM_WARPS}"
    assert b_atoms_qk % NUM_WARPS == 0, f"b_atoms_qk={b_atoms_qk} must divide NUM_WARPS={NUM_WARPS}"
    assert b_atoms_v % NUM_WARPS == 0, f"b_atoms_v={b_atoms_v} must divide NUM_WARPS={NUM_WARPS}"
    a_per_warp_qk = a_atoms_qk // NUM_WARPS
    b_per_warp_qk = b_atoms_qk // NUM_WARPS
    b_per_warp_pv = b_atoms_v // NUM_WARPS

    # SMEM sizing (bf16 = 2 B): Q+K+V+P = 16+16+16+8 KiB = 56 KiB at D=128
    # for a single K/V bank; K/V are doubled for the 2-stage pipeline
    # (Q/P/s_red stay single).
    q_smem_elems = BLOCK_M * bk_qk
    k_smem_elems = BLOCK_N * bk_qk
    v_smem_elems = D * bk_pv
    p_smem_elems = BLOCK_M * BLOCK_N
    k_smem_elems_staged = k_smem_elems * 2
    v_smem_elems_staged = v_smem_elems * 2

    # Dense: BHSD phys-padded. Varlen packed: Q/K/O [total, H(or Hkv), D] with
    # token stride H*D; V host-transposed to [Hkv, D, total] (total is runtime).
    if varlen:
        q_row_stride = H * D
        k_row_stride = Hkv * D
        o_row_stride = H * D
        # v_row_stride filled at runtime from total_tokens.
        q_batch_stride = 0
        q_head_stride = D
        k_batch_stride = 0
        k_head_stride = D
        v_batch_stride = 0
        v_head_stride = 0
        o_batch_stride = 0
        o_head_stride = D
    else:
        q_row_stride = D
        k_row_stride = D
        v_row_stride = Skv_phys  # V host-transposed to [B, Hkv, D, Skv_phys]
        o_row_stride = D
        q_batch_stride = H * Sq_phys * D
        q_head_stride = Sq_phys * D
        k_batch_stride = Hkv * Skv_phys * D
        k_head_stride = Skv_phys * D
        v_batch_stride = Hkv * D * Skv_phys
        v_head_stride = D * Skv_phys
        o_batch_stride = H * Sq_phys * D
        o_head_stride = Sq_phys * D

    # ROWS_PER_LANE = WARP_ATOMS_M * 4: each lane owns 4 rows per warp-M atom
    # (row m = warp_m_id * 32 + mma_m * 16 + ei * 4 + lane_row). See TV
    # analysis in the docstring above.
    ROWS_PER_LANE = WARP_ATOMS_M * 4
    # ``m_running`` init: log2-domain "very negative". Value chosen so
    # ``exp2(NEG_LARGE_F - m_new) = 0`` in fp32 for any bounded ``m_new``, but
    # not so extreme that ``__nv_exp2f``'s polynomial approximation returns
    # NaN under ``fastmath=fast`` (NInf/AFN).
    NEG_LARGE_F = -60.0
    LOG2E = 1.4426950408889634
    scale_log2e = float(sm_scale_f32) * LOG2E
    shuffle_steps = int(math.log2(TCU_LANE_COLS))
    has_softcap = softcap is not None
    has_swa = window_size is not None
    softcap_f32 = float(softcap) if has_softcap else 0.0
    window_size_i = int(window_size) if has_swa else 0

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def flex_attn_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        CuSeqLen: fx.Tensor,
        SeqLens: fx.Tensor,
        total_tokens: fx.Int32,
    ):
        # Declared inside the kernel body so @fx.struct annotations resolve
        # against the local closure.
        # NB: fx.Array[T, size, align] -- the third param is ALIGNMENT (bytes),
        # not a second shape dim. Passing a shape tuple e.g. ``[BLOCK_M, WARPS_N]``
        # trips ``recast_iter`` with "alignment must be a positive multiple of
        # element byte size (4), got 2" whenever WARPS_N happens to be < the
        # element size. Keep fp32 reductions as flat 1-D arrays and shape them
        # with ``.view(make_layout(...))`` at use site.
        @fx.struct
        class FlexAttnSmem:
            s_red_max: fx.Array[fx.Float32, BLOCK_M * WARPS_N]
            s_red_sum: fx.Array[fx.Float32, BLOCK_M * WARPS_N]
            q: fx.Array[elem_dtype, q_smem_elems]
            k: fx.Array[elem_dtype, k_smem_elems_staged]
            v: fx.Array[elem_dtype, v_smem_elems_staged]
            p: fx.Array[elem_dtype, p_smem_elems]

        bh_idx = fx.block_idx.x
        q_tile_idx = fx.block_idx.y
        tid = fx.thread_idx.x
        warp_id = tid // fx.Int32(WARP_SIZE)
        lane_id = fx.Int32(fx.lane_id)
        lane_col = lane_id % fx.Int32(TCU_LANE_COLS)
        lane_row = lane_id // fx.Int32(TCU_LANE_COLS)
        warp_m_id = warp_id // fx.Int32(WARPS_N)
        warp_n_id = warp_id % fx.Int32(WARPS_N)
        b_idx = bh_idx // fx.Int32(H)
        h_idx = bh_idx % fx.Int32(H)
        hkv_idx = h_idx // fx.Int32(group_size)

        fm_fast = arith.FastMathFlags.fast
        c_zero_f = fx.Float32(0.0)
        c_neg_large = fx.Float32(NEG_LARGE_F)
        c_scale_log2e = fx.Float32(scale_log2e)
        c_sm_scale = fx.Float32(float(sm_scale_f32))
        c_log2e = fx.Float32(LOG2E)
        c_softcap = fx.Float32(softcap_f32)
        c_window = fx.Int32(window_size_i)
        c_block_n = fx.Int32(BLOCK_N)
        q_start = q_tile_idx * fx.Int32(BLOCK_M)

        # Dense: compile-time logical Skv. Varlen: per-seq length from cu_seqlens.
        if fx.const_expr(varlen):
            cu_view = fx.make_view(
                fx.get_iter(CuSeqLen),
                fx.make_layout((B + 1,), (1,)),
            )
            seqlens_view = fx.make_view(
                fx.get_iter(SeqLens),
                fx.make_layout((B,), (1,)),
            )
            tok_base = fx.memref_load(cu_view, b_idx)
            seqlen = fx.memref_load(seqlens_view, b_idx)
            c_skv_logical = seqlen
            active = q_start < seqlen
            v_row_stride_r = total_tokens
            q_ptr = fx.add_offset(
                fx.get_iter(Q),
                fx.make_int_tuple(tok_base * fx.Int32(q_row_stride) + h_idx * fx.Int32(D)),
            )
            k_ptr = fx.add_offset(
                fx.get_iter(K),
                fx.make_int_tuple(tok_base * fx.Int32(k_row_stride) + hkv_idx * fx.Int32(D)),
            )
            o_ptr = fx.add_offset(
                fx.get_iter(O),
                fx.make_int_tuple(tok_base * fx.Int32(o_row_stride) + h_idx * fx.Int32(D)),
            )
            v_ptr = fx.add_offset(
                fx.get_iter(V),
                fx.make_int_tuple(hkv_idx * fx.Int32(D) * total_tokens + tok_base),
            )
            Q_bh = fx.make_view(q_ptr, fx.make_layout((Sq_phys, D), (q_row_stride, 1)))
            K_bh = fx.make_view(k_ptr, fx.make_layout((Skv_phys, D), (k_row_stride, 1)))
            O_bh = fx.make_view(o_ptr, fx.make_layout((Sq_phys, D), (o_row_stride, 1)))
            V_bh = fx.make_view(v_ptr, fx.make_layout((D, Skv_phys), (v_row_stride_r, 1)))
        else:
            c_skv_logical = fx.Int32(Skv)
            active = fx.Int32(1) != fx.Int32(0)
            # Phys-padded 4D views. V is host-transposed to [B, Hkv, D, Skv_phys]
            # so both MMA operands stay k-major "tn".
            Q_view = fx.make_view(
                fx.get_iter(Q),
                fx.make_layout((B, H, Sq_phys, D), (q_batch_stride, q_head_stride, q_row_stride, 1)),
            )
            K_view = fx.make_view(
                fx.get_iter(K),
                fx.make_layout((B, Hkv, Skv_phys, D), (k_batch_stride, k_head_stride, k_row_stride, 1)),
            )
            V_view = fx.make_view(
                fx.get_iter(V),
                fx.make_layout((B, Hkv, D, Skv_phys), (v_batch_stride, v_head_stride, v_row_stride, 1)),
            )
            O_view = fx.make_view(
                fx.get_iter(O),
                fx.make_layout((B, H, Sq_phys, D), (o_batch_stride, o_head_stride, o_row_stride, 1)),
            )
            Q_bh = fx.slice(Q_view, (b_idx, h_idx, None, None))
            K_bh = fx.slice(K_view, (b_idx, hkv_idx, None, None))
            V_bh = fx.slice(V_view, (b_idx, hkv_idx, None, None))
            O_bh = fx.slice(O_view, (b_idx, h_idx, None, None))
            v_row_stride_r = fx.Int32(v_row_stride)

        gQ = fx.slice(fx.flat_divide(Q_bh, (BLOCK_M, bk_qk)), (None, None, q_tile_idx, 0))
        gK_all = fx.flat_divide(K_bh, (BLOCK_N, bk_qk))
        gV_all = fx.flat_divide(V_bh, (D, bk_pv))
        gO = fx.slice(fx.flat_divide(O_bh, (BLOCK_M, D)), (None, None, q_tile_idx, 0))

        smem = fx.SharedAllocator(static=True).allocate(FlexAttnSmem).peek()
        q_smem = smem.q.ptr
        k_smem_base = smem.k.ptr
        v_smem_base = smem.v.ptr
        p_smem = smem.p.ptr
        # 2-D view over the flat 1-D fp32 arrays; row-major, WARPS_N-wide.
        s_red_max = smem.s_red_max.view(fx.make_layout((BLOCK_M, WARPS_N), (WARPS_N, 1)))
        s_red_sum = smem.s_red_sum.view(fx.make_layout((BLOCK_M, WARPS_N), (WARPS_N, 1)))

        mma_atom = fx.make_mma_atom(ixdl.MRMma(ATOM_M, ATOM_N, ATOM_K_B16, elem_dtype, elem_dtype, fx.Float32))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (1, 1, 1)))
        thr_mma = tiled_mma.thr_slice(lane_id)

        copy_atom_s2r_a = fx.make_copy_atom(fx.UniversalCopy32b(), elem_dtype)
        copy_atom_s2r_b = fx.make_copy_atom(fx.UniversalCopy32b(), elem_dtype)
        tiled_copy_a = fx.make_tiled_copy_A(copy_atom_s2r_a, tiled_mma)
        tiled_copy_b = fx.make_tiled_copy_B(copy_atom_s2r_b, tiled_mma)
        thr_copy_a = tiled_copy_a.get_slice(lane_id)
        thr_copy_b = tiled_copy_b.get_slice(lane_id)

        # P is staged via tiled_copy_C on the write side and tiled_copy_A on the
        # read side, both against a plain logical (BLOCK_M, BLOCK_N) SMEM view.
        # The two TV layouts differ in per-lane element ownership, but each is a
        # bijection between (thr, val) and logical (m, n); they compose correctly
        # as long as the SMEM has plain logical layout (no SME swizzle).
        # Use 16b copy atom for the C-TV write: the C fragment's 4 vals are at
        # codomains {0, 4, 8, 12} (in the (16, 16) col-major atom), i.e., 4
        # bf16 slots apart -- NOT byte-contiguous. A 32b (2-bf16 packed) copy
        # would silently corrupt half the elements. The MMA2 A-side read uses
        # 32b because A-TV vals are at codomains {0, 1, 8, 9} -- pair (0, 1)
        # and pair (8, 9) are contiguous in memory.
        copy_atom_p_r2s = fx.make_copy_atom(fx.UniversalCopy16b(), elem_dtype)
        tiled_copy_p_r2s = fx.make_tiled_copy_C(copy_atom_p_r2s, tiled_mma)
        thr_copy_p_r2s = tiled_copy_p_r2s.get_slice(lane_id)

        g2s_sme = mr_g2s_sme_config(
            a_mn_major=False,
            b_mn_major=False,
            elem_dtype=elem_dtype,
            row_atom=ixdl.MRAsyncCpRow16b,
            row_swizzle=ixdl.SMESwizzle.Row16b,
        )
        vpr = MR_GEMM_GEOM.values_per_sme_row
        tile_smem_row = fx.make_tile(SMEM_ROWS, vpr)

        def _k_stage_ptr(stage):
            return fx.add_offset(k_smem_base, fx.make_int_tuple(stage * fx.Int32(k_smem_elems)))

        def _v_stage_ptr(stage):
            return fx.add_offset(v_smem_base, fx.make_int_tuple(stage * fx.Int32(v_smem_elems)))

        def _issue_k(kv_idx, stage):
            gK = fx.slice(gK_all, (None, None, kv_idx, 0))
            sme_K = ixdl.make_sme_gmem_tensor(gK, leading_stride=k_row_stride)
            mr_gemm_g2s_issue_b_warp(
                a_mn_major=False,
                b_mn_major=False,
                warp_id=warp_id,
                b_per_warp=b_per_warp_qk,
                b_cta_gmem_view=fx.zipped_divide(sme_K, tile_smem_row),
                g2s_sme=g2s_sme,
                smem_b=_k_stage_ptr(stage),
                elem_dtype=elem_dtype,
                bm=BLOCK_M,
                bn=BLOCK_N,
                bk=bk_qk,
                geom=MR_GEMM_GEOM,
            )

        def _issue_v(kv_idx, stage):
            gV = fx.slice(gV_all, (None, None, 0, kv_idx))
            sme_V = ixdl.make_sme_gmem_tensor(gV, leading_stride=v_row_stride_r)
            mr_gemm_g2s_issue_b_warp(
                a_mn_major=False,
                b_mn_major=False,
                warp_id=warp_id,
                b_per_warp=b_per_warp_pv,
                b_cta_gmem_view=fx.zipped_divide(sme_V, tile_smem_row),
                g2s_sme=g2s_sme,
                smem_b=_v_stage_ptr(stage),
                elem_dtype=elem_dtype,
                bm=BLOCK_M,
                bn=D,
                bk=bk_pv,
                geom=MR_GEMM_GEOM,
            )

        # ---- Fragment templates -------------------------------------------
        # O_acc: MMA2's (m, n) = (BLOCK_M, D). Template from gO warp view.
        gO_warp = fx.slice(
            fx.flat_divide(gO, (ATOM_M * WARP_ATOMS_M, ATOM_N * WARP_ATOMS_N_PV)),
            (None, None, warp_m_id, warp_n_id),
        )
        gO_atoms = fx.flat_divide(gO_warp, (ATOM_M, ATOM_N))

        O_acc = []
        for mma_m in fx.range_constexpr(WARP_ATOMS_M):
            row = []
            for mma_n in fx.range_constexpr(WARP_ATOMS_N_PV):
                c_tile = fx.slice(gO_atoms, (None, None, mma_m, mma_n))
                frag = thr_mma.make_fragment_C(c_tile)
                frag.fill(0)
                row.append(frag)
            O_acc.append(row)

        # ---- P SMEM view: COL-MAJOR (M inner, stride (1, BLOCK_M)) ---------
        # CuTe TV composition (both ``make_tiled_copy_C.partition_S`` for the
        # register-to-SMEM P store AND ``make_tiled_copy_A.partition_S`` for
        # the SMEM-to-register MMA2 A load) treats the codomain of the atom TV
        # as a col-major flat index in the (16, 16) tile. Using a row-major
        # view here silently mismatches the C-write codomain with the A-read
        # codomain (they land on DIFFERENT physical LDS words) so MMA2 reads
        # 4-of-8 rows worth of stale/garbage data. Keep this col-major.
        p_full = fx.make_view(
            p_smem,
            fx.make_layout((BLOCK_M, BLOCK_N), (1, BLOCK_M)),
        )
        p_write_warp = fx.slice(
            fx.flat_divide(p_full, (ATOM_M * WARP_ATOMS_M, ATOM_N * WARP_ATOMS_N)),
            (None, None, warp_m_id, warp_n_id),
        )
        p_write_atoms = fx.flat_divide(p_write_warp, (ATOM_M, ATOM_N))

        # MMA2 A-side read: each warp needs the FULL BLOCK_N in K, so slice P
        # only along M (warp_m_id) -- all warps in a warp_m row group share the
        # 32 x BLOCK_N P strip.
        p_read_warp = fx.slice(
            fx.flat_divide(p_full, (ATOM_M * WARP_ATOMS_M, BLOCK_N)),
            (None, None, warp_m_id, 0),
        )
        p_read_atoms = fx.flat_divide(p_read_warp, (ATOM_M, ATOM_K_B16))

        S_frags = []
        for mma_m in fx.range_constexpr(WARP_ATOMS_M):
            row = []
            for mma_n in fx.range_constexpr(WARP_ATOMS_N):
                c_tile = fx.slice(p_write_atoms, (None, None, mma_m, mma_n))
                row.append(thr_mma.make_fragment_C(c_tile))
            S_frags.append(row)

        # ---- G2S Q (once, loop-invariant) --------------------------------
        sme_Q = ixdl.make_sme_gmem_tensor(gQ, leading_stride=q_row_stride)
        mr_gemm_g2s_issue_a_warp(
            a_mn_major=False,
            b_mn_major=False,
            warp_id=warp_id,
            a_per_warp=a_per_warp_qk,
            a_cta_gmem_view=fx.zipped_divide(sme_Q, tile_smem_row),
            g2s_sme=g2s_sme,
            smem_a=q_smem,
            elem_dtype=elem_dtype,
            bm=BLOCK_M,
            bn=BLOCK_N,
            bk=bk_qk,
            geom=MR_GEMM_GEOM,
        )
        ixdl.cp_async_commit_group()
        ixdl.cp_async_wait_group(0)
        fx.gpu.barrier()

        # ---- KV loop (runtime fx.range) -----------------------------------
        # Causal fully-masked tiles are dropped by shortening the trip count:
        # kv_end = min(num_kv_tiles, ceil(q_tile_end / BLOCK_N)). With
        # BLOCK_M == BLOCK_N this is min(num_kv_tiles, q_tile_idx + 1).
        # SWA empty-window tiles are NOT skipped here (element mask still applies).
        #
        # m_running / l_running are flat-packed across the scf.for boundary
        # (ROWS_PER_LANE fp32 each). O_acc stays outside and is updated in place.
        #
        # Light 2-stage K/V pipeline: prologue loads K0; after QK issue K_{i+1}
        # into the other bank; after P-write issue V_i; one wait drains both.
        c_num_kv = fx.Int32(num_kv_tiles)
        if fx.const_expr(is_causal):
            kv_end_cand = q_tile_idx + fx.Int32(1)
            kv_end = (kv_end_cand < c_num_kv).select(kv_end_cand, c_num_kv)
        else:
            kv_end = c_num_kv
        # Varlen: also drop tiles past this sequence's logical length.
        if fx.const_expr(varlen):
            kv_tiles_seq = (seqlen + c_block_n - fx.Int32(1)) // c_block_n
            kv_end = (kv_end < kv_tiles_seq).select(kv_end, kv_tiles_seq)
            # Inactive q-tiles (q_start >= seqlen): skip the KV loop entirely.
            kv_end = active.select(kv_end, fx.Int32(0))

        # Prologue: K0 into stage 0 (skipped when kv_end == 0).
        if kv_end > fx.Int32(0):
            _issue_k(fx.Int32(0), fx.Int32(0))
            ixdl.cp_async_commit_group()
            ixdl.cp_async_wait_group(0)
            fx.gpu.barrier()

        init_state = [c_neg_large for _ in range(ROWS_PER_LANE)] + [c_zero_f for _ in range(ROWS_PER_LANE)]
        loop_results = init_state
        for kv_idx, state in fx.range(fx.Int32(0), kv_end, fx.Int32(1), init=init_state):
            m_prev = [state[slot] for slot in range(ROWS_PER_LANE)]
            l_prev = [state[ROWS_PER_LANE + slot] for slot in range(ROWS_PER_LANE)]
            kv_i = fx.Int32(kv_idx)
            comp_stage = kv_i % fx.Int32(2)
            k_smem_cur = _k_stage_ptr(comp_stage)
            v_smem_cur = _v_stage_ptr(comp_stage)

            # ---- MMA1: S = Q @ K^T ----------------------------------------
            for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                for mma_n in fx.range_constexpr(WARP_ATOMS_N):
                    S_frags[mma_m][mma_n].fill(0)

            for mma_k in fx.range_constexpr(k_atoms_qk):
                a_frags, b_frags = mr_gemm_s2r_load_mma_k(
                    a_mn_major=False,
                    b_mn_major=False,
                    mma_k=mma_k,
                    g2s_sme=g2s_sme,
                    smem_a=q_smem,
                    smem_b=k_smem_cur,
                    elem_dtype=elem_dtype,
                    warp_m_id=warp_m_id,
                    warp_n_id=warp_n_id,
                    warp_atoms_m=WARP_ATOMS_M,
                    warp_atoms_n=WARP_ATOMS_N,
                    copy_atom_a=copy_atom_s2r_a,
                    copy_atom_b=copy_atom_s2r_b,
                    thr_copy_a=thr_copy_a,
                    thr_copy_b=thr_copy_b,
                    thr_mma=thr_mma,
                    bm=BLOCK_M,
                    bn=BLOCK_N,
                    bk=bk_qk,
                    geom=MR_GEMM_GEOM,
                )
                for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                    for mma_n in fx.range_constexpr(WARP_ATOMS_N):
                        fx.gemm(
                            mma_atom,
                            S_frags[mma_m][mma_n],
                            a_frags[mma_m],
                            b_frags[mma_n],
                            S_frags[mma_m][mma_n],
                        )

            # Prefetch K_{i+1} into the other bank so G2S overlaps softmax / P.
            next_kv = kv_i + fx.Int32(1)
            if next_kv < kv_end:
                _issue_k(next_kv, comp_stage ^ fx.Int32(1))
                ixdl.cp_async_commit_group()

            # ---- Scale / softcap / enter log2 domain ----------------------
            # S *= sm_scale -> optional softcap -> masks -> softmax.
            # Softmax uses exp2, so we enter log2 domain with *log2e after
            # softcap. When softcap is unset, fuse *(sm_scale*log2e).
            if fx.const_expr(has_softcap):
                for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                    for mma_n in fx.range_constexpr(WARP_ATOMS_N):
                        acc = S_frags[mma_m][mma_n]
                        old = Vec(acc.load())
                        acc.store(
                            Vec.from_elements(
                                [
                                    c_log2e
                                    * (
                                        c_softcap
                                        * fmath.tanh(
                                            (old[ei] * c_sm_scale) / c_softcap,
                                            fastmath=fm_fast,
                                        )
                                    )
                                    for ei in range(4)
                                ],
                                fx.Float32,
                            )
                        )
            else:
                for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                    for mma_n in fx.range_constexpr(WARP_ATOMS_N):
                        acc = S_frags[mma_m][mma_n]
                        acc.store(Vec(acc.load()) * c_scale_log2e)

            # ---- Causal mask ----------------------------------------------
            # Set S[q, k] = NEG_LARGE_F where ``kv_global > q_global``. Applied
            # AFTER entering the log2 domain above, so the sentinel lives in
            # the same domain as everything downstream and matches the
            # m_running init: exp2(NEG_LARGE_F - m_new) ~= 0 for any bounded
            # m_new. ``is_causal=True`` enforces ``Sq == Skv`` at compile
            # time, so q_global and kv_global share the same axis.
            #
            # Fragment coord recap (col-major C-TV, see docstring above):
            #   m = warp_m_id * (WARP_ATOMS_M * ATOM_M) + mma_m * ATOM_M
            #       + ei * 4 + lane_row       -- varies per ei
            #   n = warp_n_id * (WARP_ATOMS_N * ATOM_N) + mma_n * ATOM_N
            #       + lane_col                -- ei-invariant
            if fx.const_expr(is_causal):
                q_block_base = q_tile_idx * fx.Int32(BLOCK_M) + warp_m_id * fx.Int32(WARP_ATOMS_M * ATOM_M)
                kv_block_base = kv_i * fx.Int32(BLOCK_N) + warp_n_id * fx.Int32(WARP_ATOMS_N * ATOM_N)
                # Inner ``ei`` loop is a Python list comprehension (like the
                # O_acc alpha-scale below): a bare ``for ei in range(4):`` is
                # picked up by the AST rewriter as a dynamic for loop and
                # rejects Python-list loop-carried state.
                for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                    for mma_n in fx.range_constexpr(WARP_ATOMS_N):
                        acc = S_frags[mma_m][mma_n]
                        old = Vec(acc.load())
                        kv_g = kv_block_base + fx.Int32(mma_n * ATOM_N) + lane_col
                        acc.store(
                            Vec.from_elements(
                                [
                                    (kv_g > q_block_base + fx.Int32(mma_m * ATOM_M + ei * 4) + lane_row).select(
                                        c_neg_large, old[ei]
                                    )
                                    for ei in range(4)
                                ],
                                fx.Float32,
                            )
                        )

            # ---- Sliding-window mask --------------------------------------
            # Mask where ``q_global - kv_global > window_size``
            # (distance == window_size remains visible). Independent of
            # ``is_causal`` -- callers wanting a causal window must set both.
            if fx.const_expr(has_swa):
                q_block_base = q_tile_idx * fx.Int32(BLOCK_M) + warp_m_id * fx.Int32(WARP_ATOMS_M * ATOM_M)
                kv_block_base = kv_i * fx.Int32(BLOCK_N) + warp_n_id * fx.Int32(WARP_ATOMS_N * ATOM_N)
                for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                    for mma_n in fx.range_constexpr(WARP_ATOMS_N):
                        acc = S_frags[mma_m][mma_n]
                        old = Vec(acc.load())
                        kv_g = kv_block_base + fx.Int32(mma_n * ATOM_N) + lane_col
                        acc.store(
                            Vec.from_elements(
                                [
                                    (
                                        (q_block_base + fx.Int32(mma_m * ATOM_M + ei * 4) + lane_row) - kv_g > c_window
                                    ).select(c_neg_large, old[ei])
                                    for ei in range(4)
                                ],
                                fx.Float32,
                            )
                        )

            # ---- KV tail mask ---------------------------------------------
            # Phys pad runs full BLOCK_N G2S tiles; mask columns beyond the
            # logical Skv so pad K/V never enter softmax. Folded away when
            # Skv == Skv_phys.
            if fx.const_expr(has_kv_tail):
                kv_block_base = kv_i * fx.Int32(BLOCK_N) + warp_n_id * fx.Int32(WARP_ATOMS_N * ATOM_N)
                for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                    for mma_n in fx.range_constexpr(WARP_ATOMS_N):
                        acc = S_frags[mma_m][mma_n]
                        old = Vec(acc.load())
                        kv_g = kv_block_base + fx.Int32(mma_n * ATOM_N) + lane_col
                        acc.store(
                            Vec.from_elements(
                                [(kv_g >= c_skv_logical).select(c_neg_large, old[ei]) for ei in range(4)],
                                fx.Float32,
                            )
                        )

            # ---- Local rowmax (within this warp) --------------------------
            # C TV (version A): each of a lane's 4 fp32 elements sits in a
            # DIFFERENT row (m = mma_m * 16 + ei * 4 + lane_row) but the SAME
            # column (n = lane_col). Per-row max thus reduces over ``mma_n``
            # (pairing matched ``ei`` across mma_n atoms) and then over the 16
            # ``lane_col`` values via ``shuffle_xor(8, 4, 2, 1)``.
            local_max_flat = []
            for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                for ei in fx.range_constexpr(4):
                    v = Vec(S_frags[mma_m][0].load())[ei]
                    for mma_n in fx.range_constexpr(1, WARP_ATOMS_N):
                        v = v.maximumf(Vec(S_frags[mma_m][mma_n].load())[ei])
                    for sh in fx.range_constexpr(shuffle_steps):
                        off = TCU_LANE_COLS // (2 << sh)
                        v = v.maximumf(v.shuffle_xor(off, WARP_SIZE))
                    local_max_flat.append(v)

            # ---- Cross-warp reduce over WARPS_N ---------------------------
            # SMEM is (BLOCK_M, WARPS_N). Each lane owns 4 * WARP_ATOMS_M rows
            # (m = warp_m_id * 32 + mma_m * 16 + ei * 4 + lane_row); after the
            # cross-lane_col reduce, all 16 lane_cols hold the same value per
            # row, so gate the store on lane_col == 0. The fixed tile has
            # WARPS_N > 1 (asserted at file top) so the reduce is unconditional;
            # wrapping in ``if WARPS_N > 1`` trips the AST rewriter
            # (Python-constant conditions still lower to scf.if scopes and
            # branch-local ``m_row = [...]`` does not leak to the outer scope).
            for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                for ei in fx.range_constexpr(4):
                    row = warp_m_id * fx.Int32(WARP_ATOMS_M * ATOM_M) + fx.Int32(mma_m * ATOM_M + ei * 4) + lane_row
                    if lane_col == fx.Int32(0):
                        fx.memref_store(
                            local_max_flat[mma_m * 4 + ei],
                            s_red_max,
                            (row, warp_n_id),
                        )
            fx.gpu.barrier()
            m_row = []
            for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                for ei in fx.range_constexpr(4):
                    row = warp_m_id * fx.Int32(WARP_ATOMS_M * ATOM_M) + fx.Int32(mma_m * ATOM_M + ei * 4) + lane_row
                    v = c_neg_large
                    for wn in fx.range_constexpr(WARPS_N):
                        v = v.maximumf(fx.memref_load(s_red_max, (row, fx.Int32(wn))))
                    m_row.append(v)
            fx.gpu.barrier()

            # ---- m_new = max(m_prev, m_row); alpha = exp2(m_prev - m_new) --
            m_new = [m_prev[slot].maximumf(m_row[slot]) for slot in range(ROWS_PER_LANE)]
            alpha = [(m_prev[slot] - m_new[slot]).exp2(fastmath=fm_fast) for slot in range(ROWS_PER_LANE)]

            # ---- Scale O_acc by alpha ------------------------------------
            # Each fragment element ``ei`` corresponds to row
            # ``mma_m * 16 + ei * 4 + lane_row``; slot index into alpha is
            # ``mma_m * 4 + ei``.
            for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                for mma_n in fx.range_constexpr(WARP_ATOMS_N_PV):
                    acc = O_acc[mma_m][mma_n]
                    old = Vec(acc.load())
                    acc.store(
                        Vec.from_elements(
                            [old[ei] * alpha[mma_m * 4 + ei] for ei in range(4)],
                            fx.Float32,
                        )
                    )

            # ---- P = exp2(S - m_new) in-place on S_frags -----------------
            for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                for mma_n in fx.range_constexpr(WARP_ATOMS_N):
                    acc = S_frags[mma_m][mma_n]
                    old = Vec(acc.load())
                    acc.store(
                        Vec.from_elements(
                            [(old[ei] - m_new[mma_m * 4 + ei]).exp2(fastmath=fm_fast) for ei in range(4)],
                            fx.Float32,
                        )
                    )

            # ---- Local rowsum + cross-warp reduce -------------------------
            local_sum_flat = []
            for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                for ei in fx.range_constexpr(4):
                    v = Vec(S_frags[mma_m][0].load())[ei]
                    for mma_n in fx.range_constexpr(1, WARP_ATOMS_N):
                        v = v + Vec(S_frags[mma_m][mma_n].load())[ei]
                    for sh in fx.range_constexpr(shuffle_steps):
                        off = TCU_LANE_COLS // (2 << sh)
                        v = v + v.shuffle_xor(off, WARP_SIZE)
                    local_sum_flat.append(v)

            for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                for ei in fx.range_constexpr(4):
                    row = warp_m_id * fx.Int32(WARP_ATOMS_M * ATOM_M) + fx.Int32(mma_m * ATOM_M + ei * 4) + lane_row
                    if lane_col == fx.Int32(0):
                        fx.memref_store(
                            local_sum_flat[mma_m * 4 + ei],
                            s_red_sum,
                            (row, warp_n_id),
                        )
            fx.gpu.barrier()
            s_row = []
            for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                for ei in fx.range_constexpr(4):
                    row = warp_m_id * fx.Int32(WARP_ATOMS_M * ATOM_M) + fx.Int32(mma_m * ATOM_M + ei * 4) + lane_row
                    v = c_zero_f
                    for wn in fx.range_constexpr(WARPS_N):
                        v = v + fx.memref_load(s_red_sum, (row, fx.Int32(wn)))
                    s_row.append(v)
            fx.gpu.barrier()

            # ---- l_new = alpha * l_prev + s_row --------------------------
            l_new = [alpha[slot] * l_prev[slot] + s_row[slot] for slot in range(ROWS_PER_LANE)]

            # ---- Write P (cast to bf16) to SMEM via tiled_copy_C ---------
            # Follows kernels/gemm/iluvatar/epilogue.py::mr_hgemm_epilogue_store_tiled
            # but targets an SMEM slice instead of gmem. Note that ``p_full``
            # is COL-MAJOR (see the make_view above); the CuTe TV composition
            # in partition_S / partition_D expects that convention.
            for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                for mma_n in fx.range_constexpr(WARP_ATOMS_N):
                    acc = S_frags[mma_m][mma_n]
                    p_tile = fx.slice(p_write_atoms, (None, None, mma_m, mma_n))
                    frag = fx.make_fragment_like(acc, elem_dtype.ir_type)
                    frag.store(Vec(acc.load()).to(elem_dtype))
                    fx.copy(
                        copy_atom_p_r2s,
                        thr_copy_p_r2s.retile(frag),
                        thr_copy_p_r2s.partition_S(p_tile),
                        pred=None,
                    )
            fx.gpu.barrier()

            # ---- G2S V tile (current); wait also drains any K prefetch ----
            _issue_v(kv_i, comp_stage)
            ixdl.cp_async_commit_group()
            ixdl.cp_async_wait_group(0)
            fx.gpu.barrier()

            # ---- MMA2: O += P @ V ----------------------------------------
            # P read: plain logical (ATOM_M, ATOM_K_B16) slice; NOT via
            # mr_gemm_s2r_a_tile (that helper assumes SME-swizzled Q/K SMEM).
            # V read: standard SME path via mr_gemm_s2r_b_tile.
            for mma_k in fx.range_constexpr(k_atoms_pv):
                a_frags_pv = []
                for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                    p_a_tile = fx.slice(p_read_atoms, (None, None, mma_m, mma_k))
                    a_frags_pv.append(
                        mr_gemm_s2r_copy_a(
                            copy_atom=copy_atom_s2r_a,
                            thr_copy_a=thr_copy_a,
                            thr_mma=thr_mma,
                            smem_a_tile=p_a_tile,
                        )
                    )
                b_frags_pv = []
                for mma_n in fx.range_constexpr(WARP_ATOMS_N_PV):
                    v_tile = mr_gemm_s2r_b_tile(
                        a_mn_major=False,
                        b_mn_major=False,
                        mma_n=mma_n,
                        mma_k=mma_k,
                        g2s_sme=g2s_sme,
                        smem_b=v_smem_cur,
                        elem_dtype=elem_dtype,
                        warp_n_id=warp_n_id,
                        warp_atoms_n=WARP_ATOMS_N_PV,
                        bm=BLOCK_M,
                        bn=D,
                        bk=bk_pv,
                        geom=MR_GEMM_GEOM,
                    )
                    b_frags_pv.append(
                        mr_gemm_s2r_copy_b(
                            copy_atom=copy_atom_s2r_b,
                            thr_copy_b=thr_copy_b,
                            thr_mma=thr_mma,
                            smem_b_tile=v_tile,
                        )
                    )
                for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                    for mma_n in fx.range_constexpr(WARP_ATOMS_N_PV):
                        fx.gemm(
                            mma_atom,
                            O_acc[mma_m][mma_n],
                            a_frags_pv[mma_m],
                            b_frags_pv[mma_n],
                            O_acc[mma_m][mma_n],
                        )


            # Barrier before next iter may overwrite P / consume prefetched K.
            fx.gpu.barrier()
            loop_results = yield m_new + l_new

        l_final = [loop_results[ROWS_PER_LANE + slot] for slot in range(ROWS_PER_LANE)]

        # ---- Post-loop: normalise and write O -----------------------------
        # ``l_final`` is the last yielded rowsum pack. Element ``ei`` of an
        # (mma_m, mma_n_pv) fragment is at row ``mma_m * 16 + ei * 4 +
        # lane_row``, matching slot index ``mma_m * 4 + ei``.
        def _normalize_o():
            for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                for mma_n in fx.range_constexpr(WARP_ATOMS_N_PV):
                    acc = O_acc[mma_m][mma_n]
                    old = Vec(acc.load())
                    acc.store(
                        Vec.from_elements(
                            [old[ei] / l_final[mma_m * 4 + ei] for ei in range(4)],
                            fx.Float32,
                        )
                    )

        if fx.const_expr(varlen):
            # Inactive CTAs skipped the KV loop (kv_end=0); do not divide by the
            # zero rowsum init or write into a neighbor sequence's rows.
            if active:
                _normalize_o()
                # Predicated shfl-style store (inlined so AST rewriter sees the
                # ``if q_row < seqlen``). Row stride is packed ``H*D``, not D.
                # ``total_tokens`` (V leading stride) must be a multiple of 32.
                c_row = fx.Int32(o_row_stride)
                c_warp_n = ATOM_N * WARP_ATOMS_N_PV
                lane_voffset = lane_row * (c_row // fx.Int32(2)) + lane_col
                lane_select0 = lane_row * fx.Int32(TCU_LANE_COLS) + (lane_col * fx.Int32(2)) % fx.Int32(
                    TCU_LANE_COLS
                )
                lane_select1 = lane_select0 + fx.Int32(1)
                lane_em = lane_col // fx.Int32(8)
                width_i32 = fx.Int32(WARP_SIZE)
                mask_lo = fx.Int32(0xFFFF)
                mask_hi = fx.Int32(0xFFFF0000)
                row_base = q_start + warp_m_id * fx.Int32(ATOM_M * WARP_ATOMS_M)
                c_warp_ptr = fx.get_iter(gO_warp)
                c_byte_ptr = fx.recast_iter(
                    fx.PointerType.get(fx.Int8.ir_type, c_warp_ptr.memspace),
                    c_warp_ptr,
                )
                for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                    phys_m = mma_m * TCU_LANE_COLS
                    for ei in fx.range_constexpr(4):
                        for phys_n in fx.range_constexpr(0, c_warp_n, TCU_LANE_COLS * 2):
                            mma_n0 = phys_n // TCU_LANE_COLS
                            mma_n1 = mma_n0 + 1
                            tile_half_soffset = fx.Int32(phys_m + ei * 4) * c_row + fx.Int32(phys_n)
                            f32_0 = Vec(O_acc[mma_m][mma_n0].load())[ei]
                            f32_1 = Vec(O_acc[mma_m][mma_n1].load())[ei]
                            h0 = f32_0.to(elem_dtype)
                            h1 = f32_1.to(elem_dtype)
                            hval_i32 = Vec(Vec.from_elements([h0, h1], elem_dtype)).bitcast(fx.Int32)[0]
                            hvall = hval_i32.shuffle_idx(lane_select0, width_i32)
                            hvalh = hval_i32.shuffle_idx(lane_select1, width_i32)
                            val0 = (hvall & mask_lo) | (hvalh << fx.Int32(16))
                            val1 = hvall.shrui(fx.Int32(16)) | (hvalh & mask_hi)
                            val = fx.arith.select(
                                fx.arith.cmpi(fx.arith.CmpIPredicate.ne, lane_em, fx.Int32(0)),
                                val1,
                                val0,
                            )
                            store_byte_off = lane_voffset * fx.Int32(4) + tile_half_soffset * fx.Int32(2)
                            store_ptr = fx.recast_iter(
                                fx.PointerType.get(fx.Int32.ir_type, c_warp_ptr.memspace),
                                fx.add_offset(c_byte_ptr, fx.make_int_tuple(store_byte_off)),
                            )
                            q_row = row_base + fx.Int32(phys_m + ei * 4) + lane_row
                            if q_row < seqlen:
                                fx.ptr_store(val, store_ptr)
        else:
            _normalize_o()
            mr_hgemm_epilogue_store_tiled(
                lane_id=lane_id,
                accs=O_acc,
                gC_warp=gO_warp,
                tiled_mma=tiled_mma,
                warp_atoms_m=WARP_ATOMS_M,
                warp_atoms_n=WARP_ATOMS_N_PV,
                out_dtype=elem_dtype,
            )

    @flyc.jit
    def launch_flex_attn(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        CuSeqLen: fx.Tensor,
        SeqLens: fx.Tensor,
        total_tokens: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        flex_attn_kernel(Q, K, V, O, CuSeqLen, SeqLens, total_tokens).launch(
            grid=(B * H, num_q_tiles, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    def launch_flex_attn_checked(Q, K, V, O, cu_seqlens=None, seq_lens=None, stream=fx.Stream(None)):  # noqa: E741
        """Host entry: enforce shapes before the JIT launch.

        Dense (``varlen=False``): phys-padded BHSD tensors; cu/seq_lens ignored.
        Varlen: packed ``Q/O [total, H, D]``, ``K [total, Hkv, D]``,
        ``V [Hkv, D, total]`` (host-transposed), plus:
          * ``cu_seqlens`` int32 ``[B+1]`` -- **physical** token offsets (each
            start must be a multiple of 32 for SME G2S)
          * ``seq_lens`` int32 ``[B]`` -- **logical** lengths used for masking
        ``total`` must be a multiple of 32.
        """
        q_shape = _tensor_shape(Q)
        k_shape = _tensor_shape(K)
        v_shape = _tensor_shape(V)
        o_shape = _tensor_shape(O)
        if varlen:
            if cu_seqlens is None or seq_lens is None:
                raise ValueError("varlen launch requires cu_seqlens [B+1] and seq_lens [B] (int32)")
            cu_shape = _tensor_shape(cu_seqlens)
            sl_shape = _tensor_shape(seq_lens)
            if len(cu_shape) != 1 or cu_shape[0] != B + 1:
                raise ValueError(f"cu_seqlens shape must be ({B + 1},), got {cu_shape}")
            if len(sl_shape) != 1 or sl_shape[0] != B:
                raise ValueError(f"seq_lens shape must be ({B},), got {sl_shape}")
            total = q_shape[0] if len(q_shape) == 3 else -1
            expect_q = (total, H, D)
            expect_k = (total, Hkv, D)
            expect_v = (Hkv, D, total)
            expect_o = (total, H, D)
            if q_shape != expect_q:
                raise ValueError(f"varlen Q shape must be [total, H, D]={expect_q}, got {q_shape}")
            if k_shape != expect_k:
                raise ValueError(f"varlen K shape must be [total, Hkv, D]={expect_k}, got {k_shape}")
            if v_shape != expect_v:
                raise ValueError(
                    f"varlen V shape must be [Hkv, D, total]={expect_v} (host-transposed), got {v_shape}"
                )
            if o_shape != expect_o:
                raise ValueError(f"varlen O shape must be [total, H, D]={expect_o}, got {o_shape}")
            if total < 1:
                raise ValueError(f"varlen total_tokens must be positive, got {total}")
            if total % 32 != 0:
                raise ValueError(
                    f"varlen total_tokens (V leading dim) must be a multiple of 32 for SME G2S, got {total}"
                )
            return launch_flex_attn(
                Q, K, V, O, cu_seqlens, seq_lens, fx.Int32(total), stream=stream
            )

        expect_q = (B, H, Sq_phys, D)
        expect_k = (B, Hkv, Skv_phys, D)
        expect_v = (B, Hkv, D, Skv_phys)
        expect_o = (B, H, Sq_phys, D)
        if q_shape != expect_q:
            raise ValueError(f"Q shape must be {expect_q} (Sq_phys-padded), got {q_shape}")
        if k_shape != expect_k:
            raise ValueError(f"K shape must be {expect_k} (Skv_phys-padded), got {k_shape}")
        if v_shape != expect_v:
            raise ValueError(f"V shape must be {expect_v} (host-transposed + Skv_phys-padded), got {v_shape}")
        if o_shape != expect_o:
            raise ValueError(f"O shape must be {expect_o} (Sq_phys-padded), got {o_shape}")
        # Dense: CuSeqLen/SeqLens unused; pass O as a typed placeholder.
        return launch_flex_attn(Q, K, V, O, O, O, fx.Int32(0), stream=stream)

    return launch_flex_attn_checked


def _compile_iluvatar_qk_dot_dev(
    B: int,
    H: int,
    Sq: int,
    Skv: int,
    D: int,
    *,
    dtype: str = "bf16",
    sm_scale: float | None = None,
) -> Callable:
    """Build a development launcher that writes ``S = Q @ K^T * sm_scale``.

    This helper is not part of the stable API. It isolates the QK^T half of
    flex-attention for debugging.

    Args:
        B, H, Sq, Skv, D: See ``compile_iluvatar_flex_attention``. Same
            envelope as the fused path (D in {64, 128}, dtype in {f16, bf16},
            MHA, Sq/Skv aligned to BLOCK).
        dtype: ``"f16"`` or ``"bf16"``.
        sm_scale: Query scale; ``None`` -> ``1 / sqrt(D)``.

    Returns:
        ``launch_fn(Q, K, S_out, stream=None)``.
    """
    _validate_scope(
        B=B,
        H=H,
        Hkv=H,
        Sq=Sq,
        Skv=Skv,
        D=D,
        dtype=dtype,
        is_causal=False,
        window_size=None,
        softcap=None,
    )
    _validate_qk_dot_subset(H=H, Hkv=H, Sq=Sq, Skv=Skv)

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    sm_scale_f32 = float(sm_scale)

    return _build_qk_dot_launcher(B=B, H=H, Sq=Sq, Skv=Skv, D=D, dtype=dtype, sm_scale_f32=sm_scale_f32)


def compile_iluvatar_flex_attention(
    B: int,
    H: int,
    Sq: int,
    Skv: int,
    D: int,
    *,
    Hkv: int | None = None,
    dtype: str = "bf16",
    is_causal: bool = False,
    window_size: int | None = None,
    softcap: float | None = None,
    sm_scale: float | None = None,
    varlen: bool = False,
) -> Callable:
    """Compile a fused flex-attention forward kernel for the Iluvatar backend.

    This is a variant subset (causal / SWA / softcap), not a general score_mod
    compiler.

    Supported scope: Q/K/V/O = f16 or bf16, D in {64, 128}, GQA
    (``H % Hkv == 0``), arbitrary ``Sq``/``Skv`` (callers pass phys-padded
    contiguous tensors; see below), ``is_causal`` / ``window_size`` / ``softcap``
    in any combination (``is_causal`` requires ``Sq == Skv``).

    Dense phys pad contract:
        ``Sq_phys = ceil(Sq / 64) * 64``, ``Skv_phys = ceil(Skv / 64) * 64``.
        Launch expects contiguous
        ``Q,O: [B, H, Sq_phys, D]``, ``K: [B, Hkv, Skv_phys, D]``,
        ``V: [B, Hkv, D, Skv_phys]`` (host transpose of natural K-major V).
        Logical prefix ``[:Sq]`` / ``[:Skv]`` holds real data; pad rows/cols
        may be anything. Score columns ``kv >= Skv`` are masked in-kernel.

    Varlen contract (``varlen=True``, self-attn only):
        ``B`` is ``num_seqs``, ``Sq``/``Skv`` are the same ``max_seqlen``.
        Launch expects packed ``Q,O: [total_tokens, H, D]``,
        ``K: [total_tokens, Hkv, D]``, ``V: [Hkv, D, total_tokens]``
        (host transpose of natural ``[total, Hkv, D]``), plus
        ``cu_seqlens`` int32 ``[B+1]`` (**physical** token offsets; each start
        and ``total_tokens`` must be multiples of 32 for SME G2S) and
        ``seq_lens`` int32 ``[B]`` (**logical** lengths for masking).
        Grid is ``(B * H, ceil(max_seqlen / BLOCK_M), 1)``. Between sequences,
        pad so the next start is 32-aligned; trailing pad so ``total`` is
        32-aligned and covers the last partial G2S tile.

    Args:
        B, H, Sq, Skv, D: Logical shapes (compile-time constants). Dense:
            ``Sq``/``Skv`` are logical lengths used for masking. Varlen:
            ``B=num_seqs``, ``Sq=Skv=max_seqlen``.
        Hkv: KV head count; ``None`` means MHA (``Hkv = H``).
        dtype: ``"f16"`` or ``"bf16"``.
        is_causal: Enable lower-triangular causal mask; requires ``Sq == Skv``.
        window_size: Sliding-window radius; mask where ``q - kv > window_size``.
            Independent of ``is_causal``.
        softcap: Gemma-2 softcap; ``S = softcap * tanh(S / softcap)`` after
            ``sm_scale``, before log2-domain entry.
        sm_scale: Query scale; ``None`` defaults to ``1 / sqrt(D)``.
        varlen: If True, compile the packed ``cu_seqlens`` self-attn path.

    Returns:
        Dense: ``launch_fn(Q, K, V, O, stream=None)``.
        Varlen: ``launch_fn(Q, K, V, O, cu_seqlens=..., seq_lens=..., stream=None)``.
    """
    if Hkv is None:
        Hkv = H

    _validate_scope(
        B=B,
        H=H,
        Hkv=Hkv,
        Sq=Sq,
        Skv=Skv,
        D=D,
        dtype=dtype,
        is_causal=is_causal,
        window_size=window_size,
        softcap=softcap,
    )
    if varlen and Sq != Skv:
        raise ValueError(f"varlen self-attn requires Sq == Skv (max_seqlen); got Sq={Sq}, Skv={Skv}")

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    sm_scale = float(sm_scale)

    return _build_flex_attention_launcher(
        B=B,
        H=H,
        Hkv=Hkv,
        Sq=Sq,
        Skv=Skv,
        D=D,
        dtype=dtype,
        sm_scale_f32=sm_scale,
        is_causal=is_causal,
        window_size=window_size,
        softcap=softcap,
        varlen=varlen,
    )


__all__ = ["compile_iluvatar_flex_attention", "BLOCK_M", "BLOCK_N"]
