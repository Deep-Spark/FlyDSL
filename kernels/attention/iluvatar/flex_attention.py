# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar flex-attention forward kernel (V1).

This is a **variant subset** of PyTorch flex_attention (causal / SWA / softcap),
not a general ``score_mod`` compiler. See ``docs/iluvatar_flex_attention_v1_plan.md``.

Landed:
* PR-1a/1b/1c: fused forward, bf16 / D=128 / MHA / aligned self-attn, ``is_causal``.
* PR-2a: ``softcap`` (Gemma-2) + ``window_size`` (SWA); independent of causal.
* PR-2b: ``dtype`` in {f16, bf16} and ``D`` in {64, 128}.
* PR-2c: GQA, cross-attn (``Sq != Skv``), and arbitrary ``Sq``/``Skv`` via phys pad.
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

# --- PR-1 fixed tile parameters -----------------------------------------------
# Frozen internal constants per plan Q10 (c). PR-2 keeps them internal.
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
# PR-1b's cross-warp softmax reduce is written unconditionally (no
# ``if WARPS_N > 1``) to sidestep AST-rewriter branch-scoping; enforce here.
assert WARPS_N > 1, "PR-1b requires WARPS_N > 1 for the SMEM rowmax/rowsum reduce"


# --- Supported variants (V1 / PR-2c) ------------------------------------------
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
    """Validate compile-time inputs against the V1 envelope (plan §2.4)."""
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
    """Keep the PR-1a qk_dot helper on MHA + BLOCK-aligned self-attn only."""
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
    """Compile a plain ``S = (Q @ K^T) * sm_scale`` launcher (PR-1a dev helper).

    Layouts:
      * Q, K : ``[B, H, S*, D]`` bf16 (or f16), k-major in D.
      * S    : ``[B, H, Sq, Skv]`` fp32 output.

    One CTA processes one (b, h, q-tile) triple; loops over kv-tiles internally.
    Single-buffered K (no G2S/S2R pipelining) for PR-1a — perf tuning is a PR-1b
    concern once end-to-end numerical correctness is established.
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
    bk = D  # inner MMA K axis == full head dim; no D-axis pipeline in PR-1a.
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

        # Template S tile bound to (q_tile_idx, 0) — only its shape is used to size the
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


def _build_flex_attention_launcher(  # noqa: C901  (PR-1b: readability > cyclomatic score)
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
) -> Callable:
    """Compile the flex-attention forward launcher (PR-1 .. PR-2c).

    FA-1 chain: Q resident in SMEM; per-KV-tile MMA1 (Q @ K^T) -> online softmax
    with rowmax/rowsum cross-warp reduce -> P via SMEM (bf16, plain logical
    ``(BLOCK_M, BLOCK_N)`` layout — NOT SME-swizzled; the write side uses C
    operand TV via ``make_tiled_copy_C.partition_S`` and the read side uses A
    operand TV via ``make_tiled_copy_A.partition_S``, so both agree on the
    logical MN coordinates without a permutation) -> MMA2 (P @ V, host-side
    transposed V so both operands are k-major "tn") -> divide by rowsum ->
    write bf16 O back to gmem.

    Score mods (plan §2.3), after MMA1:
      * softcap set: ``S *= sm_scale`` → ``softcap * tanh(S/softcap)`` → ``S *= log2e``
      * softcap None: fused ``S *= (sm_scale * log2e)`` (bit-exact with PR-1)
      * then optional causal (``kv > q``) and SWA (``q - kv > window_size``) masks
        in the log2 domain with ``NEG_LARGE_F``.
      * PR-2c: when ``Skv`` is not a multiple of ``BLOCK_N``, also mask
        ``kv_g >= Skv`` (folded with ``const_expr`` so aligned shapes stay
        bit-exact). Callers must pass contiguous phys-padded Q/K/V/O.

    Loop-carried state per lane: one ``(m_running, l_running)`` fp32 pair per
    ``mma_m`` (i.e. ``ROWS_PER_LANE = WARP_ATOMS_M = 2`` pairs = 4 scalars),
    packed flat on the ``fx.range(init=...)`` boundary. ``m_running`` is
    initialised to ``-1e30`` (not ``-inf``) so
    ``alpha = exp2(m_prev - m_new)`` stays finite under fastmath in the first
    KV iteration.

    MRMma C-fragment TV (from ``lib/Dialect/FlyIXDL/MR/MmaAtom.cpp``
    ``getThrValLayoutC = FxLayout(FxShape(FxThr(16, 4), FxVal(4)),
    FxStride(FxThr(16, 1), FxVal(4)))``): with CuTe column-major codomain
    convention (M inner, N outer, ``flat = m + n * 16``), the mapping is
    ``(m, n) = (lane_row + 4 * ei, lane_col)`` — i.e., 4 accums per lane occupy
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

    elem_dtype = _DTYPE_STR_TO_FX[dtype]
    Sq_phys = _phys_seq(Sq, BLOCK_M)
    Skv_phys = _phys_seq(Skv, BLOCK_N)
    num_q_tiles = Sq_phys // BLOCK_M
    num_kv_tiles = Skv_phys // BLOCK_N
    has_kv_tail = Skv < Skv_phys
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

    # SMEM sizing (bf16 = 2 B). PR-1b totals: Q+K+V+P = 8+8+8+4 KiW = 56 KiB @ D=128.
    q_smem_elems = BLOCK_M * bk_qk
    k_smem_elems = BLOCK_N * bk_qk
    v_smem_elems = D * bk_pv
    p_smem_elems = BLOCK_M * BLOCK_N

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
    def flex_attn_kernel(Q: fx.Tensor, K: fx.Tensor, V: fx.Tensor, O: fx.Tensor):  # noqa: E741
        # Declared inside the kernel body so @fx.struct annotations resolve against
        # the local closure (see the PR-1a guardrail comment above for the PEP 563
        # rationale).
        # NB: fx.Array[T, size, align] — the third param is ALIGNMENT (bytes),
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
            k: fx.Array[elem_dtype, k_smem_elems]
            v: fx.Array[elem_dtype, v_smem_elems]
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
        c_skv_logical = fx.Int32(Skv)

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

        gQ = fx.slice(fx.flat_divide(Q_bh, (BLOCK_M, bk_qk)), (None, None, q_tile_idx, 0))
        gK_all = fx.flat_divide(K_bh, (BLOCK_N, bk_qk))
        gV_all = fx.flat_divide(V_bh, (D, bk_pv))
        gO = fx.slice(fx.flat_divide(O_bh, (BLOCK_M, D)), (None, None, q_tile_idx, 0))

        smem = fx.SharedAllocator(static=True).allocate(FlexAttnSmem).peek()
        q_smem = smem.q.ptr
        k_smem = smem.k.ptr
        v_smem = smem.v.ptr
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
        # bf16 slots apart — NOT byte-contiguous. A 32b (2-bf16 packed) copy
        # would silently corrupt half the elements. The MMA2 A-side read uses
        # 32b because A-TV vals are at codomains {0, 1, 8, 9} — pair (0, 1)
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
        # only along M (warp_m_id) — all warps in a warp_m row group share the
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

        # ---- KV loop: unrolled at Python trace time (num_kv_tiles fixed) --
        # PR-1b uses range_constexpr so m_running / l_running propagate as
        # plain Python-scope lists of DSL scalars — no scf.for loop-carried
        # state chain needed. When num_kv_tiles grows large we can revisit
        # with fx.range(init=...), but for the initial correctness pass this
        # keeps side effects on register-backed fragments (O_acc / S_frags)
        # inside the same MLIR region.
        m_prev = [c_neg_large for _ in range(ROWS_PER_LANE)]
        l_prev = [c_zero_f for _ in range(ROWS_PER_LANE)]
        for kv_idx_const in fx.range_constexpr(num_kv_tiles):
            kv_idx = fx.Int32(kv_idx_const)

            # ---- G2S K tile ------------------------------------------------
            gK = fx.slice(gK_all, (None, None, kv_idx, 0))
            sme_K = ixdl.make_sme_gmem_tensor(gK, leading_stride=k_row_stride)
            mr_gemm_g2s_issue_b_warp(
                a_mn_major=False,
                b_mn_major=False,
                warp_id=warp_id,
                b_per_warp=b_per_warp_qk,
                b_cta_gmem_view=fx.zipped_divide(sme_K, tile_smem_row),
                g2s_sme=g2s_sme,
                smem_b=k_smem,
                elem_dtype=elem_dtype,
                bm=BLOCK_M,
                bn=BLOCK_N,
                bk=bk_qk,
                geom=MR_GEMM_GEOM,
            )
            ixdl.cp_async_commit_group()
            ixdl.cp_async_wait_group(0)
            fx.gpu.barrier()

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

            # ---- Scale / softcap / enter log2 domain ----------------------
            # Plan §2.3: S *= sm_scale → optional softcap → masks → softmax.
            # Softmax uses exp2, so we enter log2 domain with *log2e after
            # softcap. When softcap is unset, keep the fused *(sm_scale*log2e)
            # path bit-exact with PR-1.
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

            # ---- Causal mask (PR-1c) --------------------------------------
            # Set S[q, k] = NEG_LARGE_F where ``kv_global > q_global``. Applied
            # AFTER entering the log2 domain above, so the sentinel lives in
            # the same domain as everything downstream and matches the
            # m_running init: exp2(NEG_LARGE_F - m_new) ≈ 0 for any bounded
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
                kv_block_base = kv_idx * fx.Int32(BLOCK_N) + warp_n_id * fx.Int32(WARP_ATOMS_N * ATOM_N)
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

            # ---- Sliding-window mask (PR-2a) ------------------------------
            # Plan §2.3: mask where ``q_global - kv_global > window_size``
            # (distance == window_size remains visible). Independent of
            # ``is_causal`` — callers wanting a causal window must set both.
            if fx.const_expr(has_swa):
                q_block_base = q_tile_idx * fx.Int32(BLOCK_M) + warp_m_id * fx.Int32(WARP_ATOMS_M * ATOM_M)
                kv_block_base = kv_idx * fx.Int32(BLOCK_N) + warp_n_id * fx.Int32(WARP_ATOMS_N * ATOM_N)
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

            # ---- KV tail mask (PR-2c) -------------------------------------
            # Phys pad runs full BLOCK_N G2S tiles; mask columns beyond the
            # logical Skv so pad K/V never enter softmax. Folded away when
            # Skv == Skv_phys (aligned path stays bit-exact with PR-2b).
            if fx.const_expr(has_kv_tail):
                kv_block_base = kv_idx * fx.Int32(BLOCK_N) + warp_n_id * fx.Int32(WARP_ATOMS_N * ATOM_N)
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
            # row, so gate the store on lane_col == 0. PR-1 fixed tile has
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

            # ---- G2S V tile -----------------------------------------------
            # V is transposed to shape (D, Skv); kv_idx selects the (D, BLOCK_N)
            # tile whose leading stride is Skv (contiguous BLOCK_N).
            gV = fx.slice(gV_all, (None, None, 0, kv_idx))
            sme_V = ixdl.make_sme_gmem_tensor(gV, leading_stride=v_row_stride)
            mr_gemm_g2s_issue_b_warp(
                a_mn_major=False,
                b_mn_major=False,
                warp_id=warp_id,
                b_per_warp=b_per_warp_pv,
                b_cta_gmem_view=fx.zipped_divide(sme_V, tile_smem_row),
                g2s_sme=g2s_sme,
                smem_b=v_smem,
                elem_dtype=elem_dtype,
                bm=BLOCK_M,
                bn=D,
                bk=bk_pv,
                geom=MR_GEMM_GEOM,
            )
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
                        smem_b=v_smem,
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

            # ---- Barrier before next iter's G2S overwrites K / P ---------
            fx.gpu.barrier()

            m_prev = m_new
            l_prev = l_new

        # ---- Post-loop: normalise and write O -----------------------------
        # ``l_prev`` holds l_final (the accumulated rowsum). Element ``ei`` of
        # an (mma_m, mma_n_pv) fragment is at row ``mma_m * 16 + ei * 4 +
        # lane_row``, matching slot index ``mma_m * 4 + ei``.
        l_final = list(l_prev)
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
        stream: fx.Stream = fx.Stream(None),
    ):
        flex_attn_kernel(Q, K, V, O).launch(
            grid=(B * H, num_q_tiles, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    def launch_flex_attn_checked(Q, K, V, O, stream=fx.Stream(None)):  # noqa: E741
        """Host entry: enforce phys-pad / GQA shapes before the JIT launch."""
        q_shape = _tensor_shape(Q)
        k_shape = _tensor_shape(K)
        v_shape = _tensor_shape(V)
        o_shape = _tensor_shape(O)
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
        return launch_flex_attn(Q, K, V, O, stream=stream)

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
    """Dev-only PR-1a helper: build a launcher that writes ``S = Q @ K^T * sm_scale``.

    Not part of the stable API. Exists to bisect the QK^T half of flex-attention
    before the softmax + PV halves land in PR-1b. Removed once
    ``compile_iluvatar_flex_attention`` is fully implemented.

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
) -> Callable:
    """Compile a fused flex-attention forward kernel for the Iluvatar backend.

    See ``docs/iluvatar_flex_attention_v1_plan.md`` for the V1 decision record.
    This is a variant subset (causal / SWA / softcap), not a general score_mod
    compiler.

    V1 scope (through PR-2c): Q/K/V/O = f16 or bf16, D in {64, 128}, GQA
    (``H % Hkv == 0``), arbitrary ``Sq``/``Skv`` (callers pass phys-padded
    contiguous tensors; see below), ``is_causal`` / ``window_size`` / ``softcap``
    in any combination (``is_causal`` requires ``Sq == Skv``).

    Phys pad contract:
        ``Sq_phys = ceil(Sq / 64) * 64``, ``Skv_phys = ceil(Skv / 64) * 64``.
        Launch expects contiguous
        ``Q,O: [B, H, Sq_phys, D]``, ``K: [B, Hkv, Skv_phys, D]``,
        ``V: [B, Hkv, D, Skv_phys]`` (host transpose of natural K-major V).
        Logical prefix ``[:Sq]`` / ``[:Skv]`` holds real data; pad rows/cols
        may be anything. Score columns ``kv >= Skv`` are masked in-kernel.

    Args:
        B, H, Sq, Skv, D: Logical shapes (compile-time constants). ``Sq``/``Skv``
            are logical lengths used for masking; physical tensor ranks use
            the padded sizes above.
        Hkv: KV head count; ``None`` means MHA (``Hkv = H``).
        dtype: ``"f16"`` or ``"bf16"``.
        is_causal: Enable lower-triangular causal mask; requires ``Sq == Skv``.
        window_size: Sliding-window radius; mask where ``q - kv > window_size``.
            Independent of ``is_causal``.
        softcap: Gemma-2 softcap; ``S = softcap * tanh(S / softcap)`` after
            ``sm_scale``, before log2-domain entry.
        sm_scale: Query scale; ``None`` defaults to ``1 / sqrt(D)``.

    Returns:
        A launcher ``launch_fn(Q, K, V, O, stream=None)`` that validates phys
        shapes then runs the kernel. Compare outputs on the logical ``O[..., :Sq, :]``
        prefix.
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
    )


__all__ = ["compile_iluvatar_flex_attention", "BLOCK_M", "BLOCK_N"]
