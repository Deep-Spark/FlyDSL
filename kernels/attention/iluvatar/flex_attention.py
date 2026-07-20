# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar flex-attention forward kernel (V1).

PR-1 lands the V1 scope incrementally via three internal sub-steps
(see ``docs/iluvatar_flex_attention_v1_plan.md`` §5.4):

* PR-1a (this commit): ``_compile_iluvatar_qk_dot_dev`` — QK^T-only dev helper,
  writes ``S = Q @ K^T * sm_scale`` (fp32) to gmem. Used to bisect the QK^T half
  before adding softmax / PV. Will be removed / inlined into
  ``compile_iluvatar_flex_attention`` in PR-1b.
* PR-1b: Full ``compile_iluvatar_flex_attention`` with online softmax + P via SMEM
  + P@V, no causal.
* PR-1c: Add ``is_causal=True``.
"""

# NOTE: do NOT add ``from __future__ import annotations`` — @fx.struct field
# annotations must be evaluated eagerly against the enclosing function scope so
# closure-captured tile constants (elem_dtype / q_smem_elems / k_smem_elems)
# resolve. PEP 563 stringifies them and the DSL layout probe then raises
# NameError inside the class body. Mirrors the guardrail comment in
# kernels/gemm/iluvatar/mr/hgemm.py.

import math
from typing import Callable

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.ixdl as ixdl
from flydsl.expr.typing import Vector as Vec
from kernels.gemm.iluvatar.common import GemmLayout
from kernels.gemm.iluvatar.mr.common import (
    ATOM_K_B16,
    ATOM_M,
    ATOM_N,
    MR_GEMM_GEOM,
    SMEM_ROWS,
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


# --- Supported variants (PR-1 subset) -----------------------------------------
_PR1_SUPPORTED_DTYPES = ("bf16",)
_PR1_SUPPORTED_D = (128,)
_DTYPE_STR_TO_FX = {"f16": fx.Float16, "bf16": fx.BFloat16}


def _validate_v1_scope(
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
    """Enforce the full V1 legal input envelope (per plan §2.4)."""
    if B <= 0 or H <= 0 or Hkv <= 0 or Sq <= 0 or Skv <= 0 or D <= 0:
        raise ValueError(f"all shape dims must be positive; got B={B} H={H} Hkv={Hkv} Sq={Sq} Skv={Skv} D={D}")
    if dtype not in ("f16", "bf16"):
        raise ValueError(f"dtype must be one of 'f16'/'bf16', got {dtype!r}")
    if D not in (64, 128):
        raise ValueError(f"D must be 64 or 128, got {D}")
    if H % Hkv != 0:
        raise ValueError(f"H ({H}) must be divisible by Hkv ({Hkv})")
    if is_causal and Sq != Skv:
        raise ValueError(f"is_causal=True requires Sq == Skv (self-attention); got Sq={Sq}, Skv={Skv}")
    if softcap is not None and softcap <= 0:
        raise ValueError(f"softcap must be > 0 when set, got {softcap}")
    if window_size is not None and window_size <= 0:
        raise ValueError(f"window_size must be > 0 when set, got {window_size}")


def _validate_pr1_scope(
    *,
    dtype: str,
    D: int,
    H: int,
    Hkv: int,
    Sq: int,
    Skv: int,
    is_causal: bool,
    window_size: int | None,
    softcap: float | None,
) -> None:
    """PR-1 additionally restricts the legal envelope. Removed in PR-2."""
    if dtype not in _PR1_SUPPORTED_DTYPES:
        raise NotImplementedError(f"PR-1 supports dtype {_PR1_SUPPORTED_DTYPES}; got {dtype!r}. f16 lands in PR-2.")
    if D not in _PR1_SUPPORTED_D:
        raise NotImplementedError(f"PR-1 supports D {_PR1_SUPPORTED_D}; got {D}. D=64 lands in PR-2.")
    if H != Hkv:
        raise NotImplementedError(f"PR-1 supports MHA only (Hkv=H); got H={H}, Hkv={Hkv}. GQA lands in PR-2.")
    if Sq != Skv:
        raise NotImplementedError(
            f"PR-1 supports self-attention only (Sq=Skv); got Sq={Sq}, Skv={Skv}. " "Cross-attention lands in PR-2."
        )
    if window_size is not None:
        raise NotImplementedError("PR-1 does not support sliding_window; lands in PR-2.")
    if softcap is not None:
        raise NotImplementedError("PR-1 does not support softcap; lands in PR-2.")
    if Sq % BLOCK_M != 0:
        raise ValueError(
            f"PR-1 requires strict alignment: Sq ({Sq}) must be a multiple of BLOCK_M ({BLOCK_M}). "
            "Tail masking lands in PR-2."
        )
    if Skv % BLOCK_N != 0:
        raise ValueError(
            f"PR-1 requires strict alignment: Skv ({Skv}) must be a multiple of BLOCK_N ({BLOCK_N}). "
            "Tail masking lands in PR-2."
        )
    if not is_causal:
        # PR-1b lands the non-causal path; PR-1c re-enables is_causal. The check stays only
        # to guard against callers who somehow reach here in the intermediate PR-1b state.
        pass


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
        B, H, Sq, Skv, D: See ``compile_iluvatar_flex_attention``. Same PR-1
            envelope (D == 128, dtype == "bf16", MHA, Sq/Skv aligned to BLOCK).
        dtype: ``"bf16"`` (PR-1 supports this only).
        sm_scale: Query scale; ``None`` -> ``1 / sqrt(D)``.

    Returns:
        ``launch_fn(Q, K, S_out, stream=None)``.
    """
    _validate_v1_scope(
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
    _validate_pr1_scope(
        dtype=dtype,
        D=D,
        H=H,
        Hkv=H,
        Sq=Sq,
        Skv=Skv,
        is_causal=False,
        window_size=None,
        softcap=None,
    )

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

    PR-1a landed only the QK^T half via ``_compile_iluvatar_qk_dot_dev``; the full
    fused kernel body (softmax + PV + epilogue) lands in PR-1b, causal masking in
    PR-1c. This entry point raises ``NotImplementedError`` until PR-1b.

    Args:
        B, H, Sq, Skv, D: See plan §2.4. All compile-time constants.
        Hkv: KV head count; ``None`` means MHA (Hkv = H). PR-1 requires Hkv == H.
        dtype: ``"f16"`` or ``"bf16"``. PR-1 accepts only ``"bf16"``.
        is_causal: Enable lower-triangular causal mask. PR-1 forces True; PR-1c
            actually implements it.
        window_size: Symmetric sliding-window radius; PR-1 must be ``None``.
        softcap: Gemma-2 softcap constant; PR-1 must be ``None``.
        sm_scale: Query scale; ``None`` defaults to ``1 / sqrt(D)``.

    Returns:
        A launcher ``launch_fn(Q, K, V, O, stream=None)``.
    """
    if Hkv is None:
        Hkv = H

    _validate_v1_scope(
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
    _validate_pr1_scope(
        dtype=dtype,
        D=D,
        H=H,
        Hkv=Hkv,
        Sq=Sq,
        Skv=Skv,
        is_causal=is_causal,
        window_size=window_size,
        softcap=softcap,
    )

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    sm_scale = float(sm_scale)

    raise NotImplementedError(
        "compile_iluvatar_flex_attention body lands in PR-1b (adds softmax + PV) "
        "and PR-1c (adds is_causal). PR-1a currently exposes only "
        "_compile_iluvatar_qk_dot_dev for QK^T bisection."
    )


__all__ = ["compile_iluvatar_flex_attention"]
