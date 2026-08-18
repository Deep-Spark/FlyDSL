# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar flex-attention forward (dense / varlen / paged + score_mod + BlockMask).

Supports f16/bf16, D in {64,128,256}, MHA/GQA, causal / SWA / softcap, varlen,
paged KV, dense alibi/score_bias, ``score_mod=TracedScoreMod`` on dense, varlen
(V3-5), and paged (V3-7a), dense ``block_mask`` / ``mask_mod`` sparse KV skip
(V3-3), and varlen ``mask_mod`` / packed BlockMask (V3-6). Optional ``tile_config``
and dense ``return_lse``.
"""

import math
from typing import Callable, Mapping, Optional

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.ixdl as ixdl
from flydsl.expr import arith
from flydsl.expr import math as fmath
from flydsl.expr.trace_mod import TracedMaskMod, TracedScoreMod
from flydsl.expr.typing import Vector as Vec
from kernels.attention.iluvatar.block_mask import (
    FlexBlockMask,
    PackedVarlenBlockMask,
    create_block_mask,
    create_block_masks_varlen,
    pack_block_masks_varlen,
)
from kernels.gemm.iluvatar.common import GemmLayout
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

# NB: kernels.gemm.iluvatar.epilogue is imported lazily inside the launcher build
# to avoid pulling ``byte_permute`` / ``stp_vs_b32`` (recent ixdl intrinsics) at
# module import time; that keeps the validation-only tests runnable against stale
# ``build-fly/python_packages`` Python bindings until a rebuild happens.

WARP_SIZE = 64

# --- Default tile parameters (overridable via tile_config) --------------------
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
_SUPPORTED_D = (64, 128, 256)
_SUPPORTED_BLOCK = (32, 64)
_DTYPE_STR_TO_FX = {"f16": fx.Float16, "bf16": fx.BFloat16}
_ELEM_BYTES_B16 = 2


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _phys_seq(seq: int, block: int) -> int:
    """Pad logical sequence length up to a multiple of ``block`` for G2S tiles."""
    return _ceil_div(seq, block) * block


def _normalize_tile_config(
    tile_config: Mapping[str, int] | None,
    *,
    paged: bool = False,
) -> tuple[int, int]:
    """Validate optional ``tile_config`` and return ``(block_m, block_n)``.

    Whitelist is ``{32,64} x {32,64}``. Paged KV requires ``block_n == 64``
    because ``page_size`` equals the KV tile. ``WARPS_M/N`` stay fixed at 2;
    ``WARP_ATOMS_*`` are derived as ``block // (ATOM_* * WARPS_*)``.
    """
    if tile_config is None:
        return BLOCK_M, BLOCK_N
    if not isinstance(tile_config, Mapping):
        raise ValueError(f"tile_config must be a mapping, got {type(tile_config)!r}")
    extra = set(tile_config) - {"block_m", "block_n"}
    if extra:
        raise ValueError(f"tile_config only allows block_m/block_n; unexpected keys {sorted(extra)}")
    block_m = int(tile_config.get("block_m", BLOCK_M))
    block_n = int(tile_config.get("block_n", BLOCK_N))
    if block_m not in _SUPPORTED_BLOCK or block_n not in _SUPPORTED_BLOCK:
        raise ValueError(
            f"tile_config block_m/block_n must be in {_SUPPORTED_BLOCK}; " f"got block_m={block_m}, block_n={block_n}"
        )
    if paged and block_n != BLOCK_N:
        raise ValueError(f"paged path requires block_n={BLOCK_N} (page_size), got {block_n}")
    warp_atoms_m = block_m // (ATOM_M * WARPS_M)
    warp_atoms_n = block_n // (ATOM_N * WARPS_N)
    if warp_atoms_m < 1 or block_m != ATOM_M * warp_atoms_m * WARPS_M:
        raise ValueError(f"block_m={block_m} is not compatible with ATOM_M={ATOM_M}, WARPS_M={WARPS_M}")
    if warp_atoms_n < 1 or block_n != ATOM_N * warp_atoms_n * WARPS_N:
        raise ValueError(f"block_n={block_n} is not compatible with ATOM_N={ATOM_N}, WARPS_N={WARPS_N}")
    return block_m, block_n


def _flex_attn_smem_bytes(
    D: int,
    kv_stages: int,
    *,
    block_m: int = BLOCK_M,
    block_n: int = BLOCK_N,
    elem_bytes: int = _ELEM_BYTES_B16,
) -> int:
    """Bytes for Q + staged K/V + P + s_red under the given tile geometry."""
    q = block_m * D * elem_bytes
    k = kv_stages * block_n * D * elem_bytes
    v = kv_stages * D * block_n * elem_bytes
    p = block_m * block_n * elem_bytes
    s_red = 2 * block_m * WARPS_N * 4  # fp32 rowmax/rowsum scratch
    return q + k + v + p + s_red


def _choose_kv_stages(
    D: int,
    *,
    block_m: int = BLOCK_M,
    block_n: int = BLOCK_N,
    elem_bytes: int = _ELEM_BYTES_B16,
) -> int:
    """Pick the largest ``kv_stages`` in {2, 1} that fits the CTA SMEM cap."""
    for stages in (2, 1):
        if (
            _flex_attn_smem_bytes(D, stages, block_m=block_m, block_n=block_n, elem_bytes=elem_bytes)
            <= DEFAULT_SMEM_CAP_BYTES
        ):
            return stages
    need = _flex_attn_smem_bytes(D, 1, block_m=block_m, block_n=block_n, elem_bytes=elem_bytes)
    raise ValueError(
        f"flex-attention SMEM {need} B for D={D} tile=({block_m},{block_n}) "
        f"exceeds CTA cap {DEFAULT_SMEM_CAP_BYTES} B"
    )


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
    allow_causal_cross_len: bool = False,
) -> None:
    """Validate compile-time inputs against the supported V1/V2 envelope."""
    if B <= 0 or H <= 0 or Hkv <= 0 or Sq <= 0 or Skv <= 0 or D <= 0:
        raise ValueError(f"all shape dims must be positive; got B={B} H={H} Hkv={Hkv} Sq={Sq} Skv={Skv} D={D}")
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"dtype must be one of {_SUPPORTED_DTYPES}, got {dtype!r}")
    if D not in _SUPPORTED_D:
        raise ValueError(f"D must be one of {_SUPPORTED_D}, got {D}")
    if H % Hkv != 0:
        raise ValueError(f"H ({H}) must be divisible by Hkv ({Hkv})")
    if is_causal and Sq != Skv and not allow_causal_cross_len:
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
    paged: bool = False,
    has_alibi: bool = False,
    has_score_bias: bool = False,
    score_mod: Optional[TracedScoreMod] = None,
    mask_mod: Optional[TracedMaskMod] = None,
    has_block_mask: bool = False,
    block_m: int = BLOCK_M,
    block_n: int = BLOCK_N,
    return_lse: bool = False,
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

    Score modifications after MMA1 (linear domain then log2):
      * ``S *= sm_scale``
      * optional ``+ alibi`` or ``+ score_bias`` (dense only; mutually exclusive)
      * optional softcap ``tanh``
      * optional ``score_mod.apply`` (dense + varlen; after softcap)
      * ``S *= log2e`` (when softcap/bias/score_mod unset, fuse ``*(sm_scale*log2e)``)
      * then optional causal / SWA / KV-tail masks in the log2 domain with
        ``NEG_LARGE_F``.
      * when ``Skv`` is not a multiple of ``BLOCK_N``, also mask
        ``kv_g >= Skv``. Callers must pass contiguous phys-padded Q/K/V/O.

    Loop-carried state per lane: ``ROWS_PER_LANE = WARP_ATOMS_M * 4`` fp32
    ``m_running`` values and the same count of ``l_running`` values, flat-packed
    on the ``fx.range(init=...)`` / ``yield`` boundary. ``m_running`` is
    initialised to ``NEG_LARGE_F`` so
    ``alpha = exp2(m_prev - m_new)`` stays finite under fastmath in the first
    KV iteration. Causal fully-masked KV tiles are skipped by shortening the
    trip count to ``kv_end``; K/V SMEM uses ``kv_stages`` in {1,2} chosen to fit
    the CTA shared-memory cap (2-stage when possible; 1-stage for large D).

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

    if varlen and paged:
        raise ValueError("paged and varlen are mutually exclusive")
    if varlen and Sq != Skv:
        raise ValueError(f"varlen self-attn requires Sq == Skv (max_seqlen); got Sq={Sq}, Skv={Skv}")
    if has_alibi and has_score_bias:
        raise ValueError("alibi_slopes and score_bias are mutually exclusive")
    if (has_alibi or has_score_bias) and (varlen or paged):
        raise ValueError("alibi/score_bias are dense-only in V2-4 (not supported with varlen/paged)")
    if score_mod is not None and not isinstance(score_mod, TracedScoreMod):
        raise ValueError(f"score_mod must be TracedScoreMod or None, got {type(score_mod).__name__}")
    if mask_mod is not None and not isinstance(mask_mod, TracedMaskMod):
        raise ValueError(f"mask_mod must be TracedMaskMod or None, got {type(mask_mod).__name__}")
    if (mask_mod is not None or has_block_mask) and paged:
        raise ValueError("block_mask/mask_mod are not supported with paged")

    has_score_mod = score_mod is not None
    # Captured into the nested kernel closure for JIT cache keying (forbid id(fn)).
    score_mod_fingerprint = score_mod.fingerprint if has_score_mod else ""
    score_mod_obj = score_mod
    has_mask_mod = mask_mod is not None
    mask_mod_fingerprint = mask_mod.fingerprint if has_mask_mod else ""
    mask_mod_obj = mask_mod
    has_block_mask = bool(has_block_mask)

    # Shadow module defaults so the nested kernel closure binds these ints.
    BLOCK_M = int(block_m)
    BLOCK_N = int(block_n)
    WARP_ATOMS_M = BLOCK_M // (ATOM_M * WARPS_M)
    WARP_ATOMS_N = BLOCK_N // (ATOM_N * WARPS_N)
    assert BLOCK_M == ATOM_M * WARP_ATOMS_M * WARPS_M
    assert BLOCK_N == ATOM_N * WARP_ATOMS_N * WARPS_N

    elem_dtype = _DTYPE_STR_TO_FX[dtype]
    Sq_phys = _phys_seq(Sq, BLOCK_M)
    Skv_phys = _phys_seq(Skv, BLOCK_N)
    num_q_tiles = Sq_phys // BLOCK_M
    num_kv_tiles = Skv_phys // BLOCK_N
    max_num_pages = num_kv_tiles  # page_size == BLOCK_N
    # Varlen/paged always apply a runtime kv>=seqlen mask; dense only when phys-padded.
    has_kv_tail = bool(varlen) or bool(paged) or (Skv < Skv_phys)
    group_size = H // Hkv
    PAGE_SIZE = BLOCK_N
    # K cache [NumBlocks, PAGE_SIZE, Hkv, D]; V kernel-facing [NumBlocks, Hkv, D, PAGE_SIZE].
    page_elems_k = PAGE_SIZE * Hkv * D
    page_elems_v = Hkv * D * PAGE_SIZE
    k_page_token_stride = Hkv * D
    v_page_leading = PAGE_SIZE  # (D, PAGE_SIZE):(PAGE_SIZE, 1)

    bk_qk = D  # MMA1 inner K = head dim
    bk_pv = BLOCK_N  # MMA2 inner K = kv-tile size
    k_atoms_qk = bk_qk // ATOM_K_B16
    k_atoms_pv = bk_pv // ATOM_K_B16

    # Warp partition:
    #   MMA1 (S = Q @ K^T): S extent per CTA = (BLOCK_M, BLOCK_N);
    #     WARPS_M x WARPS_N warp grid; warp_atoms derived from block size.
    #   MMA2 (O += P @ V):  O extent per CTA = (BLOCK_M, D);
    #     same warp grid; warp_m_id still splits M,
    #     but warp_n_id splits D (not BLOCK_N), so each warp owns
    #     (BLOCK_M/WARPS_M) x (D / WARPS_N) of O ->
    #     warp_atoms_n_pv = D / (ATOM_N * WARPS_N)
    #     (4 at D=128, 8 at D=256).
    WARP_ATOMS_N_PV = D // (ATOM_N * WARPS_N)
    assert D % (ATOM_N * WARPS_N) == 0, f"D={D} must be divisible by ATOM_N*WARPS_N={ATOM_N * WARPS_N}"
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

    # SMEM: pick largest kv_stages in {2,1} under the CTA cap for this tile x D.
    KV_STAGES = _choose_kv_stages(D, block_m=BLOCK_M, block_n=BLOCK_N, elem_bytes=_ELEM_BYTES_B16)
    q_smem_elems = BLOCK_M * bk_qk
    k_smem_elems = BLOCK_N * bk_qk
    v_smem_elems = D * bk_pv
    p_smem_elems = BLOCK_M * BLOCK_N
    k_smem_elems_staged = k_smem_elems * KV_STAGES
    v_smem_elems_staged = v_smem_elems * KV_STAGES
    smem_bytes = _flex_attn_smem_bytes(D, KV_STAGES, block_m=BLOCK_M, block_n=BLOCK_N, elem_bytes=_ELEM_BYTES_B16)
    assert smem_bytes <= DEFAULT_SMEM_CAP_BYTES, (
        f"flex-attention SMEM {smem_bytes} B exceeds cap {DEFAULT_SMEM_CAP_BYTES} B "
        f"(D={D}, tile=({BLOCK_M},{BLOCK_N}), kv_stages={KV_STAGES})"
    )

    # Dense: BHSD phys-padded. Varlen packed: Q/K/O [total, H(or Hkv), D] with
    # token stride H*D; V host-transposed to [Hkv, D, total] (total is runtime).
    # Paged: Q/O dense BHSD; K pages [NumBlocks, PAGE, Hkv, D]; V pages
    # host-transposed per page to [NumBlocks, Hkv, D, PAGE] for MMA2 "tn".
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
    elif paged:
        q_row_stride = D
        k_row_stride = k_page_token_stride  # within a page: (PAGE, D):(Hkv*D, 1)
        v_row_stride = v_page_leading  # within a page: (D, PAGE):(PAGE, 1)
        o_row_stride = D
        q_batch_stride = H * Sq_phys * D
        q_head_stride = Sq_phys * D
        k_batch_stride = 0
        k_head_stride = D
        v_batch_stride = 0
        v_head_stride = D * PAGE_SIZE
        o_batch_stride = H * Sq_phys * D
        o_head_stride = Sq_phys * D
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
    # (row m = warp_m_id * (WARP_ATOMS_M * ATOM_M) + mma_m * ATOM_M + ei * 4 +
    # lane_row). See TV analysis in the docstring above.
    ROWS_PER_LANE = WARP_ATOMS_M * 4
    # ``m_running`` init: log2-domain "very negative". Value chosen so
    # ``exp2(NEG_LARGE_F - m_new) = 0`` in fp32 for any bounded ``m_new``, but
    # not so extreme that ``__nv_exp2f``'s polynomial approximation returns
    # NaN under ``fastmath=fast`` (NInf/AFN).
    NEG_LARGE_F = -60.0
    LOG2E = 1.4426950408889634
    LN2 = math.log(2.0)
    NEG_INF_F = float("-inf")
    scale_log2e = float(sm_scale_f32) * LOG2E
    shuffle_steps = int(math.log2(TCU_LANE_COLS))
    has_softcap = softcap is not None
    has_swa = window_size is not None
    has_bias = bool(has_alibi) or bool(has_score_bias)
    # Unfused scale path when softcap, additive bias, or score_mod is present.
    use_unfused_scale = bool(has_softcap) or bool(has_bias) or bool(has_score_mod)
    softcap_f32 = float(softcap) if has_softcap else 0.0
    window_size_i = int(window_size) if has_swa else 0
    do_return_lse = bool(return_lse)

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def flex_attn_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        O: fx.Tensor,  # noqa: E741
        LSE: fx.Tensor,
        CuSeqLen: fx.Tensor,
        SeqLens: fx.Tensor,
        AlibiSlopes: fx.Tensor,
        ScoreBias: fx.Tensor,
        KvNumBlocks: fx.Tensor,
        KvIndices: fx.Tensor,
        KvIsFull: fx.Tensor,
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

        # Dense additive bias (V2-4): load alibi slope once per head.
        if fx.const_expr(has_alibi):
            alibi_view = fx.make_view(
                fx.get_iter(AlibiSlopes),
                fx.make_layout((H,), (1,)),
            )
            alibi_slope = fx.memref_load(alibi_view, h_idx)
        else:
            alibi_slope = c_zero_f

        # Dense: compile-time logical Skv. Varlen: per-seq length from cu_seqlens.
        # Paged: seq_lens_kv from SeqLens; block_table in CuSeqLen slot.
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
            c_causal_delta = fx.Int32(0)
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
            # return_lse is dense-only; placeholder keeps SSA defined.
            LSE_bh = fx.make_view(fx.get_iter(LSE), fx.make_layout((1,), (1,)))
        elif fx.const_expr(paged):
            bt_view = fx.make_view(
                fx.get_iter(CuSeqLen),
                fx.make_layout((B, max_num_pages), (max_num_pages, 1)),
            )
            seqlens_kv_view = fx.make_view(
                fx.get_iter(SeqLens),
                fx.make_layout((B,), (1,)),
            )
            seqlen_kv = fx.memref_load(seqlens_kv_view, b_idx)
            c_skv_logical = seqlen_kv
            # Cross-length causal: mask kv > q + (seqlen_kv - Sq).
            c_causal_delta = seqlen_kv - fx.Int32(Sq)
            active = fx.Int32(1) != fx.Int32(0)
            seqlen = fx.Int32(Sq)  # Q rows are dense phys-padded; store mask uses Sq
            Q_view = fx.make_view(
                fx.get_iter(Q),
                fx.make_layout((B, H, Sq_phys, D), (q_batch_stride, q_head_stride, q_row_stride, 1)),
            )
            O_view = fx.make_view(
                fx.get_iter(O),
                fx.make_layout((B, H, Sq_phys, D), (o_batch_stride, o_head_stride, o_row_stride, 1)),
            )
            Q_bh = fx.slice(Q_view, (b_idx, h_idx, None, None))
            O_bh = fx.slice(O_view, (b_idx, h_idx, None, None))
            # K/V pages are gathered in _issue_k / _issue_v; placeholders unused.
            K_bh = Q_bh
            V_bh = Q_bh
            v_row_stride_r = fx.Int32(v_row_stride)
            LSE_bh = fx.make_view(fx.get_iter(LSE), fx.make_layout((1,), (1,)))
        else:
            c_skv_logical = fx.Int32(Skv)
            c_causal_delta = fx.Int32(0)
            active = fx.Int32(1) != fx.Int32(0)
            seqlen = fx.Int32(Sq)
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
            if fx.const_expr(do_return_lse):
                LSE_view = fx.make_view(
                    fx.get_iter(LSE),
                    fx.make_layout((B, H, Sq_phys), (H * Sq_phys, Sq_phys, 1)),
                )
                LSE_bh = fx.slice(LSE_view, (b_idx, h_idx, None))
            else:
                LSE_bh = fx.make_view(fx.get_iter(LSE), fx.make_layout((1,), (1,)))

        gQ = fx.slice(fx.flat_divide(Q_bh, (BLOCK_M, bk_qk)), (None, None, q_tile_idx, 0))
        if fx.const_expr(paged):
            gK_all = gQ  # unused; page gather uses K/V base + block_table
            gV_all = gQ
        else:
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
            if fx.const_expr(paged):
                page_id = fx.memref_load(bt_view, (b_idx, kv_idx))
                k_page_ptr = fx.add_offset(
                    fx.get_iter(K),
                    fx.make_int_tuple(page_id * fx.Int32(page_elems_k) + hkv_idx * fx.Int32(D)),
                )
                gK = fx.make_view(
                    k_page_ptr,
                    fx.make_layout((BLOCK_N, D), (k_page_token_stride, 1)),
                )
                sme_K = ixdl.make_sme_gmem_tensor(gK, leading_stride=k_page_token_stride)
            else:
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
            if fx.const_expr(paged):
                page_id = fx.memref_load(bt_view, (b_idx, kv_idx))
                v_page_ptr = fx.add_offset(
                    fx.get_iter(V),
                    fx.make_int_tuple(page_id * fx.Int32(page_elems_v) + hkv_idx * fx.Int32(D * PAGE_SIZE)),
                )
                gV = fx.make_view(
                    v_page_ptr,
                    fx.make_layout((D, BLOCK_N), (PAGE_SIZE, 1)),
                )
                sme_V = ixdl.make_sme_gmem_tensor(gV, leading_stride=PAGE_SIZE)
            else:
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
        # Dense (no block_mask): causal fully-masked tiles are dropped by
        # shortening kv_end. With block_mask: trip count comes from
        # kv_num_blocks[q_tile]; kv_i is looked up from kv_indices.
        #
        # m_running / l_running are flat-packed across the scf.for boundary
        # (ROWS_PER_LANE fp32 each). O_acc stays outside and is updated in place.
        #
        # Light K/V pipeline: with kv_stages=2, prologue loads K0; after QK
        # issue K_{i+1} into the other bank; after P-write issue V_i; one wait
        # drains both. With kv_stages=1 (large D), both banks collapse to stage
        # 0 -- K_{i+1} may still prefetch into the same K bank after MMA1.
        c_num_kv = fx.Int32(num_kv_tiles)
        if fx.const_expr(has_block_mask):
            if fx.const_expr(varlen):
                # V3-6: batched tables [num_seqs, max_q_tiles, ...]; index by seq_id.
                kv_num_view = fx.make_view(
                    fx.get_iter(KvNumBlocks),
                    fx.make_layout((B, num_q_tiles), (num_q_tiles, 1)),
                )
                kv_idx_view = fx.make_view(
                    fx.get_iter(KvIndices),
                    fx.make_layout(
                        (B, num_q_tiles, num_kv_tiles),
                        (num_q_tiles * num_kv_tiles, num_kv_tiles, 1),
                    ),
                )
                kv_full_view = fx.make_view(
                    fx.get_iter(KvIsFull),
                    fx.make_layout(
                        (B, num_q_tiles, num_kv_tiles),
                        (num_q_tiles * num_kv_tiles, num_kv_tiles, 1),
                    ),
                )
                kv_trip = fx.memref_load(kv_num_view, (b_idx, q_tile_idx))
            else:
                kv_num_view = fx.make_view(
                    fx.get_iter(KvNumBlocks),
                    fx.make_layout((num_q_tiles,), (1,)),
                )
                kv_idx_view = fx.make_view(
                    fx.get_iter(KvIndices),
                    fx.make_layout((num_q_tiles, num_kv_tiles), (num_kv_tiles, 1)),
                )
                kv_full_view = fx.make_view(
                    fx.get_iter(KvIsFull),
                    fx.make_layout((num_q_tiles, num_kv_tiles), (num_kv_tiles, 1)),
                )
                kv_trip = fx.memref_load(kv_num_view, q_tile_idx)
            _mask_mod_fp = mask_mod_fingerprint
        else:
            if fx.const_expr(is_causal):
                # Self (delta=0): kv_end = q_tile + 1. Cross-length paged: allow
                # kv up through q_tile_end + causal_delta.
                q_tile_end = q_start + fx.Int32(BLOCK_M)
                kv_end_cand = (q_tile_end + c_causal_delta + c_block_n - fx.Int32(1)) // c_block_n
                kv_end = (kv_end_cand < c_num_kv).select(kv_end_cand, c_num_kv)
                kv_end = (kv_end > fx.Int32(0)).select(kv_end, fx.Int32(0))
            else:
                kv_end = c_num_kv
            # Varlen / paged: also drop tiles past this sequence's logical KV length.
            if fx.const_expr(varlen or paged):
                kv_tiles_seq = (c_skv_logical + c_block_n - fx.Int32(1)) // c_block_n
                kv_end = (kv_end < kv_tiles_seq).select(kv_end, kv_tiles_seq)
            if fx.const_expr(varlen):
                # Inactive q-tiles (q_start >= seqlen): skip the KV loop entirely.
                kv_end = active.select(kv_end, fx.Int32(0))
            kv_trip = kv_end

        # Prologue: first K into stage 0 (skipped when kv_trip == 0).
        if kv_trip > fx.Int32(0):
            if fx.const_expr(has_block_mask):
                if fx.const_expr(varlen):
                    kv0 = fx.memref_load(kv_idx_view, (b_idx, q_tile_idx, fx.Int32(0)))
                else:
                    kv0 = fx.memref_load(kv_idx_view, (q_tile_idx, fx.Int32(0)))
                _issue_k(kv0, fx.Int32(0))
            else:
                _issue_k(fx.Int32(0), fx.Int32(0))
            ixdl.cp_async_commit_group()
            ixdl.cp_async_wait_group(0)
            fx.gpu.barrier()

        init_state = [c_neg_large for _ in range(ROWS_PER_LANE)] + [c_zero_f for _ in range(ROWS_PER_LANE)]
        loop_results = init_state
        for sparse_i, state in fx.range(fx.Int32(0), kv_trip, fx.Int32(1), init=init_state):
            m_prev = [state[slot] for slot in range(ROWS_PER_LANE)]
            l_prev = [state[ROWS_PER_LANE + slot] for slot in range(ROWS_PER_LANE)]
            if fx.const_expr(has_block_mask):
                if fx.const_expr(varlen):
                    kv_i = fx.memref_load(kv_idx_view, (b_idx, q_tile_idx, sparse_i))
                    tile_full_i = fx.memref_load(kv_full_view, (b_idx, q_tile_idx, sparse_i))
                else:
                    kv_i = fx.memref_load(kv_idx_view, (q_tile_idx, sparse_i))
                    tile_full_i = fx.memref_load(kv_full_view, (q_tile_idx, sparse_i))
                tile_is_full = tile_full_i != fx.Int32(0)
            else:
                kv_i = fx.Int32(sparse_i)
                tile_is_full = fx.Int32(0) != fx.Int32(0)
            if fx.const_expr(KV_STAGES == 2):
                # fx.range IV may be index; force i32 before remui.
                comp_stage = fx.Int32(sparse_i) % fx.Int32(2)
                prefetch_stage = comp_stage ^ fx.Int32(1)
            else:
                comp_stage = fx.Int32(0)
                prefetch_stage = fx.Int32(0)
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

            # Prefetch K_{next} so G2S overlaps softmax / P.
            next_s = fx.Int32(sparse_i) + fx.Int32(1)
            if next_s < kv_trip:
                if fx.const_expr(has_block_mask):
                    if fx.const_expr(varlen):
                        next_kv = fx.memref_load(kv_idx_view, (b_idx, q_tile_idx, next_s))
                    else:
                        next_kv = fx.memref_load(kv_idx_view, (q_tile_idx, next_s))
                else:
                    next_kv = next_s
                _issue_k(next_kv, prefetch_stage)
                ixdl.cp_async_commit_group()

            # ---- Scale / bias / softcap / score_mod / enter log2 domain ----
            # Order: S *= sm_scale -> optional alibi|score_bias -> softcap
            # -> optional score_mod -> *log2e. When none of softcap/bias/score_mod:
            # fuse *(sm_scale*log2e). score_mod=None keeps the legacy unfused
            # branches below bit-identical.
            if fx.const_expr(use_unfused_scale):
                q_block_base = q_tile_idx * fx.Int32(BLOCK_M) + warp_m_id * fx.Int32(WARP_ATOMS_M * ATOM_M)
                kv_block_base = kv_i * fx.Int32(BLOCK_N) + warp_n_id * fx.Int32(WARP_ATOMS_N * ATOM_N)
                # Touch fingerprint so it stays in the kernel closure (JIT key).
                _score_mod_fp = score_mod_fingerprint
                if fx.const_expr(has_score_mod):
                    # Linear-domain mods first, then score_mod.apply, then *log2e.
                    if fx.const_expr(has_alibi):
                        for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                            for mma_n in fx.range_constexpr(WARP_ATOMS_N):
                                acc = S_frags[mma_m][mma_n]
                                old = Vec(acc.load())
                                kv_g = kv_block_base + fx.Int32(mma_n * ATOM_N) + lane_col
                                if fx.const_expr(has_softcap):
                                    acc.store(
                                        Vec.from_elements(
                                            [
                                                c_softcap
                                                * fmath.tanh(
                                                    (
                                                        old[ei] * c_sm_scale
                                                        + (
                                                            c_zero_f
                                                            - alibi_slope
                                                            * (
                                                                q_block_base
                                                                + fx.Int32(mma_m * ATOM_M + ei * 4)
                                                                + lane_row
                                                                - kv_g
                                                            ).to(fx.Float32)
                                                        )
                                                    )
                                                    / c_softcap,
                                                    fastmath=fm_fast,
                                                )
                                                for ei in range(4)
                                            ],
                                            fx.Float32,
                                        )
                                    )
                                else:
                                    acc.store(
                                        Vec.from_elements(
                                            [
                                                old[ei] * c_sm_scale
                                                + (
                                                    c_zero_f
                                                    - alibi_slope
                                                    * (
                                                        q_block_base
                                                        + fx.Int32(mma_m * ATOM_M + ei * 4)
                                                        + lane_row
                                                        - kv_g
                                                    ).to(fx.Float32)
                                                )
                                                for ei in range(4)
                                            ],
                                            fx.Float32,
                                        )
                                    )
                    elif fx.const_expr(has_score_bias):
                        score_bias_view = fx.make_view(
                            fx.get_iter(ScoreBias),
                            fx.make_layout(
                                (B, H, Sq_phys, Skv_phys),
                                (H * Sq_phys * Skv_phys, Sq_phys * Skv_phys, Skv_phys, 1),
                            ),
                        )
                        for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                            for mma_n in fx.range_constexpr(WARP_ATOMS_N):
                                acc = S_frags[mma_m][mma_n]
                                old = Vec(acc.load())
                                kv_g = kv_block_base + fx.Int32(mma_n * ATOM_N) + lane_col
                                if fx.const_expr(has_softcap):
                                    acc.store(
                                        Vec.from_elements(
                                            [
                                                c_softcap
                                                * fmath.tanh(
                                                    (
                                                        old[ei] * c_sm_scale
                                                        + fx.memref_load(
                                                            score_bias_view,
                                                            (
                                                                b_idx,
                                                                h_idx,
                                                                q_block_base
                                                                + fx.Int32(mma_m * ATOM_M + ei * 4)
                                                                + lane_row,
                                                                kv_g,
                                                            ),
                                                        )
                                                    )
                                                    / c_softcap,
                                                    fastmath=fm_fast,
                                                )
                                                for ei in range(4)
                                            ],
                                            fx.Float32,
                                        )
                                    )
                                else:
                                    acc.store(
                                        Vec.from_elements(
                                            [
                                                old[ei] * c_sm_scale
                                                + fx.memref_load(
                                                    score_bias_view,
                                                    (
                                                        b_idx,
                                                        h_idx,
                                                        q_block_base + fx.Int32(mma_m * ATOM_M + ei * 4) + lane_row,
                                                        kv_g,
                                                    ),
                                                )
                                                for ei in range(4)
                                            ],
                                            fx.Float32,
                                        )
                                    )
                    elif fx.const_expr(has_softcap):
                        for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                            for mma_n in fx.range_constexpr(WARP_ATOMS_N):
                                acc = S_frags[mma_m][mma_n]
                                old = Vec(acc.load())
                                acc.store(
                                    Vec.from_elements(
                                        [
                                            c_softcap
                                            * fmath.tanh(
                                                (old[ei] * c_sm_scale) / c_softcap,
                                                fastmath=fm_fast,
                                            )
                                            for ei in range(4)
                                        ],
                                        fx.Float32,
                                    )
                                )
                    else:
                        # score_mod only: S *= sm_scale
                        for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                            for mma_n in fx.range_constexpr(WARP_ATOMS_N):
                                acc = S_frags[mma_m][mma_n]
                                acc.store(Vec(acc.load()) * c_sm_scale)
                    # score_mod.apply then enter log2 domain
                    for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                        for mma_n in fx.range_constexpr(WARP_ATOMS_N):
                            acc = S_frags[mma_m][mma_n]
                            old = Vec(acc.load())
                            kv_g = kv_block_base + fx.Int32(mma_n * ATOM_N) + lane_col
                            acc.store(
                                Vec.from_elements(
                                    [
                                        c_log2e
                                        * score_mod_obj.apply(
                                            old[ei],
                                            b_idx,
                                            h_idx,
                                            q_block_base + fx.Int32(mma_m * ATOM_M + ei * 4) + lane_row,
                                            kv_g,
                                        )
                                        for ei in range(4)
                                    ],
                                    fx.Float32,
                                )
                            )
                elif fx.const_expr(has_alibi):
                    for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                        for mma_n in fx.range_constexpr(WARP_ATOMS_N):
                            acc = S_frags[mma_m][mma_n]
                            old = Vec(acc.load())
                            kv_g = kv_block_base + fx.Int32(mma_n * ATOM_N) + lane_col
                            if fx.const_expr(has_softcap):
                                acc.store(
                                    Vec.from_elements(
                                        [
                                            c_log2e
                                            * (
                                                c_softcap
                                                * fmath.tanh(
                                                    (
                                                        old[ei] * c_sm_scale
                                                        + (
                                                            c_zero_f
                                                            - alibi_slope
                                                            * (
                                                                q_block_base
                                                                + fx.Int32(mma_m * ATOM_M + ei * 4)
                                                                + lane_row
                                                                - kv_g
                                                            ).to(fx.Float32)
                                                        )
                                                    )
                                                    / c_softcap,
                                                    fastmath=fm_fast,
                                                )
                                            )
                                            for ei in range(4)
                                        ],
                                        fx.Float32,
                                    )
                                )
                            else:
                                acc.store(
                                    Vec.from_elements(
                                        [
                                            (
                                                old[ei] * c_sm_scale
                                                + (
                                                    c_zero_f
                                                    - alibi_slope
                                                    * (
                                                        q_block_base
                                                        + fx.Int32(mma_m * ATOM_M + ei * 4)
                                                        + lane_row
                                                        - kv_g
                                                    ).to(fx.Float32)
                                                )
                                            )
                                            * c_log2e
                                            for ei in range(4)
                                        ],
                                        fx.Float32,
                                    )
                                )
                elif fx.const_expr(has_score_bias):
                    score_bias_view = fx.make_view(
                        fx.get_iter(ScoreBias),
                        fx.make_layout(
                            (B, H, Sq_phys, Skv_phys),
                            (H * Sq_phys * Skv_phys, Sq_phys * Skv_phys, Skv_phys, 1),
                        ),
                    )
                    for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                        for mma_n in fx.range_constexpr(WARP_ATOMS_N):
                            acc = S_frags[mma_m][mma_n]
                            old = Vec(acc.load())
                            kv_g = kv_block_base + fx.Int32(mma_n * ATOM_N) + lane_col
                            if fx.const_expr(has_softcap):
                                acc.store(
                                    Vec.from_elements(
                                        [
                                            c_log2e
                                            * (
                                                c_softcap
                                                * fmath.tanh(
                                                    (
                                                        old[ei] * c_sm_scale
                                                        + fx.memref_load(
                                                            score_bias_view,
                                                            (
                                                                b_idx,
                                                                h_idx,
                                                                q_block_base
                                                                + fx.Int32(mma_m * ATOM_M + ei * 4)
                                                                + lane_row,
                                                                kv_g,
                                                            ),
                                                        )
                                                    )
                                                    / c_softcap,
                                                    fastmath=fm_fast,
                                                )
                                            )
                                            for ei in range(4)
                                        ],
                                        fx.Float32,
                                    )
                                )
                            else:
                                acc.store(
                                    Vec.from_elements(
                                        [
                                            (
                                                old[ei] * c_sm_scale
                                                + fx.memref_load(
                                                    score_bias_view,
                                                    (
                                                        b_idx,
                                                        h_idx,
                                                        q_block_base + fx.Int32(mma_m * ATOM_M + ei * 4) + lane_row,
                                                        kv_g,
                                                    ),
                                                )
                                            )
                                            * c_log2e
                                            for ei in range(4)
                                        ],
                                        fx.Float32,
                                    )
                                )
                else:
                    # softcap only (no additive bias, no score_mod)
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

            # ---- Element masks (FULL tiles may skip via select, not scf.if) -
            # Touch mask_mod fingerprint for JIT key (Python str, not IR).
            if fx.const_expr(has_mask_mod):
                _mask_mod_fp = mask_mod_fingerprint
            do_elem = fx.Int32(1)
            if fx.const_expr(has_block_mask):
                do_elem = tile_is_full.select(fx.Int32(0), fx.Int32(1))
            do_elem_b = do_elem != fx.Int32(0)

            # ---- Causal mask ----------------------------------------------
            # Set S[q, k] = NEG_LARGE_F where ``kv_global > q_global``.
            if fx.const_expr(is_causal):
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
                                        do_elem_b
                                        & (
                                            kv_g
                                            > q_block_base
                                            + fx.Int32(mma_m * ATOM_M + ei * 4)
                                            + lane_row
                                            + c_causal_delta
                                        )
                                    ).select(c_neg_large, old[ei])
                                    for ei in range(4)
                                ],
                                fx.Float32,
                            )
                        )

            # ---- Sliding-window mask --------------------------------------
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
                                        do_elem_b
                                        & (
                                            (q_block_base + fx.Int32(mma_m * ATOM_M + ei * 4) + lane_row) - kv_g
                                            > c_window
                                        )
                                    ).select(c_neg_large, old[ei])
                                    for ei in range(4)
                                ],
                                fx.Float32,
                            )
                        )

            # ---- KV tail mask ---------------------------------------------
            if fx.const_expr(has_kv_tail):
                kv_block_base = kv_i * fx.Int32(BLOCK_N) + warp_n_id * fx.Int32(WARP_ATOMS_N * ATOM_N)
                for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                    for mma_n in fx.range_constexpr(WARP_ATOMS_N):
                        acc = S_frags[mma_m][mma_n]
                        old = Vec(acc.load())
                        kv_g = kv_block_base + fx.Int32(mma_n * ATOM_N) + lane_col
                        acc.store(
                            Vec.from_elements(
                                [(do_elem_b & (kv_g >= c_skv_logical)).select(c_neg_large, old[ei]) for ei in range(4)],
                                fx.Float32,
                            )
                        )

            # ---- mask_mod element holes (V3-3) ----------------------------
            if fx.const_expr(has_mask_mod):
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
                                    do_elem_b.select(
                                        mask_mod_obj.apply(
                                            b_idx,
                                            h_idx,
                                            q_block_base + fx.Int32(mma_m * ATOM_M + ei * 4) + lane_row,
                                            kv_g,
                                        ).select(old[ei], c_neg_large),
                                        old[ei],
                                    )
                                    for ei in range(4)
                                ],
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

        c_ln2 = fx.Float32(LN2)
        c_neg_inf = fx.Float32(NEG_INF_F)

        l_final = [loop_results[ROWS_PER_LANE + slot] for slot in range(ROWS_PER_LANE)]
        m_final = [loop_results[slot] for slot in range(ROWS_PER_LANE)]

        # Optional natural-log LSE: LSE = m_log2 * ln2 + ln(l). Dense-only.
        if fx.const_expr(do_return_lse):
            if active:
                for mma_m in fx.range_constexpr(WARP_ATOMS_M):
                    for ei in fx.range_constexpr(4):
                        slot = mma_m * 4 + ei
                        row = warp_m_id * fx.Int32(WARP_ATOMS_M * ATOM_M) + fx.Int32(mma_m * ATOM_M + ei * 4) + lane_row
                        q_row = q_start + row
                        l_val = l_final[slot]
                        m_val = m_final[slot]
                        # Empty / fully-masked row -> -inf.
                        lse_val = (l_val > c_zero_f).select(
                            m_val * c_ln2 + fmath.log(l_val, fastmath=fm_fast),
                            c_neg_inf,
                        )
                        if lane_col == fx.Int32(0):
                            if q_row < fx.Int32(Sq_phys):
                                fx.memref_store(lse_val, LSE_bh, q_row)

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
                lane_select0 = lane_row * fx.Int32(TCU_LANE_COLS) + (lane_col * fx.Int32(2)) % fx.Int32(TCU_LANE_COLS)
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
        LSE: fx.Tensor,
        CuSeqLen: fx.Tensor,
        SeqLens: fx.Tensor,
        AlibiSlopes: fx.Tensor,
        ScoreBias: fx.Tensor,
        KvNumBlocks: fx.Tensor,
        KvIndices: fx.Tensor,
        KvIsFull: fx.Tensor,
        total_tokens: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        flex_attn_kernel(
            Q,
            K,
            V,
            O,
            LSE,
            CuSeqLen,
            SeqLens,
            AlibiSlopes,
            ScoreBias,
            KvNumBlocks,
            KvIndices,
            KvIsFull,
            total_tokens,
        ).launch(
            grid=(B * H, num_q_tiles, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    def launch_flex_attn_checked(
        Q,
        K,
        V,
        O,  # noqa: E741
        cu_seqlens=None,
        seq_lens=None,
        block_table=None,
        seq_lens_kv=None,
        alibi_slopes=None,
        score_bias=None,
        block_mask: FlexBlockMask | None = None,
        lse=None,
        stream=fx.Stream(None),
    ):
        """Host entry: enforce shapes before the JIT launch.

        Dense: phys-padded BHSD; optional dense-only ``alibi_slopes`` ``[H]`` fp32
        or phys-padded ``score_bias`` ``[B,H,Sq_phys,Skv_phys]`` (mutually exclusive).
        Optional ``block_mask``: dense ``FlexBlockMask``, or varlen
        ``PackedVarlenBlockMask`` / ``Sequence[FlexBlockMask]`` when compiled
        with ``has_block_mask=True``. When compiled with ``return_lse=True``,
        pass or allocate fp32 ``lse [B,H,Sq_phys]``; the launch returns that
        tensor. Paged rejects block_mask; alibi/score_bias / return_lse stay
        dense-only.
        """
        q_shape = _tensor_shape(Q)
        k_shape = _tensor_shape(K)
        v_shape = _tensor_shape(V)
        o_shape = _tensor_shape(O)

        def _bias_placeholders():
            ali = alibi_slopes if alibi_slopes is not None else O
            sb = score_bias if score_bias is not None else O
            return ali, sb

        def _lse_placeholder():
            # Unused when return_lse=False (kernel builds a 1-element view).
            return O if lse is None else lse

        def _block_mask_placeholders():
            if has_block_mask:
                if varlen:
                    packed = block_mask
                    if isinstance(packed, (list, tuple)):
                        packed = pack_block_masks_varlen(packed, max_seqlen_q=Sq, max_seqlen_kv=Skv)
                    if not isinstance(packed, PackedVarlenBlockMask):
                        raise TypeError(
                            "varlen has_block_mask requires PackedVarlenBlockMask "
                            f"or Sequence[FlexBlockMask], got {type(packed).__name__}"
                        )
                    if int(packed.block_m) != int(BLOCK_M) or int(packed.block_n) != int(BLOCK_N):
                        raise ValueError(
                            f"block_mask tile ({packed.block_m}x{packed.block_n}) must match "
                            f"compile tile ({BLOCK_M}x{BLOCK_N})"
                        )
                    if int(packed.max_q_tiles) != int(num_q_tiles) or int(packed.max_kv_tiles) != int(num_kv_tiles):
                        raise ValueError(
                            f"packed block_mask tiles ({packed.max_q_tiles},{packed.max_kv_tiles}) "
                            f"incompatible with compile ({num_q_tiles},{num_kv_tiles})"
                        )
                    if int(packed.num_seqs) != int(B):
                        raise ValueError(f"packed block_mask num_seqs={packed.num_seqs} must match B={B}")
                    return packed.kv_num_blocks, packed.kv_indices, packed.kv_is_full
                if block_mask is None:
                    raise ValueError("compiled with has_block_mask=True requires block_mask=FlexBlockMask")
                if not isinstance(block_mask, FlexBlockMask):
                    raise TypeError(f"block_mask must be FlexBlockMask, got {type(block_mask).__name__}")
                if int(block_mask.block_m) != int(BLOCK_M) or int(block_mask.block_n) != int(BLOCK_N):
                    raise ValueError(
                        f"block_mask tile ({block_mask.block_m}x{block_mask.block_n}) must match "
                        f"compile tile ({BLOCK_M}x{BLOCK_N})"
                    )
                if int(block_mask.num_q_tiles) != int(num_q_tiles) or int(block_mask.num_kv_tiles) != int(num_kv_tiles):
                    raise ValueError(
                        f"block_mask tiles ({block_mask.num_q_tiles},{block_mask.num_kv_tiles}) "
                        f"incompatible with compile ({num_q_tiles},{num_kv_tiles})"
                    )
                return block_mask.kv_num_blocks, block_mask.kv_indices, block_mask.kv_is_full
            if block_mask is not None:
                raise ValueError("block_mask passed but kernel compiled without has_block_mask=True")
            return O, O, O

        if varlen:
            if do_return_lse:
                raise ValueError("return_lse is dense-only (not supported with varlen)")
            if cu_seqlens is None or seq_lens is None:
                raise ValueError("varlen launch requires cu_seqlens [B+1] and seq_lens [B] (int32)")
            if alibi_slopes is not None or score_bias is not None:
                raise ValueError("alibi/score_bias are dense-only (not supported with varlen)")
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
                raise ValueError(f"varlen V shape must be [Hkv, D, total]={expect_v} (host-transposed), got {v_shape}")
            if o_shape != expect_o:
                raise ValueError(f"varlen O shape must be [total, H, D]={expect_o}, got {o_shape}")
            if total < 1:
                raise ValueError(f"varlen total_tokens must be positive, got {total}")
            if total % 32 != 0:
                raise ValueError(
                    f"varlen total_tokens (V leading dim) must be a multiple of 32 for SME G2S, got {total}"
                )
            ali, sb = _bias_placeholders()
            kn, ki, kf = _block_mask_placeholders()
            launch_flex_attn(
                Q,
                K,
                V,
                O,
                _lse_placeholder(),
                cu_seqlens,
                seq_lens,
                ali,
                sb,
                kn,
                ki,
                kf,
                fx.Int32(total),
                stream=stream,
            )
            return None

        if paged:
            if do_return_lse:
                raise ValueError("return_lse is dense-only (not supported with paged)")
            if block_table is None or seq_lens_kv is None:
                raise ValueError("paged launch requires block_table [B, max_num_pages] and seq_lens_kv [B] (int32)")
            if alibi_slopes is not None or score_bias is not None:
                raise ValueError("alibi/score_bias are dense-only (not supported with paged)")
            if has_block_mask or block_mask is not None:
                raise ValueError("block_mask is not supported with paged")
            bt_shape = _tensor_shape(block_table)
            sl_shape = _tensor_shape(seq_lens_kv)
            if len(bt_shape) != 2 or bt_shape[0] != B or bt_shape[1] != max_num_pages:
                raise ValueError(f"block_table shape must be ({B}, {max_num_pages}), got {bt_shape}")
            if len(sl_shape) != 1 or sl_shape[0] != B:
                raise ValueError(f"seq_lens_kv shape must be ({B},), got {sl_shape}")
            expect_q = (B, H, Sq_phys, D)
            expect_o = (B, H, Sq_phys, D)
            if q_shape != expect_q:
                raise ValueError(f"paged Q shape must be {expect_q} (Sq_phys-padded), got {q_shape}")
            if o_shape != expect_o:
                raise ValueError(f"paged O shape must be {expect_o} (Sq_phys-padded), got {o_shape}")
            if len(k_shape) != 4 or k_shape[1:] != (PAGE_SIZE, Hkv, D):
                raise ValueError(f"paged K shape must be [NumBlocks, {PAGE_SIZE}, {Hkv}, {D}], got {k_shape}")
            if len(v_shape) != 4 or v_shape[1:] != (Hkv, D, PAGE_SIZE):
                raise ValueError(
                    f"paged V shape must be [NumBlocks, {Hkv}, {D}, {PAGE_SIZE}] "
                    f"(per-page host-transposed), got {v_shape}"
                )
            if k_shape[0] != v_shape[0]:
                raise ValueError(f"paged K/V NumBlocks mismatch: K={k_shape[0]}, V={v_shape[0]}")
            ali, sb = _bias_placeholders()
            kn, ki, kf = _block_mask_placeholders()
            launch_flex_attn(
                Q,
                K,
                V,
                O,
                _lse_placeholder(),
                block_table,
                seq_lens_kv,
                ali,
                sb,
                kn,
                ki,
                kf,
                fx.Int32(0),
                stream=stream,
            )
            return None

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

        if alibi_slopes is not None and score_bias is not None:
            raise ValueError("alibi_slopes and score_bias are mutually exclusive")
        if has_alibi:
            if alibi_slopes is None:
                raise ValueError("compiled with has_alibi=True requires alibi_slopes [H]")
            as_shape = _tensor_shape(alibi_slopes)
            if len(as_shape) != 1 or as_shape[0] != H:
                raise ValueError(f"alibi_slopes shape must be ({H},), got {as_shape}")
        elif alibi_slopes is not None:
            raise ValueError("alibi_slopes passed but kernel compiled without has_alibi=True")
        if has_score_bias:
            if score_bias is None:
                raise ValueError(
                    "compiled with has_score_bias=True requires score_bias "
                    f"[B, H, Sq_phys, Skv_phys]=[{B}, {H}, {Sq_phys}, {Skv_phys}]"
                )
            sb_shape = _tensor_shape(score_bias)
            expect_sb = (B, H, Sq_phys, Skv_phys)
            if sb_shape != expect_sb:
                raise ValueError(f"score_bias shape must be {expect_sb} (phys-padded), got {sb_shape}")
        elif score_bias is not None:
            raise ValueError("score_bias passed but kernel compiled without has_score_bias=True")

        ali, sb = _bias_placeholders()
        if do_return_lse:
            if lse is None:
                import torch

                lse_tensor = torch.empty(B, H, Sq_phys, device=Q.device, dtype=torch.float32)
            else:
                lse_shape = _tensor_shape(lse)
                expect_lse = (B, H, Sq_phys)
                if lse_shape != expect_lse:
                    raise ValueError(f"lse shape must be {expect_lse}, got {lse_shape}")
                lse_tensor = lse
        else:
            lse_tensor = _lse_placeholder()
        kn, ki, kf = _block_mask_placeholders()
        launch_flex_attn(Q, K, V, O, lse_tensor, O, O, ali, sb, kn, ki, kf, fx.Int32(0), stream=stream)
        return lse_tensor if do_return_lse else None

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
    paged: bool = False,
    has_alibi: bool = False,
    has_score_bias: bool = False,
    score_mod: Optional[TracedScoreMod] = None,
    mask_mod: Optional[TracedMaskMod] = None,
    has_block_mask: bool = False,
    tile_config: Mapping[str, int] | None = None,
    return_lse: bool = False,
) -> Callable:
    """Compile a fused flex-attention forward kernel for the Iluvatar backend.

    Dense and varlen paths support ``score_mod=TracedScoreMod``. Dense also
    supports optional BlockMask sparse KV iteration (``has_block_mask=True`` at
    compile + ``FlexBlockMask`` as ``block_mask=`` at launch, with optional
    ``mask_mod`` for element holes).

    Supported scope: Q/K/V/O = f16 or bf16, D in {64, 128, 256}, GQA
    (``H % Hkv == 0``), arbitrary ``Sq``/``Skv`` (callers pass phys-padded
    contiguous tensors; see below), ``is_causal`` / ``window_size`` / ``softcap``
    in any combination. Dense ``is_causal`` requires ``Sq == Skv``; paged
    causal allows ``Sq != Skv`` (cross-length via ``seq_lens_kv - Sq``).

    Dense phys pad contract:
        ``Sq_phys = ceil(Sq / BLOCK_M) * BLOCK_M``,
        ``Skv_phys = ceil(Skv / BLOCK_N) * BLOCK_N`` (default tile 64x64).
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

    Paged contract (``paged=True``; mutually exclusive with ``varlen``):
        ``Sq``/``Skv`` are ``max_seqlen_q`` / ``max_seqlen_kv``. Q/O stay dense
        phys-padded. KV is linear paged with ``page_size == BLOCK_N``
        (must stay 64):
        ``K: [NumBlocks, 64, Hkv, D]``,
        ``V: [NumBlocks, Hkv, D, 64]`` (per-page host transpose for MMA2,
        mirroring dense V), plus ``block_table`` int32 ``[B, max_num_pages]``
        and ``seq_lens_kv`` int32 ``[B]``. Kernel gathers pages per KV tile
        (not gather-to-dense). Causal uses ``delta = seq_lens_kv[b] - Sq``.

    Args:
        B, H, Sq, Skv, D: Logical shapes (compile-time constants). Dense:
            ``Sq``/``Skv`` are logical lengths used for masking. Varlen:
            ``B=num_seqs``, ``Sq=Skv=max_seqlen``. Paged: ``Sq``/``Skv`` are
            max sequence lengths for Q and KV.
        Hkv: KV head count; ``None`` means MHA (``Hkv = H``).
        dtype: ``"f16"`` or ``"bf16"``.
        is_causal: Enable causal mask; dense requires ``Sq == Skv``.
        window_size: Sliding-window radius; mask where ``q - kv > window_size``.
            Independent of ``is_causal``.
        softcap: Gemma-2 softcap; ``S = softcap * tanh(S / softcap)`` after
            ``sm_scale``, before log2-domain entry.
        sm_scale: Query scale; ``None`` defaults to ``1 / sqrt(D)``.
        varlen: If True, compile the packed ``cu_seqlens`` self-attn path.
        paged: If True, compile the ``block_table`` paged-KV path.
        has_alibi: Dense-only; enable ``alibi_slopes [H]`` additive bias.
        has_score_bias: Dense-only; enable phys-padded ``score_bias`` tensor.
            Mutually exclusive with ``has_alibi``; not supported with varlen/paged.
        score_mod: ``TracedScoreMod`` (or ``None``) for dense, varlen, and paged.
            Inlined after softcap and before ``*log2e``. Closure scalars only;
            not a replacement for ``alibi_slopes=[H]``. Indices are sequence-local
            (``batch`` is seq id on varlen, batch item on dense/paged).
        mask_mod: ``TracedMaskMod`` for element holes (with ``has_block_mask``
            on dense/varlen). Same mod used in ``create_block_mask``. Not
            supported with paged (V3-7b).
        has_block_mask: Dense/varlen. If True, launch must pass ``block_mask=``
            (dense ``FlexBlockMask`` or varlen packed tables). Not supported
            with paged (V3-7b).
        tile_config: Optional ``{"block_m": int, "block_n": int}`` with values
            in ``{32, 64}``. Default is ``64x64``. Paged requires ``block_n=64``.
        return_lse: Dense-only. If True, launch writes fp32 ``LSE [B,H,Sq_phys]``
            in natural log (``m_log2 * ln2 + ln(l)``) and returns that tensor.

    Returns:
        Dense: ``launch_fn(Q, K, V, O, stream=None, alibi_slopes=..., score_bias=...,
        block_mask=..., lse=...)``. When ``return_lse=True``, returns the LSE tensor
        (allocates if ``lse`` omitted); otherwise returns ``None``.
        Varlen: ``launch_fn(Q, K, V, O, cu_seqlens=..., seq_lens=..., stream=None)``.
        Paged: ``launch_fn(Q, K, V, O, block_table=..., seq_lens_kv=..., stream=None)``.
    """
    if Hkv is None:
        Hkv = H

    if varlen and paged:
        raise ValueError("paged and varlen are mutually exclusive")
    if has_alibi and has_score_bias:
        raise ValueError("alibi_slopes and score_bias are mutually exclusive")
    if (has_alibi or has_score_bias) and (varlen or paged):
        raise ValueError("alibi/score_bias are dense-only in V2-4 (not supported with varlen/paged)")
    if score_mod is not None and not isinstance(score_mod, TracedScoreMod):
        raise ValueError(f"score_mod must be TracedScoreMod or None, got {type(score_mod).__name__}")
    if mask_mod is not None and not isinstance(mask_mod, TracedMaskMod):
        raise ValueError(f"mask_mod must be TracedMaskMod or None, got {type(mask_mod).__name__}")
    if (mask_mod is not None or has_block_mask) and paged:
        raise ValueError("block_mask/mask_mod are not supported with paged")
    if return_lse and (varlen or paged):
        raise ValueError("return_lse is dense-only (not supported with varlen/paged)")

    block_m, block_n = _normalize_tile_config(tile_config, paged=bool(paged))

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
        allow_causal_cross_len=bool(paged),
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
        paged=paged,
        has_alibi=has_alibi,
        has_score_bias=has_score_bias,
        score_mod=score_mod,
        mask_mod=mask_mod,
        has_block_mask=bool(has_block_mask),
        block_m=block_m,
        block_n=block_n,
        return_lse=return_lse,
    )


__all__ = [
    "compile_iluvatar_flex_attention",
    "create_block_mask",
    "create_block_masks_varlen",
    "pack_block_masks_varlen",
    "FlexBlockMask",
    "PackedVarlenBlockMask",
    "BLOCK_M",
    "BLOCK_N",
    "_SUPPORTED_BLOCK",
    "_normalize_tile_config",
]
