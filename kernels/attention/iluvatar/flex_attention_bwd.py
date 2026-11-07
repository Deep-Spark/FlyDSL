# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar flex-attention backward (V2-8, dense MHA, correctness-first).

Flash-style fused bwd: preprocess ``delta = rowsum(dO * O)``, then Q-tile
parallel main kernel recomputes scores from ``LSE`` (natural log), writes
``dQ`` directly and atomically accumulates ``dK`` / ``dV``.

Scope: dense MHA, D in {64, 128}, f16/bf16, causal / SWA / softcap, fixed
64x64 tiles. Not an MMA-optimized path -- SMEM + thread loops for numerical
bring-up against ``torch.autograd.grad``.
"""

import math
from collections.abc import Callable

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith
from flydsl.expr import math as fmath
from kernels.attention.iluvatar.flex_attention import (
    BLOCK_M,
    BLOCK_N,
    BLOCK_THREADS,
    WARP_SIZE,
    _phys_seq,
    _tensor_shape,
)

_SUPPORTED_DTYPES = ("f16", "bf16")
_SUPPORTED_D = (64, 128)
_DTYPE_STR_TO_FX = {"f16": fx.Float16, "bf16": fx.BFloat16}


def _validate_bwd_scope(
    *,
    B: int,
    H: int,
    Sq: int,
    Skv: int,
    D: int,
    dtype: str,
    is_causal: bool,
    window_size: int | None,
    softcap: float | None,
) -> None:
    if B < 1 or H < 1 or Sq < 1 or Skv < 1:
        raise ValueError(f"B/H/Sq/Skv must be >= 1, got B={B}, H={H}, Sq={Sq}, Skv={Skv}")
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"dtype must be one of {_SUPPORTED_DTYPES}, got {dtype!r}")
    if D not in _SUPPORTED_D:
        raise ValueError(f"D must be one of {_SUPPORTED_D}, got {D}")
    if is_causal and Sq != Skv:
        raise ValueError(f"dense causal bwd requires Sq == Skv, got Sq={Sq}, Skv={Skv}")
    if window_size is not None and window_size < 0:
        raise ValueError(f"window_size must be >= 0 when set, got {window_size}")
    if softcap is not None and softcap <= 0:
        raise ValueError(f"softcap must be > 0 when set, got {softcap}")


def _build_delta_kernel(*, B: int, H: int, Sq_phys: int, D: int, dtype: str):
    elem_dtype = _DTYPE_STR_TO_FX[dtype]
    num_q_tiles = Sq_phys // BLOCK_M

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def flex_attn_bwd_delta(O: fx.Tensor, dO: fx.Tensor, Delta: fx.Tensor):  # noqa: E741
        bh_idx = fx.block_idx.x
        q_tile_idx = fx.block_idx.y
        tid = fx.thread_idx.x
        lane_id = fx.Int32(fx.lane_id)
        warp_id = tid // fx.Int32(WARP_SIZE)
        b_idx = bh_idx // fx.Int32(H)
        h_idx = bh_idx % fx.Int32(H)
        q_start = q_tile_idx * fx.Int32(BLOCK_M)
        fm_fast = arith.FastMathFlags.fast
        c_zero = fx.Float32(0.0)

        @fx.struct
        class _DeltaSmem:
            partials: fx.Array[fx.Float32, BLOCK_THREADS]

        smem = fx.SharedAllocator(static=True).allocate(_DeltaSmem).peek()
        partials = smem.partials.view(fx.make_layout(BLOCK_THREADS, 1))

        O_view = fx.make_view(
            fx.get_iter(O),
            fx.make_layout((B, H, Sq_phys, D), (H * Sq_phys * D, Sq_phys * D, D, 1)),
        )
        dO_view = fx.make_view(
            fx.get_iter(dO),
            fx.make_layout((B, H, Sq_phys, D), (H * Sq_phys * D, Sq_phys * D, D, 1)),
        )
        Delta_view = fx.make_view(
            fx.get_iter(Delta),
            fx.make_layout((B, H, Sq_phys), (H * Sq_phys, Sq_phys, 1)),
        )
        O_bh = fx.slice(O_view, (b_idx, h_idx, None, None))
        dO_bh = fx.slice(dO_view, (b_idx, h_idx, None, None))
        Delta_bh = fx.slice(Delta_view, (b_idx, h_idx, None))

        # One row per iteration; all threads reduce along D.
        for row_off in fx.range_constexpr(BLOCK_M):
            q_row = q_start + fx.Int32(row_off)
            thread_sum = c_zero
            for d_base in fx.range_constexpr(0, D, BLOCK_THREADS):
                d = tid + fx.Int32(d_base)
                if d < fx.Int32(D):
                    o_v = fx.Float32(fx.memref_load(O_bh, (q_row, d)))
                    do_v = fx.Float32(fx.memref_load(dO_bh, (q_row, d)))
                    thread_sum = thread_sum + o_v * do_v
            # Warp then block reduce.
            reduced = thread_sum
            for sh_exp in fx.range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << sh_exp)
                reduced = reduced + reduced.shuffle_xor(off, WARP_SIZE)
            if lane_id == fx.Int32(0):
                fx.memref_store(reduced, partials, warp_id)
            fx.gpu.barrier()
            block_sum = c_zero
            if warp_id == fx.Int32(0):
                n_warps = fx.Int32(BLOCK_THREADS // WARP_SIZE)
                part = (lane_id < n_warps).select(fx.memref_load(partials, lane_id), c_zero)
                for sh_exp in fx.range_constexpr(int(math.log2(WARP_SIZE))):
                    off = WARP_SIZE // (2 << sh_exp)
                    part = part + part.shuffle_xor(off, WARP_SIZE)
                block_sum = part
                if lane_id == fx.Int32(0):
                    if q_row < fx.Int32(Sq_phys):
                        fx.memref_store(block_sum, Delta_bh, q_row)
            fx.gpu.barrier()

    @flyc.jit
    def launch_delta(O: fx.Tensor, dO: fx.Tensor, Delta: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        flex_attn_bwd_delta(O, dO, Delta).launch(
            grid=(B * H, num_q_tiles, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_delta


def _build_bwd_main_kernel(  # noqa: C901
    *,
    B: int,
    H: int,
    Sq: int,
    Skv: int,
    Sq_phys: int,
    Skv_phys: int,
    D: int,
    dtype: str,
    sm_scale_f32: float,
    is_causal: bool,
    window_size: int | None,
    softcap: float | None,
):
    elem_dtype = _DTYPE_STR_TO_FX[dtype]
    num_q_tiles = Sq_phys // BLOCK_M
    num_kv_tiles = Skv_phys // BLOCK_N
    has_softcap = softcap is not None
    has_swa = window_size is not None
    softcap_f32 = float(softcap) if has_softcap else 0.0
    window_size_i = int(window_size) if has_swa else 0

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def bwd_main_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        dO: fx.Tensor,
        LSE: fx.Tensor,
        Delta: fx.Tensor,
        dQ: fx.Tensor,
        dK: fx.Tensor,
        dV: fx.Tensor,
    ):
        bh_idx = fx.block_idx.x
        q_tile_idx = fx.block_idx.y
        tid = fx.thread_idx.x
        b_idx = bh_idx // fx.Int32(H)
        h_idx = bh_idx % fx.Int32(H)
        q_start = q_tile_idx * fx.Int32(BLOCK_M)
        fm_fast = arith.FastMathFlags.fast
        c_zero = fx.Float32(0.0)
        c_one = fx.Float32(1.0)
        c_sm = fx.Float32(float(sm_scale_f32))
        c_softcap = fx.Float32(softcap_f32)
        c_window = fx.Int32(window_size_i)
        c_skv = fx.Int32(Skv)
        c_sq = fx.Int32(Sq)
        c_log2e = fx.Float32(1.4426950408889634)

        atomic_add = fx.make_copy_atom(fx.UniversalAtomicAdd(elem_dtype), elem_dtype)
        scratch = fx.make_rmem_tensor(1, elem_dtype)

        Q_view = fx.make_view(
            fx.get_iter(Q),
            fx.make_layout((B, H, Sq_phys, D), (H * Sq_phys * D, Sq_phys * D, D, 1)),
        )
        K_view = fx.make_view(
            fx.get_iter(K),
            fx.make_layout((B, H, Skv_phys, D), (H * Skv_phys * D, Skv_phys * D, D, 1)),
        )
        # V / dV host layout: [B, H, D, Skv_phys]
        V_view = fx.make_view(
            fx.get_iter(V),
            fx.make_layout((B, H, D, Skv_phys), (H * D * Skv_phys, D * Skv_phys, Skv_phys, 1)),
        )
        dO_view = fx.make_view(
            fx.get_iter(dO),
            fx.make_layout((B, H, Sq_phys, D), (H * Sq_phys * D, Sq_phys * D, D, 1)),
        )
        LSE_view = fx.make_view(
            fx.get_iter(LSE),
            fx.make_layout((B, H, Sq_phys), (H * Sq_phys, Sq_phys, 1)),
        )
        Delta_view = fx.make_view(
            fx.get_iter(Delta),
            fx.make_layout((B, H, Sq_phys), (H * Sq_phys, Sq_phys, 1)),
        )
        dQ_view = fx.make_view(
            fx.get_iter(dQ),
            fx.make_layout((B, H, Sq_phys, D), (H * Sq_phys * D, Sq_phys * D, D, 1)),
        )
        # Flat views for scalar atomics into dK / dV.
        dK_flat = fx.make_view(
            fx.get_iter(dK),
            fx.make_layout((B * H * Skv_phys * D,), (1,)),
        )
        dV_flat = fx.make_view(
            fx.get_iter(dV),
            fx.make_layout((B * H * D * Skv_phys,), (1,)),
        )
        dK_div = fx.logical_divide(dK_flat, fx.make_layout(1, 1))
        dV_div = fx.logical_divide(dV_flat, fx.make_layout(1, 1))

        Q_bh = fx.slice(Q_view, (b_idx, h_idx, None, None))
        K_bh = fx.slice(K_view, (b_idx, h_idx, None, None))
        V_bh = fx.slice(V_view, (b_idx, h_idx, None, None))
        dO_bh = fx.slice(dO_view, (b_idx, h_idx, None, None))
        LSE_bh = fx.slice(LSE_view, (b_idx, h_idx, None))
        Delta_bh = fx.slice(Delta_view, (b_idx, h_idx, None))
        dQ_bh = fx.slice(dQ_view, (b_idx, h_idx, None, None))

        bh_base_k = (b_idx * fx.Int32(H) + h_idx) * fx.Int32(Skv_phys * D)
        bh_base_v = (b_idx * fx.Int32(H) + h_idx) * fx.Int32(D * Skv_phys)

        # Correctness-first: thread ``tid < BLOCK_M`` owns one query row.
        if tid < fx.Int32(BLOCK_M):
            q_row = q_start + tid
            if q_row < c_sq:
                lse_i = fx.memref_load(LSE_bh, q_row)
                delta_i = fx.memref_load(Delta_bh, q_row)

                for d in fx.range_constexpr(D):
                    fx.memref_store(c_zero.to(elem_dtype), dQ_bh, (q_row, fx.Int32(d)))

                kv_end = fx.Int32(num_kv_tiles)
                if fx.const_expr(is_causal):
                    kv_end_cand = (q_row + fx.Int32(BLOCK_N)) // fx.Int32(BLOCK_N)
                    kv_end = (kv_end_cand < fx.Int32(num_kv_tiles)).select(
                        kv_end_cand, fx.Int32(num_kv_tiles)
                    )

                for kv_tile in fx.range(fx.Int32(0), kv_end, fx.Int32(1)):
                    kv_base = kv_tile * fx.Int32(BLOCK_N)
                    for j_off in fx.range_constexpr(BLOCK_N):
                        kv_idx = kv_base + fx.Int32(j_off)
                        if kv_idx < c_skv:
                            dot = c_zero
                            for d in fx.range_constexpr(D):
                                qv = fx.Float32(fx.memref_load(Q_bh, (q_row, fx.Int32(d))))
                                kv = fx.Float32(fx.memref_load(K_bh, (kv_idx, fx.Int32(d))))
                                dot = dot + qv * kv
                            x = dot * c_sm
                            if fx.const_expr(has_softcap):
                                s_nat = c_softcap * fmath.tanh(x / c_softcap, fastmath=fm_fast)
                            else:
                                s_nat = x

                            masked = fx.Int32(0)
                            if fx.const_expr(is_causal):
                                masked = (kv_idx > q_row).select(fx.Int32(1), masked)
                            if fx.const_expr(has_swa):
                                too_far = (q_row - kv_idx) > c_window
                                masked = too_far.select(fx.Int32(1), masked)

                            if masked == fx.Int32(0):
                                # P = exp(S_nat - LSE) = exp2((S_nat - LSE) * log2e)
                                p = ((s_nat - lse_i) * c_log2e).exp2(fastmath=fm_fast)
                                dp = c_zero
                                for d in fx.range_constexpr(D):
                                    dov = fx.Float32(fx.memref_load(dO_bh, (q_row, fx.Int32(d))))
                                    vv = fx.Float32(fx.memref_load(V_bh, (fx.Int32(d), kv_idx)))
                                    dp = dp + dov * vv
                                ds = p * (dp - delta_i)
                                if fx.const_expr(has_softcap):
                                    t = s_nat / c_softcap
                                    dx = ds * (c_one - t * t)
                                else:
                                    dx = ds
                                dx_scaled = dx * c_sm

                                for d in fx.range_constexpr(D):
                                    qv = fx.Float32(fx.memref_load(Q_bh, (q_row, fx.Int32(d))))
                                    kv = fx.Float32(fx.memref_load(K_bh, (kv_idx, fx.Int32(d))))
                                    dov = fx.Float32(fx.memref_load(dO_bh, (q_row, fx.Int32(d))))
                                    old_dq = fx.Float32(fx.memref_load(dQ_bh, (q_row, fx.Int32(d))))
                                    fx.memref_store(
                                        (old_dq + dx_scaled * kv).to(elem_dtype),
                                        dQ_bh,
                                        (q_row, fx.Int32(d)),
                                    )
                                    dk_off = bh_base_k + kv_idx * fx.Int32(D) + fx.Int32(d)
                                    fx.memref_store((dx_scaled * qv).to(elem_dtype), scratch, 0)
                                    fx.copy_atom_call(
                                        atomic_add,
                                        scratch,
                                        fx.slice(dK_div, (None, dk_off)),
                                    )
                                    # V layout (d, kv): flat = d * Skv_phys + kv
                                    dv_off = bh_base_v + fx.Int32(d) * fx.Int32(Skv_phys) + kv_idx
                                    fx.memref_store((p * dov).to(elem_dtype), scratch, 0)
                                    fx.copy_atom_call(
                                        atomic_add,
                                        scratch,
                                        fx.slice(dV_div, (None, dv_off)),
                                    )

    @flyc.jit
    def launch_main(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        dO: fx.Tensor,
        LSE: fx.Tensor,
        Delta: fx.Tensor,
        dQ: fx.Tensor,
        dK: fx.Tensor,
        dV: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        bwd_main_kernel(Q, K, V, dO, LSE, Delta, dQ, dK, dV).launch(
            grid=(B * H, num_q_tiles, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_main


def compile_iluvatar_flex_attention_bwd(
    B: int,
    H: int,
    Sq: int,
    Skv: int,
    D: int,
    *,
    dtype: str = "bf16",
    is_causal: bool = False,
    window_size: int | None = None,
    softcap: float | None = None,
    sm_scale: float | None = None,
) -> Callable:
    """Compile dense MHA flex-attention backward for Iluvatar.

    Returns ``launch(Q, K, V, O, dO, LSE, dQ, dK, dV, stream=None)``.

    Layouts match dense fwd phys-pad: ``Q/O/dO/dQ [B,H,Sq_phys,D]``,
    ``K/dK [B,H,Skv_phys,D]``, ``V/dV [B,H,D,Skv_phys]``, ``LSE`` fp32
    ``[B,H,Sq_phys]`` (natural log). ``dK``/``dV`` must be zero-initialized
    (same dtype as Q); they are accumulated with ``UniversalAtomicAdd``.
    """
    _validate_bwd_scope(
        B=B,
        H=H,
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

    Sq_phys = _phys_seq(Sq, BLOCK_M)
    Skv_phys = _phys_seq(Skv, BLOCK_N)

    launch_delta = _build_delta_kernel(B=B, H=H, Sq_phys=Sq_phys, D=D, dtype=dtype)
    launch_main = _build_bwd_main_kernel(
        B=B,
        H=H,
        Sq=Sq,
        Skv=Skv,
        Sq_phys=Sq_phys,
        Skv_phys=Skv_phys,
        D=D,
        dtype=dtype,
        sm_scale_f32=sm_scale,
        is_causal=is_causal,
        window_size=window_size,
        softcap=softcap,
    )

    def launch_bwd(Q, K, V, O, dO, LSE, dQ, dK, dV, stream=None):  # noqa: E741
        import torch

        expect_q = (B, H, Sq_phys, D)
        expect_k = (B, H, Skv_phys, D)
        expect_v = (B, H, D, Skv_phys)
        expect_lse = (B, H, Sq_phys)
        for name, t, exp in (
            ("Q", Q, expect_q),
            ("K", K, expect_k),
            ("V", V, expect_v),
            ("O", O, expect_q),
            ("dO", dO, expect_q),
            ("dQ", dQ, expect_q),
            ("dK", dK, expect_k),
            ("dV", dV, expect_v),
        ):
            shape = _tensor_shape(t)
            if shape != exp:
                raise ValueError(f"{name} shape must be {exp}, got {shape}")
        if _tensor_shape(LSE) != expect_lse:
            raise ValueError(f"LSE shape must be {expect_lse}, got {_tensor_shape(LSE)}")

        if stream is None:
            stream = fx.Stream(None)

        delta = torch.empty(B, H, Sq_phys, device=Q.device, dtype=torch.float32)
        launch_delta(O, dO, delta, stream=stream)
        launch_main(Q, K, V, dO, LSE, delta, dQ, dK, dV, stream=stream)

    return launch_bwd


__all__ = ["compile_iluvatar_flex_attention_bwd"]
