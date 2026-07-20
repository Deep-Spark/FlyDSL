# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar fused TopK gating Softmax (FlyIXDL).

Ports the TensorRT-LLM / vLLM ``topkGatingSoftmax`` algorithm to the Iluvatar
backend using the same sub-warp butterfly Softmax + iterative argmax-then-mask
TopK as ``kernels/moe/topk_gating_softmax_kernel.py``, but with:

* ``WARP_SIZE = 64`` (ivcore11 hardware warp)
* ``fx.make_view`` + scalar indexing (same memory style as ``gemv.py``)
* no ROCDL BufferCopy / ``buffer_ops`` dependency
* no ``delayed_softmax`` path (upstream full-select delayed path is known-bad
  on IVCORE11; see iluvatar-adapt-base ``topk-moe-full-select``)

Semantics match ``ixformer.inference.functions.moe_topk_softmax``:

  probs = softmax(gating, dim=-1)
  weights, ids = topk(probs, k)
  if renormalize: weights /= weights.sum(-1)

Outputs: ``topk_weights`` (f32), ``topk_ids`` (i32). Optional
``token_expert_indices`` with ``tei[t, k] = k * num_tokens + t``.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import Int32, T
from kernels.gemm.iluvatar.common import WARP_SIZE

KERNEL_NAME = "iluvatar_topk_gating_softmax"

WARPS_PER_BLOCK = 16
BLOCK_THREADS = WARPS_PER_BLOCK * WARP_SIZE  # 1024

_SUPPORTED_DTYPES = ("f32", "f16", "bf16")


def _dtype_to_elem(dtype_str: str):
    if dtype_str == "f32":
        return fx.Float32
    if dtype_str == "f16":
        return fx.Float16
    if dtype_str == "bf16":
        return fx.BFloat16
    raise ValueError(f"dtype_str must be one of {_SUPPORTED_DTYPES}, got {dtype_str!r}")


def _pick_layout(num_experts: int):
    """Pick (VPT, THREADS_PER_TOKEN) for the multi-token-per-block path.

    Constraints mirror the ROCm gating kernel:
      - VPT power-of-2 in [1, 16]
      - THREADS_PER_TOKEN = num_experts // VPT is a power of 2 <= WARP_SIZE
      - prefer largest VPT
    """
    for vpt in (16, 8, 4, 2, 1):
        if num_experts % vpt != 0:
            continue
        tpt = num_experts // vpt
        if tpt > WARP_SIZE or tpt < 1:
            continue
        if (tpt & (tpt - 1)) != 0:
            continue
        return vpt, tpt
    return None, None


def _compute_layout(num_experts: int, topk: int):
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if topk > num_experts:
        raise ValueError(f"topk={topk} > num_experts={num_experts}")

    vpt, tpt = _pick_layout(num_experts)
    if vpt is None:
        raise ValueError(
            f"num_experts={num_experts} is not supported: need num_experts // VPT to be "
            f"a power of 2 <= WARP_SIZE={WARP_SIZE} for some VPT in {{16,8,4,2,1}}."
        )

    tokens_per_warp = WARP_SIZE // tpt
    tokens_per_block = WARPS_PER_BLOCK * tokens_per_warp
    return dict(
        VPT=vpt,
        THREADS_PER_TOKEN=tpt,
        TOKENS_PER_WARP=tokens_per_warp,
        TOKENS_PER_BLOCK=tokens_per_block,
    )


def build_iluvatar_topk_gating_softmax(
    num_experts: int,
    topk: int,
    dtype_str: str = "f32",
    renormalize: bool = True,
    emit_tei: bool = False,
):
    """Build a fused TopK gating Softmax launcher for Iluvatar.

    Args:
        num_experts: Columns in gating logits (must fit the PoT layout).
        topk: Experts selected per token.
        dtype_str: Input gating dtype (``f32`` / ``f16`` / ``bf16``).
        renormalize: Rescale selected weights to sum to 1.
        emit_tei: Also write ``token_expert_indices[t, k] = k * num_tokens + t``.

    Returns:
        ``launch(gating, weights, ids, num_tokens, tei=None, *, stream=None)``.
        When ``emit_tei=False``, ``tei`` is ignored.
    """
    if dtype_str not in _SUPPORTED_DTYPES:
        raise ValueError(f"dtype_str must be one of {_SUPPORTED_DTYPES}, got {dtype_str!r}")

    layout = _compute_layout(num_experts, topk)
    VPT = layout["VPT"]
    THREADS_PER_TOKEN = layout["THREADS_PER_TOKEN"]
    TOKENS_PER_WARP = layout["TOKENS_PER_WARP"]
    TOKENS_PER_BLOCK = layout["TOKENS_PER_BLOCK"]
    _dtype_to_elem(dtype_str)  # validate dtype early

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def topk_gating_softmax_kernel(
        GatingOutput: fx.Tensor,
        TopkWeights: fx.Tensor,
        TopkIds: fx.Tensor,
        TokenExpertIndices: fx.Tensor,
        i32_num_tokens: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        fm_fast = arith.FastMathFlags.fast
        c_zero_f = fx.Float32(0.0)
        c_neg_inf = fx.Float32(float("-inf"))
        c_one_f = fx.Float32(1.0)
        c_eps = fx.Float32(1e-20)

        c_warp = fx.Int32(WARP_SIZE)
        c_tpt = fx.Int32(THREADS_PER_TOKEN)
        c_tpw = fx.Int32(TOKENS_PER_WARP)
        c_tpb = fx.Int32(TOKENS_PER_BLOCK)
        c_vpt = fx.Int32(VPT)
        c_experts = fx.Int32(num_experts)
        c_topk = fx.Int32(topk)

        warp_id = tid // c_warp
        lane = fx.Int32(fx.lane_id)
        token_in_warp = lane // c_tpt
        expert_lane = lane % c_tpt
        local_token = warp_id * c_tpw + token_in_warp
        global_token = bid * c_tpb + local_token

        in_range = global_token < i32_num_tokens
        global_token_safe = in_range.select(global_token, fx.Int32(0))

        def group_reduce(x, mode):
            width_i32 = c_tpt
            w = x
            for _sh in range_constexpr(int(math.log2(THREADS_PER_TOKEN))):
                off = fx.Int32(THREADS_PER_TOKEN // (2 << _sh))
                peer = w.shuffle_xor(off, width_i32)
                if mode == "max":
                    w = w.maximumf(peer)
                else:
                    w = w.addf(peer, fastmath=fm_fast)
            return w

        def group_reduce_argmax(val, idx):
            width_i32 = c_tpt
            best_v, best_i = val, idx
            for _sh in range_constexpr(int(math.log2(THREADS_PER_TOKEN))):
                off = fx.Int32(THREADS_PER_TOKEN // (2 << _sh))
                peer_v = best_v.shuffle_xor(off, width_i32)
                peer_i = best_i.shuffle_xor(off, width_i32)
                is_greater = peer_v > best_v
                is_equal = ArithValue(peer_v) == ArithValue(best_v)
                peer_lower_idx = peer_i < best_i
                take_peer = is_greater | (is_equal & peer_lower_idx)
                best_v = take_peer.select(peer_v, best_v)
                best_i = take_peer.select(peer_i, best_i)
            return best_v, best_i

        # Row views via add_offset (dynamic token index; experts is constexpr).
        gating_row = fx.make_view(
            fx.add_offset(fx.get_iter(GatingOutput), fx.make_int_tuple(global_token_safe * c_experts)),
            fx.make_layout((num_experts,), (1,)),
        )
        weights_row = fx.make_view(
            fx.add_offset(fx.get_iter(TopkWeights), fx.make_int_tuple(global_token_safe * c_topk)),
            fx.make_layout((topk,), (1,)),
        )
        ids_row = fx.make_view(
            fx.add_offset(fx.get_iter(TopkIds), fx.make_int_tuple(global_token_safe * c_topk)),
            fx.make_layout((topk,), (1,)),
        )
        # Always materialise the TEI row view so dynamic scf.if regions do not
        # capture a Python None. Stores are gated by the compile-time emit_tei flag.
        tei_row = fx.make_view(
            fx.add_offset(
                fx.get_iter(TokenExpertIndices),
                fx.make_int_tuple(global_token_safe * c_topk),
            ),
            fx.make_layout((topk,), (1,)),
        )

        # Pass 1: load VPT experts owned by this lane + local max.
        # Contiguous ownership: columns [expert_lane * VPT, expert_lane * VPT + VPT).
        col_idx_list = []
        for v in range_constexpr(VPT):
            col_idx_list.append(expert_lane * c_vpt + fx.Int32(v))

        x_list = []
        thread_max = c_neg_inf
        for v in range_constexpr(VPT):
            # Promote to fp32 for Softmax/TopK (same pattern as Iluvatar GEMV).
            xv = fx.Float32(gating_row[col_idx_list[v]])
            x_list.append(xv)
            thread_max = thread_max.maximumf(xv)

        group_max = group_reduce(thread_max, "max")

        # Pass 2: exp(x - max) and sum (fx.exp lowers to libdevice __nv_expf).
        thread_sum = c_zero_f
        exp_list = []
        for v in range_constexpr(VPT):
            ev = fx.exp(x_list[v] - group_max, fastmath=fm_fast)
            exp_list.append(ev)
            thread_sum = thread_sum + ev

        group_sum = group_reduce(thread_sum, "sum")
        inv_sum = c_one_f / group_sum

        # Pass 3: normalize (register-resident probabilities).
        prob_list = []
        for v in range_constexpr(VPT):
            prob_list.append(exp_list[v] * inv_sum)

        # Pass 4: iterative TopK (sub-warp argmax → mask). No delayed_softmax.
        selected_weights = []
        selected_indices = []
        selected_sum = c_zero_f

        for _k in range_constexpr(topk):
            thread_best_val = c_neg_inf
            thread_best_idx = fx.Int32(-1)
            for v in range_constexpr(VPT):
                pv = prob_list[v]
                ci = col_idx_list[v]
                is_better = pv > thread_best_val
                thread_best_val = is_better.select(pv, thread_best_val)
                thread_best_idx = is_better.select(ci, thread_best_idx)

            best_val, best_idx = group_reduce_argmax(thread_best_val, thread_best_idx)
            selected_weights.append(best_val)
            selected_indices.append(best_idx)
            selected_sum = selected_sum + best_val

            for v in range_constexpr(VPT):
                ci = col_idx_list[v]
                is_winner = ArithValue(ci) == ArithValue(best_idx)
                prob_list[v] = is_winner.select(c_neg_inf, prob_list[v])

        # Pass 5: leader lane writes outputs.
        denom = selected_sum.maximumf(c_eps)
        inv_denom = c_one_f / denom

        if (expert_lane == fx.Int32(0)) & (global_token < i32_num_tokens):
            num_tokens_v = ArithValue(i32_num_tokens)
            for k_idx in range_constexpr(topk):
                w_val = selected_weights[k_idx]
                if renormalize:
                    w_val = w_val * inv_denom
                weights_row[Int32(k_idx)] = w_val
                ids_row[Int32(k_idx)] = selected_indices[k_idx]
                if emit_tei:
                    tei_row[Int32(k_idx)] = Int32(k_idx) * num_tokens_v + global_token

    @flyc.jit
    def launch_topk_gating_softmax(
        GatingOutput: fx.Tensor,
        TopkWeights: fx.Tensor,
        TopkIds: fx.Tensor,
        TokenExpertIndices: fx.Tensor,
        num_tokens_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        c_tpb_idx = fx.Index(TOKENS_PER_BLOCK)
        c_one_idx = fx.Index(1)
        nt_idx = arith.index_cast(T.index, num_tokens_in)
        grid_x = (nt_idx - c_one_idx) // c_tpb_idx + c_one_idx

        topk_gating_softmax_kernel(
            GatingOutput,
            TopkWeights,
            TopkIds,
            TokenExpertIndices,
            num_tokens_in,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    def launch(gating, weights, ids, num_tokens, tei=None, *, stream=None):
        """Host entry: accepts int ``num_tokens`` and optional torch stream.

        When ``emit_tei=True``, ``tei`` must be a ``[num_tokens, topk]`` int32
        tensor. When ``emit_tei=False``, ``tei`` is ignored (a 1-element dummy
        is passed so the traced kernel signature stays fixed).
        """
        if emit_tei:
            if tei is None:
                raise ValueError("tei tensor is required when emit_tei=True")
        elif tei is None:
            tei = ids.new_empty((1,))

        nt = fx.Int32(int(num_tokens))
        if stream is None:
            launch_topk_gating_softmax(gating, weights, ids, tei, nt)
        else:
            launch_topk_gating_softmax(gating, weights, ids, tei, nt, stream=stream)

    launch.layout = layout
    launch.kernel_name = KERNEL_NAME
    return launch


def compile_iluvatar_topk_gating_softmax(
    *,
    num_experts: int,
    topk: int,
    dtype_str: str = "f32",
    renormalize: bool = True,
    emit_tei: bool = False,
):
    """Alias of :func:`build_iluvatar_topk_gating_softmax` (GEMV-style name)."""
    return build_iluvatar_topk_gating_softmax(
        num_experts=num_experts,
        topk=topk,
        dtype_str=dtype_str,
        renormalize=renormalize,
        emit_tei=emit_tei,
    )


__all__ = [
    "BLOCK_THREADS",
    "KERNEL_NAME",
    "WARPS_PER_BLOCK",
    "build_iluvatar_topk_gating_softmax",
    "compile_iluvatar_topk_gating_softmax",
]
