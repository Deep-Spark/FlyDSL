# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar FP16/BF16 gate-activation-and-multiply kernel.

This kernel is the nonlinear epilogue of a split-K MoE stage1 GEMM. Split-K
first accumulates the two expert projections into ``x``:

``gate = hidden @ W_gate.T`` and ``up = hidden @ W_up.T``.

Only after all K slices have been reduced is it mathematically valid to apply
the nonlinear gate:

``out = SiLU(gate) * up``.

The diagram below shows the core forward path of a routed MoE expert FFN on
a single GPU, with this kernel highlighted in its surrounding data flow:

::

    hidden states [token_num, hidden_size]
                    |
                    +--> Router + TopK --> MoE sorting --> sorted_ids
                    |
                    +--> Stage1 split-K gate projection --+
                    |       G = sum_s(X_s @ W_gate_s.T)    |
                    |                                      +--> x = concat(G, U)
                    +--> Stage1 split-K up projection -----+    [token_num*topk,
                            U = sum_s(X_s @ W_up_s.T)            2*inter_dim] b16
                                                                   |
                                                                   v
                    +----------------------------------------------+
                    | This kernel: one CTA per sorted row
                    |   1. decode sorted_ids[bid] -> token_id, slot_id
                    |   2. row = token_id * topk + slot_id
                    |   3. load G/U from x[row]
                    |   4. optionally add expert-specific bias
                    |   5. compute H = SiLU(G) * U in FP32
                    |   6. optionally apply route weight and store out[row]
                    +----------------------------------------------+
                                                                   |
                                                                   v
                    Stage2 down projection: O = H @ W_down.T
                                                                   |
                                                                   v
                    routing-weighted TopK expert reduction
                                                                   |
                                                                   v
                    MoE output [token_num, hidden_size]

The sorting kernel groups token-slot pairs by expert for the grouped GEMMs.
Consequently, CTAs walk rows in expert-sorted order, but decoded results are
written back to their original ``token * topk + slot`` rows. This kernel does
not perform the gate/up projections, down projection, Top-K reduction, or
FP4/FP8 quantization. It can optionally apply the routing weight before
Stage2.
"""

from collections.abc import Callable

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr

from kernels.gemm.iluvatar.common import WARP_SIZE

BLOCK_THREADS = 4 * WARP_SIZE
KERNEL_NAME = "iluvatar_silu_and_mul_b16"
SUPPORTED_ACTS = ("silu", "swiglu")
SUPPORTED_QUANT_MODE = "none"
SUPPORTED_DTYPES = ("f16", "bf16")
_DTYPE_FX = {"f16": fx.Float16, "bf16": fx.BFloat16}
_DTYPE_TORCH = {"f16": "torch.float16", "bf16": "torch.bfloat16"}


def _tensors_overlap(a, b) -> bool:
    a0 = int(a.data_ptr())
    a1 = a0 + int(a.numel()) * int(a.element_size())
    b0 = int(b.data_ptr())
    b1 = b0 + int(b.numel()) * int(b.element_size())
    return max(a0, b0) < min(a1, b1)


# Build the non-quantized Iluvatar split-K stage1 post-process launcher.
#
# Parameters:
#   inter_dim:
#     Width of one expert FFN branch. Stage1 produces two such branches, so
#     each input row has 2 * inter_dim elements; gate-and-multiply reduces it
#     to one inter_dim-wide row.
#   topk:
#     Number of experts selected for each token. It determines the flattened
#     token-slot row mapping: row = token_id * topk + slot_id.
#   quant_mode:
#     Compatibility option shared with the generic implementation. This
#     Iluvatar kernel accepts only "none".
#   dtype:
#     Input/output type: "f16" or "bf16". Activation math remains FP32.
#   apply_route_weight:
#     Whether to multiply each token-slot row by its sorted FP32 routing
#     weight before converting to dtype.
#   gui_layout:
#     Physical gate/up input layout. False means [gate_0:N | up_0:N]; True
#     means 16-element interleaving:
#     [gate_0:16 | up_0:16 | gate_16:32 | up_16:32 | ...].
#   act:
#     Gating equation. "silu" computes gate * sigmoid(gate) * up. "swiglu"
#     uses gate * sigmoid(1.702 * gate) * (up + 1) with clamps.
#   enable_bias:
#     Whether to add expert-specific gate/up bias before activation. Enabling
#     it requires topk_ids to recover the expert of each token-slot row.
#   swiglu_limit:
#     Clamp bound used by SwiGLU. Zero selects the default bound 7.0. A
#     nonzero value with act="silu" enables the generic kernel's limited-SiLU
#     compatibility behavior.
#
# Returns:
#   A checked launcher with the same argument order as the generic
#   silu_and_mul_fq launcher. Its out_scale argument is retained only as an
#   unused ABI placeholder.
def build_iluvatar_silu_and_mul_module(
    inter_dim: int,
    topk: int,
    quant_mode: str = "none",
    gui_layout: bool = False,
    act: str = "silu",
    enable_bias: bool = False,
    swiglu_limit: float = 0.0,
    dtype: str = "bf16",
    apply_route_weight: bool = False,
) -> Callable:
    if inter_dim <= 0:
        raise ValueError(f"inter_dim must be positive, got {inter_dim}")
    if inter_dim % 32:
        raise ValueError(f"inter_dim={inter_dim} must be divisible by 32")
    if topk <= 0 or topk > 255:
        raise ValueError(f"topk must be in [1, 255], got {topk}")
    if quant_mode != "none":
        raise ValueError(
            f"Iluvatar b16 silu_and_mul supports only quant_mode='none', got {quant_mode!r}"
        )
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"dtype must be one of {SUPPORTED_DTYPES}, got {dtype!r}"
        )
    if act not in SUPPORTED_ACTS:
        raise ValueError(f"act must be one of {SUPPORTED_ACTS}, got {act!r}")
    if swiglu_limit < 0:
        raise ValueError(f"swiglu_limit must be non-negative, got {swiglu_limit}")
    elem_dtype = _DTYPE_FX[dtype]

    # Device kernel parameters:
    #   x:
    #     Accumulated split-K stage1 result with logical shape
    #     [token_num * topk, 2 * inter_dim], FP16/BF16.
    #   out:
    #     Activated expert input for stage2 with logical shape
    #     [token_num * topk, inter_dim], same dtype as x.
    #   sorted_ids:
    #     Sorted-row -> original token-slot mapping. Low 24 bits encode
    #     token_id and high 8 bits encode the top-k slot.
    #   sorted_weights:
    #     FP32 routing weights aligned with sorted_ids. Read only when
    #     apply_route_weight=True.
    #   num_valid_ids:
    #     Number of sorted rows in the routed/padded MoE region. Sentinel rows
    #     are rejected again by token_id/slot_id bounds checks.
    #   topk_ids:
    #     Router-selected expert IDs with logical shape [token_num, topk].
    #     Read only when expert-specific bias is enabled.
    #   bias:
    #     Expert gate/up bias with logical shape
    #     [num_experts, 2 * inter_dim], FP32.
    #   token_num:
    #     Number of original input tokens; used to reject sorting sentinels.
    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def _kernel(
        x: fx.Tensor,
        out: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        num_valid_ids: fx.Tensor,
        topk_ids: fx.Tensor,
        bias: fx.Tensor,
        token_num: fx.Int32,
    ):
        # One CTA consumes one entry in the expert-sorted row list. Threads
        # within that CTA partition the inter_dim columns of the decoded row.
        bid = fx.Int32(fx.block_idx.x)
        tid = fx.Int32(fx.thread_idx.x)
        n = fx.Int32(inter_dim)
        k = fx.Int32(topk)

        # Step 1 — Decode the sorted row.
        # Recover token_id and top-k slot_id from sorted_ids[bid]. A padded
        # sorting entry conventionally encodes token_id == token_num and
        # slot_id == topk, making it fail the validity predicate below.
        fused_id = fx.Int32(sorted_ids[bid])
        token_id = fused_id & fx.Int32(0xFFFFFF)
        slot_id = fused_id.shrui(fx.Int32(24))
        is_valid = (
            (bid < fx.Int32(num_valid_ids[0]))
            & (token_id < token_num)
            & (slot_id < k)
        )

        if is_valid:
            # Step 2 — Map back to the original token-slot row.
            # Activation storage stays in original order even though CTAs are
            # launched in expert-sorted order.
            row = token_id * k + slot_id

            # Prepare Step 4: resolve the routed expert once per row so every
            # column can reuse the same expert-specific gate/up bias row.
            expert_id = fx.Int32(0)
            if const_expr(enable_bias):
                expert_id = fx.Int32(topk_ids[token_id, slot_id])
            route = (
                fx.Float32(sorted_weights[bid])
                if const_expr(apply_route_weight)
                else fx.Float32(1.0)
            )

            one = fx.Float32(1.0)
            alpha = fx.Float32(1.702)
            limit_value = float(swiglu_limit) if swiglu_limit else 7.0
            limit = fx.Float32(limit_value)
            neg_limit = fx.Float32(-limit_value)
            log2e = fx.Float32(1.4426950408889634)
            exp2_scale = fx.Float32(8388608.0)
            exp2_bias = fx.Float32(1065353216.0)
            min_exp2 = fx.Float32(-126.0)
            max_exp2 = fx.Float32(126.0)

            def exp2_approx(value):
                # Schraudolph-style approximation:
                #   2^v ~= bitcast_f32(int(v * 2^23 + (127 << 23))).
                # Clamping keeps the constructed exponent in the normal FP32
                # range. This also avoids an external __nv_expf call, whose
                # libdevice object is incompatible with the IXDL link path.
                hi = (value > max_exp2).select(max_exp2, value)
                clamped = (hi < min_exp2).select(min_exp2, hi)
                bits = fx.Int32(clamped * exp2_scale + exp2_bias)
                return bits.bitcast(fx.Float32)

            # A 256-thread CTA covers 256 columns per iteration. Larger expert
            # widths are handled by a compile-time-unrolled grid-stride loop.
            for base in range_constexpr(0, inter_dim, BLOCK_THREADS):
                col = tid + fx.Int32(base)
                if col < n:
                    if const_expr(gui_layout):
                        # Translate a logical output column to the physical
                        # 16-element gate/up-interleaved stage1 layout.
                        block_idx = col.shrui(fx.Int32(4))
                        offset = col & fx.Int32(15)
                        gate_col = block_idx * fx.Int32(32) + offset
                        up_col = gate_col + fx.Int32(16)
                    else:
                        gate_col = col
                        up_col = col + n

                    # Step 3 — Load this column's gate/up pair.
                    # BF16 stage1 values are promoted to FP32 before bias,
                    # activation, and multiplication.
                    gate = x[row, gate_col].to(fx.Float32)
                    up = x[row, up_col].to(fx.Float32)

                    # Step 4 — Optionally add expert-specific bias.
                    if const_expr(enable_bias):
                        # Bias uses logical separated gate/up coordinates,
                        # independent of x's physical gui_layout.
                        gate = gate + bias[expert_id, col].to(fx.Float32)
                        up = up + bias[expert_id, col + n].to(fx.Float32)

                    # Step 5 — Apply the selected gate activation and perform
                    # the elementwise Hadamard product with the up branch.
                    if const_expr(act == "swiglu"):
                        # Project-specific SwiGLU:
                        # clamp(g, max=L) * sigmoid(1.702*g)
                        #     * (clamp(up, -L, L) + 1).
                        gate = (gate > limit).select(limit, gate)
                        up = (up > limit).select(limit, up)
                        up = (up < neg_limit).select(neg_limit, up)
                        sigmoid = one / (
                            one + exp2_approx(-(gate * alpha) * log2e)
                        )
                        value = gate * sigmoid * (up + one)
                    elif const_expr(swiglu_limit != 0.0):
                        gate = (gate > limit).select(limit, gate)
                        up = (up > limit).select(limit, up)
                        up = (up < neg_limit).select(neg_limit, up)
                        sigmoid = one / (
                            one + exp2_approx(-(gate * alpha) * log2e)
                        )
                        value = gate * sigmoid * up
                    else:
                        # SiLU(gate) * up, with
                        # exp(-gate) = exp2(-gate * log2(e)).
                        sigmoid = one / (one + exp2_approx(-gate * log2e))
                        value = gate * sigmoid * up

                    # Step 6 — Apply the optional routing weight, convert to
                    # the selected b16 dtype, and restore token-slot order.
                    out[row, col] = (value * route).to(elem_dtype)

    @flyc.jit
    def _launch(
        # These tensors have the same meanings as the device-kernel arguments.
        x: fx.Tensor,
        out: fx.Tensor,
        sorted_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        num_valid_ids: fx.Tensor,
        topk_ids: fx.Tensor,
        bias: fx.Tensor,
        token_num: fx.Int32,
        num_sorted_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        # Grid X follows sorted rows, while each 256-thread CTA covers the
        # inter_dim columns of one decoded token-slot row.
        _kernel(
            x,
            out,
            sorted_ids,
            sorted_weights,
            num_valid_ids,
            topk_ids,
            bias,
            token_num,
        ).launch(
            grid=(num_sorted_rows, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    def launch(
        x,
        out,
        out_scale,
        sorted_ids,
        num_valid_ids,
        topk_ids,
        bias,
        token_num: int,
        num_sorted_rows: int,
        stream=None,
        sorted_weights=None,
    ):
        if not isinstance(token_num, int) or token_num < 0:
            raise ValueError(f"token_num must be a non-negative int, got {token_num!r}")
        if not isinstance(num_sorted_rows, int) or num_sorted_rows < 0:
            raise ValueError(
                f"num_sorted_rows must be a non-negative int, got {num_sorted_rows!r}"
            )
        rows = token_num * topk
        if tuple(x.shape) != (rows, 2 * inter_dim):
            raise ValueError(
                f"expected x shape ({rows},{2 * inter_dim}), got {tuple(x.shape)}"
            )
        if tuple(out.shape) != (rows, inter_dim):
            raise ValueError(
                f"expected out shape ({rows},{inter_dim}), got {tuple(out.shape)}"
            )
        expected_torch_dtype = _DTYPE_TORCH[dtype]
        if str(x.dtype) != expected_torch_dtype:
            raise ValueError(f"x dtype must be {expected_torch_dtype}, got {x.dtype}")
        if str(out.dtype) != expected_torch_dtype:
            raise ValueError(
                f"out dtype must be {expected_torch_dtype}, got {out.dtype}"
            )
        if not x.is_contiguous() or not out.is_contiguous():
            raise ValueError("x and out must be contiguous")
        if x.device != out.device:
            raise ValueError("x and out must be on the same device")
        if _tensors_overlap(x, out):
            raise ValueError("out must not overlap with x")
        if sorted_ids.dim() != 1 or int(sorted_ids.numel()) < num_sorted_rows:
            raise ValueError(
                f"sorted_ids must be 1D with at least {num_sorted_rows} elements"
            )
        if str(sorted_ids.dtype) != "torch.int32":
            raise ValueError("sorted_ids dtype must be torch.int32")
        if int(num_valid_ids.numel()) < 1 or str(num_valid_ids.dtype) != "torch.int32":
            raise ValueError("num_valid_ids must contain at least one int32 element")
        if apply_route_weight:
            if sorted_weights is None:
                raise ValueError(
                    "sorted_weights is required when apply_route_weight=True"
                )
            if sorted_weights.dim() != 1 or int(sorted_weights.numel()) < num_sorted_rows:
                raise ValueError(
                    "sorted_weights must be 1D with at least "
                    f"{num_sorted_rows} elements"
                )
            if str(sorted_weights.dtype) != "torch.float32":
                raise ValueError("sorted_weights dtype must be torch.float32")
            if not sorted_weights.is_contiguous():
                raise ValueError("sorted_weights must be contiguous")
        else:
            sorted_weights = sorted_ids if sorted_weights is None else sorted_weights

        if enable_bias:
            if topk_ids is None or bias is None:
                raise ValueError("topk_ids and bias are required when enable_bias=True")
            if tuple(topk_ids.shape) != (token_num, topk):
                raise ValueError(f"topk_ids must have shape ({token_num},{topk})")
            if str(topk_ids.dtype) != "torch.int32":
                raise ValueError("topk_ids dtype must be torch.int32")
            if bias.dim() != 2 or int(bias.shape[1]) != 2 * inter_dim:
                raise ValueError(f"bias must have shape (experts,{2 * inter_dim})")
            if str(bias.dtype) != "torch.float32":
                raise ValueError("bias dtype must be torch.float32")
            if not topk_ids.is_contiguous() or not bias.is_contiguous():
                raise ValueError("topk_ids and bias must be contiguous")
        else:
            topk_ids = sorted_ids if topk_ids is None else topk_ids
            bias = x if bias is None else bias

        for name, tensor in (
            ("sorted_ids", sorted_ids),
            ("sorted_weights", sorted_weights),
            ("num_valid_ids", num_valid_ids),
            ("topk_ids", topk_ids),
            ("bias", bias),
        ):
            if tensor.device != x.device:
                raise ValueError(f"{name} must be on the same device as x")

        del out_scale
        if num_sorted_rows == 0:
            return out
        # Convert dynamic scalar launch dimensions to FlyDSL ABI values.
        args = (
            x,
            out,
            sorted_ids,
            sorted_weights,
            num_valid_ids,
            topk_ids,
            bias,
            fx.Int32(token_num),
            fx.Int32(num_sorted_rows),
        )
        _launch(*args) if stream is None else _launch(*args, stream=stream)
        return out

    launch.kernel_name = f"{KERNEL_NAME}_{dtype}"
    launch.block_threads = BLOCK_THREADS
    return launch


def compile_iluvatar_silu_and_mul(**kwargs) -> Callable:
    """Alias of :func:`build_iluvatar_silu_and_mul_module`."""
    return build_iluvatar_silu_and_mul_module(**kwargs)


__all__ = [
    "BLOCK_THREADS",
    "KERNEL_NAME",
    "SUPPORTED_ACTS",
    "SUPPORTED_DTYPES",
    "SUPPORTED_QUANT_MODE",
    "build_iluvatar_silu_and_mul_module",
    "compile_iluvatar_silu_and_mul",
]
