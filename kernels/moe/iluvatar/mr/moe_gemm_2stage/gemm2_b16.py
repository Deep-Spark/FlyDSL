# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar MR b16 MoE stage2 and non-atomic reduction composition.

The diagram below places the Stage2 GEMM in the core forward path of a routed
MoE expert FFN on one GPU. This module computes a separate down-projection for
each routed slot and can compose it with the following TopK reduction::

    hidden states X [tokens, model_dim]
                    |
                    +--> Router + TopK --> MoE sorting
                    |                         |
                    |                         +--> sorted_token_ids
                    |                         +--> sorted_expert_ids
                    |                         +--> sorted_weights
                    |                                  |
                    v                                  |
    Stage1 gate/up projection: [G, U] = X @ W1[e].T <---+
                    |
                    v
    gated activation: H[token, slot] = SiLU(G) * U
    H: [tokens, topk, inter_dim]
                    |
                    v
    +------------------------------------------------------------------+
    | This module: Stage2 down projection                              |
    |                                                                  |
    |   For each routed (token, slot) assigned to Expert e:             |
    |     O_slot = H[token, slot] @ W2[e].T                             |
    |     optionally: O_slot = O_slot * sorted_weights                  |
    |                                                                  |
    |   W2:  [experts, model_dim, inter_dim]                            |
    |   Out: [tokens, topk, model_dim]                                  |
    |   compute: gathered-A + Expert W2 -> FP32 MR MMA -> b16           |
    +------------------------------------------------------------------+
                    |
                    v
    +------------------------------------------------------------------+
    | Non-atomic TopK reduction (MoeGemm2Mode.REDUCE)                   |
    |   O[token, :] = sum_slot O_slot[token, slot, :]                   |
    +------------------------------------------------------------------+
                    |
                    v
    MoE output [tokens, model_dim]

``MoeGemm2Mode.PER_SLOT`` stops after the highlighted Stage2 projection.
``MoeGemm2Mode.REDUCE`` additionally launches the reduction shown below it.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx

from .gemm_common_b16 import (
    B16_DTYPES,
    build_grouped_b16_kernel,
    validate_grouped_b16_config,
)
from .reduction_b16 import compile_iluvatar_mr_moe_reduction_b16

DEFAULT_WARPS_M = 2
DEFAULT_WARPS_N = 4
DEFAULT_WARP_ATOMS_M = 1
DEFAULT_WARP_ATOMS_N = 2
DEFAULT_K_ATOMS = 2
DEFAULT_STAGES = 2


class MoeGemm2Mode:
    """Stage2 output strategy."""

    ATOMIC = "atomic"
    REDUCE = "reduce"
    PER_SLOT = "per_slot"


@functools.lru_cache(maxsize=256)
def compile_iluvatar_mr_moe_gemm2_b16(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    dtype: str = "bf16",
    apply_route_weight: bool = True,
    accumulate: bool = False,
    warps_m: int = DEFAULT_WARPS_M,
    warps_n: int = DEFAULT_WARPS_N,
    warp_atoms_m: int = DEFAULT_WARP_ATOMS_M,
    warp_atoms_n: int = DEFAULT_WARP_ATOMS_N,
    k_atoms: int = DEFAULT_K_ATOMS,
    stages: int = DEFAULT_STAGES,
):
    """Compile the dense Iluvatar/MR/b16 MoE down projection.

    Returned launcher signature::

        launch(Out, X, W2, sorted_token_ids, sorted_expert_ids,
               sorted_weights, num_valid_ids, tokens, num_expert_blocks,
               stream=None)

    ``X`` is ``[tokens,topk,inter_dim]`` and contiguous dense ``W2`` is
    ``[experts,model_dim,inter_dim]``. FP32 accumulators are converted to
    ``dtype`` at the epilogue. With ``accumulate=False``, ``Out`` is
    ``[tokens,topk,model_dim]``. Iluvatar b16 atomic accumulation is rejected
    because device validation showed unreliable concurrent updates; use
    :func:`compile_iluvatar_mr_moe_gemm2_b16_ex` in ``REDUCE`` mode.
    """
    if dtype not in B16_DTYPES:
        raise ValueError(f"dtype must be one of {B16_DTYPES}, got {dtype!r}")
    if accumulate:
        raise ValueError("Iluvatar b16 atomic accumulation is not supported; " "use MoeGemm2Mode.REDUCE")
    config = validate_grouped_b16_config(
        N=model_dim,
        K=inter_dim,
        dtype=dtype,
        topk=topk,
        experts=experts,
        warps_m=warps_m,
        warps_n=warps_n,
        warp_atoms_m=warp_atoms_m,
        warp_atoms_n=warp_atoms_n,
        k_atoms=k_atoms,
        stages=stages,
    )

    # Stage2 uses grouped_b16_kernel from gemm_common_b16.py; its concrete tensor
    # contract here is (all tensors dense row-major unless noted otherwise):
    #   Out:
    #     - [tokens, topk, model_dim], dtype = FP16/BF16
    #     - strides [topk*model_dim, model_dim, 1]
    #     - contains one Expert contribution per routed slot; a following
    #       reduction sums the topk dimension
    #   X:
    #     - [tokens, topk, inter_dim], dtype = FP16/BF16
    #     - strides [topk*inter_dim, inter_dim, 1]
    #   W2:
    #     - [experts, model_dim, inter_dim], dtype = FP16/BF16
    #     - strides [model_dim*inter_dim, inter_dim, 1]
    #   sorted_token_ids:
    #     - contiguous int32 [num_sorted_rows]
    #     - packed as (topk_slot << 24) | token
    #   sorted_expert_ids:
    #     - contiguous int32 [num_expert_blocks], one Expert per bm rows
    #   sorted_weights:
    #     - contiguous FP32 [num_sorted_rows], multiplied into each contribution
    #       only when apply_route_weight=True
    #   num_valid_ids:
    #     - contiguous int32 [>=1]; element 0 is the padded valid-row count
    # Accumulation inside each dot product is FP32 before conversion to Out.
    kernel = build_grouped_b16_kernel(
        N=model_dim,
        K=inter_dim,
        topk=topk,
        dtype=dtype,
        input_has_slots=True,
        apply_route_weight=apply_route_weight,
        accumulate=bool(accumulate),
        warps_m=warps_m,
        warps_n=warps_n,
        warp_atoms_m=warp_atoms_m,
        warp_atoms_n=warp_atoms_n,
        k_atoms=k_atoms,
        stages=stages,
        config=config,
    )

    @flyc.jit
    def launch_jit(
        Out: fx.Tensor,
        X: fx.Tensor,
        W2: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        num_valid_ids: fx.Tensor,
        tokens_in: fx.Int32,
        num_expert_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel(
            Out,
            X,
            W2,
            sorted_token_ids,
            sorted_expert_ids,
            sorted_weights,
            num_valid_ids,
            tokens_in,
        ).launch(
            grid=(config.n_tiles, num_expert_blocks, 1),
            block=(config.threads, 1, 1),
            stream=stream,
        )

    def launch(
        Out,
        X,
        W2,
        sorted_token_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        tokens,
        num_expert_blocks,
        stream=None,
    ):
        if apply_route_weight and sorted_weights is None:
            raise ValueError("sorted_weights is required when apply_route_weight=True")
        weights = sorted_token_ids if sorted_weights is None else sorted_weights
        args = (
            Out,
            X,
            W2,
            sorted_token_ids,
            sorted_expert_ids,
            weights,
            num_valid_ids,
            fx.Int32(int(tokens)),
            fx.Int32(int(num_expert_blocks)),
        )
        launch_jit(*args) if stream is None else launch_jit(*args, stream=stream)
        return Out

    launch.accumulate = bool(accumulate)
    launch.dtype = dtype
    launch.bm, launch.bn, launch.bk = config.bm, config.bn, config.bk
    return launch


class _Gemm2Reduction:
    """Allocation-free composition of per-slot GEMM2 and top-k reduction."""

    mode = MoeGemm2Mode.REDUCE

    def __init__(self, gemm2, reduction, topk: int, model_dim: int, dtype: str):
        self.gemm2 = gemm2
        self.reduction = reduction
        self.topk = topk
        self.model_dim = model_dim
        self.dtype = dtype

    def workspace_shape(self, tokens: int):
        return (int(tokens), self.topk, self.model_dim)

    def __call__(
        self,
        Out,
        Workspace,
        X,
        W2,
        sorted_token_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        tokens,
        num_expert_blocks,
        valid_mask=None,
        stream=None,
    ):
        """Run GEMM2 then reduction using caller-owned b16 ``Workspace``."""
        self.gemm2(
            Workspace,
            X,
            W2,
            sorted_token_ids,
            sorted_expert_ids,
            sorted_weights,
            num_valid_ids,
            tokens,
            num_expert_blocks,
            stream=stream,
        )
        self.reduction(Workspace, Out, valid_mask, tokens, stream=stream)
        return Out


def compile_iluvatar_mr_moe_gemm2_b16_ex(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    dtype: str = "bf16",
    apply_route_weight: bool = True,
    mode: str = MoeGemm2Mode.REDUCE,
    use_valid_mask: bool = False,
    warps_m: int = DEFAULT_WARPS_M,
    warps_n: int = DEFAULT_WARPS_N,
    warp_atoms_m: int = DEFAULT_WARP_ATOMS_M,
    warp_atoms_n: int = DEFAULT_WARP_ATOMS_N,
    k_atoms: int = DEFAULT_K_ATOMS,
    stages: int = DEFAULT_STAGES,
):
    """Compile stage2 in atomic, per-slot, or per-slot-plus-reduction mode.

    Reduce mode returns an allocation-free wrapper whose call adds
    ``Workspace`` immediately after ``Out``. The workspace contract is
    ``[tokens,topk,model_dim]`` in ``dtype``. Per-slot mode returns the raw
    non-atomic launcher. Atomic mode is intentionally rejected on Iluvatar b16.
    """
    if mode not in (
        MoeGemm2Mode.ATOMIC,
        MoeGemm2Mode.REDUCE,
        MoeGemm2Mode.PER_SLOT,
    ):
        raise ValueError(f"unknown GEMM2 mode {mode!r}")
    common = dict(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        dtype=dtype,
        apply_route_weight=apply_route_weight,
        warps_m=warps_m,
        warps_n=warps_n,
        warp_atoms_m=warp_atoms_m,
        warp_atoms_n=warp_atoms_n,
        k_atoms=k_atoms,
        stages=stages,
    )
    if mode == MoeGemm2Mode.ATOMIC:
        raise ValueError("MoeGemm2Mode.ATOMIC is not supported for Iluvatar b16; " "use MoeGemm2Mode.REDUCE")
    gemm2 = compile_iluvatar_mr_moe_gemm2_b16(accumulate=False, **common)
    if mode == MoeGemm2Mode.PER_SLOT:
        if use_valid_mask:
            raise ValueError("use_valid_mask applies only to reduce mode")
        return gemm2
    reduction = compile_iluvatar_mr_moe_reduction_b16(
        topk=topk,
        model_dim=model_dim,
        dtype=dtype,
        use_valid_mask=use_valid_mask,
    )
    return _Gemm2Reduction(gemm2, reduction, topk, model_dim, dtype)


__all__ = [
    "MoeGemm2Mode",
    "compile_iluvatar_mr_moe_gemm2_b16",
    "compile_iluvatar_mr_moe_gemm2_b16_ex",
]
