# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar MR b16 MoE stage1: dense gate/up projection plus SiLU.

The diagram below places the Stage1 GEMM in the core forward path of a routed
MoE expert FFN on one GPU. This module performs both the highlighted grouped
projection and the following SiLU-and-multiply operation::

    hidden states X [tokens, model_dim]
                    |
                    +--> Router + TopK --> MoE sorting
                    |                         |
                    |                         +--> sorted_token_ids
                    |                         +--> sorted_expert_ids
                    |                         +--> sorted_weights
                    |                                  |
                    +----------------------------------+
                                                       |
                                                       v
    +------------------------------------------------------------------+
    | This module: Stage1 gate/up projection                            |
    |                                                                  |
    |   For each routed (token, slot) assigned to Expert e:             |
    |     [G, U] = X[token] @ W1[e].T                                   |
    |                                                                  |
    |   W1:        [experts, 2*inter_dim, model_dim]                    |
    |   Workspace: [tokens, topk, 2*inter_dim] = concat(G, U)           |
    |   compute:   gathered-A + Expert W1 -> FP32 MR MMA -> b16         |
    +------------------------------------------------------------------+
                                                       |
                                                       v
    +------------------------------------------------------------------+
    | Shared silu_and_mul kernel                                        |
    |   H[token, slot] = SiLU(G) * U                                    |
    |   optionally: H = H * sorted_weights                              |
    |   Out: [tokens, topk, inter_dim]                                   |
    +------------------------------------------------------------------+
                                                       |
                                                       v
                    Stage2 down projection: O_slot = H @ W2[e].T
                                                       |
                                                       v
                    routing-weighted TopK reduction
                                                       |
                                                       v
                    MoE output [tokens, model_dim]
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from kernels.moe.iluvatar.silu_and_mul import compile_iluvatar_silu_and_mul

from .gemm_common_b16 import (
    B16_DTYPES,
    build_grouped_b16_kernel,
    validate_grouped_b16_config,
)

DEFAULT_WARPS_M = 2
DEFAULT_WARPS_N = 4
DEFAULT_WARP_ATOMS_M = 1
DEFAULT_WARP_ATOMS_N = 2
DEFAULT_K_ATOMS = 2
DEFAULT_STAGES = 2


@functools.lru_cache(maxsize=256)
def compile_iluvatar_mr_moe_gemm1_b16(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    dtype: str = "bf16",
    apply_route_weight: bool = False,
    warps_m: int = DEFAULT_WARPS_M,
    warps_n: int = DEFAULT_WARPS_N,
    warp_atoms_m: int = DEFAULT_WARP_ATOMS_M,
    warp_atoms_n: int = DEFAULT_WARP_ATOMS_N,
    k_atoms: int = DEFAULT_K_ATOMS,
    stages: int = DEFAULT_STAGES,
):
    """Compile Iluvatar/MR/b16 MoE stage1.

    Returned launcher signature::

        launch(Out, Workspace, X, W1, sorted_token_ids, sorted_expert_ids,
               sorted_weights, num_valid_ids, tokens, num_expert_blocks,
               stream=None)

    ``X`` is contiguous ``[tokens,model_dim]`` and dense ``W1`` is
    ``[experts,2*inter_dim,model_dim]``. ``Workspace`` must be caller-owned,
    contiguous, have shape ``[tokens,topk,2*inter_dim]``, and use ``dtype``.
    ``Out`` is ``[tokens,topk,inter_dim]``. The projection accumulates in FP32;
    a second kernel applies SiLU(gate)*up in FP32 and optionally the route
    weight. No allocation occurs in either compile or launch.
    """
    if dtype not in B16_DTYPES:
        raise ValueError(f"dtype must be one of {B16_DTYPES}, got {dtype!r}")
    if inter_dim <= 0:
        raise ValueError(f"inter_dim must be positive, got {inter_dim}")
    config = validate_grouped_b16_config(
        N=2 * inter_dim,
        K=model_dim,
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
    # Stage1 projection specializes grouped_b16_kernel as follows:
    #   X:         [tokens, model_dim], row-major [model_dim, 1]
    #   W1:        [experts, 2*inter_dim, model_dim], row-major
    #              [2*inter_dim*model_dim, model_dim, 1]
    #   Workspace: [tokens, topk, 2*inter_dim], row-major
    #              [topk*2*inter_dim, 2*inter_dim, 1]
    # The first inter_dim columns are gate and the second inter_dim columns
    # are up. Accumulation is FP32; Workspace is converted to FP16/BF16.
    projection = build_grouped_b16_kernel(
        N=2 * inter_dim,
        K=model_dim,
        topk=topk,
        dtype=dtype,
        input_has_slots=False,
        apply_route_weight=False,
        accumulate=False,
        warps_m=warps_m,
        warps_n=warps_n,
        warp_atoms_m=warp_atoms_m,
        warp_atoms_n=warp_atoms_n,
        k_atoms=k_atoms,
        stages=stages,
        config=config,
    )
    activation = compile_iluvatar_silu_and_mul(
        inter_dim=inter_dim,
        topk=topk,
        quant_mode="none",
        gui_layout=False,
        act="silu",
        enable_bias=False,
        dtype=dtype,
        apply_route_weight=apply_route_weight,
    )

    @flyc.jit
    def launch_projection_jit(
        Workspace: fx.Tensor,
        X: fx.Tensor,
        W1: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        tokens_in: fx.Int32,
        num_expert_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        projection(
            Workspace,
            X,
            W1,
            sorted_token_ids,
            sorted_expert_ids,
            sorted_token_ids,
            num_valid_ids,
            tokens_in,
        ).launch(
            grid=(config.n_tiles, num_expert_blocks, 1),
            block=(config.threads, 1, 1),
            stream=stream,
        )

    def launch(
        Out,
        Workspace,
        X,
        W1,
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
        projection_args = (
            Workspace,
            X,
            W1,
            sorted_token_ids,
            sorted_expert_ids,
            num_valid_ids,
            fx.Int32(int(tokens)),
            fx.Int32(int(num_expert_blocks)),
        )
        if stream is None:
            launch_projection_jit(*projection_args)
        else:
            launch_projection_jit(*projection_args, stream=stream)

        rows = int(tokens) * topk
        activation(
            Workspace.view(rows, 2 * inter_dim),
            Out.view(rows, inter_dim),
            None,
            sorted_token_ids,
            num_valid_ids,
            None,
            None,
            int(tokens),
            int(num_expert_blocks) * config.bm,
            stream=stream,
            sorted_weights=weights,
        )
        return Out

    launch.workspace_shape = lambda tokens: (int(tokens), topk, 2 * inter_dim)
    launch.dtype = dtype
    launch.bm, launch.bn, launch.bk = config.bm, config.bn, config.bk
    return launch


__all__ = ["compile_iluvatar_mr_moe_gemm1_b16"]
