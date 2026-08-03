# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Shared correctness-first b16 grouped projection for the two MoE stages."""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from kernels.gemm.iluvatar.common import WARP_SIZE, parse_major_pattern
from kernels.gemm.iluvatar.mr.common import (
    ATOM_K_B16,
    ATOM_M,
    ATOM_N,
    DEFAULT_SMEM_CAP_BYTES,
    MR_GEMM_GEOM,
    sme_atom_counts,
)

B16_DTYPES = ("f16", "bf16")
_DTYPE_FX = {"f16": fx.Float16, "bf16": fx.BFloat16}


@dataclass(frozen=True)
class GroupedB16Config:
    bm: int
    bn: int
    bk: int
    threads: int
    n_tiles: int
    smem_bytes: int


def validate_grouped_b16_config(
    *,
    N: int,
    K: int,
    dtype: str,
    topk: int,
    experts: int,
    warps_m: int,
    warps_n: int,
    warp_atoms_m: int,
    warp_atoms_n: int,
    k_atoms: int,
    stages: int,
) -> GroupedB16Config:
    """Validate the public b16 contract and derive the routing CTA geometry."""
    if dtype not in B16_DTYPES:
        raise ValueError(f"dtype must be one of {B16_DTYPES}, got {dtype!r}")
    if min(N, K, topk, experts) <= 0:
        raise ValueError(f"N, K, topk and experts must be positive, got {N}, {K}, {topk}, {experts}")
    if topk > 255:
        raise ValueError(f"packed routing slot is 8-bit; topk must be <=255, got {topk}")
    if stages != 2:
        raise ValueError(f"only a two-stage MR-compatible pipeline is supported, got stages={stages}")
    if min(warps_m, warps_n, warp_atoms_m, warp_atoms_n, k_atoms) <= 0:
        raise ValueError("warp counts, atom counts and k_atoms must be positive")

    bm = ATOM_M * warp_atoms_m * warps_m
    bn = ATOM_N * warp_atoms_n * warps_n
    bk = ATOM_K_B16 * k_atoms
    threads = warps_m * warps_n * WARP_SIZE
    if threads > 1024:
        raise ValueError(f"CTA has {threads} threads; Iluvatar limit is 1024")
    if K % bk:
        raise ValueError(f"K={K} must be divisible by bk={bk}")
    if N % ATOM_N:
        raise ValueError(f"N={N} must be divisible by the MR output atom width {ATOM_N}")
    if bk % MR_GEMM_GEOM.values_per_sme_row:
        raise ValueError(
            f"bk={bk} must be divisible by SME row width " f"{MR_GEMM_GEOM.values_per_sme_row}; use an even k_atoms"
        )

    # Preserve the MR geometry restrictions in the public API so the scalar V1
    # can be replaced by a gathered-A MR mainloop without changing callers.
    _, b_atoms, _, _ = sme_atom_counts(
        parse_major_pattern("tn"),
        bm,
        bn,
        bk,
        values_per_sme_row=MR_GEMM_GEOM.values_per_sme_row,
    )
    num_warps = warps_m * warps_n
    if b_atoms % num_warps:
        raise ValueError(f"B SME chunk count {b_atoms} must divide across {num_warps} warps")
    smem_bytes = stages * (bm + bn) * bk * 2
    if smem_bytes > DEFAULT_SMEM_CAP_BYTES:
        raise ValueError(f"CTA shared memory {smem_bytes} B exceeds {DEFAULT_SMEM_CAP_BYTES} B")
    return GroupedB16Config(
        bm=bm,
        bn=bn,
        bk=bk,
        threads=threads,
        n_tiles=(N + bn - 1) // bn,
        smem_bytes=smem_bytes,
    )


def build_grouped_b16_kernel(
    *,
    N: int,
    K: int,
    topk: int,
    dtype: str,
    input_has_slots: bool,
    apply_route_weight: bool,
    accumulate: bool,
    warps_m: int,
    warps_n: int,
    warp_atoms_m: int,
    warp_atoms_n: int,
    k_atoms: int,
    stages: int,
    config: GroupedB16Config,
):
    """Build a dense ``W[E,N,K]`` routed projection with FP32 accumulation.

    V1 deliberately uses a scalar dot-product mainloop. An earlier gathered-A
    MRMma prototype produced invalid b16 accumulator lanes on ivcore11; keeping
    this verified path is preferable to silently returning incorrect values.
    The CTA/routing API remains MR-compatible for a later optimized mainloop.
    """
    del warps_m, warps_n, warp_atoms_m, warp_atoms_n, k_atoms, stages
    if accumulate:
        raise ValueError(
            "Iluvatar b16 atomic accumulation is not supported; " "use per-slot output followed by reduction"
        )
    elem_dtype = _DTYPE_FX[dtype]
    bm, bn, threads = config.bm, config.bn, config.threads

    # Kernel tensor contract (all layouts are dense row-major):
    #   Out:
    #     - shape [tokens, topk, N], dtype = elem_dtype (FP16 or BF16)
    #     - strides [topk*N, N, 1]
    #     - Stage1 specialization: N = 2*inter_dim, storing [gate | up]
    #     - Stage2 specialization: N = model_dim, storing one down-projection
    #       contribution for each (token, top-k slot)
    #   X:
    #     - input_has_slots=False: [tokens, K], strides [K, 1]
    #     - input_has_slots=True:  [tokens, topk, K],
    #       strides [topk*K, K, 1]
    #     - dtype = elem_dtype
    #   W:
    #     - shape [experts, N, K], strides [N*K, K, 1]
    #     - dtype = elem_dtype; W[e, n, :] is one Expert output row
    #   sorted_token_ids:
    #     - contiguous int32 [num_sorted_rows]
    #     - low 24 bits = token, high 8 bits = top-k slot
    #   sorted_expert_ids:
    #     - contiguous int32 [num_expert_blocks]
    #     - one Expert ID for every bm consecutive sorted rows
    #   sorted_weights:
    #     - contiguous FP32 [num_sorted_rows], aligned with sorted_token_ids
    #   num_valid_ids:
    #     - contiguous int32 [>=1]; element 0 is the padded valid-row count
    #   tokens_in:
    #     - scalar int32 containing the original token count
    #
    # Grid layout:
    #   blockIdx.x selects a bn-wide output-column tile;
    #   blockIdx.y selects one bm-row Expert group.
    @flyc.kernel(known_block_size=[threads, 1, 1])
    def grouped_b16_kernel(
        Out: fx.Tensor,
        X: fx.Tensor,
        W: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        num_valid_ids: fx.Tensor,
        tokens_in: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        n_base = fx.Int32(fx.block_idx.x) * fx.Int32(bn)
        expert_block = fx.Int32(fx.block_idx.y)
        m_base = expert_block * fx.Int32(bm)
        expert = fx.Int32(sorted_expert_ids[expert_block])
        tokens = fx.Int32(tokens_in)
        col = n_base + tid

        if (tid < fx.Int32(bn)) & (col < fx.Int32(N)):
            for local_m in fx.range(0, bm, 1):
                sorted_row = m_base + fx.Int32(local_m)
                fused = fx.Int32(sorted_token_ids[sorted_row])
                token = fused & fx.Int32(0xFFFFFF)
                slot = fused.shrui(fx.Int32(24))
                valid = (sorted_row < fx.Int32(num_valid_ids[0])) & (token < tokens) & (slot < fx.Int32(topk))
                if valid:
                    acc = fx.Float32(0.0)
                    for kk in fx.range(0, K, 1):
                        k = fx.Int32(kk)
                        if fx.const_expr(input_has_slots):
                            a = X[token, slot, k].to(fx.Float32)
                        else:
                            a = X[token, k].to(fx.Float32)
                        b = W[expert, col, k].to(fx.Float32)
                        acc = acc + a * b
                    if fx.const_expr(apply_route_weight):
                        acc = acc * fx.Float32(sorted_weights[sorted_row])
                    Out[token, slot, col] = acc.to(elem_dtype)

    return grouped_b16_kernel


__all__ = [
    "B16_DTYPES",
    "GroupedB16Config",
    "build_grouped_b16_kernel",
    "validate_grouped_b16_config",
]
