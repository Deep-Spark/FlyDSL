# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Ixinfer V5-layout BF16 paged decode fast path.

This path deliberately has a narrow contract: one decode token, head size 128,
HND cache layout, and at most 16 query heads per KV head.  Its sequence tile is
one 16-token page and its K/V staging uses the two-buffer pipeline shared with
the experimental pipelined MMA kernel.
"""

from kernels.attention.iluvatar.flash_attn_kvcache_mma_pipe import (
    build_mma_decode_attention_kernel,
)

V5_CHUNK = 16


def build_v5_decode_attention_kernel(
    *,
    batch_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seqlen_k: int,
    page_block_size: int,
    num_groups: int,
):
    """Build a single-wave, double-buffered HND V5 decode kernel."""
    if head_dim != 128:
        raise ValueError("V5 decode requires head_dim == 128")
    if num_heads % num_kv_heads != 0 or not 1 <= num_heads // num_kv_heads <= 16:
        raise ValueError("V5 decode requires 1 <= num_heads / num_kv_heads <= 16")
    if page_block_size != V5_CHUNK:
        raise ValueError("V5 decode currently requires page_block_size == 16")
    if max_seqlen_k % V5_CHUNK:
        raise ValueError("V5 decode requires a 16-aligned maximum context length")

    return build_mma_decode_attention_kernel(
        batch_size=batch_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_seqlen_k=max_seqlen_k,
        paged=True,
        page_block_size=page_block_size,
        upstream_cache_layout=False,
        num_splits=num_groups,
        block_n=V5_CHUNK,
    )
