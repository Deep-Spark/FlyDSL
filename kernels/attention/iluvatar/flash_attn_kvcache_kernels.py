# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Device-kernel assembly for Iluvatar KV-cache FlashAttention."""

import functools
from typing import Optional, Tuple

import flydsl.compiler as flyc
import flydsl.expr as fx

from .flash_attn_decode_mma import build_mma_decode_attention_kernel
from .flash_attn_kvcache_scalar import build_scalar_kvcache_kernels
from .flash_attn_kvcache_split_reduce import (
    build_split_reduce_kernels,
)

ATTN_THREADS = 256
WARP_SIZE = 64


@functools.lru_cache(maxsize=None)
def build_flash_attn_with_kvcache_module(
    *,
    batch_size: int,
    seqlen_q: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seqlen_k: int,
    dtype_str: str = "bf16",
    paged: bool = False,
    page_block_size: int = 16,
    has_rotary: bool = False,
    rotary_cols: int = 0,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    rotary_interleaved: bool = True,
    update_cache: bool = True,
    num_splits: int = 1,
    upstream_cache_layout: bool = False,
    use_mma_decode: bool = False,
    use_pipelined_mma_decode: bool = False,
    use_simt_decode: bool = False,
    simt_k_warps: int = 1,
    mma_block_n: int = 128,
    softmax_scale: Optional[float] = None,
    softcap: float = 0.0,
    run_attention: bool = True,
    cache_strides: tuple[int, int, int, int] | None = None,
):
    if dtype_str not in ("bf16", "f16"):
        raise ValueError(f"dtype_str must be 'bf16' or 'f16', got {dtype_str!r}")
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    if paged and page_block_size % 16 != 0:
        raise ValueError("paged cache block size must be divisible by 16")
    if has_rotary and rotary_cols not in (head_dim, head_dim // 2):
        raise ValueError("rotary_cos/sin last dimension must be head_dim or head_dim // 2")
    if num_splits < 1:
        raise ValueError("num_splits must be >= 1")
    if use_pipelined_mma_decode and (not paged or upstream_cache_layout or page_block_size != 16):
        raise ValueError("pipelined MMA decode requires a 16-token HND paged cache")

    update_cache_kernel, attention_kernel, split_attention_kernel = build_scalar_kvcache_kernels(
        seqlen_q=seqlen_q,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_seqlen_k=max_seqlen_k,
        dtype_str=dtype_str,
        paged=paged,
        page_block_size=page_block_size,
        has_rotary=has_rotary,
        rotary_cols=rotary_cols,
        causal=causal,
        window_size=window_size,
        rotary_interleaved=rotary_interleaved,
        num_splits=num_splits,
        upstream_cache_layout=upstream_cache_layout,
        softmax_scale=softmax_scale,
        softcap=softcap,
        cache_strides=cache_strides if paged else None,
    )
    split_reduce_kernel, pipelined_split_reduce_kernel, reduce_threads, pipelined_reduce_warps = (
        build_split_reduce_kernels(head_dim=head_dim, num_splits=num_splits, dtype_str=dtype_str)
    )
    if use_simt_decode:
        from .flash_attn_decode_simt import build_simt_decode_attention_kernel

        simt_decode_kernel, simt_threads, simt_smem, simt_grid = build_simt_decode_attention_kernel(
            batch_size=batch_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seqlen_k=max_seqlen_k,
            page_block_size=page_block_size,
            num_splits=num_splits,
            k_warps=simt_k_warps,
            cache_strides=cache_strides,
        )
    elif use_pipelined_mma_decode:
        from .flash_attn_decode_mma_pipelined import build_pipelined_mma_decode_attention_kernel

        mma_decode_kernel, mma_threads, mma_smem, mma_grid = build_pipelined_mma_decode_attention_kernel(
            batch_size=batch_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seqlen_k=max_seqlen_k,
            paged=True,
            page_block_size=page_block_size,
            upstream_cache_layout=False,
            num_splits=num_splits,
            block_n=page_block_size,
        )
    elif use_mma_decode:
        mma_decode_kernel, mma_threads, mma_smem, mma_grid = build_mma_decode_attention_kernel(
            batch_size=batch_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seqlen_k=max_seqlen_k,
            paged=paged,
            page_block_size=page_block_size,
            upstream_cache_layout=upstream_cache_layout,
            num_splits=num_splits,
            block_n=mma_block_n,
        )

    if use_simt_decode and not update_cache:

        @flyc.jit
        def launch_simt_decode(
            QWork: fx.Tensor,
            KCache: fx.Tensor,
            VCache: fx.Tensor,
            CacheSeqLens: fx.Tensor,
            BlockTable: fx.Tensor,
            Out: fx.Tensor,
            GroupMax: fx.Tensor,
            GroupSum: fx.Tensor,
            PartialOut: fx.Tensor,
            stream: fx.Stream = fx.Stream(None),
        ):
            simt_decode_kernel(
                QWork, KCache, VCache, CacheSeqLens, BlockTable, Out, GroupMax, GroupSum, PartialOut
            ).launch(
                grid=simt_grid,
                block=(simt_threads, 1, 1),
                smem=simt_smem,
                stream=stream,
            )
            if num_splits > 1:
                if head_dim == 256:
                    split_reduce_kernel(GroupMax, GroupSum, PartialOut, Out).launch(
                        grid=(seqlen_q, batch_size, num_heads),
                        block=(reduce_threads, 1, 1),
                        stream=stream,
                    )
                else:
                    pipelined_split_reduce_kernel(GroupMax, GroupSum, PartialOut, Out).launch(
                        grid=(seqlen_q, batch_size, num_heads),
                        block=(pipelined_reduce_warps * WARP_SIZE, 1, 1),
                        stream=stream,
                    )

        return launch_simt_decode

    @flyc.jit
    def launch_flash_attn_with_kvcache(
        QKV: fx.Tensor,
        QWork: fx.Tensor,
        KCache: fx.Tensor,
        VCache: fx.Tensor,
        CacheSeqLens: fx.Tensor,
        BlockTable: fx.Tensor,
        CacheLeftpad: fx.Tensor,
        RotaryCos: fx.Tensor,
        RotarySin: fx.Tensor,
        Out: fx.Tensor,
        GroupMax: fx.Tensor,
        GroupSum: fx.Tensor,
        PartialOut: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        if update_cache:
            update_cache_kernel(
                QKV,
                QWork,
                KCache,
                VCache,
                CacheSeqLens,
                BlockTable,
                CacheLeftpad,
                RotaryCos,
                RotarySin,
            ).launch(
                grid=(max(num_heads, num_kv_heads), seqlen_q, batch_size),
                block=(max(head_dim, 1), 1, 1),
                stream=stream,
            )
        if run_attention and use_simt_decode:
            simt_decode_kernel(
                QWork, KCache, VCache, CacheSeqLens, BlockTable, Out, GroupMax, GroupSum, PartialOut
            ).launch(
                grid=simt_grid,
                block=(simt_threads, 1, 1),
                smem=simt_smem,
                stream=stream,
            )
            if num_splits > 1:
                if head_dim == 256:
                    split_reduce_kernel(GroupMax, GroupSum, PartialOut, Out).launch(
                        grid=(seqlen_q, batch_size, num_heads),
                        block=(reduce_threads, 1, 1),
                        stream=stream,
                    )
                else:
                    pipelined_split_reduce_kernel(GroupMax, GroupSum, PartialOut, Out).launch(
                        grid=(seqlen_q, batch_size, num_heads),
                        block=(pipelined_reduce_warps * WARP_SIZE, 1, 1),
                        stream=stream,
                    )
        elif run_attention and (use_mma_decode or use_pipelined_mma_decode):
            mma_decode_kernel(
                QWork, KCache, VCache, CacheSeqLens, BlockTable, Out, GroupMax, GroupSum, PartialOut
            ).launch(
                grid=mma_grid,
                block=(mma_threads, 1, 1),
                smem=mma_smem,
                stream=stream,
            )
            if num_splits > 1:
                if use_pipelined_mma_decode:
                    pipelined_split_reduce_kernel(GroupMax, GroupSum, PartialOut, Out).launch(
                        grid=(seqlen_q, batch_size, num_heads),
                        block=(pipelined_reduce_warps * WARP_SIZE, 1, 1),
                        stream=stream,
                    )
                else:
                    split_reduce_kernel(GroupMax, GroupSum, PartialOut, Out).launch(
                        grid=(seqlen_q, batch_size, num_heads),
                        block=(reduce_threads, 1, 1),
                        stream=stream,
                    )
        elif run_attention and num_splits == 1:
            attention_kernel(QWork, KCache, VCache, CacheSeqLens, BlockTable, CacheLeftpad, Out).launch(
                grid=(seqlen_q, batch_size, num_heads),
                block=(ATTN_THREADS, 1, 1),
                stream=stream,
            )
        elif run_attention:
            split_attention_kernel(
                QWork, KCache, VCache, CacheSeqLens, BlockTable, CacheLeftpad, GroupMax, GroupSum, PartialOut
            ).launch(
                grid=(seqlen_q, batch_size, num_heads * num_splits),
                block=(ATTN_THREADS, 1, 1),
                stream=stream,
            )
            split_reduce_kernel(GroupMax, GroupSum, PartialOut, Out).launch(
                grid=(seqlen_q, batch_size, num_heads),
                block=(reduce_threads, 1, 1),
                stream=stream,
            )

    return launch_flash_attn_with_kvcache
