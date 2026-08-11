# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Launch-plan caches and host-side decode planning for KV-cache attention."""

import os
from typing import NamedTuple

import torch

import flydsl.expr as fx

_COMPILED_LAUNCH_CACHE: dict = {}
# Prepared common-case SIMT decode launches. The key contains the complete
# static tensor signature and stream, while tensor addresses stay dynamic.
_SIMT_DECODE_FAST_CACHE: dict = {}
# Reused host-side placeholders (read-only for the kernels) to avoid per-call allocs.
_PLACEHOLDER_CACHE: dict = {}


def get_compiled_launch(key):
    return _COMPILED_LAUNCH_CACHE.get(key)


def store_compiled_launch(key, compiled) -> None:
    _COMPILED_LAUNCH_CACHE[key] = compiled


def get_simt_decode_entry(key):
    return _SIMT_DECODE_FAST_CACHE.get(key)


def store_simt_decode_entry(key, entry) -> None:
    _SIMT_DECODE_FAST_CACHE[key] = entry


def get_placeholder(key):
    return _PLACEHOLDER_CACHE.get(key)


def store_placeholder(key, value) -> None:
    _PLACEHOLDER_CACHE[key] = value


def clear_kvcache_caches() -> None:
    _COMPILED_LAUNCH_CACHE.clear()
    _SIMT_DECODE_FAST_CACHE.clear()
    _PLACEHOLDER_CACHE.clear()


def _kvcache_strict_checks() -> bool:
    """Device-syncing metadata validation. Off by default: each `.item()` can cost tens
    of microseconds and dominates decode latency vs the MMA kernel itself.

    Enable with ``FLYDSL_KVCACHE_STRICT_CHECKS=1`` for debugging / unit tests that
    assert on invalid ``cache_seqlens`` / ``block_table`` values. Negative padded
    ``block_table`` entries also require strict mode so the wrapper can disable MMA.
    """
    return os.environ.get("FLYDSL_KVCACHE_STRICT_CHECKS", "0") == "1"


def _cached_zeros_i32(batch_size: int, device) -> torch.Tensor:
    key = ("zeros_i32", device, batch_size)
    cached = _PLACEHOLDER_CACHE.get(key)
    if cached is None:
        cached = torch.zeros((batch_size,), device=device, dtype=torch.int32)
        _PLACEHOLDER_CACHE[key] = cached
    return cached


def _cached_rotary_dummy(device, dtype: torch.dtype, head_dim: int):
    key = ("rotary", device, dtype, head_dim)
    cached = _PLACEHOLDER_CACHE.get(key)
    if cached is None:
        cached = (
            torch.empty((1, head_dim), device=device, dtype=dtype),
            torch.empty((1, head_dim), device=device, dtype=dtype),
        )
        _PLACEHOLDER_CACHE[key] = cached
    return cached


def _cached_split1_bufs(device):
    key = ("split1", device)
    cached = _PLACEHOLDER_CACHE.get(key)
    if cached is None:
        group_max = torch.empty((1, 1, 1, 1), device=device, dtype=torch.float32)
        group_sum = torch.empty_like(group_max)
        partial_out = torch.empty((1, 1, 1, 1, 1), device=device, dtype=torch.float32)
        cached = (group_max, group_sum, partial_out)
        _PLACEHOLDER_CACHE[key] = cached
    return cached


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.bfloat16:
        return "bf16"
    if dtype is torch.float16:
        return "f16"
    raise TypeError(f"flash_attn_with_kvcache only supports bf16/f16, got {dtype}")


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _simt_decode_fast_key(
    q,
    k_cache,
    v_cache,
    *,
    k,
    v,
    rotary_cos,
    rotary_sin,
    cache_seqlens,
    cache_batch_idx,
    cache_leftpad,
    block_table,
    softmax_scale,
    causal,
    window_size,
    softcap,
    alibi_slopes,
    num_splits,
    return_softmax_lse,
    is_qkv_packed,
    force_upstream_cache_layout,
    stream,
    out,
    max_context_len,
):
    """Return the static signature for the common GQA=2 SIMT decode path."""
    if (
        k is not None
        or v is not None
        or rotary_cos is not None
        or rotary_sin is not None
        or cache_batch_idx is not None
        or cache_leftpad is not None
        or alibi_slopes is not None
        or return_softmax_lse
        or is_qkv_packed
        or force_upstream_cache_layout is not False
        or stream is not None
        or not isinstance(cache_seqlens, torch.Tensor)
        or not isinstance(block_table, torch.Tensor)
        or not isinstance(num_splits, int)
        or window_size != (-1, -1)
        or softcap != 0.0
        or os.environ.get("FLYDSL_KVCACHE_STRICT_CHECKS", "0") == "1"
        or os.environ.get("FLYDSL_KVCACHE_SIMT_DECODE", "1") != "1"
    ):
        return None
    if not q.is_cuda:
        return None
    if max_context_len is not None and (
        isinstance(max_context_len, bool) or not isinstance(max_context_len, int) or max_context_len <= 0
    ):
        return None
    # Match plan_decode_launch's 32-token SIMT bucket so decode steps that
    # grow one token at a time reuse the same compiled launch/workspace.
    bucketed_context_len = (
        None if max_context_len is None else ((int(max_context_len) + 31) // 32) * 32
    )
    current_stream = torch.cuda.current_stream(q.device)
    key = (
        q.device,
        q.dtype,
        tuple(q.shape),
        tuple(q.stride()),
        k_cache.device,
        k_cache.dtype,
        tuple(k_cache.shape),
        tuple(k_cache.stride()),
        v_cache.device,
        v_cache.dtype,
        tuple(v_cache.shape),
        tuple(v_cache.stride()),
        cache_seqlens.device,
        cache_seqlens.dtype,
        tuple(cache_seqlens.shape),
        tuple(cache_seqlens.stride()),
        block_table.device,
        block_table.dtype,
        tuple(block_table.shape),
        tuple(block_table.stride()),
        None if out is None else (out.device, out.dtype, tuple(out.shape), tuple(out.stride())),
        softmax_scale,
        causal,
        num_splits,
        bucketed_context_len,
        current_stream.cuda_stream,
    )
    return key, current_stream


class _SimtDecodeLaunchPlan:
    """Validated SIMT decode launch reused by integration-specific adapters."""

    __slots__ = ("compiled", "cached_out", "group_max", "group_sum", "partial_out", "stream")

    def __init__(self, entry, current_stream):
        self.compiled, self.cached_out, self.group_max, self.group_sum, self.partial_out = entry
        self.stream = fx.Stream(current_stream)

    def __call__(self, q, k_cache, v_cache, cache_seqlens, block_table, out):
        out_buf = self.cached_out if out is None else out
        self.compiled(
            q,
            k_cache,
            v_cache,
            cache_seqlens,
            block_table,
            out_buf,
            self.group_max,
            self.group_sum,
            self.partial_out,
            self.stream,
        )
        return out_buf


def _get_simt_decode_launch_plan(
    q,
    k_cache,
    v_cache,
    cache_seqlens,
    block_table,
    *,
    softmax_scale,
    causal,
    num_splits,
    out,
    max_context_len,
):
    """Return a plan only after the matching public call validated and cached it."""
    probe = _simt_decode_fast_key(
        q,
        k_cache,
        v_cache,
        k=None,
        v=None,
        rotary_cos=None,
        rotary_sin=None,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=None,
        cache_leftpad=None,
        block_table=block_table,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        num_splits=num_splits,
        return_softmax_lse=False,
        is_qkv_packed=False,
        force_upstream_cache_layout=False,
        stream=None,
        out=out,
        max_context_len=max_context_len,
    )
    if probe is None:
        return None
    entry = _SIMT_DECODE_FAST_CACHE.get(probe[0])
    return None if entry is None else _SimtDecodeLaunchPlan(entry, probe[1])


def _cached_split_bufs(device, batch_size: int, seqlen_q: int, num_heads: int, num_splits: int, head_dim: int):
    """Reuse split-KV workspaces; decode writes every element before reading it."""
    key = ("split", device, batch_size, seqlen_q, num_heads, num_splits, head_dim)
    cached = _PLACEHOLDER_CACHE.get(key)
    if cached is None:
        group_max = torch.empty(
            (batch_size, seqlen_q, num_heads, num_splits),
            device=device,
            dtype=torch.float32,
        )
        group_sum = torch.empty_like(group_max)
        partial_out = torch.empty(
            (batch_size, seqlen_q, num_heads, num_splits, head_dim),
            device=device,
            dtype=torch.float32,
        )
        cached = (group_max, group_sum, partial_out)
        _PLACEHOLDER_CACHE[key] = cached
    return cached


def select_decode_backends(
    *,
    has_cache_leftpad: bool,
    softmax_scale: float,
    default_softmax_scale: float,
    softcap: float,
    head_dim: int,
    dtype: torch.dtype,
    seqlen_q: int,
    num_heads: int,
    num_kv_heads: int,
    kernel_paged: bool,
    has_padded_block_table: bool,
    upstream_cache_layout: bool,
    page_block_size: int,
    max_seqlen_k: int,
    window_size,
    k_cache_contiguous: bool,
    v_cache_contiguous: bool,
):
    """Choose SIMT, pipelined HND MMA, and baseline MMA decode eligibility."""
    mma_decode_enabled = os.environ.get("FLYDSL_KVCACHE_MMA_DECODE", "1") == "1"
    simt_decode_enabled = os.environ.get("FLYDSL_KVCACHE_SIMT_DECODE", "1") == "1"
    mma_block_n = int(os.environ.get("FLYDSL_KVCACHE_MMA_BN", "32"))
    common = (
        not has_cache_leftpad
        and softmax_scale == default_softmax_scale
        and softcap == 0.0
        and dtype is torch.bfloat16
        and seqlen_q == 1
        and not has_padded_block_table
        and window_size == (-1, -1)
    )
    contiguous_cache = k_cache_contiguous and v_cache_contiguous
    d128_gqa2 = (
        head_dim == 128
        and num_heads == num_kv_heads * 2
        and page_block_size == 16
        and max_seqlen_k >= 1024
    )
    d256_gqa4 = (
        head_dim == 256
        and num_heads == num_kv_heads * 4
        and page_block_size % 16 == 0
    )
    use_simt_decode = (
        simt_decode_enabled
        and common
        and (contiguous_cache or d256_gqa4)
        and (d128_gqa2 or d256_gqa4)
        and kernel_paged
        and not upstream_cache_layout
        and max_seqlen_k % 16 == 0
    )
    use_pipelined_mma_decode = (
        not use_simt_decode
        and mma_decode_enabled
        and common
        and contiguous_cache
        and 1 <= (num_heads // num_kv_heads) <= 16
        and kernel_paged
        and not upstream_cache_layout
        and page_block_size == 16
        and max_seqlen_k >= 1024
        and max_seqlen_k % 16 == 0
    )
    use_mma_decode = (
        not use_simt_decode
        and not use_pipelined_mma_decode
        and mma_decode_enabled
        and common
        and contiguous_cache
        and 1 <= (num_heads // num_kv_heads) <= 16
        and max_seqlen_k % 128 == 0
    )
    return use_simt_decode, use_pipelined_mma_decode, use_mma_decode, mma_block_n


class DecodeLaunchPlan(NamedTuple):
    use_simt_decode: bool
    use_pipelined_mma_decode: bool
    use_mma_decode: bool
    mma_block_n: int
    planning_max_seqlen_k: int
    effective_num_splits: int
    simt_k_warps: int


def plan_decode_launch(
    *,
    has_cache_leftpad: bool,
    softmax_scale: float,
    default_softmax_scale: float,
    softcap: float,
    head_dim: int,
    dtype: torch.dtype,
    batch_size: int,
    seqlen_q: int,
    num_heads: int,
    num_kv_heads: int,
    num_splits: int,
    kernel_paged: bool,
    has_padded_block_table: bool,
    upstream_cache_layout: bool,
    page_block_size: int,
    max_seqlen_k: int,
    planning_max_seqlen_k: int,
    max_context_len,
    window_size,
    k_cache_contiguous: bool,
    v_cache_contiguous: bool,
) -> DecodeLaunchPlan:
    """Choose the decode backend, context bucket, and split configuration."""
    use_simt_decode, use_pipelined_mma_decode, use_mma_decode, mma_block_n = select_decode_backends(
        has_cache_leftpad=has_cache_leftpad,
        softmax_scale=softmax_scale,
        default_softmax_scale=default_softmax_scale,
        softcap=softcap,
        head_dim=head_dim,
        dtype=dtype,
        seqlen_q=seqlen_q,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        kernel_paged=kernel_paged,
        has_padded_block_table=has_padded_block_table,
        upstream_cache_layout=upstream_cache_layout,
        page_block_size=page_block_size,
        max_seqlen_k=max_seqlen_k,
        window_size=window_size,
        k_cache_contiguous=k_cache_contiguous,
        v_cache_contiguous=v_cache_contiguous,
    )
    if use_simt_decode and max_context_len is not None:
        # Plan in 32-token work units so 513/1025-token steps are not bucketed
        # into their 1024/2048 power-of-two capacity sizes.
        planning_max_seqlen_k = min(
            max_seqlen_k,
            ((planning_max_seqlen_k + 31) // 32) * 32,
        )
    elif use_pipelined_mma_decode and max_context_len is not None:
        planning_max_seqlen_k = min(
            max_seqlen_k,
            1 << (planning_max_seqlen_k - 1).bit_length(),
        )

    simt_k_warps = 1
    if use_simt_decode:
        from kernels.attention.iluvatar.mma_decode_splits import compute_qwen_simt_decode_config

        planned_splits, simt_k_warps = compute_qwen_simt_decode_config(
            batch_size=batch_size,
            seqlen_q=seqlen_q,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            max_seqlen_k=planning_max_seqlen_k,
        )
        if num_splits == 0:
            effective_num_splits = planned_splits
        elif num_splits == 1:
            effective_num_splits = 1
            pages = _ceil_div(planning_max_seqlen_k, page_block_size)
            simt_k_warps = min(16, 1 << ((pages - 1).bit_length() - 1))
        else:
            effective_num_splits = min(num_splits, planning_max_seqlen_k // page_block_size)
            simt_k_warps = 1
    elif use_pipelined_mma_decode:
        from kernels.attention.iluvatar.mma_decode_splits import compute_pipelined_mma_decode_config

        if num_splits == 0:
            effective_num_splits, _, _ = compute_pipelined_mma_decode_config(
                batch_size=batch_size,
                seqlen_q=seqlen_q,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                max_seqlen_k=planning_max_seqlen_k,
            )
            active_kv_tiles = _ceil_div(planning_max_seqlen_k, page_block_size)
            effective_num_splits = min(effective_num_splits, max(1, active_kv_tiles // 2))
            if batch_size * seqlen_q * num_kv_heads <= 8 and active_kv_tiles <= 32:
                effective_num_splits = 1
        else:
            effective_num_splits = min(max(1, num_splits), planning_max_seqlen_k // page_block_size)
    elif use_mma_decode:
        from kernels.attention.iluvatar.mma_decode_splits import compute_mma_decode_num_splits

        if num_splits == 0:
            effective_num_splits = compute_mma_decode_num_splits(
                batch_size=batch_size,
                seqlen_q=seqlen_q,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                max_seqlen_k=planning_max_seqlen_k,
                block_n=mma_block_n,
            )
        elif num_splits >= 2:
            effective_num_splits = min(num_splits, planning_max_seqlen_k // mma_block_n)
            while effective_num_splits > 1 and (planning_max_seqlen_k // mma_block_n) % effective_num_splits != 0:
                effective_num_splits -= 1
        else:
            effective_num_splits = 1
    elif num_splits == 0:
        low_parallelism = batch_size * seqlen_q * num_heads < 64
        effective_num_splits = 8 if planning_max_seqlen_k >= 2048 else 4
        if seqlen_q != 1 or planning_max_seqlen_k < 1024 or not low_parallelism:
            effective_num_splits = 1
    else:
        effective_num_splits = num_splits

    return DecodeLaunchPlan(
        use_simt_decode=use_simt_decode,
        use_pipelined_mma_decode=use_pipelined_mma_decode,
        use_mma_decode=use_mma_decode,
        mma_block_n=mma_block_n,
        planning_max_seqlen_k=planning_max_seqlen_k,
        effective_num_splits=max(1, min(effective_num_splits, max_seqlen_k)),
        simt_k_warps=simt_k_warps,
    )
