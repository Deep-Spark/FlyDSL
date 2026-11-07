# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Launch-plan caches and host-side decode planning for KV-cache attention."""

import os
from collections import OrderedDict
from typing import NamedTuple

import torch

import flydsl.expr as fx

_MAX_COMPILED_LAUNCHES = 256
_MAX_SIMT_FAST_ENTRIES = 512
_MAX_WORKSPACE_ENTRIES_PER_DEVICE = 32
_COMPILED_LAUNCH_CACHE: OrderedDict = OrderedDict()
# Prepared common-case SIMT decode launches. The key contains the complete
# static tensor signature and stream, while tensor addresses stay dynamic.
_SIMT_DECODE_FAST_CACHE: OrderedDict = OrderedDict()
# Reused host-side placeholders (read-only for the kernels) to avoid per-call allocs.
_PLACEHOLDER_CACHE: dict = {}


def _should_cache_workspace(key, device) -> bool:
    if key in _PLACEHOLDER_CACHE:
        return True
    entries = sum(
        1
        for cached_key in _PLACEHOLDER_CACHE
        if cached_key[0] in ("split", "split1")
        and len(cached_key) > 1
        and cached_key[1] == device
    )
    return entries < _MAX_WORKSPACE_ENTRIES_PER_DEVICE


def get_compiled_launch(key):
    value = _COMPILED_LAUNCH_CACHE.get(key)
    if value is not None:
        _COMPILED_LAUNCH_CACHE.move_to_end(key)
    return value


def store_compiled_launch(key, compiled) -> None:
    _COMPILED_LAUNCH_CACHE[key] = compiled
    _COMPILED_LAUNCH_CACHE.move_to_end(key)
    while len(_COMPILED_LAUNCH_CACHE) > _MAX_COMPILED_LAUNCHES:
        _COMPILED_LAUNCH_CACHE.popitem(last=False)


def get_simt_decode_entry(key):
    value = _SIMT_DECODE_FAST_CACHE.get(key)
    if value is not None:
        _SIMT_DECODE_FAST_CACHE.move_to_end(key)
    return value


def store_simt_decode_entry(key, entry) -> None:
    _SIMT_DECODE_FAST_CACHE[key] = entry
    _SIMT_DECODE_FAST_CACHE.move_to_end(key)
    while len(_SIMT_DECODE_FAST_CACHE) > _MAX_SIMT_FAST_ENTRIES:
        _SIMT_DECODE_FAST_CACHE.popitem(last=False)


def get_placeholder(key):
    return _PLACEHOLDER_CACHE.get(key)


def store_placeholder(key, value) -> None:
    _PLACEHOLDER_CACHE[key] = value


def clear_kvcache_caches(*, include_builders: bool = False) -> None:
    """Clear runtime state; optionally release cached kernel-builder modules.

    Builder modules are retained by default because some Iluvatar compiler
    versions are not safe when modules are repeatedly destroyed and rebuilt in
    one process. Their LRU is bounded independently.
    """
    _COMPILED_LAUNCH_CACHE.clear()
    _SIMT_DECODE_FAST_CACHE.clear()
    _PLACEHOLDER_CACHE.clear()
    from .flash_attn_prefill import clear_prefill_caches

    clear_prefill_caches()
    if include_builders:
        from .flash_attn_kvcache_kernels import build_flash_attn_with_kvcache_module

        build_flash_attn_with_kvcache_module.cache_clear()


def _kvcache_strict_checks() -> bool:
    """Device-syncing metadata validation. Off by default: each `.item()` can cost tens
    of microseconds and dominates decode latency vs the MMA kernel itself.

    Enable with ``FLYDSL_KVCACHE_STRICT_CHECKS=1`` for debugging / unit tests that
    assert on invalid ``cache_seqlens`` / ``block_table`` values. Negative padded
    ``block_table`` entries also require strict mode so the wrapper can disable MMA.
    """
    return os.environ.get("FLYDSL_KVCACHE_STRICT_CHECKS", "0") == "1"


def _cached_zeros_i32(batch_size: int, device) -> torch.Tensor:
    key = ("zeros_i32", device)
    cached = _PLACEHOLDER_CACHE.get(key)
    if cached is None or cached.shape[0] < batch_size:
        cap = batch_size if cached is None else max(batch_size, cached.shape[0])
        cached = torch.zeros((cap,), device=device, dtype=torch.int32)
        _PLACEHOLDER_CACHE[key] = cached
    return cached if cached.shape[0] == batch_size else cached[:batch_size]


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


def _cached_split1_bufs(device, stream_ptr: int):
    key = ("split1", device, stream_ptr)
    cached = _PLACEHOLDER_CACHE.get(key)
    if cached is None:
        group_max = torch.empty((1, 1, 1, 1), device=device, dtype=torch.float32)
        group_sum = torch.empty_like(group_max)
        partial_out = torch.empty((1, 1, 1, 1, 1), device=device, dtype=torch.float32)
        cached = (group_max, group_sum, partial_out)
        if _should_cache_workspace(key, device):
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


def _simt_occupancy_bucket(batch_size: int) -> int:
    """Batch class that changes when D256 auto split count would change."""
    if batch_size >= 16:
        return 16
    if batch_size >= 8:
        return 8
    if batch_size >= 4:
        return 4
    return 1


def _simt_fast_schedule(q, k_cache, max_context_len) -> tuple[int, int]:
    """Return ``(groups, k_warps)`` for fast/pin keys, matching the planner.

    Context length is not a compile specialization; only the split/k-warp
    schedule is. Growing decode by one token therefore reuses the launch.
    """
    from kernels.attention.iluvatar.mma_decode_splits import compute_qwen_simt_decode_config

    page_block_size = int(k_cache.shape[2])
    capacity = int(k_cache.shape[0]) * page_block_size
    if max_context_len is None:
        planned_k = capacity
    else:
        planned_k = min(int(max_context_len), capacity)
        # Same 32-token work unit as plan_decode_launch, so the key's
        # (groups, k_warps) matches the compiled split count.
        planned_k = ((planned_k + 31) // 32) * 32
    planned_k = max(16, planned_k)
    return compute_qwen_simt_decode_config(
        batch_size=int(q.shape[0]),
        seqlen_q=int(q.shape[1]),
        num_heads=int(q.shape[2]),
        num_kv_heads=int(k_cache.shape[1]),
        head_dim=int(q.shape[-1]),
        max_seqlen_k=planned_k,
    )


def _simt_runtime_metadata_valid(
    q,
    k_cache,
    v_cache,
    cache_seqlens,
    block_table,
    out,
) -> bool:
    """Cheap host-only guards required before a dynamic-batch fast launch."""
    if (
        not isinstance(q, torch.Tensor)
        or not isinstance(k_cache, torch.Tensor)
        or not isinstance(v_cache, torch.Tensor)
        or not isinstance(cache_seqlens, torch.Tensor)
        or not isinstance(block_table, torch.Tensor)
        or q.ndim != 4
        or k_cache.ndim != 4
        or v_cache.ndim != 4
        or cache_seqlens.ndim != 1
        or block_table.ndim != 2
    ):
        return False
    batch_size = q.shape[0]
    if cache_seqlens.shape[0] != batch_size or block_table.shape[0] != batch_size:
        return False
    if out is not None and (
        not isinstance(out, torch.Tensor) or tuple(out.shape) != tuple(q.shape)
    ):
        return False
    tensors = (q, k_cache, v_cache, cache_seqlens, block_table)
    if any(t.device != q.device for t in tensors):
        return False
    if q.dtype != torch.bfloat16 or k_cache.dtype != q.dtype or v_cache.dtype != q.dtype:
        return False
    if cache_seqlens.dtype != torch.int32 or block_table.dtype != torch.int32:
        return False
    if (
        not q.is_contiguous()
        or not k_cache.is_contiguous()
        or not v_cache.is_contiguous()
        or not cache_seqlens.is_contiguous()
        or block_table.stride(1) != 1
    ):
        return False
    if tuple(k_cache.shape) != tuple(v_cache.shape):
        return False
    if out is not None and (
        out.device != q.device
        or out.dtype != q.dtype
        or not out.is_contiguous()
    ):
        return False
    return True


def _simt_plan_signature(q, k_cache, v_cache, block_table):
    return (
        q.device,
        q.dtype,
        tuple(q.shape[1:]),
        tuple(q.stride()[1:]),
        tuple(k_cache.shape),
        tuple(k_cache.stride()),
        tuple(v_cache.shape),
        tuple(v_cache.stride()),
        tuple(block_table.shape[1:]),
        tuple(block_table.stride()),
    )


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
        or not isinstance(num_splits, int)
        or window_size != (-1, -1)
        or softcap != 0.0
        or os.environ.get("FLYDSL_KVCACHE_STRICT_CHECKS", "0") == "1"
        or os.environ.get("FLYDSL_KVCACHE_SIMT_DECODE", "1") != "1"
    ):
        return None
    if not _simt_runtime_metadata_valid(
        q, k_cache, v_cache, cache_seqlens, block_table, out
    ):
        return None
    if not q.is_cuda:
        return None
    if max_context_len is not None and (
        isinstance(max_context_len, bool) or not isinstance(max_context_len, int) or max_context_len <= 0
    ):
        return None
    current_stream = torch.cuda.current_stream(q.device)
    # Batch is a launch-grid / workspace extent, not a kernel specialization.
    # Keep inner shapes and all strides so layout changes still miss.
    key = (
        q.device,
        q.dtype,
        tuple(q.shape[1:]),
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
        tuple(cache_seqlens.stride()),
        block_table.device,
        block_table.dtype,
        tuple(block_table.shape[1:]),
        tuple(block_table.stride()),
        None
        if out is None
        else (out.device, out.dtype, tuple(out.shape[1:]), tuple(out.stride())),
        softmax_scale,
        causal,
        num_splits,
        _simt_occupancy_bucket(q.shape[0]),
        _simt_fast_schedule(q, k_cache, max_context_len),
    )
    return key, current_stream


class _SimtDecodeLaunchPlan:
    """Validated SIMT decode launch reused by integration-specific adapters."""

    __slots__ = (
        "compiled",
        "num_splits",
        "seqlen_q",
        "num_heads",
        "head_dim",
        "stream",
        "stream_ptr",
        "signature",
    )

    def __init__(self, entry, current_stream, q, k_cache, v_cache, block_table):
        self.compiled, self.num_splits, self.seqlen_q, self.num_heads, self.head_dim = entry
        self.stream = fx.Stream(current_stream)
        self.stream_ptr = current_stream.cuda_stream
        self.signature = _simt_plan_signature(q, k_cache, v_cache, block_table)

    def _split_bufs(self, q):
        if self.num_splits > 1:
            return _cached_split_bufs(
                q.device,
                q.shape[0],
                self.seqlen_q,
                self.num_heads,
                self.num_splits,
                self.head_dim,
                torch.cuda.current_stream(q.device).cuda_stream,
            )
        return _cached_split1_bufs(q.device, torch.cuda.current_stream(q.device).cuda_stream)

    def __call__(self, q, k_cache, v_cache, cache_seqlens, block_table, out):
        if not _simt_runtime_metadata_valid(
            q, k_cache, v_cache, cache_seqlens, block_table, out
        ):
            raise ValueError(
                "SIMT decode plan requires cache_seqlens/block_table/out "
                "leading dimensions to match q batch"
            )
        if _simt_plan_signature(q, k_cache, v_cache, block_table) != self.signature:
            raise ValueError("SIMT decode plan was called with incompatible tensor metadata")
        # Returned tensors must have ordinary PyTorch ownership semantics.  In
        # particular, a later call through this cached launch plan must not
        # overwrite an output retained by its caller.
        out_buf = torch.empty(q.shape, device=q.device, dtype=q.dtype) if out is None else out
        group_max, group_sum, partial_out = self._split_bufs(q)
        current_stream = torch.cuda.current_stream(q.device)
        launch_stream = (
            self.stream
            if current_stream.cuda_stream == self.stream_ptr
            else fx.Stream(current_stream)
        )
        self.compiled(
            q,
            k_cache,
            v_cache,
            cache_seqlens,
            block_table,
            out_buf,
            group_max,
            group_sum,
            partial_out,
            launch_stream,
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
    entry = get_simt_decode_entry(probe[0])
    if entry is None or len(entry) != 5:
        return None
    return _SimtDecodeLaunchPlan(
        entry, probe[1], q, k_cache, v_cache, block_table
    )


def _cached_split_bufs(
    device,
    batch_size: int,
    seqlen_q: int,
    num_heads: int,
    num_splits: int,
    head_dim: int,
    stream_ptr: int,
):
    """Reuse split-KV workspaces; decode writes every element before reading it.

    Capacity grows with the largest batch seen on this stream so mixed-batch
    decode does not reallocate or specialize on the current ``B``.
    """
    key = ("split", device, stream_ptr, seqlen_q, num_heads, num_splits, head_dim)
    cached = _PLACEHOLDER_CACHE.get(key)
    if cached is None or cached[0].shape[0] < batch_size:
        cap = batch_size if cached is None else max(batch_size, cached[0].shape[0])
        group_max = torch.empty(
            (cap, seqlen_q, num_heads, num_splits),
            device=device,
            dtype=torch.float32,
        )
        group_sum = torch.empty_like(group_max)
        partial_out = torch.empty(
            (cap, seqlen_q, num_heads, num_splits, head_dim),
            device=device,
            dtype=torch.float32,
        )
        cached = (group_max, group_sum, partial_out)
        if _should_cache_workspace(key, device):
            _PLACEHOLDER_CACHE[key] = cached
    if cached[0].shape[0] == batch_size:
        return cached
    return tuple(tensor[:batch_size] for tensor in cached)


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
    mma_block_n = 32
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
        and head_dim == 128
        and 1 <= (num_heads // num_kv_heads) <= 16
        and kernel_paged
        and not upstream_cache_layout
        and page_block_size == 16
        and max_seqlen_k >= 1024
        and max_seqlen_k % 16 == 0
    )
    mma_candidate = (
        not use_simt_decode
        and not use_pipelined_mma_decode
        and mma_decode_enabled
        and common
        and contiguous_cache
        and head_dim == 128
        and 1 <= (num_heads // num_kv_heads) <= 16
    )
    use_mma_decode = False
    if mma_candidate:
        raw_mma_block_n = os.environ.get("FLYDSL_KVCACHE_MMA_BN", "32")
        try:
            mma_block_n = int(raw_mma_block_n)
        except (TypeError, ValueError) as exc:
            raise ValueError("FLYDSL_KVCACHE_MMA_BN must be one of 16, 32, 64, or 128") from exc
        if mma_block_n not in (16, 32, 64, 128):
            raise ValueError("FLYDSL_KVCACHE_MMA_BN must be one of 16, 32, 64, or 128")
        use_mma_decode = max_seqlen_k % mma_block_n == 0
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
            if head_dim != 256:
                pages = _ceil_div(planning_max_seqlen_k, page_block_size)
                simt_k_warps = min(16, 1 << max(0, (pages - 1).bit_length() - 1))
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
