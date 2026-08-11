# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Correctness-first FlyDSL ``flash_attn_with_kvcache`` implementation.

This module mirrors the MR flash-attention KV-cache contract closely enough for
native FlyDSL testing:

* upstream input:   q [B, S_q, H_q, D], optional k/v [B, S_new, H_kv, D]
* upstream cache:   dense [B, S_cache, H_kv, D] or paged [num_blocks, page, H_kv, D]
* MR input:         packed QKV [B, S_q, H_q + 2 * H_kv, D]
* MR cache:         dense [B, H_kv, S_cache, D] or paged [num_blocks, H_kv, page, D]
* cache_seqlens follows upstream when k/v are separate (old lengths) and MR when packed
  (visible lengths after the packed K/V tokens are present).

The kernels are intentionally simple.  A first launch rotates/copies Q and
updates K/V cache; a second launch computes one attention row per CTA.
"""

import math
from typing import Optional, Tuple

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

from .flash_attn_kvcache_kernels import (
    build_flash_attn_with_kvcache_module,
)
from .flash_attn_kvcache_planner import (
    _cached_rotary_dummy,
    _cached_split1_bufs,
    _cached_split_bufs,
    _cached_zeros_i32,
    _dtype_name,
    _kvcache_strict_checks,
    _simt_decode_fast_key,
    get_compiled_launch,
    get_placeholder,
    plan_decode_launch,
    store_compiled_launch,
    store_placeholder,
    store_simt_decode_entry,
)
from .flash_attn_paged_kv import resolve_kv_cache_layout, validate_block_table_shape


def _apply_rotary_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    interleaved: bool,
):
    cos_pos = cos[positions.long()].unsqueeze(-2).to(dtype=x.dtype, device=x.device)
    sin_pos = sin[positions.long()].unsqueeze(-2).to(dtype=x.dtype, device=x.device)
    if not interleaved:
        half = x.shape[-1] // 2
        x_first, x_second = x[..., :half], x[..., half:]
        if cos_pos.shape[-1] == x.shape[-1]:
            cos_first, cos_second = cos_pos[..., :half], cos_pos[..., half:]
            sin_first, sin_second = sin_pos[..., :half], sin_pos[..., half:]
        else:
            cos_first = cos_second = cos_pos
            sin_first = sin_second = sin_pos
        return torch.cat(
            [
                x_first * cos_first - x_second * sin_first,
                x_second * cos_second + x_first * sin_second,
            ],
            dim=-1,
        )
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    if cos_pos.shape[-1] == x.shape[-1]:
        cos_even, cos_odd = cos_pos[..., 0::2], cos_pos[..., 1::2]
        sin_even, sin_odd = sin_pos[..., 0::2], sin_pos[..., 1::2]
    else:
        cos_even = cos_odd = cos_pos
        sin_even = sin_odd = sin_pos
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos_even - x_odd * sin_even
    out[..., 1::2] = x_odd * cos_odd + x_even * sin_odd
    return out


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[torch.Tensor | int] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    alibi_slopes=None,
    num_splits: int = 0,
    return_softmax_lse: bool = False,
    alibi_mode: int = 0,
    is_qkv_packed: bool = False,
    force_upstream_cache_layout: Optional[bool] = None,
    stream=None,
    out: Optional[torch.Tensor] = None,
    max_context_len: Optional[int] = None,
):
    """FlyDSL-native forward-only KV-cache attention.

    Unsupported optional arguments are rejected instead of silently falling
    back, so callers can see exactly which MR-compatible subset is active.

    ``force_upstream_cache_layout`` overrides the cache-layout inference: by
    default the layout follows ``is_qkv_packed`` (packed -> MR/HND, separate ->
    upstream/NHD).  Pass ``True`` to force NHD ``[blocks, page, Hkv, D]`` or
    ``False`` to force HND ``[blocks, Hkv, page, D]`` regardless of packing.

    ``max_context_len`` is an optional host-known maximum visible KV length for
    this batch.  It only plans the decode split count; cache capacity remains
    the compilation and addressing bound.  vLLM supplies this value without a
    device synchronization.

    Device-syncing metadata checks (``cache_seqlens.aminmax``, referenced
    ``block_table`` gather, padded-entry scan) are off by default because they
    dominate decode latency. Set ``FLYDSL_KVCACHE_STRICT_CHECKS=1`` to enable.

    If ``out`` is provided it must be ``[B, S_q, H_q, D]`` with the same device
    and dtype as ``q``; the kernel writes into it in-place.
    """
    fast_out_was_none = out is None
    fast_probe = _simt_decode_fast_key(
        q,
        k_cache,
        v_cache,
        k=k,
        v=v,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=cache_batch_idx,
        cache_leftpad=cache_leftpad,
        block_table=block_table,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        num_splits=num_splits,
        return_softmax_lse=return_softmax_lse,
        is_qkv_packed=is_qkv_packed,
        force_upstream_cache_layout=force_upstream_cache_layout,
        stream=stream,
        out=out,
        max_context_len=max_context_len,
    )
    if fast_probe is not None:
        # Prefer the launch-plan helper so the cached Stream object is reused
        # instead of reconstructing fx.Stream on every decode step.
        from .flash_attn_kvcache_planner import _get_simt_decode_launch_plan

        fast_plan = _get_simt_decode_launch_plan(
            q,
            k_cache,
            v_cache,
            cache_seqlens,
            block_table,
            softmax_scale=softmax_scale,
            causal=causal,
            num_splits=num_splits,
            out=out,
            max_context_len=max_context_len,
        )
        if fast_plan is not None:
            return fast_plan(q, k_cache, v_cache, cache_seqlens, block_table, out)

    del alibi_mode
    out_buf = out
    if (k is None) != (v is None):
        raise ValueError("k and v must be provided together")
    if alibi_slopes is not None:
        raise NotImplementedError("ALiBi is not supported in the FlyDSL native path")
    if return_softmax_lse:
        raise NotImplementedError("return_softmax_lse is not supported in the FlyDSL native path")
    try:
        softcap = float(softcap)
    except (TypeError, ValueError) as exc:
        raise ValueError("softcap must be a finite non-negative scalar") from exc
    if not math.isfinite(softcap) or softcap < 0.0:
        raise ValueError("softcap must be a finite non-negative scalar")
    if num_splits < 0:
        raise ValueError("num_splits must be >= 0")
    if q.ndim != 4:
        raise ValueError("q must be 4D")
    if k_cache.ndim != 4 or v_cache.ndim != 4:
        raise ValueError("k_cache and v_cache must be 4D tensors")
    if q.device != k_cache.device or q.device != v_cache.device:
        raise ValueError("q, k_cache, and v_cache must be on the same device")
    if q.dtype != k_cache.dtype or q.dtype != v_cache.dtype:
        raise ValueError("q, k_cache, and v_cache must have the same dtype")
    if q.stride(-1) != 1 or k_cache.stride(-1) != 1 or v_cache.stride(-1) != 1:
        raise ValueError("q, k_cache, and v_cache must have contiguous last dimension")
    current_stream = fast_probe[1] if fast_probe is not None else torch.cuda.current_stream(q.device)
    if stream is None:
        stream = fx.Stream(current_stream)
    else:
        raw_stream = stream.value if isinstance(stream, fx.Stream) else stream
        if raw_stream is None:
            raw_stream_ptr = torch.cuda.default_stream(q.device).cuda_stream
        else:
            raw_stream_ptr = raw_stream if isinstance(raw_stream, int) else getattr(raw_stream, "cuda_stream", None)
        if raw_stream_ptr != current_stream.cuda_stream:
            raise NotImplementedError(
                "custom stream must be the current PyTorch stream so wrapper tensor operations stay ordered"
            )

    batch_size, seqlen_q = q.shape[0], q.shape[1]
    head_dim = q.shape[-1]
    cache_row_indices = None
    if cache_batch_idx is not None:
        if block_table is not None:
            raise NotImplementedError("cache_batch_idx is only supported with dense caches")
        if cache_batch_idx.shape != (batch_size,) or cache_batch_idx.dtype not in (torch.int32, torch.int64):
            raise ValueError("cache_batch_idx must be a [B] int32 or int64 tensor")
        if cache_batch_idx.device != q.device:
            raise ValueError("cache_batch_idx must be on the same device as q")
        cache_row_indices = cache_batch_idx.contiguous()
        if _kvcache_strict_checks():
            cache_batch_min, cache_batch_max = cache_row_indices.aminmax()
            if cache_batch_min.item() < 0 or cache_batch_max.item() >= k_cache.shape[0]:
                raise ValueError("cache_batch_idx entries must index dense cache rows")
        if isinstance(cache_seqlens, torch.Tensor):
            if cache_seqlens.shape != (k_cache.shape[0],) or cache_seqlens.device != q.device:
                raise ValueError("cache_seqlens must be a [cache_batch] tensor on the same device as q")
            cache_seqlens = cache_seqlens.index_select(0, cache_row_indices)
    default_softmax_scale = head_dim ** (-0.5)
    if softmax_scale is None:
        softmax_scale = default_softmax_scale
    else:
        try:
            softmax_scale = float(softmax_scale)
        except (TypeError, ValueError) as exc:
            raise ValueError("softmax_scale must be a finite scalar") from exc
        if not math.isfinite(softmax_scale):
            raise ValueError("softmax_scale must be a finite scalar")
    strict_checks = _kvcache_strict_checks()
    if cache_leftpad is None:
        cache_leftpad_for_kernel = _cached_zeros_i32(batch_size, q.device)
    else:
        if block_table is not None:
            raise NotImplementedError("cache_leftpad is only supported with dense caches")
        if cache_seqlens is None:
            raise ValueError("cache_seqlens is required with cache_leftpad")
        if cache_leftpad.shape != (batch_size,) or cache_leftpad.dtype != torch.int32:
            raise ValueError("cache_leftpad must be a [B] int32 tensor")
        if cache_leftpad.device != q.device:
            raise ValueError("cache_leftpad must be on the same device as q")
        if strict_checks and cache_leftpad.amin().item() < 0:
            raise ValueError("cache_leftpad must be non-negative")
        cache_leftpad_for_kernel = cache_leftpad.contiguous()

    update_cache = True
    upstream_cache_layout = resolve_kv_cache_layout(
        default_nhd=not is_qkv_packed,
        force_nhd=force_upstream_cache_layout,
    )
    if is_qkv_packed:
        if k is not None or v is not None:
            raise NotImplementedError("Separate k/v inputs are not supported when is_qkv_packed=True")
        num_kv_heads = k_cache.shape[2] if upstream_cache_layout else k_cache.shape[1]
        num_heads = q.shape[2] - 2 * num_kv_heads
        qkv = q.contiguous()
        k_cache_view = k_cache if cache_row_indices is None else k_cache.index_select(0, cache_row_indices)
        v_cache_view = v_cache if cache_row_indices is None else v_cache.index_select(0, cache_row_indices)
        cache_len_delta = 0
    else:
        num_heads = q.shape[2]
        if k is None:
            update_cache = False
            # NHD [blocks/B, page/S, Hkv, D] -> Hkv at dim 2;
            # HND [blocks/B, Hkv, page/S, D] -> Hkv at dim 1.
            num_kv_heads = k_cache.shape[2] if upstream_cache_layout else k_cache.shape[1]
            qkv = q.contiguous()
            cache_len_delta = 0
        else:
            if k.ndim != 4 or v.ndim != 4:
                raise ValueError("k and v must be [B, S_new, H_kv, D]")
            if k.shape != v.shape:
                raise ValueError("k and v must have matching shapes")
            if k.shape[0] != batch_size or k.shape[1] != seqlen_q or k.shape[-1] != head_dim:
                raise NotImplementedError("FlyDSL currently requires k/v to have [B, S_q, H_kv, D]")
            if k.dtype != q.dtype or v.dtype != q.dtype or k.device != q.device or v.device != q.device:
                raise ValueError("q, k, and v must have matching dtype and device")
            num_kv_heads = k.shape[2]
            qkv = torch.empty(
                (batch_size, seqlen_q, num_heads + 2 * num_kv_heads, head_dim),
                device=q.device,
                dtype=q.dtype,
            )
            qkv[:, :, :num_heads, :].copy_(q)
            qkv[:, :, num_heads : num_heads + num_kv_heads, :].copy_(k)
            qkv[:, :, num_heads + num_kv_heads :, :].copy_(v)
            cache_len_delta = seqlen_q
        # Upstream cache layout is dense [B, S, Hkv, D] or paged
        # [blocks, page, Hkv, D]. The kernels handle this layout directly.
        k_cache_view = k_cache if cache_row_indices is None else k_cache.index_select(0, cache_row_indices)
        v_cache_view = v_cache if cache_row_indices is None else v_cache.index_select(0, cache_row_indices)

    if num_heads <= 0:
        raise ValueError("number of query heads must be positive")
    if num_heads % num_kv_heads != 0:
        raise ValueError("number of Q heads must be divisible by number of KV heads")

    paged = block_table is not None
    kernel_paged = paged
    if paged:
        validate_block_table_shape(block_table, batch=batch_size, device=q.device)
        page_block_size = k_cache_view.shape[1] if upstream_cache_layout else k_cache_view.shape[2]
        if page_block_size <= 0:
            raise ValueError("paged cache page size must be positive")
        expected_cache_shape = (
            (k_cache_view.shape[0], page_block_size, num_kv_heads, head_dim)
            if upstream_cache_layout
            else (k_cache_view.shape[0], num_kv_heads, page_block_size, head_dim)
        )
        if tuple(k_cache_view.shape) != expected_cache_shape or tuple(v_cache_view.shape) != expected_cache_shape:
            raise ValueError("paged k_cache and v_cache must have matching [blocks, page, Hkv, D] cache layout")
        if k_cache_view.shape[0] <= 0:
            raise ValueError("paged cache must contain at least one physical block")
        max_seqlen_k = block_table.shape[1] * page_block_size
        block_table_for_kernel = block_table
    else:
        cache_seqlen_dim = 1 if upstream_cache_layout else 2
        expected_cache_shape = (
            (batch_size, k_cache_view.shape[cache_seqlen_dim], num_kv_heads, head_dim)
            if upstream_cache_layout
            else (batch_size, num_kv_heads, k_cache_view.shape[cache_seqlen_dim], head_dim)
        )
        if tuple(k_cache_view.shape) != expected_cache_shape or tuple(v_cache_view.shape) != expected_cache_shape:
            raise ValueError("dense k_cache and v_cache must have matching [B, S, Hkv, D] cache layout")
        if k_cache_view.shape[0] != batch_size or v_cache_view.shape[0] != batch_size:
            raise ValueError("dense cache batch must match q batch")
        page_block_size = 16
        max_seqlen_k = k_cache_view.shape[1] if upstream_cache_layout else k_cache_view.shape[2]
        if max_seqlen_k <= 0:
            raise ValueError("dense cache capacity must be positive")
        key = ("empty_bt", q.device, batch_size)
        block_table_for_kernel = get_placeholder(key)
        if block_table_for_kernel is None:
            block_table_for_kernel = torch.empty((batch_size, 1), device=q.device, dtype=torch.int32)
            store_placeholder(key, block_table_for_kernel)

    host_cache_len = None
    if cache_seqlens is None:
        host_cache_len = max_seqlen_k - cache_len_delta
        cache_seqlens = torch.full((batch_size,), host_cache_len, dtype=torch.int32, device=q.device)
    elif isinstance(cache_seqlens, int):
        host_cache_len = cache_seqlens
        cache_seqlens = torch.full((batch_size,), cache_seqlens, dtype=torch.int32, device=q.device)
    else:
        cache_seqlens = cache_seqlens.contiguous()
    if cache_seqlens.shape != (batch_size,) or cache_seqlens.dtype != torch.int32:
        raise ValueError("cache_seqlens must be an int or a [B] int32 tensor")
    if cache_seqlens.device != q.device:
        raise ValueError("cache_seqlens must be on the same device as q")
    # Host-known lengths are cheap to validate. Tensor lengths need device syncs
    # (aminmax / .item); skip those unless FLYDSL_KVCACHE_STRICT_CHECKS=1.
    if host_cache_len is not None:
        min_cache_len = max_cache_len = host_cache_len
        if min_cache_len < 0:
            raise ValueError("cache_seqlens must be non-negative")
        if is_qkv_packed:
            if update_cache and min_cache_len < seqlen_q:
                raise ValueError("packed cache_seqlens must include at least the appended QKV tokens")
            if max_cache_len > max_seqlen_k:
                raise ValueError("cache_seqlens exceeds cache capacity")
        elif max_cache_len + cache_len_delta > max_seqlen_k:
            raise ValueError("cache_seqlens plus appended K/V tokens exceeds cache capacity")
    elif strict_checks:
        min_cache_len, max_cache_len = cache_seqlens.aminmax()
        min_cache_len = min_cache_len.item()
        max_cache_len = max_cache_len.item()
        if min_cache_len < 0:
            raise ValueError("cache_seqlens must be non-negative")
        if is_qkv_packed:
            if update_cache and min_cache_len < seqlen_q:
                raise ValueError("packed cache_seqlens must include at least the appended QKV tokens")
            if max_cache_len > max_seqlen_k:
                raise ValueError("cache_seqlens exceeds cache capacity")
        elif max_cache_len + cache_len_delta > max_seqlen_k:
            raise ValueError("cache_seqlens plus appended K/V tokens exceeds cache capacity")
    if cache_leftpad is not None and strict_checks:
        if bool((cache_leftpad_for_kernel + cache_seqlens + cache_len_delta > max_seqlen_k).any().item()):
            raise ValueError("cache_leftpad plus visible KV tokens exceeds cache capacity")
    if (
        strict_checks
        and update_cache
        and cache_row_indices is not None
        and torch.unique(cache_row_indices).numel() != batch_size
    ):
        raise ValueError("cache_batch_idx must not contain duplicate indices when updating the cache")
    if cache_len_delta:
        cache_seqlens = cache_seqlens + torch.full_like(cache_seqlens, cache_len_delta)
    if max_context_len is None:
        planning_max_seqlen_k = max_seqlen_k
    else:
        if isinstance(max_context_len, bool) or not isinstance(max_context_len, int):
            raise ValueError("max_context_len must be a positive integer when provided")
        if max_context_len <= 0:
            raise ValueError("max_context_len must be a positive integer when provided")
        planning_max_seqlen_k = min(max_context_len, max_seqlen_k)
    has_padded_block_table = False
    if paged and strict_checks:
        needed_blocks = (cache_seqlens + page_block_size - 1) // page_block_size
        logical_blocks = torch.arange(block_table.shape[1], device=q.device, dtype=torch.int32)
        referenced = block_table[logical_blocks[None, :] < needed_blocks[:, None]]
        if referenced.numel():
            table_min, table_max = referenced.aminmax()
            if table_min.item() < 0 or table_max.item() >= k_cache_view.shape[0]:
                raise ValueError("referenced block_table entries must index physical cache blocks")
        has_padded_block_table = bool((block_table < 0).any().item())

    has_rotary = rotary_cos is not None or rotary_sin is not None
    if has_rotary:
        if rotary_cos is None or rotary_sin is None:
            raise ValueError("rotary_cos and rotary_sin must be provided together")
        if rotary_cos.shape != rotary_sin.shape or rotary_cos.ndim != 2:
            raise ValueError("rotary_cos and rotary_sin must have matching [S, C] shape")
        if rotary_cos.dtype != q.dtype or rotary_sin.dtype != q.dtype:
            rotary_cos = rotary_cos.to(dtype=q.dtype)
            rotary_sin = rotary_sin.to(dtype=q.dtype)
        if rotary_cos.device != q.device or rotary_sin.device != q.device:
            raise ValueError("rotary_cos and rotary_sin must be on the same device as q")
        rotary_cos = rotary_cos.contiguous()
        rotary_sin = rotary_sin.contiguous()
        rotary_cols = rotary_cos.shape[1]
        if not update_cache:
            token_offsets = torch.arange(seqlen_q, device=q.device, dtype=torch.int32)[None, :]
            if causal or window_size != (-1, -1):
                positions = cache_seqlens[:, None] + token_offsets
            else:
                positions = cache_seqlens[:, None].expand(batch_size, seqlen_q)
            q = _apply_rotary_torch(q, rotary_cos, rotary_sin, positions, interleaved=rotary_interleaved)
            qkv = q.contiguous()
            rotary_cos = None
            rotary_sin = None
            has_rotary = False
            rotary_cols = 0
            rotary_cos, rotary_sin = _cached_rotary_dummy(q.device, q.dtype, head_dim)
    else:
        rotary_cols = 0
        rotary_cos, rotary_sin = _cached_rotary_dummy(q.device, q.dtype, head_dim)

    has_cache_leftpad = cache_leftpad is not None
    decode_plan = plan_decode_launch(
        has_cache_leftpad=has_cache_leftpad,
        softmax_scale=softmax_scale,
        default_softmax_scale=default_softmax_scale,
        softcap=softcap,
        head_dim=head_dim,
        dtype=q.dtype,
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_splits=num_splits,
        kernel_paged=kernel_paged,
        has_padded_block_table=has_padded_block_table,
        upstream_cache_layout=upstream_cache_layout,
        page_block_size=page_block_size,
        max_seqlen_k=max_seqlen_k,
        planning_max_seqlen_k=planning_max_seqlen_k,
        max_context_len=max_context_len,
        window_size=window_size,
        k_cache_contiguous=k_cache_view.is_contiguous(),
        v_cache_contiguous=v_cache_view.is_contiguous(),
    )
    use_simt_decode = decode_plan.use_simt_decode
    use_pipelined_mma_decode = decode_plan.use_pipelined_mma_decode
    use_mma_decode = decode_plan.use_mma_decode
    mma_block_n = decode_plan.mma_block_n
    planning_max_seqlen_k = decode_plan.planning_max_seqlen_k
    effective_num_splits = decode_plan.effective_num_splits
    simt_k_warps = decode_plan.simt_k_warps

    if update_cache and paged:
        if is_qkv_packed:
            q_part, k_part, v_part = q.split([num_heads, num_kv_heads, num_kv_heads], dim=2)
        else:
            q_part, k_part, v_part = q, k, v
        token_offsets = torch.arange(seqlen_q, device=q.device, dtype=torch.int32)[None, :]
        first_new_position = cache_seqlens - seqlen_q
        kv_positions = first_new_position[:, None] + token_offsets
        if causal or window_size != (-1, -1):
            q_positions = kv_positions
        else:
            q_positions = first_new_position[:, None].expand(batch_size, seqlen_q)
        if has_rotary:
            q_part = _apply_rotary_torch(
                q_part,
                rotary_cos,
                rotary_sin,
                q_positions,
                interleaved=rotary_interleaved,
            )
            k_part = _apply_rotary_torch(
                k_part,
                rotary_cos,
                rotary_sin,
                kv_positions,
                interleaved=rotary_interleaved,
            )
        for token_idx in range(seqlen_q):
            positions = kv_positions[:, token_idx]
            logical_blocks = positions // page_block_size
            block_offsets = positions % page_block_size
            physical_blocks = block_table.gather(1, logical_blocks[:, None].to(torch.long))[:, 0]
            for batch_idx in range(batch_size):
                physical_block = physical_blocks[batch_idx]
                block_offset = block_offsets[batch_idx]
                if upstream_cache_layout:
                    k_cache_view[physical_block, block_offset].copy_(k_part[batch_idx, token_idx])
                    v_cache_view[physical_block, block_offset].copy_(v_part[batch_idx, token_idx])
                else:
                    k_cache_view[physical_block, :, block_offset].copy_(k_part[batch_idx, token_idx])
                    v_cache_view[physical_block, :, block_offset].copy_(v_part[batch_idx, token_idx])
        q = q_part.contiguous()
        qkv = q
        update_cache = False
        if has_rotary:
            has_rotary = False
            rotary_cols = 0
            rotary_cos, rotary_sin = _cached_rotary_dummy(q.device, q.dtype, head_dim)

    expected_out_shape = (batch_size, seqlen_q, num_heads, head_dim)
    if out_buf is not None:
        if tuple(out_buf.shape) != expected_out_shape:
            raise ValueError(f"out must have shape {expected_out_shape}, got {tuple(out_buf.shape)}")
        if out_buf.dtype != q.dtype or out_buf.device != q.device:
            raise ValueError("out must have the same dtype and device as q")
        if out_buf.stride(-1) != 1:
            raise ValueError("out must have contiguous last dimension")

    if use_simt_decode:
        # Unlike the MMA kernels, the SIMT GQA=2 kernel only loads the
        # two valid query heads owned by each KV head. It does not need the
        # 16-row tail padding below, so read Q directly and avoid a per-token
        # device-to-device copy launch.
        q_work = q if q.is_contiguous() else q.contiguous()
        if out_buf is None:
            out_key = ("mma_out", q.device, q.dtype, batch_size, seqlen_q, num_heads, head_dim)
            out_buf = get_placeholder(out_key)
            if out_buf is None:
                out_buf = torch.empty(expected_out_shape, device=q.device, dtype=q.dtype)
                store_placeholder(out_key, out_buf)
    elif use_mma_decode or use_pipelined_mma_decode:
        # Tail-pad q_work so the kernel's unguarded 16-row Q async-load stays
        # in-bounds when the last GQA group has fewer than 16 heads.
        # Reuse the padded Q scratch across calls with the same decode shape.
        q_elems = batch_size * seqlen_q * num_heads * head_dim
        q_key = ("mma_q", q.device, q.dtype, batch_size, seqlen_q, num_heads, head_dim)
        q_storage = get_placeholder(q_key)
        if q_storage is None:
            q_storage = torch.empty(q_elems + 16 * head_dim, device=q.device, dtype=q.dtype)
            store_placeholder(q_key, q_storage)
        q_work = q_storage[:q_elems].view(batch_size, seqlen_q, num_heads, head_dim)
        if not update_cache:
            q_work.copy_(q)
        if out_buf is None:
            out_key = ("mma_out", q.device, q.dtype, batch_size, seqlen_q, num_heads, head_dim)
            out_buf = get_placeholder(out_key)
            if out_buf is None:
                out_buf = torch.empty(expected_out_shape, device=q.device, dtype=q.dtype)
                store_placeholder(out_key, out_buf)
    elif not update_cache:
        q_work = q.contiguous()
        if out_buf is None:
            out_buf = torch.empty_like(q_work)
    else:
        q_work = torch.empty(expected_out_shape, device=q.device, dtype=q.dtype)
        if out_buf is None:
            out_buf = torch.empty_like(q_work)
    out = out_buf
    use_varlen_prefill = (
        paged
        and not use_mma_decode
        and not use_pipelined_mma_decode
        and not use_simt_decode
        and q.dtype is torch.bfloat16
        and head_dim == 128
        and window_size == (-1, -1)
        and softcap == 0.0
        and not has_cache_leftpad
    )
    if (
        paged
        and upstream_cache_layout
        and not use_mma_decode
        and not use_pipelined_mma_decode
        and not use_simt_decode
        and not use_varlen_prefill
    ):
        if upstream_cache_layout:
            dense_k = torch.zeros(batch_size, max_seqlen_k, num_kv_heads, head_dim, device=q.device, dtype=q.dtype)
            dense_v = torch.zeros_like(dense_k)
        else:
            dense_k = torch.zeros(batch_size, num_kv_heads, max_seqlen_k, head_dim, device=q.device, dtype=q.dtype)
            dense_v = torch.zeros_like(dense_k)
        for logical_block in range(block_table.shape[1]):
            physical_blocks = block_table[:, logical_block]
            valid_batches = torch.nonzero(physical_blocks >= 0, as_tuple=False)[:, 0]
            physical_blocks = physical_blocks.index_select(0, valid_batches).to(torch.long)
            token_start = logical_block * page_block_size
            token_end = token_start + page_block_size
            if upstream_cache_layout:
                dense_k[valid_batches, token_start:token_end] = k_cache_view.index_select(0, physical_blocks)
                dense_v[valid_batches, token_start:token_end] = v_cache_view.index_select(0, physical_blocks)
            else:
                dense_k[valid_batches, :, token_start:token_end] = k_cache_view.index_select(0, physical_blocks)
                dense_v[valid_batches, :, token_start:token_end] = v_cache_view.index_select(0, physical_blocks)
        k_cache_view = dense_k
        v_cache_view = dense_v
        kernel_paged = False
        block_table_for_kernel = torch.empty((batch_size, 1), device=q.device, dtype=torch.int32)

    dtype_str = _dtype_name(q.dtype)
    if effective_num_splits > 1:
        group_max, group_sum, partial_out = _cached_split_bufs(
            q.device,
            batch_size,
            seqlen_q,
            num_heads,
            effective_num_splits,
            head_dim,
        )
    else:
        group_max, group_sum, partial_out = _cached_split1_bufs(q.device)
    launcher = build_flash_attn_with_kvcache_module(
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_seqlen_k=planning_max_seqlen_k if use_simt_decode else max_seqlen_k,
        dtype_str=dtype_str,
        paged=kernel_paged,
        page_block_size=page_block_size,
        has_rotary=has_rotary,
        rotary_cols=rotary_cols,
        causal=causal,
        window_size=window_size,
        rotary_interleaved=rotary_interleaved,
        update_cache=update_cache,
        num_splits=effective_num_splits,
        upstream_cache_layout=upstream_cache_layout,
        use_mma_decode=use_mma_decode,
        use_pipelined_mma_decode=use_pipelined_mma_decode,
        use_simt_decode=use_simt_decode,
        simt_k_warps=simt_k_warps,
        mma_block_n=mma_block_n,
        softmax_scale=softmax_scale,
        softcap=softcap,
        run_attention=not use_varlen_prefill,
        cache_strides=tuple(k_cache_view.stride()) if kernel_paged else None,
    )
    if use_simt_decode and not update_cache:
        launch_args = (
            q_work,
            k_cache_view,
            v_cache_view,
            cache_seqlens,
            block_table_for_kernel,
            out,
            group_max,
            group_sum,
            partial_out,
            stream,
        )
    else:
        launch_args = (
            qkv,
            q_work,
            k_cache_view,
            v_cache_view,
            cache_seqlens,
            block_table_for_kernel,
            cache_leftpad_for_kernel,
            rotary_cos,
            rotary_sin,
            out,
            group_max,
            group_sum,
            partial_out,
            stream,
        )
    static_simt_decode = fast_probe is not None and use_simt_decode and not update_cache
    compile_key = (
        launcher,
        q.device,
        (
            q_work.dtype,
            tuple(q_work.shape),
            tuple(q_work.stride()),
            tuple(k_cache_view.shape),
            tuple(k_cache_view.stride()),
            tuple(v_cache_view.shape),
            tuple(v_cache_view.stride()),
            tuple(cache_seqlens.shape),
            tuple(cache_seqlens.stride()),
            tuple(block_table_for_kernel.shape),
            tuple(block_table_for_kernel.stride()),
            tuple(out.shape),
            tuple(out.stride()),
        )
        if static_simt_decode
        else None,
    )
    compiled = get_compiled_launch(compile_key)
    if compiled is None:
        if static_simt_decode:
            compile_args = tuple(
                flyc.from_torch_tensor(arg) if isinstance(arg, torch.Tensor) else arg for arg in launch_args
            )
        else:
            compile_args = launch_args
        compiled = flyc.compile(launcher, *compile_args)
        store_compiled_launch(compile_key, compiled)
    compiled(*launch_args)
    if fast_probe is not None and use_simt_decode and not update_cache:
        fast_entry = (
            compiled,
            out if fast_out_was_none else None,
            group_max,
            group_sum,
            partial_out,
        )
        store_simt_decode_entry(fast_probe[0], fast_entry)
    if use_varlen_prefill:
        from .flash_attn_prefill import _prefill

        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, seqlen_q, device=q.device, dtype=torch.int32)
        prefill_out = _prefill(
            q_work.reshape(-1, num_heads, head_dim),
            k_cache_view,
            v_cache_view,
            cu_seqlens_q,
            cache_seqlens,
            block_table_for_kernel,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_block_size=page_block_size,
            causal=causal,
            upstream_cache_layout=upstream_cache_layout,
            softmax_scale=softmax_scale,
            batch=batch_size,
            max_seqlen_q=seqlen_q,
            out=out.reshape(-1, num_heads, head_dim),
            stream=stream,
        )
        out = prefill_out.reshape(batch_size, seqlen_q, num_heads, head_dim)
    if update_cache and cache_row_indices is not None:
        cache_row_indices_long = cache_row_indices.to(torch.long)
        k_cache.index_copy_(0, cache_row_indices_long, k_cache_view)
        v_cache.index_copy_(0, cache_row_indices_long, v_cache_view)
    return out
