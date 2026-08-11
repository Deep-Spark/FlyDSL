# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FlyDSL-native ``flash_attn_varlen_func`` for vLLM's unified attention path.

Mirrors the public ``flash_attn_varlen_func`` (see
``flash-attention/flash_attn/flash_attn_interface.py``) for the subset vLLM v1
actually drives:

* Packed varlen queries ``q [total_q, num_heads, head_dim]`` with
  ``cu_seqlens_q [batch + 1]``.
* Either KV source:
  - *Paged* cache (``block_table`` given), layout selected by
    ``kv_cache_layout``: HND (default)
    ``[num_blocks, num_kv_heads, page, head_dim]`` (vLLM
    ``VLLM_KV_CACHE_LAYOUT=HND``) or NHD
    ``[num_blocks, page, num_kv_heads, head_dim]`` (vLLM default).  KV length
    from ``seqused_k [batch]`` (or derived from ``cu_seqlens_k``).
  - *Dense* packed varlen ``k/v [total_k, num_kv_heads, head_dim]``
    (``block_table is None``), addressed by ``cu_seqlens_k [batch + 1]``.  This
    is the fresh-prefill (no prefix cache) path of ``flash_attn_varlen_func``.
* Bottom-right causal masking, GQA, bf16, ``head_dim in (128, 256)``.

Unsupported options (dropout, sliding window, softcap, ALiBi, returning the
softmax LSE / attention probabilities) are rejected rather than silently
mis-handled, matching the philosophy of ``flash_attn_with_kvcache``.

Dispatch:
* ``max_seqlen_q == 1`` (pure decode) routes to ``flash_attn_with_kvcache``.
* otherwise the varlen prefill kernel (``flash_attn_varlen_mma``) runs.
"""

import math
from typing import Optional, Tuple

import torch

import flydsl.expr as fx

from .flash_attn_paged_kv import validate_block_table_shape, validate_kv_cache_layout
from .flash_attn_prefill import (
    _get_varlen_prefill_launch_plan as _get_varlen_prefill_launch_plan,
)
from .flash_attn_prefill import _prefill
from .flash_attn_varlen_mma import HEAD_DIM


def _prefill_via_padded_kvcache(
    q,
    k_cache,
    v_cache,
    cu_seqlens_q,
    seqused_k,
    block_table,
    *,
    max_seqlen_q,
    max_seqlen_k,
    causal,
    softmax_scale,
    upstream_cache_layout,
    out,
    stream,
):
    """Run generic paged varlen prefill through the scalar KV-cache kernel.

    The KV-cache kernel accepts a rectangular ``[B, Sq, H, D]`` query tensor,
    while vLLM supplies packed varlen queries. Right-aligning each sequence in
    a ``max_seqlen_q`` rectangle preserves bottom-right causal positions:

      padded_sq = max_seqlen_q - query_len + local_sq
      kv_pos    = kv_len - max_seqlen_q + padded_sq
                = kv_len - query_len + local_sq

    This provides a correctness-first path for head dimensions not handled by
    the tensor-core varlen kernel (currently D=256 for Qwen3.5). The packing
    uses device-side PyTorch indexing and therefore does not synchronize query
    lengths to the host.
    """
    from .flash_attn_kvcache import flash_attn_with_kvcache

    batch = cu_seqlens_q.numel() - 1
    total_q, num_heads, head_dim = q.shape
    token = torch.arange(total_q, device=q.device, dtype=torch.int32)
    # ``right=True`` assigns a token at a sequence boundary to the next row.
    batch_idx = torch.searchsorted(cu_seqlens_q[1:], token, right=True)
    query_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    local_idx = token - cu_seqlens_q.index_select(0, batch_idx)
    padded_idx = max_seqlen_q - query_lens.index_select(0, batch_idx) + local_idx

    q_padded = torch.zeros(
        (batch, max_seqlen_q, num_heads, head_dim),
        device=q.device,
        dtype=q.dtype,
    )
    q_padded[batch_idx.long(), padded_idx.long()] = q
    out_padded = torch.empty_like(q_padded)

    flash_attn_with_kvcache(
        q_padded,
        k_cache,
        v_cache,
        cache_seqlens=seqused_k,
        block_table=block_table,
        softmax_scale=softmax_scale,
        causal=causal,
        force_upstream_cache_layout=upstream_cache_layout,
        stream=stream,
        out=out_padded,
        max_context_len=max_seqlen_k,
    )

    result = out_padded[batch_idx.long(), padded_idx.long()]
    if out is not None:
        out.copy_(result)
        return out
    return result


def _decode_dispatch(
    q,
    k_cache,
    v_cache,
    seqused_k,
    block_table,
    *,
    num_heads,
    num_kv_heads,
    head_dim,
    causal,
    out,
    batch,
    upstream_cache_layout,
    stream,
):
    """Route pure-decode (max_seqlen_q == 1) to ``flash_attn_with_kvcache``."""
    from .flash_attn_kvcache import flash_attn_with_kvcache

    # Packed [batch, num_heads, D] -> decode wrapper's [B, 1, Hq, D].
    q_dec = q.reshape(batch, 1, num_heads, head_dim).contiguous()
    res = flash_attn_with_kvcache(
        q_dec,
        k_cache,
        v_cache,
        cache_seqlens=seqused_k,
        block_table=block_table,
        causal=causal,
        force_upstream_cache_layout=upstream_cache_layout,
        stream=stream,
    )
    res = res.reshape(batch, num_heads, head_dim)
    if out is not None:
        out.copy_(res)
        return out
    return res


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k=None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes=None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    block_table=None,
    *,
    seqused_k=None,
    out=None,
    stream=None,
    use_decode_kernel: bool = True,
    kv_cache_layout: str = "HND",
    strict_checks: bool = True,
):
    """Forward-only varlen attention over a paged KV cache (vLLM path).

    ``q``: ``[total_q, num_heads, head_dim]`` bf16.
    ``k``/``v``: paged cache, layout selected by ``kv_cache_layout``:
        * ``"HND"`` (default): ``[num_blocks, num_kv_heads, page, head_dim]``
          (vLLM ``VLLM_KV_CACHE_LAYOUT=HND``).
        * ``"NHD"``: ``[num_blocks, page, num_kv_heads, head_dim]``
          (vLLM default layout).
    ``cu_seqlens_q``: ``[batch + 1]`` int32.
    ``seqused_k`` (preferred) or ``cu_seqlens_k``: per-sequence KV length.
    ``block_table``: ``[batch, max_blocks]`` int32 (required; paged only).
    """
    upstream_cache_layout = validate_kv_cache_layout(kv_cache_layout)
    if dropout_p != 0.0:
        raise NotImplementedError("dropout is not supported in the FlyDSL varlen path")
    if window_size != (-1, -1):
        raise NotImplementedError("sliding window is not supported yet")
    if softcap != 0.0:
        raise NotImplementedError("softcap is not supported yet")
    if alibi_slopes is not None:
        raise NotImplementedError("ALiBi is not supported yet")
    if return_attn_probs:
        raise NotImplementedError("return_attn_probs / softmax LSE is not supported yet")
    if q.ndim != 3:
        raise ValueError("q must be [total_q, num_heads, head_dim]")
    if q.dtype is not torch.bfloat16:
        raise NotImplementedError("only bf16 is supported")
    if q.shape[-1] not in (HEAD_DIM, 256):
        raise NotImplementedError(
            f"only head_dim in ({HEAD_DIM}, 256) is supported"
        )
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, v must share dtype")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, v must be on the same device")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("q, k, v must have contiguous last dimension")

    paged = block_table is not None
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    if paged:
        if k.ndim != 4 or v.ndim != 4:
            raise ValueError(
                "paged k/v must be 4D "
                "(HND [num_blocks, num_kv_heads, page, head_dim] or "
                "NHD [num_blocks, page, num_kv_heads, head_dim])"
            )
        if upstream_cache_layout:  # NHD [num_blocks, page, num_kv_heads, head_dim]
            page_block_size = k.shape[1]
            num_kv_heads = k.shape[2]
        else:  # HND [num_blocks, num_kv_heads, page, head_dim]
            num_kv_heads = k.shape[1]
            page_block_size = k.shape[2]
        if k.shape != v.shape:
            raise ValueError("paged k and v must have matching shapes")
        if k.stride() != v.stride():
            raise ValueError("paged k and v must have matching strides")
        if k.shape[0] <= 0 or k.shape[-1] != head_dim:
            raise ValueError("paged k/v must have non-empty blocks and match q head_dim")
    else:
        # Dense (non-paged) packed varlen k/v: [total_k, num_kv_heads, head_dim].
        if k.ndim != 3 or v.ndim != 3:
            raise ValueError("dense (non-paged) k/v must be 3D [total_k, num_kv_heads, head_dim]")
        if cu_seqlens_k is None:
            raise ValueError("dense (non-paged) varlen requires cu_seqlens_k")
        num_kv_heads = k.shape[1]
        page_block_size = 16  # unused for dense; must satisfy build asserts
        if k.shape != v.shape or k.shape[-1] != head_dim:
            raise ValueError("dense k/v must have matching [total_k, num_kv_heads, head_dim] shapes")
    if num_heads % num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads")
    if page_block_size % 16 != 0:
        raise ValueError("page_block_size must be a multiple of 16")

    cu_seqlens_q = cu_seqlens_q.contiguous().to(torch.int32)
    batch = cu_seqlens_q.numel() - 1
    if cu_seqlens_q.device != q.device:
        raise ValueError("cu_seqlens_q must be on the same device as q")
    if batch <= 0:
        raise ValueError("cu_seqlens_q must contain batch + 1 entries")
    if strict_checks and cu_seqlens_q[0].item() != 0:
        raise ValueError("cu_seqlens_q must contain batch + 1 entries and start at zero")
    if strict_checks and bool((cu_seqlens_q[1:] < cu_seqlens_q[:-1]).any().item()):
        raise ValueError("cu_seqlens_q must be non-decreasing")
    if strict_checks and cu_seqlens_q[-1].item() != q.shape[0]:
        raise ValueError("cu_seqlens_q must end at total_q")
    if paged:
        validate_block_table_shape(block_table, batch=batch, device=q.device)
        block_table = block_table.contiguous().to(torch.int32)

    # Per-sequence KV length; dense also needs cu_seqlens_k for the KV start.
    ck_i32 = None
    seqused_is_cumulative = False
    if cu_seqlens_k is not None:
        ck_i32 = cu_seqlens_k.contiguous().to(torch.int32)
    if seqused_k is not None:
        seqused_k = seqused_k.contiguous().to(torch.int32)
    elif ck_i32 is not None:
        # vLLM already supplies cumulative KV offsets. In the unchecked paged
        # prefill path, let the kernel take adjacent differences directly so
        # every attention layer avoids a temporary tensor and subtraction.
        if paged and not strict_checks and max_seqlen_q not in (None, 1):
            seqused_k = ck_i32
            seqused_is_cumulative = True
        else:
            seqused_k = (ck_i32[1:] - ck_i32[:-1]).contiguous().to(torch.int32)
    else:
        raise ValueError("either seqused_k or cu_seqlens_k must be provided")
    expected_seq_entries = batch + 1 if seqused_is_cumulative else batch
    if seqused_k.numel() != expected_seq_entries:
        expected = "`batch + 1`" if seqused_is_cumulative else "`batch`"
        raise ValueError(f"KV sequence metadata must have {expected} entries")
    if not paged and ck_i32.numel() != batch + 1:
        raise ValueError("cu_seqlens_k must have `batch + 1` entries")
    if seqused_k.device != q.device or (ck_i32 is not None and ck_i32.device != q.device):
        raise ValueError("KV sequence metadata must be on the same device as q")
    if strict_checks and not seqused_is_cumulative and bool((seqused_k < 0).any().item()):
        raise ValueError("seqused_k must be non-negative")
    if paged:
        if strict_checks:
            needed_blocks = (seqused_k + page_block_size - 1) // page_block_size
            if bool((needed_blocks > block_table.shape[1]).any().item()):
                raise ValueError("seqused_k exceeds block_table capacity")
            logical_blocks = torch.arange(block_table.shape[1], device=q.device, dtype=torch.int32)
            referenced = block_table[logical_blocks[None, :] < needed_blocks[:, None]]
            if referenced.numel():
                ref_min, ref_max = referenced.aminmax()
                if ref_min.item() < 0 or ref_max.item() >= k.shape[0]:
                    raise ValueError("referenced block_table entries must index physical cache blocks")
            zero_length_rows = seqused_k == 0
            if bool(zero_length_rows.any().item()):
                block_table = block_table.clone()
                block_table[zero_length_rows, 0] = 0
    else:
        if strict_checks and (ck_i32[0].item() != 0 or bool((ck_i32[1:] < ck_i32[:-1]).any().item())):
            raise ValueError("cu_seqlens_k must be non-decreasing and start at zero")
        if strict_checks and ck_i32[-1].item() > k.shape[0]:
            raise ValueError("cu_seqlens_k exceeds dense K/V capacity")
        if strict_checks and bool((ck_i32[:-1] + seqused_k > ck_i32[1:]).any().item()):
            raise ValueError("seqused_k exceeds its dense K/V sequence span")

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    else:
        softmax_scale = float(softmax_scale)

    if max_seqlen_q is None:
        max_seqlen_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
    elif strict_checks and bool((cu_seqlens_q[1:] - cu_seqlens_q[:-1] > max_seqlen_q).any().item()):
        raise ValueError("max_seqlen_q is smaller than an actual query sequence")
    if max_seqlen_k is None:
        seq_lengths = ck_i32[1:] - ck_i32[:-1] if seqused_is_cumulative else seqused_k
        max_seqlen_k = int(seq_lengths.max().item())
    elif strict_checks and not seqused_is_cumulative and bool((seqused_k > max_seqlen_k).any().item()):
        raise ValueError("max_seqlen_k is smaller than an actual KV sequence")

    if q.is_cuda:
        current_stream = torch.cuda.current_stream(q.device)
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

    # Decode fast-path (paged only): every sequence has exactly one query token.
    # Decode kernels compile in the canonical 1/sqrt(D) scale and do not take
    # a scale argument.  Even a near-default explicit scale must use prefill,
    # which bakes that exact value into its kernel.
    default_scale = softmax_scale == 1.0 / math.sqrt(head_dim)
    decode_capacity = block_table.shape[1] * page_block_size if paged else 0
    if (
        paged
        and use_decode_kernel
        and max_seqlen_q == 1
        and q.shape[0] == batch
        and default_scale
        and decode_capacity >= 128
        and decode_capacity % 128 == 0
    ):
        return _decode_dispatch(
            q,
            k,
            v,
            seqused_k,
            block_table,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            causal=causal,
            out=out,
            batch=batch,
            upstream_cache_layout=upstream_cache_layout,
            stream=stream,
        )

    return _prefill(
        q,
        k,
        v,
        cu_seqlens_q,
        seqused_k,
        block_table,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_block_size=page_block_size,
        causal=causal,
        upstream_cache_layout=upstream_cache_layout,
        softmax_scale=softmax_scale,
        batch=batch,
        max_seqlen_q=max_seqlen_q,
        out=out,
        stream=stream,
        paged=paged,
        cu_seqlens_k=ck_i32,
        seqused_is_cumulative=seqused_is_cumulative,
    )
