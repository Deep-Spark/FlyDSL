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
    ``[num_blocks, num_kv_heads, page, head_dim]`` -- matching ixformer's paged
    varlen and vLLM with ``VLLM_KV_CACHE_LAYOUT=HND`` -- or NHD
    ``[num_blocks, page, num_kv_heads, head_dim]`` (vLLM's default).  KV length
    from ``seqused_k [batch]`` (or derived from ``cu_seqlens_k``).
  - *Dense* packed varlen ``k/v [total_k, num_kv_heads, head_dim]``
    (``block_table is None``), addressed by ``cu_seqlens_k [batch + 1]``.  This
    is the fresh-prefill (no prefix cache) path of ``flash_attn_varlen_func``.
* Bottom-right causal masking, GQA, bf16, ``head_dim == 128``.

Unsupported options (dropout, sliding window, softcap, ALiBi, returning the
softmax LSE / attention probabilities) are rejected rather than silently
mis-handled, matching the philosophy of ``flash_attn_with_kvcache``.

Dispatch:
* ``max_seqlen_q == 1`` (pure decode) is routed to the proven decode kernels via
  ``flash_attn_with_kvcache`` (library / MMA tensor-core decode).
* otherwise the varlen prefill kernel (``flash_attn_varlen_mma``) runs.
"""

import math
from typing import Optional, Tuple

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

from .flash_attn_varlen_mma import (
    HEAD_DIM,
    bm_for,
    build_flash_attn_varlen_kernel,
    max_q_tiles_for,
    select_num_warps,
)

# Compiled prefill launchers keyed by static shape / config.
_PREFILL_CACHE: dict = {}


def _build_prefill_launcher(
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_block_size: int,
    causal: bool,
    upstream_cache_layout: bool,
    softmax_scale: float,
    batch: int,
    max_q_tiles: int,
    num_warps: int,
    paged: bool = True,
):
    kernel, threads, smem, tile = build_flash_attn_varlen_kernel(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_block_size=page_block_size,
        causal=causal,
        upstream_cache_layout=upstream_cache_layout,
        softmax_scale=softmax_scale,
        num_warps=num_warps,
        paged=paged,
    )
    # Grid (heads, batch, q_tiles): head fastest balances causal load per warp.
    grid = (num_heads, batch, max_q_tiles)

    @flyc.jit
    def launch(Q, K, V, CuQ, SeqK, BT, Oo, stream=fx.Stream(None)):
        kernel(Q, K, V, CuQ, SeqK, BT, Oo).launch(grid=grid, block=(threads, 1, 1), smem=smem, stream=stream)

    return launch


def _prefill(
    q,
    k,
    v,
    cu_seqlens_q,
    seqused_k,
    block_table,
    *,
    num_heads,
    num_kv_heads,
    head_dim,
    page_block_size,
    causal,
    upstream_cache_layout,
    softmax_scale,
    batch,
    max_seqlen_q,
    out,
    stream,
    paged=True,
    cu_seqlens_k=None,
):
    total_q = q.shape[0]
    num_warps = select_num_warps(max_seqlen_q)
    bm = bm_for(num_warps)
    max_q_tiles = max_q_tiles_for(max_seqlen_q, num_warps)

    # Tail-pad Q so the kernel's unguarded BM-row async load stays in bounds for
    # the final tile of the last sequence.
    q_pad = torch.empty((total_q + bm, num_heads, head_dim), device=q.device, dtype=q.dtype)
    q_pad[:total_q].copy_(q)

    if paged:
        k_flat = k.reshape(-1)
        v_flat = v.reshape(-1)
        # BlockTable slot carries the paged block table.
        bt_arg = block_table
    else:
        # Dense packed [total_k, Hkv, D]; tail-pad by one BM tile so the last
        # sequence's clamped 16-row SME gather stays in bounds.  The pad rows
        # MUST be zeroed: they back masked (out-of-range) keys, and the mask is
        # applied as an additive -inf bias, so a NaN/garbage K would give
        # ``NaN + (-inf) = NaN`` (and ``0 * NaN = NaN`` in the P*V accumulate).
        # Zeroed pad -> score 0 -> masked to -inf -> exp 0, and 0*0 = 0.
        total_k = k.shape[0]
        k_pad = torch.zeros((total_k + bm, num_kv_heads, head_dim), device=k.device, dtype=k.dtype)
        k_pad[:total_k].copy_(k)
        v_pad = torch.zeros((total_k + bm, num_kv_heads, head_dim), device=v.device, dtype=v.dtype)
        v_pad[:total_k].copy_(v)
        k_flat = k_pad.reshape(-1)
        v_flat = v_pad.reshape(-1)
        # BlockTable slot instead carries cu_seqlens_k [batch + 1].
        bt_arg = cu_seqlens_k

    user_out = out
    # Tail-pad the output by one BM tile as a defensive guard: the epilogue store
    # is row-guarded, but padding removes any risk of a stray BM-row write past
    # ``total_q`` clobbering neighbouring allocations (e.g. the KV cache).
    out_buf = torch.empty((total_q + bm, num_heads, head_dim), device=q.device, dtype=q.dtype)

    # The scale is compiled into the prefill kernel.  Preserve its exact Python
    # float value here: rounding can alias distinct caller-supplied scales onto
    # a launcher compiled with a different constant.
    key = (
        num_heads,
        num_kv_heads,
        head_dim,
        page_block_size,
        bool(causal),
        bool(upstream_cache_layout),
        float(softmax_scale),
        batch,
        max_q_tiles,
        num_warps,
        bool(paged),
    )
    args = (q_pad.reshape(-1), k_flat, v_flat, cu_seqlens_q, seqused_k, bt_arg, out_buf.reshape(-1), stream)

    compiled = _PREFILL_CACHE.get(key)
    if compiled is None:
        launcher = _build_prefill_launcher(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_block_size=page_block_size,
            causal=causal,
            upstream_cache_layout=upstream_cache_layout,
            softmax_scale=softmax_scale,
            batch=batch,
            max_q_tiles=max_q_tiles,
            num_warps=num_warps,
            paged=paged,
        )
        compiled = flyc.compile(launcher, *args)
        _PREFILL_CACHE[key] = compiled
    compiled(*args)
    result = out_buf[:total_q]
    if user_out is not None:
        user_out.copy_(result)
        return user_out
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
    """Route pure-decode (max_seqlen_q == 1) to the proven decode kernels."""
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
):
    """Forward-only varlen attention over a paged KV cache (vLLM path).

    ``q``: ``[total_q, num_heads, head_dim]`` bf16.
    ``k``/``v``: paged cache, layout selected by ``kv_cache_layout``:
        * ``"HND"`` (default): ``[num_blocks, num_kv_heads, page, head_dim]``
          -- matches ixformer's paged varlen and vLLM with
          ``VLLM_KV_CACHE_LAYOUT=HND``.
        * ``"NHD"``: ``[num_blocks, page, num_kv_heads, head_dim]``
          -- vLLM's default layout.
    ``cu_seqlens_q``: ``[batch + 1]`` int32.
    ``seqused_k`` (preferred) or ``cu_seqlens_k``: per-sequence KV length.
    ``block_table``: ``[batch, max_blocks]`` int32 (required; paged only).
    """
    if kv_cache_layout not in ("HND", "NHD"):
        raise ValueError("kv_cache_layout must be 'HND' or 'NHD'")
    upstream_cache_layout = kv_cache_layout == "NHD"
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
    if q.shape[-1] != HEAD_DIM:
        raise NotImplementedError(f"only head_dim == {HEAD_DIM} is supported")
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
    if batch <= 0 or cu_seqlens_q[0].item() != 0:
        raise ValueError("cu_seqlens_q must contain batch + 1 entries and start at zero")
    if bool((cu_seqlens_q[1:] < cu_seqlens_q[:-1]).any().item()):
        raise ValueError("cu_seqlens_q must be non-decreasing")
    if cu_seqlens_q[-1].item() != q.shape[0]:
        raise ValueError("cu_seqlens_q must end at total_q")
    if paged:
        if (
            block_table.ndim != 2
            or block_table.shape[0] != batch
            or block_table.shape[1] <= 0
            or block_table.device != q.device
        ):
            raise ValueError(
                f"block_table must be [batch, max_blocks] on the same device as q; "
                f"got shape={tuple(block_table.shape)}, device={block_table.device}, "
                f"batch={batch}, q.device={q.device}"
            )
        block_table = block_table.contiguous().to(torch.int32)

    # Per-sequence KV length; dense also needs cu_seqlens_k for the KV start.
    ck_i32 = None
    if cu_seqlens_k is not None:
        ck_i32 = cu_seqlens_k.contiguous().to(torch.int32)
    if seqused_k is not None:
        seqused_k = seqused_k.contiguous().to(torch.int32)
    elif ck_i32 is not None:
        seqused_k = (ck_i32[1:] - ck_i32[:-1]).contiguous().to(torch.int32)
    else:
        raise ValueError("either seqused_k or cu_seqlens_k must be provided")
    if seqused_k.numel() != batch:
        raise ValueError("seqused_k must have `batch` entries")
    if not paged and ck_i32.numel() != batch + 1:
        raise ValueError("cu_seqlens_k must have `batch + 1` entries")
    if seqused_k.device != q.device or (ck_i32 is not None and ck_i32.device != q.device):
        raise ValueError("KV sequence metadata must be on the same device as q")
    if bool((seqused_k < 0).any().item()):
        raise ValueError("seqused_k must be non-negative")
    if paged:
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
        if ck_i32[0].item() != 0 or bool((ck_i32[1:] < ck_i32[:-1]).any().item()):
            raise ValueError("cu_seqlens_k must be non-decreasing and start at zero")
        if ck_i32[-1].item() > k.shape[0]:
            raise ValueError("cu_seqlens_k exceeds dense K/V capacity")
        if bool((ck_i32[:-1] + seqused_k > ck_i32[1:]).any().item()):
            raise ValueError("seqused_k exceeds its dense K/V sequence span")

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    else:
        softmax_scale = float(softmax_scale)

    if max_seqlen_q is None:
        max_seqlen_q = int((cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
    elif bool((cu_seqlens_q[1:] - cu_seqlens_q[:-1] > max_seqlen_q).any().item()):
        raise ValueError("max_seqlen_q is smaller than an actual query sequence")
    if max_seqlen_k is None:
        max_seqlen_k = int(seqused_k.max().item())
    elif bool((seqused_k > max_seqlen_k).any().item()):
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
    )
