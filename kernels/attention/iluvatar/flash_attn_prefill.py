# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Prepared varlen prefill launchers and JIT plan cache."""

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

from .flash_attn_varlen_mma import (
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
    seqused_is_cumulative: bool = False,
    cache_strides=None,
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
        seqused_is_cumulative=seqused_is_cumulative,
        cache_strides=cache_strides,
    )
    # Grid (heads, batch, q_tiles): head fastest balances causal load per warp.
    grid = (num_heads, batch, max_q_tiles)

    @flyc.jit
    def launch(Q, K, V, CuQ, SeqK, BT, Oo, stream=fx.Stream(None)):
        kernel(
            fx.get_iter(Q),
            fx.get_iter(K),
            fx.get_iter(V),
            fx.get_iter(CuQ),
            fx.get_iter(SeqK),
            BT,
            fx.get_iter(Oo),
        ).launch(grid=grid, block=(threads, 1, 1), smem=smem, stream=stream)

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
    seqused_is_cumulative=False,
):
    total_q = q.shape[0]
    num_warps = select_num_warps(max_seqlen_q, head_dim)
    bm = bm_for(num_warps)
    max_q_tiles = max_q_tiles_for(max_seqlen_q, num_warps)

    # The Q DMA always reads a complete BM tile. A single packed sequence can
    # therefore use Q directly only when its final tile is also complete.
    # General varlen batches retain the padded fallback because individual
    # sequence lengths are only available in device metadata here.
    if batch == 1 and total_q % bm == 0:
        q_pad = q
    else:
        q_pad = torch.empty((total_q + bm, num_heads, head_dim), device=q.device, dtype=q.dtype)
        q_pad[:total_q].copy_(q)

    if paged:
        k_arg = k
        v_arg = v
        cache_strides = tuple(k.stride())
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
        k_arg = k_pad
        v_arg = v_pad
        cache_strides = None
        # BlockTable slot instead carries cu_seqlens_k [batch + 1].
        bt_arg = cu_seqlens_k
    block_table_layout = (tuple(bt_arg.shape), tuple(bt_arg.stride()))

    user_out = out
    # The kernel's epilogue store is row-guarded, so write directly into a
    # caller-provided output. Keep a padded allocation only for returned output.
    out_buf = (
        torch.empty((total_q + bm, num_heads, head_dim), device=q.device, dtype=q.dtype)
        if user_out is None
        else user_out
    )

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
        bool(seqused_is_cumulative),
        cache_strides,
        block_table_layout,
    )
    args = (q_pad.reshape(-1), k_arg, v_arg, cu_seqlens_q, seqused_k, bt_arg, out_buf.reshape(-1), stream)

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
            seqused_is_cumulative=seqused_is_cumulative,
            cache_strides=cache_strides,
        )
        compiled = flyc.compile(launcher, *args)
        _PREFILL_CACHE[key] = compiled
    compiled(*args)
    result = out_buf[:total_q]
    if user_out is not None:
        return user_out
    return result


class _VarlenPrefillLaunchPlan:
    """Prepared B=1 paged prefill launch with dynamic tensor addresses."""

    __slots__ = ("compiled", "stream")

    def __init__(self, compiled, current_stream):
        self.compiled = compiled
        self.stream = fx.Stream(current_stream)

    def __call__(self, q, k, v, cu_seqlens_q, seqused_k, block_table, out):
        self.compiled(
            q.reshape(-1),
            k,
            v,
            cu_seqlens_q,
            seqused_k,
            block_table,
            out.reshape(-1),
            self.stream,
        )
        return out


def _get_varlen_prefill_launch_plan(
    q,
    k,
    v,
    cu_seqlens_q,
    seqused_k,
    block_table,
    out,
    *,
    max_seqlen_q,
    softmax_scale,
    causal,
    seqused_is_cumulative=False,
):
    """Return a plan only after the matching public call populated the JIT cache."""
    del v, seqused_k
    batch = cu_seqlens_q.numel() - 1
    num_warps = select_num_warps(max_seqlen_q, q.shape[2])
    bm = bm_for(num_warps)
    if not q.is_cuda or batch != 1 or q.shape[0] % bm != 0 or out is None or k.ndim != 4 or block_table.ndim != 2:
        return None

    num_heads = q.shape[1]
    head_dim = q.shape[2]
    num_kv_heads = k.shape[1]
    page_block_size = k.shape[2]
    key = (
        num_heads,
        num_kv_heads,
        head_dim,
        page_block_size,
        bool(causal),
        False,
        float(softmax_scale),
        batch,
        max_q_tiles_for(max_seqlen_q, num_warps),
        num_warps,
        True,
        bool(seqused_is_cumulative),
        tuple(k.stride()),
        (tuple(block_table.shape), tuple(block_table.stride())),
    )
    compiled = _PREFILL_CACHE.get(key)
    if compiled is None:
        return None
    return _VarlenPrefillLaunchPlan(compiled, torch.cuda.current_stream(q.device))
