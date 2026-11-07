# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Prepared varlen prefill launchers and JIT plan cache."""

import os
from collections import OrderedDict

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
_MAX_PREFILL_LAUNCHES = 128
_PREFILL_CACHE: OrderedDict = OrderedDict()
# Grow-only Q DMA pads so multi-batch prefill does not malloc every layer.
_Q_PAD_CACHE: dict = {}
# Grow-only dense KV staging. Each packed sequence gets a zero-filled 16-row
# tail so masked P*V work can never consume NaN/Inf from the following sequence.
_DENSE_KV_PAD_CACHE: dict = {}
_MAX_STAGING_ENTRIES_PER_DEVICE = 16


def _should_cache_staging(cache, key, device) -> bool:
    if key in cache:
        return True
    entries = sum(
        1
        for cached_key in cache
        if len(cached_key) > 1 and cached_key[1] == device
    )
    return entries < _MAX_STAGING_ENTRIES_PER_DEVICE


def _compile_context_key(tensor):
    return (
        tensor.device,
        os.environ.get("FLYDSL_COMPILE_BACKEND"),
        os.environ.get("FLYDSL_RUNTIME_KIND"),
        os.environ.get("ARCH"),
    )


def _prefill_cache_get(key):
    compiled = _PREFILL_CACHE.get(key)
    if compiled is not None and hasattr(_PREFILL_CACHE, "move_to_end"):
        _PREFILL_CACHE.move_to_end(key)
    return compiled


def _prefill_cache_store(key, compiled) -> None:
    _PREFILL_CACHE[key] = compiled
    if hasattr(_PREFILL_CACHE, "move_to_end"):
        _PREFILL_CACHE.move_to_end(key)
        while len(_PREFILL_CACHE) > _MAX_PREFILL_LAUNCHES:
            _PREFILL_CACHE.popitem(last=False)


def clear_prefill_caches() -> None:
    _PREFILL_CACHE.clear()
    _Q_PAD_CACHE.clear()
    _DENSE_KV_PAD_CACHE.clear()


def _cached_q_pad(q, bm: int):
    """Copy packed Q into a reused buffer with one extra BM tile for DMA."""
    total_q = q.shape[0]
    num_heads, head_dim = q.shape[1], q.shape[2]
    need = total_q + bm
    key = (
        "qpad",
        q.device,
        q.dtype,
        num_heads,
        head_dim,
        torch.cuda.current_stream(q.device).cuda_stream,
    )
    cached = _Q_PAD_CACHE.get(key)
    if cached is None or cached.shape[0] < need:
        cap = need if cached is None else max(need, cached.shape[0])
        cached = torch.empty((cap, num_heads, head_dim), device=q.device, dtype=q.dtype)
        if _should_cache_staging(_Q_PAD_CACHE, key, q.device):
            _Q_PAD_CACHE[key] = cached
    padded = cached[:need]
    padded[:total_q].copy_(q)
    return padded


def _build_dense_stage_launcher(
    *,
    batch: int,
    total_k: int,
    num_kv_heads: int,
    head_dim: int,
    blocks_per_seq: int,
):
    row_elems = num_kv_heads * head_dim
    threads = 256

    @flyc.kernel
    def flash_attn_prefill_stage(
        K: fx.Pointer,
        V: fx.Pointer,
        CuK: fx.Pointer,
        SeqK: fx.Pointer,
        KOut: fx.Pointer,
        VOut: fx.Pointer,
        PaddedCu: fx.Pointer,
        SafeSeq: fx.Pointer,
    ):
        block = fx.Int32(fx.block_idx.x)
        b = block // fx.Int32(blocks_per_seq)
        shard = block % fx.Int32(blocks_per_seq)
        tid = fx.Int32(fx.thread_idx.x)

        def load_i32(ptr, idx):
            return fx.ptr_load(
                fx.add_offset(ptr, fx.make_int_tuple(idx)), fx.Int32
            )

        src_start = load_i32(CuK, b)
        src_start = (src_start > fx.Int32(0)).select(src_start, fx.Int32(0))
        src_start = (src_start < fx.Int32(total_k)).select(
            src_start, fx.Int32(total_k)
        )
        src_end = load_i32(CuK, b + fx.Int32(1))
        src_end = (src_end > src_start).select(src_end, src_start)
        src_end = (src_end < fx.Int32(total_k)).select(
            src_end, fx.Int32(total_k)
        )
        span = src_end - src_start
        span = (span > fx.Int32(0)).select(span, fx.Int32(0))
        requested = load_i32(SeqK, b)
        requested = (requested > fx.Int32(0)).select(requested, fx.Int32(0))
        visible = (requested < span).select(requested, span)
        padded_span = ((span + fx.Int32(15)) // fx.Int32(16)) * fx.Int32(16)

        dst_start = fx.Int32(0)
        for prior in fx.range(fx.Int32(0), b, fx.Int32(1)):
            prior_start = load_i32(CuK, prior)
            prior_start = (prior_start > fx.Int32(0)).select(
                prior_start, fx.Int32(0)
            )
            prior_start = (prior_start < fx.Int32(total_k)).select(
                prior_start, fx.Int32(total_k)
            )
            prior_end = load_i32(CuK, prior + fx.Int32(1))
            prior_end = (prior_end > prior_start).select(prior_end, prior_start)
            prior_end = (prior_end < fx.Int32(total_k)).select(
                prior_end, fx.Int32(total_k)
            )
            prior_span = prior_end - prior_start
            dst_start = dst_start + (
                (prior_span + fx.Int32(15)) // fx.Int32(16)
            ) * fx.Int32(16)

        if shard == fx.Int32(0) and tid == fx.Int32(0):
            fx.ptr_store(dst_start, fx.add_offset(PaddedCu, fx.make_int_tuple(b)))
            fx.ptr_store(visible, fx.add_offset(SafeSeq, fx.make_int_tuple(b)))
            if b == fx.Int32(batch - 1):
                fx.ptr_store(
                    dst_start + padded_span,
                    fx.add_offset(PaddedCu, fx.make_int_tuple(b + fx.Int32(1))),
                )

        total_elems = padded_span * fx.Int32(row_elems)
        elem_start = shard * fx.Int32(threads) + tid
        elem_step = fx.Int32(threads * blocks_per_seq)
        for elem in fx.range(elem_start, total_elems, elem_step):
            token = elem // fx.Int32(row_elems)
            inner = elem % fx.Int32(row_elems)
            dst_elem = (
                fx.Int64(dst_start + token) * fx.Int64(row_elems)
                + fx.Int64(inner)
            )
            if token < visible:
                src_elem = (
                    fx.Int64(src_start + token) * fx.Int64(row_elems)
                    + fx.Int64(inner)
                )
                k_value = fx.ptr_load(
                    fx.add_offset(K, fx.make_int_tuple(src_elem)), fx.BFloat16
                )
                v_value = fx.ptr_load(
                    fx.add_offset(V, fx.make_int_tuple(src_elem)), fx.BFloat16
                )
                fx.ptr_store(
                    k_value, fx.add_offset(KOut, fx.make_int_tuple(dst_elem))
                )
                fx.ptr_store(
                    v_value, fx.add_offset(VOut, fx.make_int_tuple(dst_elem))
                )
            else:
                fx.ptr_store(
                    fx.BFloat16(0.0),
                    fx.add_offset(KOut, fx.make_int_tuple(dst_elem)),
                )
                fx.ptr_store(
                    fx.BFloat16(0.0),
                    fx.add_offset(VOut, fx.make_int_tuple(dst_elem)),
                )

    @flyc.jit
    def launch(K, V, CuK, SeqK, KOut, VOut, PaddedCu, SafeSeq, stream=fx.Stream(None)):
        flash_attn_prefill_stage(
            fx.get_iter(K),
            fx.get_iter(V),
            fx.get_iter(CuK),
            fx.get_iter(SeqK),
            fx.get_iter(KOut),
            fx.get_iter(VOut),
            fx.get_iter(PaddedCu),
            fx.get_iter(SafeSeq),
        ).launch(
            grid=(batch * blocks_per_seq, 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    return launch


def _stage_dense_kv(k, v, cu_seqlens_k, seqused_k, *, batch: int, bm: int):
    total_k, num_kv_heads, head_dim = k.shape
    blocks_per_seq = min(
        128,
        max(
            1,
            (total_k * num_kv_heads * head_dim + batch * 1023)
            // (batch * 1024),
        ),
    )
    need = total_k + batch * 15 + bm
    key = (
        "dense_kv",
        k.device,
        k.dtype,
        batch,
        num_kv_heads,
        head_dim,
        torch.cuda.current_stream(k.device).cuda_stream,
    )
    cached = _DENSE_KV_PAD_CACHE.get(key)
    if cached is None or cached[0].shape[0] < need:
        cap = need if cached is None else max(need, cached[0].shape[0])
        k_pad = torch.empty(
            (cap, num_kv_heads, head_dim), device=k.device, dtype=k.dtype
        )
        v_pad = torch.empty_like(k_pad)
        padded_cu = torch.empty(batch + 1, device=k.device, dtype=torch.int32)
        safe_seq = torch.empty(batch, device=k.device, dtype=torch.int32)
        cached = (k_pad, v_pad, padded_cu, safe_seq)
        if _should_cache_staging(_DENSE_KV_PAD_CACHE, key, k.device):
            _DENSE_KV_PAD_CACHE[key] = cached
    k_pad, v_pad, padded_cu, safe_seq = cached
    k_stage = k_pad[:need]
    v_stage = v_pad[:need]
    compile_key = (
        "dense_stage",
        _compile_context_key(k),
        k.dtype,
        batch,
        total_k,
        blocks_per_seq,
        num_kv_heads,
        head_dim,
    )
    compiled = _prefill_cache_get(compile_key)
    stream = fx.Stream(torch.cuda.current_stream(k.device))
    args = (
        k,
        v,
        cu_seqlens_k,
        seqused_k,
        k_stage,
        v_stage,
        padded_cu,
        safe_seq,
        stream,
    )
    if compiled is None:
        launcher = _build_dense_stage_launcher(
            batch=batch,
            total_k=total_k,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            blocks_per_seq=blocks_per_seq,
        )
        compiled = flyc.compile(launcher, *args)
        _prefill_cache_store(compile_key, compiled)
    else:
        compiled(*args)
    return k_stage, v_stage, padded_cu, safe_seq


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
    if total_q == 0:
        return out if out is not None else torch.empty_like(q)
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
        q_pad = _cached_q_pad(q, bm)

    if paged:
        k_arg = k
        v_arg = v
        seq_arg = seqused_k
        cache_strides = tuple(k.stride())
        # BlockTable slot carries the paged block table.
        bt_arg = block_table
    else:
        k_arg, v_arg, padded_cu, seq_arg = _stage_dense_kv(
            k,
            v,
            cu_seqlens_k,
            seqused_k,
            batch=batch,
            bm=bm,
        )
        cache_strides = None
        # BlockTable slot instead carries padded dense starts [batch + 1].
        bt_arg = padded_cu
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
        _compile_context_key(q),
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
    args = (
        q_pad.reshape(-1),
        k_arg,
        v_arg,
        cu_seqlens_q,
        seq_arg,
        bt_arg,
        out_buf.reshape(-1),
        stream,
    )

    compiled = _prefill_cache_get(key)
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
        _prefill_cache_store(key, compiled)
    else:
        compiled(*args)
    result = out_buf[:total_q]
    if user_out is not None:
        return user_out
    return result


def _prefill_plan_signature(
    q, k, v, cu_seqlens_q, seqused_k, block_table, out
):
    tensors = (q, k, v, cu_seqlens_q, seqused_k, block_table, out)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return None
    return tuple(
        (tensor.device, tensor.dtype, tuple(tensor.shape), tuple(tensor.stride()))
        for tensor in tensors
    )


class _VarlenPrefillLaunchPlan:
    """Prepared paged prefill launch with dynamic tensor addresses."""

    __slots__ = (
        "compiled",
        "stream",
        "stream_ptr",
        "bm",
        "direct_q",
        "signature",
    )

    def __init__(
        self,
        compiled,
        current_stream,
        bm: int,
        direct_q: bool,
        signature,
    ):
        self.compiled = compiled
        self.stream = fx.Stream(current_stream)
        self.stream_ptr = current_stream.cuda_stream
        self.bm = bm
        self.direct_q = direct_q
        self.signature = signature

    def __call__(self, q, k, v, cu_seqlens_q, seqused_k, block_table, out):
        if (
            _prefill_plan_signature(
                q, k, v, cu_seqlens_q, seqused_k, block_table, out
            )
            != self.signature
        ):
            raise ValueError(
                "varlen prefill plan was called with incompatible tensor metadata"
            )
        q_arg = q if self.direct_q else _cached_q_pad(q, self.bm)
        current_stream = torch.cuda.current_stream(q.device)
        launch_stream = (
            self.stream
            if current_stream.cuda_stream == self.stream_ptr
            else fx.Stream(current_stream)
        )
        self.compiled(
            q_arg.reshape(-1),
            k,
            v,
            cu_seqlens_q,
            seqused_k,
            block_table,
            out.reshape(-1),
            launch_stream,
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
    batch = cu_seqlens_q.numel() - 1
    num_warps = select_num_warps(max_seqlen_q, q.shape[2])
    bm = bm_for(num_warps)
    if not q.is_cuda or batch <= 0 or out is None or k.ndim != 4 or block_table.ndim != 2:
        return None
    direct_q = batch == 1 and q.shape[0] % bm == 0

    num_heads = q.shape[1]
    head_dim = q.shape[2]
    num_kv_heads = k.shape[1]
    page_block_size = k.shape[2]
    key = (
        _compile_context_key(q),
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
    compiled = _prefill_cache_get(key)
    if compiled is None:
        return None
    return _VarlenPrefillLaunchPlan(
        compiled,
        torch.cuda.current_stream(q.device),
        bm,
        direct_q,
        _prefill_plan_signature(
            q, k, v, cu_seqlens_q, seqused_k, block_table, out
        ),
    )
