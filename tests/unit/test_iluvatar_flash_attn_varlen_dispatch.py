#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Host-side routing tests for Iluvatar varlen FlashAttention.

Needs CUDA tensors because ``validate_block_table_shape`` requires a device
int32 table; kernels themselves are monkeypatched away.
"""

import math

import pytest
import torch

from kernels.attention.iluvatar import flash_attn_varlen
from kernels.attention.iluvatar import flash_attn_prefill

pytestmark = [pytest.mark.l2_device, pytest.mark.iluvatar_lower]

if not torch.cuda.is_available():
    pytest.skip("CUDA-compatible device is not available", allow_module_level=True)


def _paged_decode_inputs(*, max_blocks: int = 8, page: int = 16, total_q: int = 1):
    device = "cuda"
    q = torch.empty((total_q, 2, 128), device=device, dtype=torch.bfloat16)
    k = torch.empty((max_blocks, 1, page, 128), device=device, dtype=torch.bfloat16)
    v = torch.empty_like(k)
    cu_seqlens_q = torch.tensor([0, total_q], device=device, dtype=torch.int32)
    seqused_k = torch.tensor([1], device=device, dtype=torch.int32)
    block_table = torch.arange(max_blocks, device=device, dtype=torch.int32).reshape(1, max_blocks)
    return q, k, v, cu_seqlens_q, seqused_k, block_table


def _dense_inputs(*, total_q: int = 1, total_k: int = 8):
    device = "cuda"
    q = torch.empty((total_q, 2, 128), device=device, dtype=torch.bfloat16)
    k = torch.empty((total_k, 1, 128), device=device, dtype=torch.bfloat16)
    v = torch.empty_like(k)
    cu_seqlens_q = torch.tensor([0, total_q], device=device, dtype=torch.int32)
    cu_seqlens_k = torch.tensor([0, total_k], device=device, dtype=torch.int32)
    return q, k, v, cu_seqlens_q, cu_seqlens_k


def _route(monkeypatch, *, expect, q, k, v, cu_seqlens_q, **kwargs):
    calls = {}

    def fake_decode(*args, **kw):
        calls["decode"] = (args, kw)
        return "decode"

    def fake_prefill(*args, **kw):
        calls["prefill"] = (args, kw)
        return "prefill"

    monkeypatch.setattr(flash_attn_varlen, "_decode_dispatch", fake_decode)
    monkeypatch.setattr(flash_attn_varlen, "_prefill", fake_prefill)

    stream = kwargs.pop("stream", torch.cuda.current_stream(q.device))
    result = flash_attn_varlen.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        stream=stream,
        **kwargs,
    )
    assert result == expect
    assert expect in calls
    other = "prefill" if expect == "decode" else "decode"
    assert other not in calls
    return calls[expect]


def test_custom_scale_bypasses_decode_and_reaches_prefill(monkeypatch):
    q, k, v, cu_seqlens_q, seqused_k, block_table = _paged_decode_inputs()
    scale = 128 ** (-0.5) + 1e-10
    _, kwargs = _route(
        monkeypatch,
        expect="prefill",
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        block_table=block_table,
        seqused_k=seqused_k,
        softmax_scale=scale,
    )
    assert kwargs["softmax_scale"] == scale


def test_explicit_canonical_scale_keeps_decode_route(monkeypatch):
    q, k, v, cu_seqlens_q, seqused_k, block_table = _paged_decode_inputs()
    _route(
        monkeypatch,
        expect="decode",
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        block_table=block_table,
        seqused_k=seqused_k,
        softmax_scale=1.0 / math.sqrt(128),
    )


def test_multi_token_query_routes_to_prefill(monkeypatch):
    q, k, v, cu_seqlens_q, seqused_k, block_table = _paged_decode_inputs(total_q=2)
    _route(
        monkeypatch,
        expect="prefill",
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=2,
        block_table=block_table,
        seqused_k=seqused_k,
        softmax_scale=1.0 / math.sqrt(128),
    )


def test_small_decode_capacity_routes_to_prefill(monkeypatch):
    # 4 blocks * 16 = 64 < 128 decode capacity gate.
    q, k, v, cu_seqlens_q, seqused_k, block_table = _paged_decode_inputs(max_blocks=4)
    _route(
        monkeypatch,
        expect="prefill",
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        block_table=block_table,
        seqused_k=seqused_k,
        softmax_scale=1.0 / math.sqrt(128),
    )


def test_dense_varlen_never_uses_decode_kernel(monkeypatch):
    q, k, v, cu_seqlens_q, cu_seqlens_k = _dense_inputs()
    _route(
        monkeypatch,
        expect="prefill",
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=1,
        softmax_scale=1.0 / math.sqrt(128),
    )


def test_use_decode_kernel_false_forces_prefill(monkeypatch):
    q, k, v, cu_seqlens_q, seqused_k, block_table = _paged_decode_inputs()
    _route(
        monkeypatch,
        expect="prefill",
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        block_table=block_table,
        seqused_k=seqused_k,
        softmax_scale=1.0 / math.sqrt(128),
        use_decode_kernel=False,
    )


@pytest.mark.parametrize(
    ("make_out", "match"),
    [
        (lambda q: torch.empty((q.shape[0] + 1, *q.shape[1:]), device=q.device, dtype=q.dtype), "out must have shape"),
        (lambda q: torch.empty(q.shape, device=q.device, dtype=torch.float16), "same dtype and device"),
        (lambda q: torch.empty((*q.shape, 2), device=q.device, dtype=q.dtype)[..., 0], "out must be contiguous"),
    ],
    ids=["shape", "dtype", "contiguity"],
)
def test_rejects_invalid_out_before_launch(monkeypatch, make_out, match):
    q, k, v, cu_seqlens_q, seqused_k, block_table = _paged_decode_inputs(total_q=2)
    monkeypatch.setattr(
        flash_attn_varlen,
        "_prefill",
        lambda *args, **kwargs: pytest.fail("invalid out reached kernel launch"),
    )

    with pytest.raises(ValueError, match=match):
        flash_attn_varlen.flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            max_seqlen_q=2,
            block_table=block_table,
            seqused_k=seqused_k,
            out=make_out(q),
            stream=torch.cuda.current_stream(q.device),
        )


def test_prefill_cache_key_includes_block_table_shape_and_stride(monkeypatch):
    monkeypatch.setattr(flash_attn_prefill, "_PREFILL_CACHE", {})
    compile_calls = []

    def fake_build(**kwargs):
        return object()

    def fake_compile(launcher, *args):
        del launcher
        compile_calls.append(args[5])
        return lambda *launch_args: None

    monkeypatch.setattr(flash_attn_prefill, "_build_prefill_launcher", fake_build)
    monkeypatch.setattr(flash_attn_prefill.flyc, "compile", fake_compile)

    q, k, v, cu_q, seqused_k, _ = _paged_decode_inputs(total_q=2)
    for block_table in (
        torch.arange(8, device=q.device, dtype=torch.int32).reshape(1, 8),
        torch.arange(16, device=q.device, dtype=torch.int32).reshape(1, 16),
        torch.arange(16, device=q.device, dtype=torch.int32).reshape(1, 16)[:, ::2],
    ):
        flash_attn_prefill._prefill(
            q,
            k,
            v,
            cu_q,
            seqused_k,
            block_table,
            num_heads=2,
            num_kv_heads=1,
            head_dim=128,
            page_block_size=16,
            causal=True,
            upstream_cache_layout=False,
            softmax_scale=1.0 / math.sqrt(128),
            batch=1,
            max_seqlen_q=2,
            out=torch.empty_like(q),
            stream=torch.cuda.current_stream(q.device),
        )

    assert len(compile_calls) == 3


def test_aligned_single_sequence_q_keeps_full_bm_guard(monkeypatch):
    captured_q = []

    monkeypatch.setattr(flash_attn_prefill, "_PREFILL_CACHE", {})
    monkeypatch.setattr(flash_attn_prefill, "_build_prefill_launcher", lambda **kwargs: object())

    def fake_compile(launcher, *args):
        del launcher
        captured_q.append(args[0])
        return lambda *launch_args: None

    monkeypatch.setattr(flash_attn_prefill.flyc, "compile", fake_compile)

    q, k, v, cu_q, seqused_k, block_table = _paged_decode_inputs(total_q=16)
    flash_attn_prefill._prefill(
        q,
        k,
        v,
        cu_q,
        seqused_k,
        block_table,
        num_heads=2,
        num_kv_heads=1,
        head_dim=128,
        page_block_size=16,
        causal=True,
        upstream_cache_layout=False,
        softmax_scale=1.0 / math.sqrt(128),
        batch=1,
        max_seqlen_q=16,
        out=torch.empty_like(q),
        stream=torch.cuda.current_stream(q.device),
    )

    bm = flash_attn_prefill.bm_for(flash_attn_prefill.select_num_warps(16, 128))
    assert captured_q[0].numel() == (q.shape[0] + bm) * q.shape[1] * q.shape[2]
