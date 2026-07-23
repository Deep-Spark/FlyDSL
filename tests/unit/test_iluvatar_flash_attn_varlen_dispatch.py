#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Host-side routing tests for Iluvatar varlen FlashAttention."""

import math

import pytest
import torch

from kernels.attention.iluvatar import flash_attn_varlen

pytestmark = [pytest.mark.l2_device]


def _decode_inputs():
    q = torch.empty((1, 2, 128), dtype=torch.bfloat16)
    k = torch.empty((8, 1, 16, 128), dtype=torch.bfloat16)
    v = torch.empty_like(k)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32)
    seqused_k = torch.tensor([1], dtype=torch.int32)
    block_table = torch.arange(8, dtype=torch.int32).reshape(1, 8)
    return q, k, v, cu_seqlens_q, seqused_k, block_table


def test_custom_scale_bypasses_decode_and_reaches_prefill(monkeypatch):
    q, k, v, cu_seqlens_q, seqused_k, block_table = _decode_inputs()
    calls = {}

    def fake_decode(*args, **kwargs):
        calls["decode"] = (args, kwargs)
        return "decode"

    def fake_prefill(*args, **kwargs):
        calls["prefill"] = (args, kwargs)
        return "prefill"

    monkeypatch.setattr(flash_attn_varlen, "_decode_dispatch", fake_decode)
    monkeypatch.setattr(flash_attn_varlen, "_prefill", fake_prefill)

    scale = 128 ** (-0.5) + 1e-10
    result = flash_attn_varlen.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        max_seqlen_q=1,
        block_table=block_table,
        seqused_k=seqused_k,
        softmax_scale=scale,
        stream=object(),
    )

    assert result == "prefill"
    assert "decode" not in calls
    assert calls["prefill"][1]["softmax_scale"] == scale


def test_explicit_canonical_scale_keeps_decode_route(monkeypatch):
    q, k, v, cu_seqlens_q, seqused_k, block_table = _decode_inputs()

    monkeypatch.setattr(flash_attn_varlen, "_decode_dispatch", lambda *args, **kwargs: "decode")
    monkeypatch.setattr(flash_attn_varlen, "_prefill", lambda *args, **kwargs: "prefill")

    result = flash_attn_varlen.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        max_seqlen_q=1,
        block_table=block_table,
        seqused_k=seqused_k,
        softmax_scale=1.0 / math.sqrt(128),
        stream=object(),
    )

    assert result == "decode"
