# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar fused NeoX RoPE and flash KV-cache correctness tests."""

import os
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.iluvatar_lower]

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        pytest.skip(f"torch is required for Iluvatar fused RoPE-cache tests: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA-compatible Iluvatar device is not available")
    return torch


def _configure_iluvatar_env(monkeypatch):
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


def test_iluvatar_fused_rope_cache_rejects_partial_warp_head_dim():
    from kernels.attention.iluvatar.fused_rope_cache_kernel import build_fused_rope_cache_module

    with pytest.raises(ValueError, match="warp size"):
        build_fused_rope_cache_module(head_dim=96)


def _rope_reference(torch, q, k, v, cos_cache, sin_cache, positions, slots, key_cache, value_cache, block_size):
    half_dim = q.shape[-1] // 2
    cos = cos_cache[positions.long()].unsqueeze(1)
    sin = sin_cache[positions.long()].unsqueeze(1)
    cos = torch.cat((cos, cos), dim=-1)
    sin = torch.cat((sin, sin), dim=-1)

    def rotate(src):
        first, second = src[..., :half_dim], src[..., half_dim:]
        return torch.cat(
            (
                first * cos[..., :half_dim] - second * sin[..., :half_dim],
                second * cos[..., half_dim:] + first * sin[..., half_dim:],
            ),
            dim=-1,
        )

    q_out = rotate(q)
    k_out = rotate(k)
    key_out = key_cache.clone()
    value_out = value_cache.clone()
    for token, slot in enumerate(slots.cpu().tolist()):
        if slot >= 0:
            block, offset = divmod(slot, block_size)
            key_out[block, offset] = k_out[token]
            value_out[block, offset] = v[token]
    return q_out, k_out, key_out, value_out


@pytest.mark.parametrize(
    "dtype_name,head_dim,rtol,atol",
    [
        ("float16", 64, 1e-2, 1e-2),
        ("bfloat16", 128, 2e-2, 2e-2),
    ],
)
def test_iluvatar_fused_rope_cache_matches_reference(monkeypatch, dtype_name, head_dim, rtol, atol):
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)

    from kernels.attention.iluvatar.fused_rope_cache_kernel import build_fused_rope_cache_module

    tokens, q_heads, kv_heads, block_size, num_blocks = 3, 4, 2, 4, 2
    dtype = getattr(torch, dtype_name)
    dtype_str = "f16" if dtype_name == "float16" else "bf16"
    torch.manual_seed(42)
    q = torch.randn((tokens, q_heads, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((tokens, kv_heads, head_dim), device="cuda", dtype=dtype)
    v = torch.randn((tokens, kv_heads, head_dim), device="cuda", dtype=dtype)
    cos_cache = torch.randn((16, head_dim // 2), device="cuda", dtype=dtype)
    sin_cache = torch.randn((16, head_dim // 2), device="cuda", dtype=dtype)
    positions = torch.tensor([1, 5, 7], device="cuda", dtype=torch.int32)
    slots = torch.tensor([0, -1, 2], device="cuda", dtype=torch.int32)
    key_cache = torch.zeros((num_blocks, block_size, kv_heads, head_dim), device="cuda", dtype=dtype)
    value_cache = torch.zeros_like(key_cache)
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    scales = torch.ones((1,), device="cuda", dtype=torch.float32)

    q_ref, k_ref, key_ref, value_ref = _rope_reference(
        torch, q, k, v, cos_cache, sin_cache, positions, slots, key_cache, value_cache, block_size
    )
    launch = build_fused_rope_cache_module(
        head_dim=head_dim,
        num_q_heads=q_heads,
        num_kv_heads=kv_heads,
        block_size=block_size,
        dtype_str=dtype_str,
    )
    launch(
        q,
        k,
        v,
        positions,
        cos_cache,
        sin_cache,
        slots,
        key_cache,
        value_cache,
        q_out,
        k_out,
        tokens,
        scales,
        scales,
        stream=torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(q_out, q_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(k_out, k_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(key_cache, key_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(value_cache, value_ref, rtol=rtol, atol=atol)
