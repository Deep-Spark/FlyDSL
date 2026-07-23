#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from ._iluvatar_flash_attn_kvcache_support import (
    _configure_iluvatar_env,  # noqa: F401
    _make_rotary_tables,
    _reference,
    flash_attn_with_kvcache,
    pytest,
    torch,
)

pytestmark = [pytest.mark.l2_device]


def test_flash_attn_with_kvcache_split_kv_decode():
    torch.manual_seed(2468)
    device = "cuda"
    dtype = torch.bfloat16
    bsz, seqlen_q, num_heads, num_kv_heads, head_dim = 1, 1, 2, 1, 16
    max_seqlen = 128
    cache_seqlens = torch.tensor([97], device=device, dtype=torch.int32)

    qkv = torch.empty(bsz, seqlen_q, num_heads + 2 * num_kv_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    k_cache = torch.empty(bsz, num_kv_heads, max_seqlen, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    v_cache = torch.empty_like(k_cache).uniform_(-1, 1)
    cos, sin = _make_rotary_tables(max_seqlen + 8, head_dim, dtype, device)

    ref, ref_k, ref_v = _reference(
        qkv,
        k_cache.clone(),
        v_cache.clone(),
        cache_seqlens,
        cos,
        sin,
        causal=True,
        window_size=(-1, -1),
    )
    out = flash_attn_with_kvcache(
        qkv,
        k_cache,
        v_cache,
        rotary_cos=cos,
        rotary_sin=sin,
        cache_seqlens=cache_seqlens,
        causal=True,
        rotary_interleaved=False,
        is_qkv_packed=True,
        num_splits=4,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(k_cache.float(), ref_k.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(v_cache.float(), ref_v.float(), atol=3e-2, rtol=3e-2)
