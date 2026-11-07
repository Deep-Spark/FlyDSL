#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from ._iluvatar_flash_attn_kvcache_support import (
    _apply_rope,
    _attention_ref,
    _configure_iluvatar_env,  # noqa: F401
    _dense_to_paged,
    _dense_to_paged_upstream,
    _make_rotary_tables,
    _paged_to_dense,
    _paged_to_dense_upstream,
    _reference,
    _reference_upstream,
    flash_attn_with_kvcache,
    fx,
    pytest,
    torch,
)

pytestmark = [pytest.mark.l2_device, pytest.mark.iluvatar_lower]


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("seqlen_q", [1, 3])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "f16"])
@pytest.mark.parametrize("rotary_interleaved", [False, True], ids=["neox", "interleaved"])
def test_flash_attn_with_kvcache(paged: bool, seqlen_q: int, dtype: torch.dtype, rotary_interleaved: bool):
    torch.manual_seed(1234 + seqlen_q + int(paged))
    device = "cuda"
    bsz, num_heads, num_kv_heads, head_dim = 2, 4, 2, 16
    max_seqlen = 8
    block_size = 16
    max_blocks = 1
    cache_seqlens = torch.tensor([5, 7], device=device, dtype=torch.int32)
    if seqlen_q > 1:
        cache_seqlens = torch.tensor([seqlen_q, seqlen_q], device=device, dtype=torch.int32)

    qkv = torch.empty(bsz, seqlen_q, num_heads + 2 * num_kv_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    k_dense = torch.empty(bsz, num_kv_heads, max_seqlen, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    v_dense = torch.empty_like(k_dense).uniform_(-1, 1)
    cos, sin = _make_rotary_tables(max_seqlen + 8, head_dim, dtype, device)

    block_table = None
    k_arg, v_arg = k_dense.clone(), v_dense.clone()
    if paged:
        block_table = torch.arange(bsz * max_blocks, device=device, dtype=torch.int32).reshape(bsz, max_blocks)
        num_blocks = bsz * max_blocks
        k_arg = _dense_to_paged(k_dense, block_table, block_size, num_blocks)
        v_arg = _dense_to_paged(v_dense, block_table, block_size, num_blocks)

    ref, ref_k, ref_v = _reference(
        qkv,
        k_arg.clone(),
        v_arg.clone(),
        cache_seqlens,
        cos,
        sin,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        rotary_interleaved=rotary_interleaved,
    )
    out = flash_attn_with_kvcache(
        qkv,
        k_arg,
        v_arg,
        rotary_cos=cos,
        rotary_sin=sin,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        causal=True,
        rotary_interleaved=rotary_interleaved,
        is_qkv_packed=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)
    if paged:
        got_k = _paged_to_dense(k_arg, block_table, ref_k.shape[2])
        got_v = _paged_to_dense(v_arg, block_table, ref_v.shape[2])
    else:
        got_k, got_v = k_arg, v_arg
    torch.testing.assert_close(got_k.float(), ref_k.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(got_v.float(), ref_v.float(), atol=3e-2, rtol=3e-2)


def test_flash_attn_with_kvcache_paged_update_crosses_page_boundary():
    torch.manual_seed(24601)
    device = "cuda"
    dtype = torch.bfloat16
    batch, seqlen_q, num_heads, num_kv_heads, head_dim = 1, 2, 4, 2, 16
    page_size, max_seqlen = 16, 32
    cache_seqlens = torch.tensor([15], device=device, dtype=torch.int32)
    q = torch.empty(batch, seqlen_q, num_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    k = torch.empty(batch, seqlen_q, num_kv_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    v = torch.empty_like(k).uniform_(-1, 1)
    dense_k = torch.empty(batch, max_seqlen, num_kv_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    dense_v = torch.empty_like(dense_k).uniform_(-1, 1)
    block_table = torch.tensor([[1, 0]], device=device, dtype=torch.int32)
    k_cache = _dense_to_paged_upstream(dense_k, block_table, page_size, 2)
    v_cache = _dense_to_paged_upstream(dense_v, block_table, page_size, 2)
    identity_cos = torch.ones(max_seqlen, head_dim // 2, device=device, dtype=dtype)
    zero_sin = torch.zeros_like(identity_cos)
    ref, ref_k, ref_v = _reference_upstream(
        q,
        k_cache.clone(),
        v_cache.clone(),
        cache_seqlens,
        identity_cos,
        zero_sin,
        k=k,
        v=v,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
    )

    out = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k=k,
        v=v,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        causal=True,
    )
    torch.cuda.synchronize()
    got_k = _paged_to_dense_upstream(k_cache, block_table, max_seqlen)
    got_v = _paged_to_dense_upstream(v_cache, block_table, max_seqlen)
    torch.testing.assert_close(got_k.float(), ref_k.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(got_v.float(), ref_v.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)


def test_flash_attn_with_kvcache_vectorized_b48_nhd_update():
    batch, page, num_heads, num_kv_heads, head_dim = 48, 16, 2, 1, 16
    q = torch.randn(batch, 1, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, 1, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    k_cache = torch.zeros(
        batch, page, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.arange(batch, device="cuda", dtype=torch.int32).view(batch, 1)
    cache_seqlens = torch.zeros(batch, device="cuda", dtype=torch.int32)

    flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k=k,
        v=v,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        causal=True,
        force_upstream_cache_layout=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(k_cache[:, 0].float(), k[:, 0].float(), atol=0, rtol=0)
    torch.testing.assert_close(v_cache[:, 0].float(), v[:, 0].float(), atol=0, rtol=0)


def test_flash_attn_with_kvcache_paged_nhd_d256_without_densify():
    torch.manual_seed(2468)
    batch, seqlen_q, num_heads, num_kv_heads, head_dim = 1, 2, 4, 1, 256
    page, max_seqlen, cache_len = 16, 32, 17
    q = torch.randn(
        batch, seqlen_q, num_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    dense_k = torch.randn(
        batch, max_seqlen, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    dense_v = torch.randn_like(dense_k)
    block_table = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
    k_cache = _dense_to_paged_upstream(dense_k, block_table, page, 2)
    v_cache = _dense_to_paged_upstream(dense_v, block_table, page, 2)
    cache_seqlens = torch.tensor([cache_len], device="cuda", dtype=torch.int32)
    ref = _attention_ref(
        q,
        dense_k,
        dense_v,
        cache_seqlens,
        causal=True,
        window_size=(-1, -1),
    )

    out = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        causal=True,
        force_upstream_cache_layout=True,
        max_context_len=cache_len,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out.float(), ref.float(), atol=5e-2, rtol=5e-2)


def test_flash_attn_with_kvcache_decode_varlen_fallback_is_nonrecursive():
    torch.manual_seed(2026)
    device = "cuda"
    dtype = torch.bfloat16
    batch, num_heads, num_kv_heads, head_dim = 1, 32, 1, 128
    q = torch.empty(batch, 1, num_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    dense_k = torch.empty(batch, 128, num_kv_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    dense_v = torch.empty_like(dense_k).uniform_(-1, 1)
    cache_seqlens = torch.tensor([97], device=device, dtype=torch.int32)
    block_table = torch.tensor([[0]], device=device, dtype=torch.int32)
    k_cache = dense_k.reshape(1, 128, num_kv_heads, head_dim)
    v_cache = dense_v.reshape_as(k_cache)
    ref = _attention_ref(
        q,
        dense_k,
        dense_v,
        cache_seqlens,
        causal=True,
        window_size=(-1, -1),
    )

    out = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        causal=True,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("with_update", [False, True])
def test_flash_attn_with_kvcache_upstream_api(paged: bool, with_update: bool):
    torch.manual_seed(4321 + int(paged) + 10 * int(with_update))
    device = "cuda"
    dtype = torch.bfloat16
    bsz, seqlen_q, num_heads, num_kv_heads, head_dim = 2, 1, 4, 2, 16
    max_seqlen = 8
    block_size = 16
    max_blocks = 1
    cache_seqlens = torch.tensor([4, 6], device=device, dtype=torch.int32)

    q = torch.empty(bsz, seqlen_q, num_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    k = torch.empty(bsz, seqlen_q, num_kv_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    v = torch.empty_like(k).uniform_(-1, 1)
    k_dense = torch.empty(bsz, max_seqlen, num_kv_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    v_dense = torch.empty_like(k_dense).uniform_(-1, 1)
    cos, sin = _make_rotary_tables(max_seqlen + 8, head_dim, dtype, device)

    block_table = None
    k_arg, v_arg = k_dense.clone(), v_dense.clone()
    if paged:
        block_table = torch.arange(bsz * max_blocks, device=device, dtype=torch.int32).reshape(bsz, max_blocks)
        num_blocks = bsz * max_blocks
        k_arg = _dense_to_paged_upstream(k_dense, block_table, block_size, num_blocks)
        v_arg = _dense_to_paged_upstream(v_dense, block_table, block_size, num_blocks)

    ref, ref_k, ref_v = _reference_upstream(
        q,
        k_arg.clone(),
        v_arg.clone(),
        cache_seqlens,
        cos,
        sin,
        k=k if with_update else None,
        v=v if with_update else None,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
        rotary_interleaved=True,
    )
    out = flash_attn_with_kvcache(
        q,
        k_arg,
        v_arg,
        k=k if with_update else None,
        v=v if with_update else None,
        rotary_cos=cos if with_update else None,
        rotary_sin=sin if with_update else None,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        causal=True,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)
    if paged:
        got_k = _paged_to_dense_upstream(k_arg, block_table, ref_k.shape[1])
        got_v = _paged_to_dense_upstream(v_arg, block_table, ref_v.shape[1])
    else:
        got_k, got_v = k_arg, v_arg
    torch.testing.assert_close(got_k.float(), ref_k.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(got_v.float(), ref_v.float(), atol=3e-2, rtol=3e-2)


def test_flash_attn_with_kvcache_dense_noncausal_neox_update_positions():
    torch.manual_seed(31415)
    device = "cuda"
    dtype = torch.bfloat16
    batch, seqlen_q, num_heads, num_kv_heads, head_dim = 1, 3, 4, 2, 16
    q = torch.empty(batch, seqlen_q, num_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    k = torch.empty(batch, seqlen_q, num_kv_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    v = torch.empty_like(k).uniform_(-1, 1)
    k_cache = torch.empty(batch, 12, num_kv_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    v_cache = torch.empty_like(k_cache).uniform_(-1, 1)
    cache_seqlens = torch.tensor([4], device=device, dtype=torch.int32)
    cos, sin = _make_rotary_tables(16, head_dim, dtype, device)
    ref, ref_k, ref_v = _reference_upstream(
        q,
        k_cache.clone(),
        v_cache.clone(),
        cache_seqlens,
        cos,
        sin,
        k=k,
        v=v,
        causal=False,
        window_size=(-1, -1),
        rotary_interleaved=False,
    )

    out = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k=k,
        v=v,
        rotary_cos=cos,
        rotary_sin=sin,
        cache_seqlens=cache_seqlens,
        causal=False,
        rotary_interleaved=False,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(k_cache.float(), ref_k.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(v_cache.float(), ref_v.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("query_order_lengths", [False, True])
def test_flash_attn_with_kvcache_dense_cache_batch_idx_update(monkeypatch, query_order_lengths: bool):
    """Dense cache rows are staged, updated, and scattered by cache_batch_idx."""
    torch.manual_seed(97531)
    device = "cuda"
    dtype = torch.bfloat16
    batch_size, cache_batch, seqlen_q = 2, 3, 1
    num_heads, num_kv_heads, head_dim, max_seqlen = 4, 2, 16, 8
    cache_batch_idx = torch.tensor([2, 0], device=device, dtype=torch.int32)
    cache_seqlens = (
        torch.tensor([6, 3], device=device, dtype=torch.int32)
        if query_order_lengths
        else torch.tensor([3, 4, 6], device=device, dtype=torch.int32)
    )

    q = torch.empty(batch_size, seqlen_q, num_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    k = torch.empty(batch_size, seqlen_q, num_kv_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    v = torch.empty_like(k).uniform_(-1, 1)
    k_cache = torch.empty(cache_batch, max_seqlen, num_kv_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    v_cache = torch.empty_like(k_cache).uniform_(-1, 1)
    cos, sin = _make_rotary_tables(max_seqlen + 1, head_dim, dtype, device)

    selected_lens = cache_seqlens if query_order_lengths else cache_seqlens.index_select(0, cache_batch_idx)
    ref_out, ref_k, ref_v = _reference_upstream(
        q,
        k_cache.index_select(0, cache_batch_idx),
        v_cache.index_select(0, cache_batch_idx),
        selected_lens,
        cos,
        sin,
        k=k,
        v=v,
        causal=True,
        window_size=(-1, -1),
    )
    expected_k = k_cache.clone()
    expected_v = v_cache.clone()
    expected_k.index_copy_(0, cache_batch_idx.to(torch.long), ref_k)
    expected_v.index_copy_(0, cache_batch_idx.to(torch.long), ref_v)

    out = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k=k,
        v=v,
        rotary_cos=cos,
        rotary_sin=sin,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=cache_batch_idx,
        causal=True,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref_out.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(k_cache.float(), expected_k.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(v_cache.float(), expected_v.float(), atol=3e-2, rtol=3e-2)

    monkeypatch.setenv("FLYDSL_KVCACHE_STRICT_CHECKS", "1")
    with pytest.raises(ValueError, match="must not contain duplicate"):
        flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=torch.tensor([0, 0], device=device, dtype=torch.int32),
            causal=True,
        )

    with pytest.raises(NotImplementedError, match="only supported with dense"):
        flash_attn_with_kvcache(
            q,
            torch.empty(2, 16, num_kv_heads, head_dim, device=device, dtype=dtype),
            torch.empty(2, 16, num_kv_heads, head_dim, device=device, dtype=dtype),
            cache_seqlens=torch.tensor([1, 1], device=device, dtype=torch.int32),
            cache_batch_idx=torch.tensor([0, 1], device=device, dtype=torch.int32),
            block_table=torch.zeros((batch_size, 1), device=device, dtype=torch.int32),
        )


def test_flash_attn_with_kvcache_dense_cache_leftpad():
    """Dense cache left padding offsets K/V addressing without changing positions."""
    torch.manual_seed(86420)
    device = "cuda"
    dtype = torch.bfloat16
    batch_size, seqlen_q, num_heads, num_kv_heads, head_dim, capacity = 2, 1, 4, 2, 16, 8
    cache_seqlens = torch.tensor([3, 4], device=device, dtype=torch.int32)
    cache_leftpad = torch.tensor([2, 1], device=device, dtype=torch.int32)
    q = torch.empty(batch_size, seqlen_q, num_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    k_cache = torch.empty(batch_size, capacity, num_kv_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    v_cache = torch.empty_like(k_cache).uniform_(-1, 1)

    logical_k = torch.zeros_like(k_cache)
    logical_v = torch.zeros_like(v_cache)
    for batch_idx in range(batch_size):
        leftpad = int(cache_leftpad[batch_idx])
        logical_k[batch_idx, : capacity - leftpad] = k_cache[batch_idx, leftpad:]
        logical_v[batch_idx, : capacity - leftpad] = v_cache[batch_idx, leftpad:]
    ref = _attention_ref(q, logical_k, logical_v, cache_seqlens, causal=True, window_size=(-1, -1))

    out = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        cache_leftpad=cache_leftpad,
        causal=True,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)


def test_flash_attn_with_kvcache_custom_softmax_scale_scalar():
    """A non-default scale takes the scalar path and changes attention scores."""
    torch.manual_seed(8765)
    device = "cuda"
    dtype = torch.bfloat16
    bsz, seqlen_q, num_heads, num_kv_heads, head_dim, max_seqlen = 1, 1, 2, 1, 16, 8
    cache_seqlens = torch.tensor([6], device=device, dtype=torch.int32)
    softmax_scale = 0.25

    q = torch.empty(bsz, seqlen_q, num_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    k_cache = torch.empty(bsz, max_seqlen, num_kv_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    v_cache = torch.empty_like(k_cache).uniform_(-1, 1)
    ref = _attention_ref(
        q,
        k_cache,
        v_cache,
        cache_seqlens,
        causal=True,
        window_size=(-1, -1),
        softmax_scale=softmax_scale,
    )

    out = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        causal=True,
        softmax_scale=softmax_scale,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)


def test_flash_attn_with_kvcache_softcap_scalar():
    torch.manual_seed(9753)
    device = "cuda"
    dtype = torch.bfloat16
    softcap = 0.75
    q = torch.empty(1, 1, 2, 16, device=device, dtype=dtype).uniform_(-1, 1)
    k_cache = torch.empty(1, 8, 1, 16, device=device, dtype=dtype).uniform_(-1, 1)
    v_cache = torch.empty_like(k_cache).uniform_(-1, 1)
    cache_seqlens = torch.tensor([6], device=device, dtype=torch.int32)
    ref = _attention_ref(q, k_cache, v_cache, cache_seqlens, causal=True, window_size=(-1, -1), softcap=softcap)

    out = flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=cache_seqlens, causal=True, softcap=softcap)
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("rotary_interleaved", [False, True], ids=["neox", "interleaved"])
def test_flash_attn_with_kvcache_attention_only_rotary(rotary_interleaved: bool):
    torch.manual_seed(8642)
    device = "cuda"
    dtype = torch.bfloat16
    batch, seqlen_q, num_heads, num_kv_heads, head_dim = 2, 3, 4, 2, 16
    q = torch.empty(batch, seqlen_q, num_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    k_cache = torch.empty(batch, 8, num_kv_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    v_cache = torch.empty_like(k_cache).uniform_(-1, 1)
    cache_seqlens = torch.tensor([5, 7], device=device, dtype=torch.int32)
    cos, sin = _make_rotary_tables(16, head_dim, dtype, device)
    positions = cache_seqlens[:, None].expand(batch, seqlen_q)
    q_rot = _apply_rope(q, cos, sin, positions, interleaved=rotary_interleaved)
    ref = _attention_ref(
        q_rot,
        k_cache,
        v_cache,
        cache_seqlens,
        causal=False,
        window_size=(-1, -1),
    )

    out = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        rotary_cos=cos,
        rotary_sin=sin,
        cache_seqlens=cache_seqlens,
        causal=False,
        rotary_interleaved=rotary_interleaved,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)


def test_flash_attn_with_kvcache_rejects_short_rotary_table():
    device = "cuda"
    dtype = torch.bfloat16
    q = torch.empty(1, 1, 2, 16, device=device, dtype=dtype)
    k_cache = torch.empty(1, 256, 1, 16, device=device, dtype=dtype)
    v_cache = torch.empty_like(k_cache)
    cos, sin = _make_rotary_tables(128, 16, dtype, device)

    with pytest.raises(ValueError, match="insufficient rows"):
        flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            rotary_cos=cos,
            rotary_sin=sin,
            cache_seqlens=128,
            causal=True,
        )


def test_flash_attn_with_kvcache_rejects_odd_rotary_head_dim():
    q = torch.empty(1, 1, 2, 15, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.empty(1, 8, 1, 15, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.empty_like(k_cache)
    cos = torch.empty(8, 7, device="cuda", dtype=torch.bfloat16)
    sin = torch.empty_like(cos)

    with pytest.raises(ValueError, match="even head dimension"):
        flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            rotary_cos=cos,
            rotary_sin=sin,
            cache_seqlens=0,
            rotary_interleaved=True,
        )


def test_flash_attn_with_kvcache_rejects_head_dim_above_256():
    q = torch.empty(1, 1, 2, 257, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.empty(1, 8, 1, 257, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.empty_like(k_cache)

    with pytest.raises(ValueError, match=r"\[1, 256\]"):
        flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=1)


def test_flash_attn_with_kvcache_packed_forced_nhd_layout():
    torch.manual_seed(7531)
    device = "cuda"
    dtype = torch.bfloat16
    batch, num_heads, num_kv_heads, head_dim = 1, 4, 2, 16
    qkv = torch.empty(batch, 1, num_heads + 2 * num_kv_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    k_cache = torch.empty(batch, 8, num_kv_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    v_cache = torch.empty_like(k_cache).uniform_(-1, 1)
    cache_seqlens = torch.tensor([5], device=device, dtype=torch.int32)
    q_part, k_new, v_new = qkv.split([num_heads, num_kv_heads, num_kv_heads], dim=2)
    ref_k, ref_v = k_cache.clone(), v_cache.clone()
    ref_k[:, 4] = k_new[:, 0]
    ref_v[:, 4] = v_new[:, 0]
    ref = _attention_ref(
        q_part,
        ref_k,
        ref_v,
        cache_seqlens,
        causal=True,
        window_size=(-1, -1),
    )

    out = flash_attn_with_kvcache(
        qkv,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        causal=True,
        is_qkv_packed=True,
        force_upstream_cache_layout=True,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(k_cache.float(), ref_k.float(), atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(v_cache.float(), ref_v.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize(
    ("is_qkv_packed", "cache_seqlens", "message"),
    [
        (False, 8, "exceeds cache capacity"),
        (True, 0, "must include at least"),
    ],
)
def test_flash_attn_with_kvcache_rejects_invalid_cache_lengths(is_qkv_packed: bool, cache_seqlens: int, message: str):
    device = "cuda"
    dtype = torch.bfloat16
    bsz, seqlen_q, num_heads, num_kv_heads, head_dim, capacity = 1, 1, 2, 1, 16, 8
    if is_qkv_packed:
        q = torch.empty(bsz, seqlen_q, num_heads + 2 * num_kv_heads, head_dim, device=device, dtype=dtype)
        k_cache = torch.empty(bsz, num_kv_heads, capacity, head_dim, device=device, dtype=dtype)
        k = v = None
    else:
        q = torch.empty(bsz, seqlen_q, num_heads, head_dim, device=device, dtype=dtype)
        k_cache = torch.empty(bsz, capacity, num_kv_heads, head_dim, device=device, dtype=dtype)
        k = torch.empty(bsz, seqlen_q, num_kv_heads, head_dim, device=device, dtype=dtype)
        v = torch.empty_like(k)
    v_cache = torch.empty_like(k_cache)

    with pytest.raises(ValueError, match=message):
        flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            cache_seqlens=cache_seqlens,
            is_qkv_packed=is_qkv_packed,
        )


def test_flash_attn_with_kvcache_rejects_invalid_paged_block_table(monkeypatch):
    monkeypatch.setenv("FLYDSL_KVCACHE_STRICT_CHECKS", "1")
    device = "cuda"
    dtype = torch.bfloat16
    q = torch.empty(1, 1, 2, 16, device=device, dtype=dtype)
    k_cache = torch.empty(1, 16, 1, 16, device=device, dtype=dtype)
    v_cache = torch.empty_like(k_cache)
    block_table = torch.tensor([[1]], device=device, dtype=torch.int32)

    with pytest.raises(ValueError, match="physical cache blocks"):
        flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=1,
            block_table=block_table,
        )


def test_flash_attn_with_kvcache_accepts_unused_padded_block_entries():
    torch.manual_seed(1122)
    device = "cuda"
    dtype = torch.bfloat16
    q = torch.empty(1, 1, 2, 16, device=device, dtype=dtype).uniform_(-1, 1)
    dense_k = torch.empty(1, 32, 1, 16, device=device, dtype=dtype).uniform_(-1, 1)
    dense_v = torch.empty_like(dense_k).uniform_(-1, 1)
    k_cache = dense_k[:, :16].clone().reshape(1, 16, 1, 16)
    v_cache = dense_v[:, :16].clone().reshape(1, 16, 1, 16)
    block_table = torch.tensor([[0, -1]], device=device, dtype=torch.int32)
    cache_seqlens = torch.tensor([5], device=device, dtype=torch.int32)
    ref = _attention_ref(
        q,
        dense_k,
        dense_v,
        cache_seqlens,
        causal=True,
        window_size=(-1, -1),
    )

    out = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        causal=True,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)


def test_flash_attn_with_kvcache_rejects_noncurrent_stream():
    q = torch.empty(1, 1, 2, 16, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.empty(1, 8, 1, 16, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.empty_like(k_cache)
    other_stream = torch.cuda.Stream()

    with pytest.raises(NotImplementedError, match="current PyTorch stream"):
        flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=1,
            stream=fx.Stream(other_stream),
        )


def test_flash_attn_with_kvcache_rejects_unsupported_cache_strides():
    q = torch.empty(1, 1, 2, 16, device="cuda", dtype=torch.bfloat16)
    dense_k = torch.empty(1, 8, 1, 17, device="cuda", dtype=torch.bfloat16)[..., :16]
    dense_v = torch.empty_like(dense_k)
    with pytest.raises(ValueError, match="canonical contiguous strides"):
        flash_attn_with_kvcache(q, dense_k, dense_v, cache_seqlens=1)

    paged_k = torch.empty(1, 1, 16, 16, device="cuda", dtype=torch.bfloat16)
    paged_v = torch.empty(1, 1, 16, 17, device="cuda", dtype=torch.bfloat16)[..., :16]
    with pytest.raises(ValueError, match="matching strides"):
        flash_attn_with_kvcache(
            q,
            paged_k,
            paged_v,
            cache_seqlens=1,
            block_table=torch.zeros((1, 1), device="cuda", dtype=torch.int32),
            force_upstream_cache_layout=False,
        )


def test_flash_attn_with_kvcache_validates_before_paged_update():
    q = torch.empty(1, 1, 2, 16, device="cuda", dtype=torch.bfloat16)
    k = torch.empty(1, 1, 1, 16, device="cuda", dtype=torch.bfloat16)
    v = torch.empty_like(k)
    k_cache = torch.zeros(1, 16, 1, 16, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    original_k, original_v = k_cache.clone(), v_cache.clone()

    with pytest.raises(ValueError, match="out must have shape"):
        flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            k=k,
            v=v,
            cache_seqlens=0,
            block_table=torch.zeros((1, 1), device="cuda", dtype=torch.int32),
            out=torch.empty(1, 1, 1, 16, device="cuda", dtype=torch.bfloat16),
        )
    torch.testing.assert_close(k_cache, original_k)
    torch.testing.assert_close(v_cache, original_v)


def test_flash_attn_with_kvcache_rejects_small_host_known_max_context():
    q = torch.empty(1, 1, 2, 16, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.empty(1, 8, 1, 16, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.empty_like(k_cache)
    with pytest.raises(ValueError, match="host-known visible cache length"):
        flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=5, max_context_len=4)
