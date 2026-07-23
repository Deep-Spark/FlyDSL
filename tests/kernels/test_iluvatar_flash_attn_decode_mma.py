#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from kernels.attention.iluvatar.flash_attn_kvcache_split_reduce import _pipelined_reduce_warp_count

from ._iluvatar_flash_attn_kvcache_support import (
    _attention_ref,
    _configure_iluvatar_env,  # noqa: F401
    _dense_to_paged,
    _dense_to_paged_upstream,
    _make_rotary_tables,
    _paged_to_dense,
    _reference,
    _reference_upstream,
    flash_attn_kvcache_module,
    flash_attn_with_kvcache,
    pytest,
    torch,
)

pytestmark = [pytest.mark.l2_device]


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 2), (4, 4), (8, 1)])
def test_flash_attn_with_kvcache_mma_decode(paged: bool, num_heads: int, num_kv_heads: int):
    """HEAD_DIM=128 bf16 decode path that dispatches to the MMA decode kernel."""
    torch.manual_seed(7 + num_heads + int(paged))
    device = "cuda"
    dtype = torch.bfloat16
    bsz, seqlen_q, head_dim = 2, 1, 128
    max_seqlen = 128
    # Use a single 128-wide page so the (separately) validated cache-update path
    # stays on its block-0 fast path; the MMA multi-block paged gather is covered
    # by test_flash_attn_with_kvcache_mma_decode_paged_multiblock.
    block_size = 128
    max_blocks = max_seqlen // block_size
    cache_seqlens = torch.tensor([128, 100], device=device, dtype=torch.int32)

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
        rotary_interleaved=False,
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


@pytest.mark.parametrize("num_heads,num_kv_heads", [(8, 2), (4, 4)])
def test_flash_attn_with_kvcache_mma_decode_paged_multiblock(num_heads: int, num_kv_heads: int):
    """Attention-only (no cache update) multi-block paged MMA decode read."""
    torch.manual_seed(99 + num_heads)
    device = "cuda"
    dtype = torch.bfloat16
    bsz, seqlen_q, head_dim = 2, 1, 128
    max_seqlen = 256
    block_size = 16
    max_blocks = max_seqlen // block_size
    cache_seqlens = torch.tensor([256, 200], device=device, dtype=torch.int32)

    # upstream attention-only path: separate q, no k/v -> update_cache disabled.
    q = torch.empty(bsz, seqlen_q, num_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    k_dense = torch.empty(bsz, max_seqlen, num_kv_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    v_dense = torch.empty_like(k_dense).uniform_(-1, 1)
    cos, sin = _make_rotary_tables(max_seqlen + 8, head_dim, dtype, device)

    block_table = torch.randperm(bsz * max_blocks, device=device, dtype=torch.int32).reshape(bsz, max_blocks)
    num_blocks = bsz * max_blocks
    k_arg = _dense_to_paged_upstream(k_dense, block_table, block_size, num_blocks)
    v_arg = _dense_to_paged_upstream(v_dense, block_table, block_size, num_blocks)

    ref, _, _ = _reference_upstream(
        q,
        k_arg.clone(),
        v_arg.clone(),
        cache_seqlens,
        cos,
        sin,
        k=None,
        v=None,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
    )
    out = flash_attn_with_kvcache(
        q,
        k_arg,
        v_arg,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        causal=True,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("num_splits", [2, 3, 5, 9, 15, 64, 114])
def test_flash_attn_with_kvcache_pipelined_hnd_uneven_group(num_splits: int):
    """Pipelined HND fast path with library-aligned split-group reductions."""
    torch.manual_seed(211)
    device = "cuda"
    dtype = torch.bfloat16
    bsz, seqlen_q, num_heads, num_kv_heads, head_dim = 1, 1, 16, 2, 128
    max_seqlen = 2048
    block_size = 16
    cache_seqlens = torch.tensor([2001], device=device, dtype=torch.int32)

    q = torch.empty(bsz, seqlen_q, num_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    k_dense = torch.empty(bsz, num_kv_heads, max_seqlen, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    v_dense = torch.empty_like(k_dense).uniform_(-1, 1)
    max_blocks = max_seqlen // block_size
    block_table = torch.arange(bsz * max_blocks, device=device, dtype=torch.int32).reshape(bsz, max_blocks)
    k_cache = _dense_to_paged(k_dense, block_table, block_size, bsz * max_blocks)
    v_cache = _dense_to_paged(v_dense, block_table, block_size, bsz * max_blocks)

    ref = _attention_ref(
        q,
        k_dense.transpose(1, 2),
        v_dense.transpose(1, 2),
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
        force_upstream_cache_layout=False,
        num_splits=num_splits,
    )
    torch.cuda.synchronize()
    # Two wide splits accumulate many more BF16 values per group than the pipelined
    # operating point, so retain the established BF16 relative tolerance while
    # allowing its observed absolute rounding envelope.
    atol = 5e-2 if num_splits < 16 else 3e-2
    torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=3e-2)


@pytest.mark.parametrize(
    ("num_kv_heads", "active_seqlen"),
    [(2, 384), (8, 256), (8, 384), (8, 512)],
)
def test_flash_attn_with_kvcache_pipelined_short_auto_split_guard(num_kv_heads: int, active_seqlen: int):
    """Short active context uses the pipelined path's direct-output mode despite 2K capacity."""
    torch.manual_seed(307)
    device = "cuda"
    dtype = torch.bfloat16
    bsz, seqlen_q, num_heads, head_dim = 1, 1, 16, 128
    max_seqlen, block_size = 2048, 16
    cache_seqlens = torch.tensor([active_seqlen], device=device, dtype=torch.int32)

    q = torch.empty(bsz, seqlen_q, num_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    k_dense = torch.empty(bsz, num_kv_heads, max_seqlen, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    v_dense = torch.empty_like(k_dense).uniform_(-1, 1)
    max_blocks = max_seqlen // block_size
    block_table = torch.arange(bsz * max_blocks, device=device, dtype=torch.int32).reshape(bsz, max_blocks)
    k_cache = _dense_to_paged(k_dense, block_table, block_size, bsz * max_blocks)
    v_cache = _dense_to_paged(v_dense, block_table, block_size, bsz * max_blocks)

    ref = _attention_ref(
        q,
        k_dense.transpose(1, 2),
        v_dense.transpose(1, 2),
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
        max_context_len=active_seqlen,
        causal=True,
        force_upstream_cache_layout=False,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("num_splits", [4, 8])
def test_flash_attn_with_kvcache_mma_decode_split(num_splits: int):
    """Flash-decoding (split-KV) MMA decode path + reduce kernel."""
    torch.manual_seed(31 + num_splits)
    device = "cuda"
    dtype = torch.bfloat16
    bsz, seqlen_q, num_heads, num_kv_heads, head_dim = 2, 1, 8, 2, 128
    max_seqlen = 512
    block_size = 16
    max_blocks = max_seqlen // block_size
    cache_seqlens = torch.tensor([512, 401], device=device, dtype=torch.int32)

    q = torch.empty(bsz, seqlen_q, num_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    k_dense = torch.empty(bsz, max_seqlen, num_kv_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    v_dense = torch.empty_like(k_dense).uniform_(-1, 1)
    cos, sin = _make_rotary_tables(max_seqlen + 8, head_dim, dtype, device)

    block_table = torch.arange(bsz * max_blocks, device=device, dtype=torch.int32).reshape(bsz, max_blocks)
    num_blocks = bsz * max_blocks
    k_arg = _dense_to_paged_upstream(k_dense, block_table, block_size, num_blocks)
    v_arg = _dense_to_paged_upstream(v_dense, block_table, block_size, num_blocks)

    ref, _, _ = _reference_upstream(
        q,
        k_arg.clone(),
        v_arg.clone(),
        cache_seqlens,
        cos,
        sin,
        k=None,
        v=None,
        causal=True,
        window_size=(-1, -1),
        block_table=block_table,
    )
    out = flash_attn_with_kvcache(
        q,
        k_arg,
        v_arg,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        causal=True,
        num_splits=num_splits,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize(
    ("softmax_scale", "expect_mma"),
    [(None, True), (0.25, False)],
)
def test_flash_attn_with_kvcache_mma_scale_dispatch(monkeypatch, softmax_scale: float | None, expect_mma: bool):
    """Only a non-default scale disables an otherwise eligible MMA decode."""
    captured = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(flash_attn_kvcache_module, "build_flash_attn_with_kvcache_module", fake_build)
    monkeypatch.setattr(flash_attn_kvcache_module.flyc, "compile", lambda *_args: lambda *_launch_args: None)
    q = torch.empty(1, 1, 2, 128, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.empty(1, 128, 1, 128, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.empty_like(k_cache)

    flash_attn_kvcache_module.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=128,
        softmax_scale=softmax_scale,
    )

    assert captured["use_mma_decode"] is expect_mma
    assert captured["use_pipelined_mma_decode"] is False


def test_pipelined_reduce_warp_count_scales_with_split_groups():
    assert _pipelined_reduce_warp_count(1) == 1
    assert _pipelined_reduce_warp_count(16) == 8
    assert _pipelined_reduce_warp_count(32) == 16
