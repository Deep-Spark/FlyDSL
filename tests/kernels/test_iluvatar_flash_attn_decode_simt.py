#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from kernels.attention.iluvatar.flash_attn_kvcache_planner import _get_simt_decode_launch_plan

from ._iluvatar_flash_attn_kvcache_support import (
    _attention_ref,
    _configure_iluvatar_env,  # noqa: F401
    _dense_to_paged,
    flash_attn_kvcache_module,
    flash_attn_with_kvcache,
    pytest,
    torch,
)

pytestmark = [pytest.mark.l2_device]


@pytest.mark.parametrize(
    ("num_splits", "active_seqlen"),
    [(16, 4001), (28, 4001), (32, 4001), (32, 1001)],
)
def test_flash_attn_with_kvcache_simt_decode_split(num_splits: int, active_seqlen: int):
    """GQA=2 SIMT decode writes mergeable split softmax states."""
    torch.manual_seed(223 + num_splits + active_seqlen)
    device = "cuda"
    dtype = torch.bfloat16
    bsz, seqlen_q, num_heads, num_kv_heads, head_dim = 1, 1, 16, 8, 128
    max_seqlen, block_size = 4096, 16
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
        causal=True,
        force_upstream_cache_layout=False,
        num_splits=num_splits,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize(
    ("active_seqlen", "strided_cache"),
    [(127, False), (513, False), (513, True), (1025, False)],
)
def test_flash_attn_with_kvcache_d256_gqa4_simt_528_page(
    active_seqlen: int,
    strided_cache: bool,
):
    """D256/GQA4 SIMT decode chunks 528-token physical pages into 16s."""
    torch.manual_seed(317 + active_seqlen)
    device = "cuda"
    dtype = torch.bfloat16
    bsz, seqlen_q, num_heads, num_kv_heads, head_dim = 1, 1, 16, 4, 256
    max_seqlen, block_size = 1056, 528
    cache_seqlens = torch.tensor([active_seqlen], device=device, dtype=torch.int32)

    q = torch.empty(bsz, seqlen_q, num_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    k_dense = torch.empty(bsz, num_kv_heads, max_seqlen, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    v_dense = torch.empty_like(k_dense).uniform_(-1, 1)
    max_blocks = max_seqlen // block_size
    block_table = torch.arange(max_blocks, device=device, dtype=torch.int32).reshape(bsz, max_blocks)
    k_cache = _dense_to_paged(k_dense, block_table, block_size, max_blocks)
    v_cache = _dense_to_paged(v_dense, block_table, block_size, max_blocks)
    if strided_cache:
        k_storage = torch.empty(
            max_blocks,
            num_kv_heads,
            block_size,
            2,
            head_dim,
            device=device,
            dtype=dtype,
        )
        v_storage = torch.empty_like(k_storage)
        k_storage[:, :, :, 0, :] = k_cache
        v_storage[:, :, :, 0, :] = v_cache
        k_cache = k_storage[:, :, :, 0, :]
        v_cache = v_storage[:, :, :, 0, :]

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
        num_splits=0,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref.float(), atol=4e-2, rtol=4e-2)


@pytest.mark.parametrize(
    ("active_seqlen", "expected_config"),
    [
        (256, (1, 4)),
        (384, (1, 4)),
        (512, (1, 4)),
        (528, (17, 1)),
        (1024, (32, 1)),
        (1536, (24, 1)),
        (2048, (32, 1)),
        (4096, (32, 1)),
        (16384, (29, 1)),
        (32768, (28, 1)),
    ],
)
def test_qwen_simt_decode_k_warp_planner(active_seqlen: int, expected_config: tuple[int, int]):
    from kernels.attention.iluvatar.mma_decode_splits import compute_qwen_simt_decode_config

    assert (
        compute_qwen_simt_decode_config(
            batch_size=1,
            seqlen_q=1,
            num_heads=16,
            num_kv_heads=8,
            head_dim=128,
            max_seqlen_k=active_seqlen,
        )
        == expected_config
    )


@pytest.mark.parametrize(
    ("active_seqlen", "expected_config"),
    [
        (32, (2, 1)),
        (128, (8, 1)),
        (512, (16, 1)),
        (528, (16, 1)),
        (1024, (16, 1)),
        (1088, (16, 1)),
        (1536, (32, 1)),
        (2048, (32, 1)),
    ],
)
def test_qwen_d256_gqa4_simt_planner(active_seqlen: int, expected_config: tuple[int, int]):
    from kernels.attention.iluvatar.mma_decode_splits import compute_qwen_simt_decode_config

    assert (
        compute_qwen_simt_decode_config(
            batch_size=1,
            seqlen_q=1,
            num_heads=16,
            num_kv_heads=4,
            head_dim=256,
            max_seqlen_k=active_seqlen,
        )
        == expected_config
    )


def test_flash_attn_with_kvcache_reuses_simt_decode_launch_plan(monkeypatch):
    """Matching SIMT decode calls bypass repeated Python dispatch and planning."""
    build_calls = []
    launches = []

    def fake_build(**kwargs):
        build_calls.append(kwargs)
        return object()

    monkeypatch.setattr(flash_attn_kvcache_module, "build_flash_attn_with_kvcache_module", fake_build)
    monkeypatch.setattr(
        flash_attn_kvcache_module.flyc,
        "compile",
        lambda *_args: lambda *launch_args: launches.append(launch_args),
    )
    q0 = torch.empty(1, 1, 16, 128, device="cuda", dtype=torch.bfloat16)
    q1 = torch.empty_like(q0)
    k_cache = torch.empty(128, 8, 16, 128, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.empty_like(k_cache)
    block_table = torch.arange(128, device="cuda", dtype=torch.int32).reshape(1, 128)
    cache_seqlens0 = torch.tensor([256], device="cuda", dtype=torch.int32)
    cache_seqlens1 = torch.tensor([384], device="cuda", dtype=torch.int32)
    kwargs = dict(
        block_table=block_table,
        max_context_len=384,
        causal=True,
        force_upstream_cache_layout=False,
    )

    flash_attn_kvcache_module.flash_attn_with_kvcache(
        q0,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens0,
        **kwargs,
    )
    plan = _get_simt_decode_launch_plan(
        q0,
        k_cache,
        v_cache,
        cache_seqlens0,
        block_table,
        softmax_scale=None,
        causal=True,
        num_splits=0,
        out=None,
        max_context_len=384,
    )
    assert plan is not None
    plan(q1, k_cache, v_cache, cache_seqlens1, block_table, None)
    flash_attn_kvcache_module.flash_attn_with_kvcache(
        q1,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens1,
        **kwargs,
    )
    flash_attn_kvcache_module.flash_attn_with_kvcache(
        q1,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens1,
        num_splits=1,
        **kwargs,
    )

    assert len(build_calls) == 2
    assert len(launches) == 4
    assert launches[1][0] is q1
    assert launches[1][3] is cache_seqlens1
