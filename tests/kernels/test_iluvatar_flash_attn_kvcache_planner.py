#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from ._iluvatar_flash_attn_kvcache_support import (
    _configure_iluvatar_env,  # noqa: F401
    flash_attn_kvcache_module,
    pytest,
    torch,
)

pytestmark = [pytest.mark.l2_device]


@pytest.fixture(autouse=True)
def _enable_decode_backends(monkeypatch):
    # Planner integration tests assert MMA vs SIMT routing; pin production
    # defaults so a dirty process env cannot disable either path.
    monkeypatch.setenv("FLYDSL_KVCACHE_MMA_DECODE", "1")
    monkeypatch.setenv("FLYDSL_KVCACHE_SIMT_DECODE", "1")


def test_mma_decode_num_splits_qwen3_shape():
    from kernels.attention.iluvatar.mma_decode_splits import (
        compute_mma_decode_num_splits,
        compute_pipelined_mma_decode_config,
    )

    assert compute_pipelined_mma_decode_config(
        batch_size=1,
        seqlen_q=1,
        num_heads=16,
        num_kv_heads=2,
        head_dim=128,
        max_seqlen_k=32768,
    ) == (64, 512, 1)
    assert (
        compute_mma_decode_num_splits(
            batch_size=1,
            seqlen_q=1,
            num_heads=16,
            num_kv_heads=8,
            head_dim=128,
            max_seqlen_k=512,
            block_n=32,
        )
        == 16
    )
    assert (
        compute_mma_decode_num_splits(
            batch_size=128,
            seqlen_q=1,
            num_heads=8,
            num_kv_heads=2,
            head_dim=128,
            max_seqlen_k=8192,
            block_n=32,
        )
        == 1
    )
    assert (
        compute_mma_decode_num_splits(
            batch_size=1,
            seqlen_q=1,
            num_heads=16,
            num_kv_heads=2,
            head_dim=128,
            max_seqlen_k=32768,
            block_n=32,
        )
        == 32
    )


@pytest.mark.parametrize("num_kv_heads", [2, 8])
def test_flash_attn_with_kvcache_uses_max_context_len_for_pipelined_split_planning(monkeypatch, num_kv_heads: int):
    """The vLLM batch maximum plans groups without changing cache capacity."""
    captured = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(flash_attn_kvcache_module, "build_flash_attn_with_kvcache_module", fake_build)
    monkeypatch.setattr(flash_attn_kvcache_module.flyc, "compile", lambda *_args: lambda *_launch_args: None)
    q = torch.empty(1, 1, 16, 128, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.empty(128, num_kv_heads, 16, 128, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.empty_like(k_cache)
    block_table = torch.arange(128, device="cuda", dtype=torch.int32).reshape(1, 128)

    flash_attn_kvcache_module.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=torch.tensor([384], device="cuda", dtype=torch.int32),
        block_table=block_table,
        max_context_len=384,
        causal=True,
        force_upstream_cache_layout=False,
    )

    # Pipelined MMA still compiles against cache capacity; SIMT compiles against
    # the 32-token planned context bucket so short vLLM batches stay compact.
    assert captured["max_seqlen_k"] == (384 if num_kv_heads == 8 else 2048)
    assert captured["use_pipelined_mma_decode"] is (num_kv_heads == 2)
    assert captured["use_simt_decode"] is (num_kv_heads == 8)
    if num_kv_heads == 8:
        assert captured["simt_k_warps"] == 4
    assert captured["num_splits"] == 1


def test_flash_attn_with_kvcache_caps_pipelined_groups_by_active_tiles(monkeypatch):
    """The pipelined MMA planner cannot create more groups than active tile work supports."""
    from kernels.attention.iluvatar import mma_decode_splits

    captured = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(mma_decode_splits, "compute_pipelined_mma_decode_config", lambda **_kwargs: (128, 0, 1))
    monkeypatch.setattr(flash_attn_kvcache_module, "build_flash_attn_with_kvcache_module", fake_build)
    monkeypatch.setattr(flash_attn_kvcache_module.flyc, "compile", lambda *_args: lambda *_launch_args: None)
    q = torch.empty(1, 1, 16, 128, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.empty(128, 2, 16, 128, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.empty_like(k_cache)
    block_table = torch.arange(128, device="cuda", dtype=torch.int32).reshape(1, 128)

    flash_attn_kvcache_module.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=torch.tensor([768], device="cuda", dtype=torch.int32),
        block_table=block_table,
        max_context_len=768,
        causal=True,
        force_upstream_cache_layout=False,
    )

    # 768 buckets to 1024: 64 active 16-token tiles permit 32 groups.
    assert captured["num_splits"] == 32


def test_flash_attn_with_kvcache_reuses_pipelined_split_workspace(monkeypatch):
    """Split pipelined MMA decode keeps its partial buffers across matching calls."""
    launches = []

    monkeypatch.setattr(
        flash_attn_kvcache_module,
        "build_flash_attn_with_kvcache_module",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        flash_attn_kvcache_module.flyc,
        "compile",
        lambda *_args: lambda *launch_args: launches.append(launch_args),
    )
    q = torch.empty(1, 1, 16, 128, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.empty(128, 2, 16, 128, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.empty_like(k_cache)
    block_table = torch.arange(128, device="cuda", dtype=torch.int32).reshape(1, 128)
    kwargs = dict(
        cache_seqlens=torch.tensor([1024], device="cuda", dtype=torch.int32),
        block_table=block_table,
        max_context_len=1024,
        causal=True,
        force_upstream_cache_layout=False,
    )

    flash_attn_kvcache_module.flash_attn_with_kvcache(q, k_cache, v_cache, **kwargs)
    flash_attn_kvcache_module.flash_attn_with_kvcache(q, k_cache, v_cache, **kwargs)

    assert len(launches) == 2
    assert launches[0][10] is launches[1][10]
    assert launches[0][11] is launches[1][11]
    assert launches[0][12] is launches[1][12]


def test_cached_simt_launch_returns_independently_owned_outputs(monkeypatch):
    calls = 0

    def fake_compiled(*launch_args):
        nonlocal calls
        calls += 1
        launch_args[5].fill_(calls)

    monkeypatch.setattr(
        flash_attn_kvcache_module,
        "build_flash_attn_with_kvcache_module",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(flash_attn_kvcache_module.flyc, "compile", lambda *_args: fake_compiled)
    q = torch.empty(1, 1, 16, 128, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.empty(64, 8, 16, 128, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.empty_like(k_cache)
    kwargs = dict(
        cache_seqlens=torch.tensor([384], device="cuda", dtype=torch.int32),
        block_table=torch.arange(64, device="cuda", dtype=torch.int32).reshape(1, 64),
        max_context_len=384,
        causal=True,
        force_upstream_cache_layout=False,
    )

    first = flash_attn_kvcache_module.flash_attn_with_kvcache(q, k_cache, v_cache, **kwargs)
    second = flash_attn_kvcache_module.flash_attn_with_kvcache(q, k_cache, v_cache, **kwargs)

    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first, torch.ones_like(first))
    torch.testing.assert_close(second, torch.full_like(second, 2))


def test_split_workspace_cache_is_scoped_to_current_stream(monkeypatch):
    launches = []
    monkeypatch.setattr(
        flash_attn_kvcache_module,
        "build_flash_attn_with_kvcache_module",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        flash_attn_kvcache_module.flyc,
        "compile",
        lambda *_args: lambda *launch_args: launches.append(launch_args),
    )
    q = torch.empty(1, 1, 16, 128, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.empty(128, 2, 16, 128, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.empty_like(k_cache)
    kwargs = dict(
        cache_seqlens=torch.tensor([1024], device="cuda", dtype=torch.int32),
        block_table=torch.arange(128, device="cuda", dtype=torch.int32).reshape(1, 128),
        max_context_len=1024,
        causal=True,
        force_upstream_cache_layout=False,
    )
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    for current in streams:
        with torch.cuda.stream(current):
            flash_attn_kvcache_module.flash_attn_with_kvcache(q, k_cache, v_cache, **kwargs)

    assert launches[0][10].data_ptr() != launches[1][10].data_ptr()
    assert launches[0][11].data_ptr() != launches[1][11].data_ptr()
    assert launches[0][12].data_ptr() != launches[1][12].data_ptr()
