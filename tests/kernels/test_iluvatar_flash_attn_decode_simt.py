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

pytestmark = [pytest.mark.l2_device, pytest.mark.iluvatar_lower]


@pytest.fixture(autouse=True)
def _enable_simt_decode(monkeypatch):
    # This module exercises the SIMT path; do not silently fall back to scalar
    # when the process env has FLYDSL_KVCACHE_SIMT_DECODE=0.
    monkeypatch.setenv("FLYDSL_KVCACHE_SIMT_DECODE", "1")
    monkeypatch.setenv("FLYDSL_KVCACHE_MMA_DECODE", "0")


@pytest.mark.parametrize(
    ("num_splits", "active_seqlen"),
    [(3, 1001), (4, 1001), (16, 4001), (28, 4001), (32, 4001), (32, 1001)],
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


def test_flash_attn_with_kvcache_simt_decode_empty_non_split():
    """An empty cache produces a finite zero output in direct-output mode."""
    torch.manual_seed(419)
    device = "cuda"
    dtype = torch.bfloat16
    bsz, seqlen_q, num_heads, num_kv_heads, head_dim = 1, 1, 16, 8, 128
    max_seqlen, block_size = 256, 16
    cache_seqlens = torch.zeros(bsz, device=device, dtype=torch.int32)

    q = torch.empty(bsz, seqlen_q, num_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    max_blocks = max_seqlen // block_size
    block_table = torch.arange(max_blocks, device=device, dtype=torch.int32).reshape(bsz, max_blocks)
    k_cache = torch.empty(max_blocks, num_kv_heads, block_size, head_dim, device=device, dtype=dtype)
    v_cache = torch.empty_like(k_cache)

    out = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        max_context_len=max_seqlen,
        causal=True,
        force_upstream_cache_layout=False,
        num_splits=1,
    )
    torch.cuda.synchronize()

    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, torch.zeros_like(out), atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("bsz", "active_seqlen", "strided_cache", "num_splits"),
    [
        (1, 127, False, 0),
        (2, 127, False, 0),
        (1, 513, False, 0),
        (1, 513, True, 0),
        (1, 1025, False, 0),
        (7, 513, False, 0),
        (8, 513, False, 0),
        (32, 513, False, 0),
        (8, 513, False, 1),
    ],
)
def test_flash_attn_with_kvcache_d256_gqa4_simt_528_page(
    bsz: int,
    active_seqlen: int,
    strided_cache: bool,
    num_splits: int,
):
    """D256/GQA4 SIMT decode chunks 528-token physical pages into 16s."""
    torch.manual_seed(317 + active_seqlen)
    device = "cuda"
    dtype = torch.bfloat16
    seqlen_q, num_heads, num_kv_heads, head_dim = 1, 16, 4, 256
    max_seqlen, block_size = 1056, 528
    cache_seqlens = torch.full((bsz,), active_seqlen, device=device, dtype=torch.int32)

    q = torch.empty(bsz, seqlen_q, num_heads, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    k_dense = torch.empty(bsz, num_kv_heads, max_seqlen, head_dim, device=device, dtype=dtype).uniform_(-1, 1)
    v_dense = torch.empty_like(k_dense).uniform_(-1, 1)
    max_blocks = max_seqlen // block_size
    total_blocks = bsz * max_blocks
    block_table = torch.arange(total_blocks, device=device, dtype=torch.int32).reshape(bsz, max_blocks)
    k_cache = _dense_to_paged(k_dense, block_table, block_size, total_blocks)
    v_cache = _dense_to_paged(v_dense, block_table, block_size, total_blocks)
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
        num_splits=num_splits,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref.float(), atol=4e-2, rtol=4e-2)


def test_d256_gqa4_plan_accepts_vllm_block_major_kv_view():
    """vLLM's block-major K/V view must populate and reuse the fast plan."""
    torch.manual_seed(421)
    device = "cuda"
    dtype = torch.bfloat16
    active_seqlen = 127
    block_size = 528
    num_blocks = 2
    q = torch.empty(1, 1, 16, 256, device=device, dtype=dtype).uniform_(-1, 1)
    k_dense = torch.empty(
        1, 4, num_blocks * block_size, 256, device=device, dtype=dtype
    ).uniform_(-1, 1)
    v_dense = torch.empty_like(k_dense).uniform_(-1, 1)
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).view(1, -1)
    contiguous_k = _dense_to_paged(k_dense, block_table, block_size, num_blocks)
    contiguous_v = _dense_to_paged(v_dense, block_table, block_size, num_blocks)
    block_major_cache = torch.empty(
        num_blocks, 2, 4, block_size, 256, device=device, dtype=dtype
    )
    block_major_cache[:, 0] = contiguous_k
    block_major_cache[:, 1] = contiguous_v
    k_cache = block_major_cache[:, 0]
    v_cache = block_major_cache[:, 1]
    cache_seqlens = torch.tensor([active_seqlen], device=device, dtype=torch.int32)
    out = torch.empty_like(q)

    assert not k_cache.is_contiguous()
    assert k_cache.stride() == (2 * 4 * block_size * 256, block_size * 256, 256, 1)
    flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        max_context_len=active_seqlen,
        causal=True,
        force_upstream_cache_layout=False,
        out=out,
    )
    plan = _get_simt_decode_launch_plan(
        q,
        k_cache,
        v_cache,
        cache_seqlens,
        block_table,
        softmax_scale=None,
        causal=True,
        num_splits=0,
        out=out,
        max_context_len=active_seqlen,
    )
    assert plan is not None
    plan_out = torch.empty_like(q)
    plan(q, k_cache, v_cache, cache_seqlens, block_table, plan_out)
    torch.cuda.synchronize()

    ref = _attention_ref(
        q,
        k_dense.transpose(1, 2),
        v_dense.transpose(1, 2),
        cache_seqlens,
        causal=True,
        window_size=(-1, -1),
    )
    torch.testing.assert_close(out.float(), ref.float(), atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(plan_out.float(), ref.float(), atol=4e-2, rtol=4e-2)


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
        (32, (1, 1)),
        (64, (1, 2)),
        (128, (1, 4)),
        (192, (1, 8)),
        (256, (1, 8)),
        (320, (16, 1)),
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

    def fake_compile(_launcher, *compile_args):
        launches.append(
            tuple(getattr(arg, "torch_tensor", arg) for arg in compile_args)
        )
        return lambda *launch_args: launches.append(launch_args)

    monkeypatch.setattr(flash_attn_kvcache_module, "build_flash_attn_with_kvcache_module", fake_build)
    monkeypatch.setattr(
        flash_attn_kvcache_module.flyc,
        "compile",
        fake_compile,
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


def test_flash_attn_with_kvcache_reuses_simt_decode_across_occupancy_class(monkeypatch):
    """Large-batch D256 decode shares one compiled launcher inside a split class."""
    build_calls = []
    launches = []

    def fake_build(**kwargs):
        build_calls.append(kwargs)
        return object()

    def fake_compile(_launcher, *compile_args):
        launches.append(
            tuple(getattr(arg, "torch_tensor", arg) for arg in compile_args)
        )
        return lambda *launch_args: launches.append(launch_args)

    monkeypatch.setattr(flash_attn_kvcache_module, "build_flash_attn_with_kvcache_module", fake_build)
    monkeypatch.setattr(
        flash_attn_kvcache_module.flyc,
        "compile",
        fake_compile,
    )
    k_cache = torch.empty(64, 4, 16, 256, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.empty_like(k_cache)
    kwargs = dict(
        max_context_len=512,
        causal=True,
        force_upstream_cache_layout=False,
    )

    def run(batch: int):
        q = torch.empty(batch, 1, 16, 256, device="cuda", dtype=torch.bfloat16)
        flash_attn_kvcache_module.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=torch.full((batch,), 512, device="cuda", dtype=torch.int32),
            block_table=torch.arange(64, device="cuda", dtype=torch.int32).reshape(1, 64).expand(batch, -1).contiguous(),
            **kwargs,
        )

    run(32)
    run(48)

    assert len(build_calls) == 1
    assert build_calls[0]["batch_size"] == 0
    assert build_calls[0]["use_simt_decode"] is True
    assert len(launches) == 2
    assert launches[0][0].shape[0] == 32
    assert launches[1][0].shape[0] == 48


def test_flash_attn_with_kvcache_reuses_simt_decode_across_context_growth(monkeypatch):
    """D256 B=48 decode reuses one launcher when context grows 512→640."""
    build_calls = []
    launches = []

    def fake_build(**kwargs):
        build_calls.append(kwargs)
        return object()

    def fake_compile(_launcher, *compile_args):
        launches.append(
            tuple(getattr(arg, "torch_tensor", arg) for arg in compile_args)
        )
        return lambda *launch_args: launches.append(launch_args)

    monkeypatch.setattr(flash_attn_kvcache_module, "build_flash_attn_with_kvcache_module", fake_build)
    monkeypatch.setattr(
        flash_attn_kvcache_module.flyc,
        "compile",
        fake_compile,
    )
    batch = 48
    k_cache = torch.empty(64, 4, 16, 256, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.empty_like(k_cache)
    q = torch.empty(batch, 1, 16, 256, device="cuda", dtype=torch.bfloat16)
    block_table = (
        torch.arange(64, device="cuda", dtype=torch.int32).reshape(1, 64).expand(batch, -1).contiguous()
    )

    def run(ctx: int):
        flash_attn_kvcache_module.flash_attn_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=torch.full((batch,), ctx, device="cuda", dtype=torch.int32),
            block_table=block_table,
            max_context_len=ctx,
            causal=True,
            force_upstream_cache_layout=False,
        )

    run(512)
    run(640)

    assert len(build_calls) == 1
    assert build_calls[0]["batch_size"] == 0
    assert build_calls[0]["max_seqlen_k"] == 1024
    assert build_calls[0]["use_simt_decode"] is True
    assert build_calls[0]["num_splits"] == 2
    assert len(launches) == 2
