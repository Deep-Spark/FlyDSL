# SPDX-License-Identifier: Apache-2.0

"""Host-side decode backend selection and planning (no kernel launch)."""

import pytest
import torch

from kernels.attention.iluvatar.flash_attn_kvcache_planner import (
    plan_decode_launch,
    select_decode_backends,
)
from kernels.attention.iluvatar.mma_decode_splits import compute_qwen_simt_decode_config


def _select(**overrides):
    args = {
        "has_cache_leftpad": False,
        "softmax_scale": 128**-0.5,
        "default_softmax_scale": 128**-0.5,
        "softcap": 0.0,
        "head_dim": 128,
        "dtype": torch.bfloat16,
        "seqlen_q": 1,
        "num_heads": 16,
        "num_kv_heads": 8,
        "kernel_paged": True,
        "has_padded_block_table": False,
        "upstream_cache_layout": False,
        "page_block_size": 16,
        "max_seqlen_k": 4096,
        "window_size": (-1, -1),
        "k_cache_contiguous": True,
        "v_cache_contiguous": True,
    }
    args.update(overrides)
    return select_decode_backends(**args)


@pytest.mark.l0_backend_agnostic
@pytest.mark.parametrize(
    ("batch_size", "max_seqlen_k", "expected_splits"),
    [
        (1, 128, 8),
        (1, 512, 16),
        (3, 1024, 11),
        (4, 1024, 8),
        (5, 1024, 7),
        (6, 1024, 6),
        (7, 1024, 5),
        (8, 1024, 4),
        (11, 1024, 3),
        (15, 1024, 3),
        (16, 1024, 2),
        (48, 128, 2),
        (48, 512, 2),
        (48, 544, 2),
        (48, 1024, 2),
        (32, 1024, 2),
    ],
)
def test_qwen_d256_simt_splits_are_batch_aware(
    batch_size: int,
    max_seqlen_k: int,
    expected_splits: int,
):
    assert compute_qwen_simt_decode_config(
        batch_size=batch_size,
        seqlen_q=1,
        num_heads=16,
        num_kv_heads=4,
        head_dim=256,
        max_seqlen_k=max_seqlen_k,
    ) == (expected_splits, 1)


@pytest.mark.l0_backend_agnostic
def test_selects_simt_for_paged_gqa_two(monkeypatch):
    monkeypatch.setenv("FLYDSL_KVCACHE_SIMT_DECODE", "1")
    assert _select()[:3] == (True, False, False)


@pytest.mark.l0_backend_agnostic
def test_selects_simt_for_d256_gqa_four_with_528_token_pages(monkeypatch):
    monkeypatch.setenv("FLYDSL_KVCACHE_SIMT_DECODE", "1")
    assert _select(
        softmax_scale=256**-0.5,
        default_softmax_scale=256**-0.5,
        head_dim=256,
        num_kv_heads=4,
        page_block_size=528,
        max_seqlen_k=528,
    )[:3] == (True, False, False)


@pytest.mark.l0_backend_agnostic
def test_selects_d256_simt_for_strided_vllm_cache(monkeypatch):
    monkeypatch.setenv("FLYDSL_KVCACHE_SIMT_DECODE", "1")
    assert _select(
        softmax_scale=256**-0.5,
        default_softmax_scale=256**-0.5,
        head_dim=256,
        num_kv_heads=4,
        k_cache_contiguous=False,
        v_cache_contiguous=False,
    )[:3] == (True, False, False)
    assert _select(k_cache_contiguous=False, v_cache_contiguous=False)[:3] == (
        False,
        False,
        False,
    )


@pytest.mark.l0_backend_agnostic
def test_selects_pipelined_then_baseline_mma(monkeypatch):
    monkeypatch.setenv("FLYDSL_KVCACHE_SIMT_DECODE", "1")
    monkeypatch.setenv("FLYDSL_KVCACHE_MMA_DECODE", "1")
    assert _select(num_heads=32)[:3] == (False, True, False)
    assert _select(num_heads=32, kernel_paged=False, max_seqlen_k=512)[:3] == (False, False, True)
    # MMA kernels hard-require D=128; other head dims must stay on scalar.
    assert _select(
        softmax_scale=16**-0.5,
        default_softmax_scale=16**-0.5,
        head_dim=16,
        num_heads=2,
        num_kv_heads=1,
        kernel_paged=False,
        max_seqlen_k=128,
    )[:3] == (False, False, False)


@pytest.mark.l0_backend_agnostic
def test_common_gates_fall_back_to_scalar(monkeypatch):
    """Anything outside the decode-fast common contract stays off SIMT/MMA."""
    monkeypatch.setenv("FLYDSL_KVCACHE_SIMT_DECODE", "1")
    monkeypatch.setenv("FLYDSL_KVCACHE_MMA_DECODE", "1")
    assert _select(seqlen_q=2)[:3] == (False, False, False)
    assert _select(softcap=1.0)[:3] == (False, False, False)
    assert _select(window_size=(64, 0))[:3] == (False, False, False)
    assert _select(has_cache_leftpad=True)[:3] == (False, False, False)
    assert _select(softmax_scale=0.1)[:3] == (False, False, False)
    assert _select(dtype=torch.float16)[:3] == (False, False, False)


@pytest.mark.l0_backend_agnostic
def test_decode_environment_switches_preserve_fallback(monkeypatch):
    monkeypatch.setenv("FLYDSL_KVCACHE_SIMT_DECODE", "0")
    monkeypatch.setenv("FLYDSL_KVCACHE_MMA_DECODE", "0")
    assert _select()[:3] == (False, False, False)


@pytest.mark.l0_backend_agnostic
def test_mma_block_size_is_validated_only_for_mma_candidate(monkeypatch):
    monkeypatch.setenv("FLYDSL_KVCACHE_SIMT_DECODE", "1")
    monkeypatch.setenv("FLYDSL_KVCACHE_MMA_DECODE", "1")
    monkeypatch.setenv("FLYDSL_KVCACHE_MMA_BN", "not-an-integer")
    # SIMT wins this shape, so an unrelated MMA tuning knob is not consulted.
    assert _select()[:3] == (True, False, False)
    with pytest.raises(ValueError, match="one of 16, 32, 64, or 128"):
        _select(kernel_paged=False, num_heads=32)

    monkeypatch.setenv("FLYDSL_KVCACHE_MMA_BN", "64")
    assert _select(kernel_paged=False, num_heads=32)[2:] == (True, 64)


@pytest.mark.l0_backend_agnostic
def test_d256_explicit_split_one_uses_direct_simt_output(monkeypatch):
    monkeypatch.setenv("FLYDSL_KVCACHE_SIMT_DECODE", "1")
    plan = plan_decode_launch(
        has_cache_leftpad=False,
        softmax_scale=256**-0.5,
        default_softmax_scale=256**-0.5,
        softcap=0.0,
        head_dim=256,
        dtype=torch.bfloat16,
        batch_size=1,
        seqlen_q=1,
        num_heads=16,
        num_kv_heads=4,
        num_splits=1,
        kernel_paged=True,
        has_padded_block_table=False,
        upstream_cache_layout=False,
        page_block_size=16,
        max_seqlen_k=16,
        planning_max_seqlen_k=16,
        max_context_len=1,
        window_size=(-1, -1),
        k_cache_contiguous=True,
        v_cache_contiguous=True,
    )
    assert plan.use_simt_decode is True
    assert plan.effective_num_splits == 1
    assert plan.simt_k_warps == 1


@pytest.mark.l0_backend_agnostic
def test_qwen_simt_uses_32_token_context_granularity(monkeypatch):
    monkeypatch.setenv("FLYDSL_KVCACHE_SIMT_DECODE", "1")

    def plan(active: int):
        return plan_decode_launch(
            has_cache_leftpad=False,
            softmax_scale=128**-0.5,
            default_softmax_scale=128**-0.5,
            softcap=0.0,
            head_dim=128,
            dtype=torch.bfloat16,
            batch_size=1,
            seqlen_q=1,
            num_heads=16,
            num_kv_heads=8,
            num_splits=0,
            kernel_paged=True,
            has_padded_block_table=False,
            upstream_cache_layout=False,
            page_block_size=16,
            max_seqlen_k=2048,
            planning_max_seqlen_k=active,
            max_context_len=active,
            window_size=(-1, -1),
            k_cache_contiguous=True,
            v_cache_contiguous=True,
        )

    assert plan(513).planning_max_seqlen_k == 544
    assert plan(513).effective_num_splits == 17
    assert plan(1025).planning_max_seqlen_k == 1056
    assert plan(1025).effective_num_splits == 17


@pytest.mark.l2_device
@pytest.mark.iluvatar_lower
def test_simt_fast_key_matches_split_schedule_not_token_bucket(monkeypatch):
    """Fast-key hashing needs real CUDA tensors; keep it out of L0 CPU CI.

    Context length is not a compile key. Growing within one D256 split
    band reuses the launch; crossing 8→16 or 16→32 groups does not.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA-compatible device is not available")
    monkeypatch.setenv("FLYDSL_KVCACHE_SIMT_DECODE", "1")
    from kernels.attention.iluvatar.flash_attn_kvcache_planner import _simt_decode_fast_key

    q = torch.empty((1, 1, 16, 256), device="cuda", dtype=torch.bfloat16)
    k = torch.empty((4, 4, 528, 256), device="cuda", dtype=torch.bfloat16)
    v = torch.empty_like(k)
    cache_seqlens = torch.zeros(1, device="cuda", dtype=torch.int32)
    block_table = torch.zeros((1, 4), device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)

    def key_for(ctx: int):
        probe = _simt_decode_fast_key(
            q,
            k,
            v,
            k=None,
            v=None,
            rotary_cos=None,
            rotary_sin=None,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=None,
            cache_leftpad=None,
            block_table=block_table,
            softmax_scale=None,
            causal=True,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            num_splits=0,
            return_softmax_lse=False,
            is_qkv_packed=False,
            force_upstream_cache_layout=False,
            stream=None,
            out=out,
            max_context_len=ctx,
        )
        assert probe is not None
        return probe[0]

    assert key_for(1025) == key_for(1024)
    assert key_for(512) == key_for(640)
    assert key_for(128) != key_for(144)
    assert key_for(1088) != key_for(1104)


@pytest.mark.l2_device
@pytest.mark.iluvatar_lower
def test_simt_fast_key_ignores_batch_within_occupancy_class(monkeypatch):
    """B=32 and B=48 share a key; B=1 (more splits) does not."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA-compatible device is not available")
    monkeypatch.setenv("FLYDSL_KVCACHE_SIMT_DECODE", "1")
    from kernels.attention.iluvatar.flash_attn_kvcache_planner import _simt_decode_fast_key

    k = torch.empty((8, 4, 16, 256), device="cuda", dtype=torch.bfloat16)
    v = torch.empty_like(k)

    def key_for(batch: int):
        q = torch.empty((batch, 1, 16, 256), device="cuda", dtype=torch.bfloat16)
        probe = _simt_decode_fast_key(
            q,
            k,
            v,
            k=None,
            v=None,
            rotary_cos=None,
            rotary_sin=None,
            cache_seqlens=torch.zeros(batch, device="cuda", dtype=torch.int32),
            cache_batch_idx=None,
            cache_leftpad=None,
            block_table=torch.zeros((batch, 8), device="cuda", dtype=torch.int32),
            softmax_scale=None,
            causal=True,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            num_splits=0,
            return_softmax_lse=False,
            is_qkv_packed=False,
            force_upstream_cache_layout=False,
            stream=None,
            out=torch.empty_like(q),
            max_context_len=512,
        )
        assert probe is not None
        return probe[0]

    assert key_for(32) == key_for(48)
    assert key_for(1) != key_for(48)
    assert key_for(8) != key_for(16)


def test_simt_runtime_metadata_rejects_stale_leading_dimensions():
    from kernels.attention.iluvatar.flash_attn_kvcache_planner import (
        _simt_runtime_metadata_valid,
    )

    q = torch.empty((48, 1, 16, 256), dtype=torch.bfloat16)
    k = torch.empty((8, 4, 16, 256), dtype=torch.bfloat16)
    v = torch.empty_like(k)
    good_lens = torch.zeros(48, dtype=torch.int32)
    good_table = torch.zeros((48, 8), dtype=torch.int32)
    good_out = torch.empty_like(q)

    assert _simt_runtime_metadata_valid(q, k, v, good_lens, good_table, good_out)
    assert not _simt_runtime_metadata_valid(
        q, k, v, good_lens[:32], good_table, good_out
    )
    assert not _simt_runtime_metadata_valid(
        q, k, v, good_lens, good_table[:32], good_out
    )
    assert not _simt_runtime_metadata_valid(
        q, k, v, good_lens, good_table, good_out[:32]
    )


@pytest.mark.l2_device
@pytest.mark.iluvatar_lower
def test_simt_plan_uses_runtime_current_stream():
    if not torch.cuda.is_available():
        pytest.skip("CUDA-compatible device is not available")
    from kernels.attention.iluvatar.flash_attn_kvcache_planner import (
        _SimtDecodeLaunchPlan,
    )

    launches = []
    current = torch.cuda.current_stream()
    q = torch.empty((1, 1, 16, 256), device="cuda", dtype=torch.bfloat16)
    k = torch.empty((8, 4, 16, 256), device="cuda", dtype=torch.bfloat16)
    v = torch.empty_like(k)
    lens = torch.ones(1, device="cuda", dtype=torch.int32)
    table = torch.zeros((1, 8), device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)
    plan = _SimtDecodeLaunchPlan(
        (lambda *args: launches.append(args), 2, 1, 16, 256),
        current,
        q,
        k,
        v,
        table,
    )
    other = torch.cuda.Stream()

    with torch.cuda.stream(other):
        plan(q, k, v, lens, table, out)

    raw_stream = launches[0][-1].value
    stream_ptr = (
        raw_stream if isinstance(raw_stream, int) else raw_stream.cuda_stream
    )
    assert stream_ptr == other.cuda_stream


def test_clear_kvcache_caches_also_clears_prefill_state():
    from kernels.attention.iluvatar import flash_attn_kvcache_planner as planner
    from kernels.attention.iluvatar import flash_attn_prefill as prefill

    planner._COMPILED_LAUNCH_CACHE["compiled"] = object()
    planner._SIMT_DECODE_FAST_CACHE["fast"] = object()
    planner._PLACEHOLDER_CACHE["workspace"] = object()
    prefill._PREFILL_CACHE["prefill"] = object()
    prefill._Q_PAD_CACHE["q"] = object()
    prefill._DENSE_KV_PAD_CACHE["kv"] = object()

    planner.clear_kvcache_caches(include_builders=True)

    assert not planner._COMPILED_LAUNCH_CACHE
    assert not planner._SIMT_DECODE_FAST_CACHE
    assert not planner._PLACEHOLDER_CACHE
    assert not prefill._PREFILL_CACHE
    assert not prefill._Q_PAD_CACHE
    assert not prefill._DENSE_KV_PAD_CACHE


def test_split_workspace_cache_is_bounded_across_streams():
    from kernels.attention.iluvatar import flash_attn_kvcache_planner as planner

    planner.clear_kvcache_caches()
    device = torch.device("cpu")
    for stream_ptr in range(planner._MAX_WORKSPACE_ENTRIES_PER_DEVICE + 8):
        planner._cached_split1_bufs(device, stream_ptr)

    entries = [
        key
        for key in planner._PLACEHOLDER_CACHE
        if key[0] == "split1" and key[1] == device
    ]
    assert len(entries) == planner._MAX_WORKSPACE_ENTRIES_PER_DEVICE
