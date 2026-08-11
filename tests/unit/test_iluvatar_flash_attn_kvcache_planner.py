# SPDX-License-Identifier: Apache-2.0

"""Host-side decode backend selection and planning (no kernel launch)."""

import pytest
import torch

from kernels.attention.iluvatar.flash_attn_kvcache_planner import (
    plan_decode_launch,
    select_decode_backends,
)


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
def test_d256_explicit_split_one_falls_back_on_short_page(monkeypatch):
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
    assert plan.use_simt_decode is False
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
def test_simt_fast_key_buckets_max_context_len(monkeypatch):
    """Fast-key hashing needs real CUDA tensors; keep it out of L0 CPU CI."""
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

    assert key_for(1025) == key_for(1056)
    assert key_for(1025) != key_for(1024)
