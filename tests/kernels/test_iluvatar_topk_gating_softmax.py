# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar fused TopK gating Softmax device tests."""

import os
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.iluvatar_lower]

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_CASES = (
    # label, tokens, experts, topk, dtype_str, renormalize
    # Token counts sized for WARPS_PER_BLOCK=16 (E=64→TPB=256, E=128→128, E=256→64)
    # so prefill/dsv2 still exercise multi-block grids.
    ("decode_f32", 1, 64, 8, "f32", True),
    ("prefill_f32", 512, 64, 8, "f32", True),
    ("dsv2_f32", 128, 256, 6, "f32", True),
    ("prefill_bf16", 256, 128, 8, "bf16", True),
    ("norenorm_f32", 16, 64, 4, "f32", False),
)


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        pytest.skip(f"torch is required for Iluvatar TopK gating tests: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA-compatible Iluvatar device is not available")
    return torch


def _configure_iluvatar_env(monkeypatch) -> None:
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


def _require_kernel():
    try:
        from kernels.moe.iluvatar.topk_gating_softmax import build_iluvatar_topk_gating_softmax
    except ModuleNotFoundError as exc:
        pytest.fail(f"failed to import iluvatar topk_gating_softmax: {exc}")
    return build_iluvatar_topk_gating_softmax


def _torch_dtype(torch, dtype_str: str):
    return {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}[dtype_str]


def _reference(torch, gating_fp32, topk: int, renormalize: bool):
    probs = torch.softmax(gating_fp32, dim=-1)
    weights, ids = torch.topk(probs, topk, dim=-1)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-20)
    return weights.to(torch.float32), ids.to(torch.int32)


@pytest.mark.parametrize(
    "label,num_tokens,num_experts,topk,dtype_str,renormalize",
    _CASES,
    ids=[c[0] for c in _CASES],
)
def test_iluvatar_topk_gating_softmax(
    label,
    num_tokens,
    num_experts,
    topk,
    dtype_str,
    renormalize,
    monkeypatch,
):
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    build_fn = _require_kernel()

    launch = build_fn(
        num_experts=num_experts,
        topk=topk,
        dtype_str=dtype_str,
        renormalize=renormalize,
        emit_tei=True,
    )

    torch.manual_seed(42)
    torch_dtype = _torch_dtype(torch, dtype_str)
    gating_fp32 = (torch.rand(num_tokens, num_experts, device="cuda", dtype=torch.float32) * 4.0) - 2.0
    # Quantize first so the reference sees the same bytes as the kernel.
    gating = gating_fp32.to(torch_dtype).contiguous()
    gating_for_ref = gating.to(torch.float32)

    ref_w, ref_ids = _reference(torch, gating_for_ref, topk, renormalize)

    weights = torch.empty(num_tokens, topk, device="cuda", dtype=torch.float32)
    ids = torch.empty(num_tokens, topk, device="cuda", dtype=torch.int32)
    tei = torch.empty(num_tokens, topk, device="cuda", dtype=torch.int32)

    stream = torch.cuda.Stream()
    launch(gating, weights, ids, num_tokens, tei=tei, stream=stream)
    torch.cuda.synchronize()

    # Tie-breaking may reorder equal probs; compare as (id, weight) multisets per row
    # and as sorted weights.
    ids_match = True
    for t in range(num_tokens):
        if set(ids[t].tolist()) != set(ref_ids[t].tolist()):
            ids_match = False
            break
    assert ids_match, f"{label}: topk ids mismatch"

    w_diff = (weights.sort(dim=-1).values - ref_w.sort(dim=-1).values).abs().max().item()
    atol = 2e-3 if dtype_str != "f32" else 1e-5
    assert w_diff < atol, f"{label}: weight max abs diff {w_diff} >= {atol}"

    for k in range(topk):
        expect = k * num_tokens + torch.arange(num_tokens, device="cuda", dtype=torch.int32)
        assert torch.equal(tei[:, k], expect), f"{label}: tei column {k} mismatch"


def test_iluvatar_topk_gating_layout_rejects_bad_experts():
    build_fn = _require_kernel()
    with pytest.raises(ValueError, match="not supported"):
        build_fn(num_experts=96, topk=4, dtype_str="f32")


def test_iluvatar_topk_gating_layout_values():
    from kernels.moe.iluvatar.topk_gating_softmax import WARPS_PER_BLOCK

    build_fn = _require_kernel()
    launch = build_fn(num_experts=128, topk=8, dtype_str="f32")
    # Prefer largest VPT: E=128 → VPT=16, TPT=8, tokens/warp=8,
    # tokens/block = WARPS_PER_BLOCK * 8 (16 → 128).
    assert launch.layout["VPT"] == 16
    assert launch.layout["THREADS_PER_TOKEN"] == 8
    assert launch.layout["TOKENS_PER_WARP"] == 8
    assert launch.layout["TOKENS_PER_BLOCK"] == WARPS_PER_BLOCK * 8
