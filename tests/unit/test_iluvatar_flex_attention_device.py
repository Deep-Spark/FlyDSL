# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar flex-attention device tests.

PR-1a covers the ``Q @ K^T * sm_scale`` sub-step only (``_compile_iluvatar_qk_dot_dev``);
the fused kernel entry point ``compile_iluvatar_flex_attention`` currently raises
``NotImplementedError`` and is tested in that state.

PR-1b/PR-1c will expand this file with the full-kernel correctness cases.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l2_device]

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        pytest.skip(f"torch is required for Iluvatar flex-attention device tests: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA-compatible Iluvatar device is not available")
    return torch


def _configure_iluvatar_env(monkeypatch) -> None:
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


def _require_flex_attn_module():
    try:
        import kernels.attention.iluvatar.flex_attention as mod
    except ModuleNotFoundError as exc:
        pytest.skip(f"kernels.attention.iluvatar.flex_attention not importable: {exc}")
    return mod


# --- PR-1a: Q @ K^T only ------------------------------------------------------


@pytest.mark.parametrize(
    "B,H,Sq,Skv",
    [
        (1, 4, 64, 64),  # smallest well-aligned tile
        (2, 4, 128, 128),  # multiple q_tiles and kv_tiles
    ],
)
def test_iluvatar_flex_attention_qk_dot_pr1a(monkeypatch, B, H, Sq, Skv):
    """PR-1a: fused ``S = (Q @ K^T) * sm_scale`` matches fp32 reference within bf16 tolerance."""
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    mod = _require_flex_attn_module()

    D = 128
    torch_dtype = torch.bfloat16
    sm_scale = 1.0 / math.sqrt(D)

    torch.manual_seed(0)
    Q = torch.randn(B, H, Sq, D, device="cuda", dtype=torch_dtype)
    K = torch.randn(B, H, Skv, D, device="cuda", dtype=torch_dtype)
    S = torch.zeros(B, H, Sq, Skv, device="cuda", dtype=torch.float32)

    launch = mod._compile_iluvatar_qk_dot_dev(B, H, Sq, Skv, D, dtype="bf16", sm_scale=sm_scale)
    launch(Q, K, S)
    torch.cuda.synchronize()

    ref = (Q.float() @ K.float().transpose(-1, -2)) * sm_scale
    torch.testing.assert_close(S, ref, rtol=2e-2, atol=2e-2)


# --- Compile-time validation --------------------------------------------------


def test_iluvatar_flex_attention_rejects_unsupported_dtype():
    mod = _require_flex_attn_module()
    with pytest.raises(NotImplementedError, match="f16 lands in PR-2"):
        mod.compile_iluvatar_flex_attention(1, 4, 64, 64, 128, dtype="f16", is_causal=True)


def test_iluvatar_flex_attention_rejects_unsupported_head_dim():
    mod = _require_flex_attn_module()
    with pytest.raises(NotImplementedError, match=r"D=64 lands in PR-2"):
        mod.compile_iluvatar_flex_attention(1, 4, 64, 64, 64, dtype="bf16", is_causal=True)


def test_iluvatar_flex_attention_rejects_gqa_in_pr1():
    mod = _require_flex_attn_module()
    with pytest.raises(NotImplementedError, match="GQA lands in PR-2"):
        mod.compile_iluvatar_flex_attention(1, 4, 64, 64, 128, Hkv=2, dtype="bf16", is_causal=True)


def test_iluvatar_flex_attention_rejects_cross_attn_in_pr1():
    mod = _require_flex_attn_module()
    with pytest.raises(NotImplementedError, match="Cross-attention lands in PR-2"):
        mod.compile_iluvatar_flex_attention(1, 4, 64, 128, 128, dtype="bf16", is_causal=False)


def test_iluvatar_flex_attention_rejects_illegal_causal_shape():
    mod = _require_flex_attn_module()
    with pytest.raises(ValueError, match="is_causal=True requires Sq == Skv"):
        mod.compile_iluvatar_flex_attention(1, 4, 64, 128, 128, dtype="bf16", is_causal=True)


def test_iluvatar_flex_attention_rejects_illegal_H_Hkv():
    mod = _require_flex_attn_module()
    with pytest.raises(ValueError, match="must be divisible by Hkv"):
        mod.compile_iluvatar_flex_attention(1, 5, 64, 64, 128, Hkv=2, dtype="bf16", is_causal=True)


def test_iluvatar_flex_attention_rejects_negative_softcap():
    mod = _require_flex_attn_module()
    with pytest.raises(ValueError, match=r"softcap must be > 0"):
        mod.compile_iluvatar_flex_attention(1, 4, 64, 64, 128, dtype="bf16", is_causal=True, softcap=-1.0)
