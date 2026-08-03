# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar flex-attention device tests.

Covers:
* PR-1a: QK^T-only helper vs ``torch.matmul``.
* PR-1b: non-causal fused forward vs ``F.scaled_dot_product_attention``.
* PR-1c: causal fused forward vs SDPA ``is_causal=True``.
* PR-2a: softcap / SWA (+ combinations) vs hand-written fp32 reference.
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


# --- PR-1b: full non-causal fused kernel --------------------------------------


@pytest.mark.parametrize(
    "B,H,Sq,Skv",
    [
        (1, 4, 64, 64),  # single q_tile / single kv_tile
        (2, 4, 128, 128),  # multiple q_tiles / multiple kv_tiles
        (1, 8, 192, 192),  # 3x3 tile grid, larger head count
    ],
)
def test_iluvatar_flex_attention_forward_pr1b(monkeypatch, B, H, Sq, Skv):
    """PR-1b: full non-causal forward matches ``F.scaled_dot_product_attention``.

    The launcher expects V pre-transposed by the host to ``[B, H, D, Skv]`` so
    both MMA operands stay k-major "tn"; the test replicates that contract.
    """
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    mod = _require_flex_attn_module()

    D = 128
    torch_dtype = torch.bfloat16
    sm_scale = 1.0 / math.sqrt(D)

    torch.manual_seed(0)
    Q = torch.randn(B, H, Sq, D, device="cuda", dtype=torch_dtype)
    K = torch.randn(B, H, Skv, D, device="cuda", dtype=torch_dtype)
    V_natural = torch.randn(B, H, Skv, D, device="cuda", dtype=torch_dtype)
    V_transposed = V_natural.transpose(-1, -2).contiguous()  # [B, H, D, Skv]
    O = torch.zeros(B, H, Sq, D, device="cuda", dtype=torch_dtype)  # noqa: E741

    launch = mod.compile_iluvatar_flex_attention(B, H, Sq, Skv, D, dtype="bf16", is_causal=False, sm_scale=sm_scale)
    launch(Q, K, V_transposed, O)
    torch.cuda.synchronize()

    ref = torch.nn.functional.scaled_dot_product_attention(
        Q.float(),
        K.float(),
        V_natural.float(),
        is_causal=False,
        scale=sm_scale,
    ).to(torch_dtype)
    torch.testing.assert_close(O, ref, rtol=2e-2, atol=2e-2)


# --- PR-1c: causal fused kernel ----------------------------------------------


@pytest.mark.parametrize(
    "B,H,Sq",
    [
        (1, 4, 64),  # single tile
        (2, 4, 128),  # multiple q_tiles / kv_tiles — exercises the triangular pattern across tiles
        (1, 8, 192),  # 3x3 tile grid, larger head count
    ],
)
def test_iluvatar_flex_attention_forward_pr1c_causal(monkeypatch, B, H, Sq):
    """PR-1c: causal forward matches ``F.scaled_dot_product_attention(is_causal=True)``.

    ``is_causal=True`` requires ``Sq == Skv``; the causal mask is applied
    after the ``(sm_scale * log2e)`` scale on ``S = Q @ K^T`` and before the
    online rowmax, using ``NEG_LARGE_F`` as the log2-domain sentinel.
    """
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    mod = _require_flex_attn_module()

    Skv = Sq
    D = 128
    torch_dtype = torch.bfloat16
    sm_scale = 1.0 / math.sqrt(D)

    torch.manual_seed(0)
    Q = torch.randn(B, H, Sq, D, device="cuda", dtype=torch_dtype)
    K = torch.randn(B, H, Skv, D, device="cuda", dtype=torch_dtype)
    V_natural = torch.randn(B, H, Skv, D, device="cuda", dtype=torch_dtype)
    V_transposed = V_natural.transpose(-1, -2).contiguous()  # [B, H, D, Skv]
    O = torch.zeros(B, H, Sq, D, device="cuda", dtype=torch_dtype)  # noqa: E741

    launch = mod.compile_iluvatar_flex_attention(B, H, Sq, Skv, D, dtype="bf16", is_causal=True, sm_scale=sm_scale)
    launch(Q, K, V_transposed, O)
    torch.cuda.synchronize()

    ref = torch.nn.functional.scaled_dot_product_attention(
        Q.float(),
        K.float(),
        V_natural.float(),
        is_causal=True,
        scale=sm_scale,
    ).to(torch_dtype)
    torch.testing.assert_close(O, ref, rtol=2e-2, atol=2e-2)


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


def test_iluvatar_flex_attention_rejects_nonpositive_window_size():
    mod = _require_flex_attn_module()
    with pytest.raises(ValueError, match=r"window_size must be > 0"):
        mod.compile_iluvatar_flex_attention(1, 4, 64, 64, 128, dtype="bf16", is_causal=True, window_size=0)


# --- PR-2a: softcap / SWA ----------------------------------------------------


def _reference_flex_attention_fp32(
    Q,
    K,
    V,
    *,
    sm_scale: float,
    is_causal: bool = False,
    window_size: int | None = None,
    softcap: float | None = None,
):
    """Plan §3.1 hand-written fp32 reference for score mods."""
    torch = _require_torch()
    S = torch.matmul(Q.float(), K.float().transpose(-1, -2)) * sm_scale
    if softcap is not None:
        S = softcap * torch.tanh(S / softcap)
    Sq = S.shape[-2]
    Skv = S.shape[-1]
    q_idx = torch.arange(Sq, device=S.device).view(Sq, 1)
    kv_idx = torch.arange(Skv, device=S.device).view(1, Skv)
    mask = torch.zeros((Sq, Skv), device=S.device, dtype=torch.bool)
    if is_causal:
        mask = mask | (kv_idx > q_idx)
    if window_size is not None:
        mask = mask | ((q_idx - kv_idx) > window_size)
    S = S.masked_fill(mask, float("-inf"))
    P = torch.softmax(S, dim=-1)
    P = torch.nan_to_num(P, nan=0.0)
    return torch.matmul(P, V.float())


def _run_flex_attn(
    monkeypatch,
    *,
    B,
    H,
    Sq,
    Skv,
    is_causal=False,
    window_size=None,
    softcap=None,
    seed=0,
):
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    mod = _require_flex_attn_module()

    D = 128
    torch_dtype = torch.bfloat16
    sm_scale = 1.0 / math.sqrt(D)

    torch.manual_seed(seed)
    Q = torch.randn(B, H, Sq, D, device="cuda", dtype=torch_dtype)
    K = torch.randn(B, H, Skv, D, device="cuda", dtype=torch_dtype)
    V_natural = torch.randn(B, H, Skv, D, device="cuda", dtype=torch_dtype)
    V_transposed = V_natural.transpose(-1, -2).contiguous()
    O = torch.zeros(B, H, Sq, D, device="cuda", dtype=torch_dtype)  # noqa: E741

    launch = mod.compile_iluvatar_flex_attention(
        B,
        H,
        Sq,
        Skv,
        D,
        dtype="bf16",
        is_causal=is_causal,
        window_size=window_size,
        softcap=softcap,
        sm_scale=sm_scale,
    )
    launch(Q, K, V_transposed, O)
    torch.cuda.synchronize()

    ref = _reference_flex_attention_fp32(
        Q,
        K,
        V_natural,
        sm_scale=sm_scale,
        is_causal=is_causal,
        window_size=window_size,
        softcap=softcap,
    ).to(torch_dtype)
    return O, ref


@pytest.mark.parametrize(
    "is_causal,window_size,softcap",
    [
        (False, 64, None),
        (False, None, 30.0),
        (True, 64, None),
        (True, None, 30.0),
        (True, 64, 30.0),
        (False, 64, 30.0),
    ],
)
def test_iluvatar_flex_attention_forward_pr2a_variants(monkeypatch, is_causal, window_size, softcap):
    out, ref = _run_flex_attn(
        monkeypatch,
        B=2,
        H=4,
        Sq=128,
        Skv=128,
        is_causal=is_causal,
        window_size=window_size,
        softcap=softcap,
    )
    torch = _require_torch()
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


def test_iluvatar_flex_attention_pr2a_softcap_none_matches_pr1b(monkeypatch):
    """No softcap / no SWA must stay aligned with the fused-scale PR-1 path."""
    torch = _require_torch()
    out, ref = _run_flex_attn(monkeypatch, B=1, H=4, Sq=64, Skv=64, is_causal=False)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
