# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar flex-attention device tests.

Covers:
* PR-1a: QK^T-only helper vs ``torch.matmul``.
* PR-1b: non-causal fused forward vs ``F.scaled_dot_product_attention``.
* PR-1c: causal fused forward vs SDPA ``is_causal=True``.
* PR-2a: softcap / SWA (+ combinations) vs hand-written fp32 reference.
* PR-2b: D=64 variant-cross + f16 dtype coverage.
* PR-2c: GQA / cross-attn / phys-pad tail + plan §3.2 shape-cross.
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


def _phys_seq(seq: int, block: int) -> int:
    return ((seq + block - 1) // block) * block


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
    with pytest.raises(ValueError, match=r"dtype must be one of"):
        mod.compile_iluvatar_flex_attention(1, 4, 64, 64, 128, dtype="fp32", is_causal=True)


def test_iluvatar_flex_attention_rejects_unsupported_head_dim():
    mod = _require_flex_attn_module()
    with pytest.raises(ValueError, match=r"D must be one of"):
        mod.compile_iluvatar_flex_attention(1, 4, 64, 64, 32, dtype="bf16", is_causal=True)


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


def test_iluvatar_flex_attention_rejects_unpadded_launch_shape(monkeypatch):
    """Launch must reject logical-sized tensors when phys pad is required."""
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    mod = _require_flex_attn_module()

    B, H, Sq, Skv, D = 1, 4, 100, 128, 64
    Sq_phys = _phys_seq(Sq, mod.BLOCK_M)
    assert Sq_phys != Sq
    launch = mod.compile_iluvatar_flex_attention(B, H, Sq, Skv, D, dtype="bf16", is_causal=False)
    Q = torch.randn(B, H, Sq, D, device="cuda", dtype=torch.bfloat16)  # not padded
    K = torch.randn(B, H, Skv, D, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(B, H, D, Skv, device="cuda", dtype=torch.bfloat16)
    O = torch.zeros(B, H, Sq, D, device="cuda", dtype=torch.bfloat16)  # noqa: E741
    with pytest.raises(ValueError, match="Sq_phys-padded"):
        launch(Q, K, V, O)


def test_iluvatar_flex_attention_qk_dot_rejects_gqa():
    mod = _require_flex_attn_module()
    with pytest.raises(NotImplementedError, match="qk_dot helper is MHA-only"):
        # Force the subset guard via the public validate path used by qk_dot.
        mod._validate_qk_dot_subset(H=4, Hkv=2, Sq=64, Skv=64)


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
    """Plan §3.1 hand-written fp32 reference for score mods (supports GQA)."""
    torch = _require_torch()
    H = Q.shape[1]
    Hkv = K.shape[1]
    if H != Hkv:
        group = H // Hkv
        K = K.repeat_interleave(group, dim=1)
        V = V.repeat_interleave(group, dim=1)
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
    D=128,
    Hkv=None,
    dtype="bf16",
    is_causal=False,
    window_size=None,
    softcap=None,
    seed=0,
):
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    mod = _require_flex_attn_module()

    if Hkv is None:
        Hkv = H
    torch_dtype = {"bf16": torch.bfloat16, "f16": torch.float16}[dtype]
    atol_rtol = 2e-2 if dtype == "bf16" else 1e-2
    if Sq >= 1024 or Skv >= 1024:
        atol_rtol *= 2
    sm_scale = 1.0 / math.sqrt(D)

    Sq_phys = _phys_seq(Sq, mod.BLOCK_M)
    Skv_phys = _phys_seq(Skv, mod.BLOCK_N)

    torch.manual_seed(seed)
    Q = torch.zeros(B, H, Sq_phys, D, device="cuda", dtype=torch_dtype)
    K = torch.zeros(B, Hkv, Skv_phys, D, device="cuda", dtype=torch_dtype)
    V_natural = torch.zeros(B, Hkv, Skv_phys, D, device="cuda", dtype=torch_dtype)
    Q[:, :, :Sq, :].copy_(torch.randn(B, H, Sq, D, device="cuda", dtype=torch_dtype))
    K[:, :, :Skv, :].copy_(torch.randn(B, Hkv, Skv, D, device="cuda", dtype=torch_dtype))
    V_natural[:, :, :Skv, :].copy_(torch.randn(B, Hkv, Skv, D, device="cuda", dtype=torch_dtype))
    V_transposed = V_natural.transpose(-1, -2).contiguous()
    O = torch.zeros(B, H, Sq_phys, D, device="cuda", dtype=torch_dtype)  # noqa: E741

    launch = mod.compile_iluvatar_flex_attention(
        B,
        H,
        Sq,
        Skv,
        D,
        Hkv=Hkv,
        dtype=dtype,
        is_causal=is_causal,
        window_size=window_size,
        softcap=softcap,
        sm_scale=sm_scale,
    )
    launch(Q, K, V_transposed, O)
    torch.cuda.synchronize()

    ref = _reference_flex_attention_fp32(
        Q[:, :, :Sq, :],
        K[:, :, :Skv, :],
        V_natural[:, :, :Skv, :],
        sm_scale=sm_scale,
        is_causal=is_causal,
        window_size=window_size,
        softcap=softcap,
    ).to(torch_dtype)
    return O[:, :, :Sq, :], ref, atol_rtol


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
    out, ref, tol = _run_flex_attn(
        monkeypatch,
        B=2,
        H=4,
        Sq=128,
        Skv=128,
        D=128,
        dtype="bf16",
        is_causal=is_causal,
        window_size=window_size,
        softcap=softcap,
    )
    torch = _require_torch()
    torch.testing.assert_close(out, ref, rtol=tol, atol=tol)


def test_iluvatar_flex_attention_pr2a_softcap_none_matches_pr1b(monkeypatch):
    """No softcap / no SWA must stay aligned with the fused-scale PR-1 path."""
    torch = _require_torch()
    out, ref, tol = _run_flex_attn(monkeypatch, B=1, H=4, Sq=64, Skv=64, is_causal=False)
    torch.testing.assert_close(out, ref, rtol=tol, atol=tol)


# --- PR-2b: f16 + D=64 -------------------------------------------------------


@pytest.mark.parametrize(
    "is_causal,window_size,softcap",
    [
        (False, None, None),
        (True, None, None),
        (False, 64, None),
        (False, None, 30.0),
        (True, 64, None),
        (True, None, 30.0),
    ],
)
def test_iluvatar_flex_attention_forward_pr2b_d64_variants(monkeypatch, is_causal, window_size, softcap):
    """Plan §3.2 variant-cross base shape with D=64."""
    out, ref, tol = _run_flex_attn(
        monkeypatch,
        B=2,
        H=4,
        Sq=256,
        Skv=256,
        D=64,
        dtype="bf16",
        is_causal=is_causal,
        window_size=window_size,
        softcap=softcap,
    )
    torch = _require_torch()
    torch.testing.assert_close(out, ref, rtol=tol, atol=tol)


def test_iluvatar_flex_attention_forward_pr2b_f16_causal(monkeypatch):
    """Plan §3.2 dtype-cross (smaller aligned shape for CI time)."""
    out, ref, tol = _run_flex_attn(
        monkeypatch,
        B=2,
        H=8,
        Sq=128,
        Skv=128,
        D=128,
        dtype="f16",
        is_causal=True,
    )
    torch = _require_torch()
    torch.testing.assert_close(out, ref, rtol=tol, atol=tol)


def test_iluvatar_flex_attention_forward_pr2b_f16_d64_smoke(monkeypatch):
    out, ref, tol = _run_flex_attn(
        monkeypatch,
        B=1,
        H=4,
        Sq=64,
        Skv=64,
        D=64,
        dtype="f16",
        is_causal=True,
    )
    torch = _require_torch()
    torch.testing.assert_close(out, ref, rtol=tol, atol=tol)


def test_iluvatar_flex_attention_qk_dot_pr2b_d64_smoke(monkeypatch):
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    mod = _require_flex_attn_module()

    B, H, Sq, Skv, D = 1, 4, 64, 64, 64
    sm_scale = 1.0 / math.sqrt(D)
    torch.manual_seed(0)
    Q = torch.randn(B, H, Sq, D, device="cuda", dtype=torch.bfloat16)
    K = torch.randn(B, H, Skv, D, device="cuda", dtype=torch.bfloat16)
    S = torch.zeros(B, H, Sq, Skv, device="cuda", dtype=torch.float32)

    launch = mod._compile_iluvatar_qk_dot_dev(B, H, Sq, Skv, D, dtype="bf16", sm_scale=sm_scale)
    launch(Q, K, S)
    torch.cuda.synchronize()

    ref = (Q.float() @ K.float().transpose(-1, -2)) * sm_scale
    torch.testing.assert_close(S, ref, rtol=2e-2, atol=2e-2)


# --- PR-2c: GQA / cross / tail + plan §3.2 shape-cross ------------------------


@pytest.mark.parametrize(
    "B,H,Hkv,Sq,Skv,D,is_causal",
    [
        (1, 4, 4, 128, 128, 64, True),  # MHA self
        (1, 4, 2, 128, 128, 64, True),  # GQA self
        (2, 8, 8, 512, 512, 128, True),  # medium self
        (1, 4, 4, 64, 256, 64, False),  # cross-attn
        # Plan table lists is_causal=True for shape-cross, but Sq≠Skv forbids it
        # (same note as #4); use False so the Skv tail path is actually exercised.
        (1, 4, 4, 128, 250, 64, False),  # unaligned Skv tail
    ],
)
def test_iluvatar_flex_attention_forward_pr2c_shape_cross(monkeypatch, B, H, Hkv, Sq, Skv, D, is_causal):
    """Plan §3.2 shape-cross (5 groups)."""
    out, ref, tol = _run_flex_attn(
        monkeypatch,
        B=B,
        H=H,
        Hkv=Hkv,
        Sq=Sq,
        Skv=Skv,
        D=D,
        dtype="bf16",
        is_causal=is_causal,
    )
    torch = _require_torch()
    torch.testing.assert_close(out, ref, rtol=tol, atol=tol)


def test_iluvatar_flex_attention_forward_pr2c_sq_tail_smoke(monkeypatch):
    """Sq not a multiple of BLOCK_M (epilogue writes phys-padded O)."""
    out, ref, tol = _run_flex_attn(
        monkeypatch,
        B=1,
        H=4,
        Sq=100,
        Skv=128,
        D=64,
        dtype="bf16",
        is_causal=False,
    )
    torch = _require_torch()
    torch.testing.assert_close(out, ref, rtol=tol, atol=tol)


def test_iluvatar_flex_attention_package_export():
    from kernels.attention.iluvatar import compile_iluvatar_flex_attention as exported

    mod = _require_flex_attn_module()
    assert exported is mod.compile_iluvatar_flex_attention
