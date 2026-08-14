# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar flex-attention device tests.

Covers:
* QK^T-only helper vs ``torch.matmul``.
* Non-causal and causal fused forward vs scaled dot-product attention.
* Softcap and sliding-window combinations vs a hand-written fp32 reference.
* f16/bf16, D=64/128/256, MHA/GQA, cross-attention, and physical-padding tails.
* Packed varlen (``cu_seqlens``) self-attn vs per-seq dense concat.
* Paged KV (``block_table``) vs gather-to-dense flex reference.
* Dense alibi / score_bias and the ``flydsl_flex_attn_func`` dispatcher.
* Dense ``score_mod=TracedScoreMod`` (V3-2) vs host ``eval_host`` reference.
* Dense ``create_block_mask`` / ``block_mask`` sparse KV skip (V3-3) vs dense path.
* Optional ``tile_config`` whitelist and dense ``autotune_iluvatar_flex_attention_tile``.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import flydsl.expr as fx
import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.iluvatar_lower]

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


def _configure_iluvatar_env(monkeypatch, *, enable_cache: bool = False) -> None:
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "1" if enable_cache else "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


def _require_flex_attn_module():
    try:
        import kernels.attention.iluvatar.flex_attention as mod
    except ModuleNotFoundError as exc:
        pytest.skip(f"kernels.attention.iluvatar.flex_attention not importable: {exc}")
    return mod


def _phys_seq(seq: int, block: int) -> int:
    return ((seq + block - 1) // block) * block


def _require_perf_enabled() -> None:
    if os.environ.get("FLYDSL_ILUVATAR_RUN_FLEX_ATTN_PERF", "").lower() not in {"1", "true", "yes", "on"}:
        pytest.skip("set FLYDSL_ILUVATAR_RUN_FLEX_ATTN_PERF=1 to run Iluvatar flex-attention perf")


def _bench_gpu_us(fn, *, warmup: int, iters: int) -> float:
    torch = _require_torch()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end) * 1000.0 / iters)


def _attention_tflops(*, B: int, H: int, Sq: int, Skv: int, D: int, latency_us: float) -> float:
    """Rough forward FLOPs: QK^T + PV ~= 4 * B * H * Sq * Skv * D."""
    if latency_us <= 0:
        return 0.0
    flops = 4.0 * B * H * Sq * Skv * D
    return flops / (latency_us * 1.0e6)


# --- Q @ K^T only -------------------------------------------------------------


@pytest.mark.parametrize(
    "B,H,Sq,Skv",
    [
        (1, 4, 64, 64),  # smallest well-aligned tile
        (2, 4, 128, 128),  # multiple q_tiles and kv_tiles
    ],
)
def test_iluvatar_flex_attention_qk_dot_pr1a(monkeypatch, B, H, Sq, Skv):
    """The fused QK score matches the fp32 reference within bf16 tolerance."""
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


# --- Full non-causal fused kernel ---------------------------------------------


@pytest.mark.parametrize(
    "B,H,Sq,Skv",
    [
        (1, 4, 64, 64),  # single q_tile / single kv_tile
        (2, 4, 128, 128),  # multiple q_tiles / multiple kv_tiles
        (1, 8, 192, 192),  # 3x3 tile grid, larger head count
    ],
)
def test_iluvatar_flex_attention_forward_pr1b(monkeypatch, B, H, Sq, Skv):
    """Full non-causal forward matches ``F.scaled_dot_product_attention``.

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


# --- Causal fused kernel ------------------------------------------------------


@pytest.mark.parametrize(
    "B,H,Sq",
    [
        (1, 4, 64),  # single tile
        (2, 4, 128),  # multiple q_tiles / kv_tiles -- exercises the triangular pattern across tiles
        (1, 8, 192),  # 3x3 tile grid, larger head count
    ],
)
def test_iluvatar_flex_attention_forward_pr1c_causal(monkeypatch, B, H, Sq):
    """Causal forward matches ``F.scaled_dot_product_attention(is_causal=True)``.

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
    with pytest.raises(ValueError, match=r"D must be one of"):
        mod.compile_iluvatar_flex_attention(1, 4, 64, 64, 96, dtype="bf16", is_causal=True)
    with pytest.raises(ValueError, match=r"D must be one of"):
        mod.compile_iluvatar_flex_attention(1, 4, 64, 64, 192, dtype="bf16", is_causal=True)


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


# --- Softcap / SWA ------------------------------------------------------------


def _reference_flex_attention_fp32(
    Q,
    K,
    V,
    *,
    sm_scale: float,
    is_causal: bool = False,
    window_size: int | None = None,
    softcap: float | None = None,
    alibi_slopes=None,
    score_bias=None,
    score_mod=None,
    mask_mod=None,
):
    """Hand-written fp32 reference for score modifications, including GQA."""
    torch = _require_torch()
    H = Q.shape[1]
    Hkv = K.shape[1]
    if H != Hkv:
        group = H // Hkv
        K = K.repeat_interleave(group, dim=1)
        V = V.repeat_interleave(group, dim=1)
    S = torch.matmul(Q.float(), K.float().transpose(-1, -2)) * sm_scale
    Sq = S.shape[-2]
    Skv = S.shape[-1]
    B = S.shape[0]
    q_idx = torch.arange(Sq, device=S.device).view(Sq, 1)
    kv_idx = torch.arange(Skv, device=S.device).view(1, Skv)
    if alibi_slopes is not None and score_bias is not None:
        raise ValueError("alibi_slopes and score_bias are mutually exclusive")
    if alibi_slopes is not None:
        # slopes [H]; bias = -slope * (q - kv)
        slopes = alibi_slopes.float().view(1, H, 1, 1)
        S = S + (-slopes * (q_idx.float() - kv_idx.float()))
    if score_bias is not None:
        sb = score_bias.float()
        if sb.shape[1] == Sq and sb.dim() == 4 and sb.shape[2] == H:
            sb = sb.permute(0, 2, 1, 3)
        S = S + sb
    if softcap is not None:
        S = softcap * torch.tanh(S / softcap)
    if score_mod is not None:
        # Dense logical prefix; matches kernel phys indices when Sq/Skv unpadded
        # or when only the logical region is compared.
        Sout = S.clone()
        for b in range(B):
            for h in range(H):
                for qi in range(Sq):
                    for ki in range(Skv):
                        Sout[b, h, qi, ki] = float(
                            score_mod.eval_host(float(S[b, h, qi, ki]), b, h, qi, ki)
                        )
        S = Sout
    mask = torch.zeros((Sq, Skv), device=S.device, dtype=torch.bool)
    if is_causal:
        # Align q=0 with kv=(Skv-Sq) so decode (Sq=1) and cross-length prefill
        # match the paged kernel's delta = seqlen_kv - Sq.
        delta = Skv - Sq
        mask = mask | (kv_idx > q_idx + delta)
    if window_size is not None:
        mask = mask | ((q_idx - kv_idx) > window_size)
    if mask_mod is not None:
        for qi in range(Sq):
            for ki in range(Skv):
                if not bool(mask_mod.eval_host(0, 0, qi, ki)):
                    mask[qi, ki] = True
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
    tile_config=None,
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

    block_m, block_n = mod._normalize_tile_config(tile_config)
    Sq_phys = _phys_seq(Sq, block_m)
    Skv_phys = _phys_seq(Skv, block_n)

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
        tile_config=tile_config,
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
    """No softcap or SWA stays aligned with the fused-scale path."""
    torch = _require_torch()
    out, ref, tol = _run_flex_attn(monkeypatch, B=1, H=4, Sq=64, Skv=64, is_causal=False)
    torch.testing.assert_close(out, ref, rtol=tol, atol=tol)


# --- f16 + D=64 ---------------------------------------------------------------


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
    """Cover score-modification variants with D=64."""
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
    """Cover the f16 causal path with an aligned CI-sized shape."""
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


# --- GQA / cross-attention / sequence tails ----------------------------------


@pytest.mark.parametrize(
    "B,H,Hkv,Sq,Skv,D,is_causal",
    [
        (1, 4, 4, 128, 128, 64, True),  # MHA self
        (1, 4, 2, 128, 128, 64, True),  # GQA self
        (2, 8, 8, 512, 512, 128, True),  # medium self
        (1, 4, 4, 64, 256, 64, False),  # cross-attn
        # Causal mode requires Sq == Skv; use False to exercise the Skv tail.
        (1, 4, 4, 128, 250, 64, False),  # unaligned Skv tail
    ],
)
def test_iluvatar_flex_attention_forward_pr2c_shape_cross(monkeypatch, B, H, Hkv, Sq, Skv, D, is_causal):
    """Cover MHA, GQA, cross-attention, and an unaligned Skv tail."""
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
    from kernels.attention.iluvatar import autotune_iluvatar_flex_attention_tile as exported_tune
    from kernels.attention.iluvatar import compile_iluvatar_flex_attention as exported
    from kernels.attention.iluvatar import flydsl_flex_attn_func as exported_func

    mod = _require_flex_attn_module()
    assert exported is mod.compile_iluvatar_flex_attention
    assert exported_func is not None
    from kernels.attention.iluvatar import flex_attn_interface as iface

    assert exported_func is iface.flydsl_flex_attn_func
    assert exported_tune is iface.autotune_iluvatar_flex_attention_tile


# --- Perf (opt-in; consumed by perf-daily-iluvatar) ----------------------------

# Fallback when FLYDSL_PERF_CONFIG_PATH is unset. Keep in sync with
# ``iluvatar_flex_attention.params.perf_config`` in
# ``.github/perf-kernels-iluvatar.json`` so daily trend keys stay stable.
_DEFAULT_FLEX_ATTN_PERF_CONFIG = {
    "shape": {"B": 2, "H": 32, "Hkv": 8, "Sq": 4096, "Skv": 4096, "D": 128},
    "warmup": 5,
    "iters": 10,
    "cases": [
        {"name": "causal", "is_causal": True, "window_size": None, "softcap": None, "dtype": "bf16"},
        {"name": "causal", "is_causal": True, "window_size": None, "softcap": None, "dtype": "f16"},
        {
            "name": "causal_swa1024",
            "is_causal": True,
            "window_size": 1024,
            "softcap": None,
            "dtype": "bf16",
        },
        {
            "name": "causal_swa1024",
            "is_causal": True,
            "window_size": 1024,
            "softcap": None,
            "dtype": "f16",
        },
        {
            "name": "causal_softcap30",
            "is_causal": True,
            "window_size": None,
            "softcap": 30.0,
            "dtype": "bf16",
        },
        {
            "name": "causal_softcap30",
            "is_causal": True,
            "window_size": None,
            "softcap": 30.0,
            "dtype": "f16",
        },
    ],
}

_FLEX_ATTN_PERF_CONFIG_ENV = "FLYDSL_PERF_CONFIG_PATH"


def _load_flex_attn_perf_config() -> dict:
    """Load suite ``perf_config`` from env path, else the hardcoded §3.3 six cases."""
    path = os.environ.get(_FLEX_ATTN_PERF_CONFIG_ENV, "").strip()
    if not path:
        return dict(_DEFAULT_FLEX_ATTN_PERF_CONFIG)
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"{_FLEX_ATTN_PERF_CONFIG_ENV} does not exist: {path}")
    with cfg_path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{_FLEX_ATTN_PERF_CONFIG_ENV} must point to a JSON object, got {type(cfg)}")
    return cfg


def _parse_flex_attn_perf_cases(cfg: dict):
    shape = cfg.get("shape") or {}
    if not isinstance(shape, dict):
        raise ValueError("perf_config.shape must be an object")
    required = ("B", "H", "Hkv", "Sq", "Skv", "D")
    missing = [k for k in required if k not in shape]
    if missing:
        raise ValueError(f"perf_config.shape missing keys: {missing}")
    B = int(shape["B"])
    H = int(shape["H"])
    Hkv = int(shape["Hkv"])
    Sq = int(shape["Sq"])
    Skv = int(shape["Skv"])
    D = int(shape["D"])

    raw_cases = cfg.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("perf_config.cases must be a non-empty list")
    cases = []
    for i, case in enumerate(raw_cases):
        if not isinstance(case, dict):
            raise ValueError(f"perf_config.cases[{i}] must be an object")
        name = str(case["name"])
        is_causal = bool(case.get("is_causal", False))
        window_size = case.get("window_size", None)
        if window_size is not None:
            window_size = int(window_size)
        softcap = case.get("softcap", None)
        if softcap is not None:
            softcap = float(softcap)
        dtype_str = str(case["dtype"])
        cases.append((name, is_causal, window_size, softcap, dtype_str))
    return B, H, Hkv, Sq, Skv, D, cases


def test_iluvatar_flex_attention_perf(monkeypatch):
    """Large-shape latency probe for daily trend (six cases by default)."""
    _require_perf_enabled()
    torch = _require_torch()
    # The kv loop is unrolled at trace time, so JIT compilation for these shapes
    # costs minutes and grows linearly with Skv. Reuse the disk cache so a daily
    # run pays it once instead of per invocation.
    _configure_iluvatar_env(monkeypatch, enable_cache=True)
    mod = _require_flex_attn_module()

    cfg = _load_flex_attn_perf_config()
    B, H, Hkv, Sq, Skv, D, cases = _parse_flex_attn_perf_cases(cfg)

    # warmup must stay >= 1: the first launch triggers JIT compilation, which
    # would otherwise dominate the reported latency. Env overrides win over json.
    warmup_default = int(cfg.get("warmup", 5))
    iters_default = int(cfg.get("iters", 10))
    warmup = max(1, int(os.environ.get("FLYDSL_ILUVATAR_FLEX_ATTN_PERF_WARMUP", str(warmup_default))))
    iters = int(os.environ.get("FLYDSL_ILUVATAR_FLEX_ATTN_PERF_ITERS", str(iters_default)))

    Sq_phys = _phys_seq(Sq, mod.BLOCK_M)
    Skv_phys = _phys_seq(Skv, mod.BLOCK_N)
    assert Sq_phys == Sq and Skv_phys == Skv

    metrics = {}
    for case_name, is_causal, window_size, softcap, dtype_str in cases:
        torch_dtype = {"bf16": torch.bfloat16, "f16": torch.float16}[dtype_str]
        sm_scale = 1.0 / math.sqrt(D)

        torch.manual_seed(123)
        Q = torch.randn(B, H, Sq_phys, D, device="cuda", dtype=torch_dtype).contiguous()
        K = torch.randn(B, Hkv, Skv_phys, D, device="cuda", dtype=torch_dtype).contiguous()
        V_natural = torch.randn(B, Hkv, Skv_phys, D, device="cuda", dtype=torch_dtype).contiguous()
        V_transposed = V_natural.transpose(-1, -2).contiguous()
        O = torch.empty(B, H, Sq_phys, D, device="cuda", dtype=torch_dtype)  # noqa: E741

        launch = mod.compile_iluvatar_flex_attention(
            B,
            H,
            Sq,
            Skv,
            D,
            Hkv=Hkv,
            dtype=dtype_str,
            is_causal=is_causal,
            window_size=window_size,
            softcap=softcap,
            sm_scale=sm_scale,
        )
        stream = torch.cuda.current_stream()

        def run_flydsl():
            launch(Q, K, V_transposed, O, stream=stream)

        flydsl_us = _bench_gpu_us(run_flydsl, warmup=warmup, iters=iters)
        tflops = _attention_tflops(B=B, H=H, Sq=Sq, Skv=Skv, D=D, latency_us=flydsl_us)

        point_metrics = {
            "latency_us": float(flydsl_us),
            "tflops": float(tflops),
        }

        compare_torch = window_size is None and softcap is None
        torch_us = None
        speedup = None
        if compare_torch:
            # SDPA expects natural V layout [B, Hkv, Skv, D]; expand GQA heads for torch.
            group = H // Hkv
            K_torch = K.repeat_interleave(group, dim=1)
            V_torch = V_natural.repeat_interleave(group, dim=1)

            def run_torch():
                torch.nn.functional.scaled_dot_product_attention(
                    Q,
                    K_torch,
                    V_torch,
                    is_causal=True,
                    scale=sm_scale,
                )

            torch_us = _bench_gpu_us(run_torch, warmup=warmup, iters=iters)
            speedup = torch_us / flydsl_us if flydsl_us > 0 else 0.0
            point_metrics["torch_latency_us"] = float(torch_us)
            point_metrics["speedup_torch"] = float(speedup)

        metric_key = f"{case_name}.{dtype_str}"
        print(
            f"[iluvatar-flex-attn-perf] key={metric_key} "
            f"shape=B{B}_H{H}_Hkv{Hkv}_Sq{Sq}_Skv{Skv}_D{D} "
            f"flydsl={flydsl_us:.1f}us tflops={tflops:.2f} "
            f"torch={f'{torch_us:.1f}us' if torch_us is not None else 'N/A'} "
            f"speedup_torch={f'{speedup:.3f}x' if speedup is not None else 'N/A'}"
        )
        metrics[metric_key] = point_metrics

    print("PERF_CASE_JSON=" + json.dumps({"metrics": metrics}, sort_keys=True))


# --- V2-1 packed varlen -------------------------------------------------------


def _pack_varlen_qkv(seqlens, H, Hkv, D, torch_dtype, *, seed=0, align=32, pad_tokens=64):
    """Pack varlen Q/K/V with 32-aligned sequence starts for SME G2S.

    Returns physical ``cu_seqlens`` (addressing) and logical ``seq_lens`` (masking).
    ``total_tokens`` is rounded up to a multiple of ``align``.
    """
    torch = _require_torch()
    torch.manual_seed(seed)
    align = int(align)
    cu_phys = [0]
    for s in seqlens:
        # Pad each sequence so the next start stays ``align``-aligned.
        phys = ((int(s) + align - 1) // align) * align
        cu_phys.append(cu_phys[-1] + phys)
    total = cu_phys[-1] + int(pad_tokens)
    total = ((total + align - 1) // align) * align
    Q = torch.zeros(total, H, D, device="cuda", dtype=torch_dtype)
    K = torch.zeros(total, Hkv, D, device="cuda", dtype=torch_dtype)
    V_nat = torch.zeros(total, Hkv, D, device="cuda", dtype=torch_dtype)
    for i, s in enumerate(seqlens):
        lo = cu_phys[i]
        Q[lo : lo + s].copy_(torch.randn(s, H, D, device="cuda", dtype=torch_dtype))
        K[lo : lo + s].copy_(torch.randn(s, Hkv, D, device="cuda", dtype=torch_dtype))
        V_nat[lo : lo + s].copy_(torch.randn(s, Hkv, D, device="cuda", dtype=torch_dtype))
    V = V_nat.permute(1, 2, 0).contiguous()  # [Hkv, D, total]
    O = torch.zeros_like(Q)  # noqa: E741
    cu_t = torch.tensor(cu_phys, dtype=torch.int32, device="cuda")
    seq_lens_t = torch.tensor(list(seqlens), dtype=torch.int32, device="cuda")
    return Q, K, V, V_nat, O, cu_t, seq_lens_t, cu_phys


def _run_varlen_flex_attn(
    monkeypatch,
    *,
    seqlens,
    H,
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
    max_seqlen = max(seqlens)
    sm_scale = 1.0 / math.sqrt(D)
    num_seqs = len(seqlens)

    Q, K, V, V_nat, O, cu_t, seq_lens_t, cu_phys = _pack_varlen_qkv(  # noqa: E741
        seqlens, H, Hkv, D, torch_dtype, seed=seed
    )

    launch = mod.compile_iluvatar_flex_attention(
        num_seqs,
        H,
        max_seqlen,
        max_seqlen,
        D,
        Hkv=Hkv,
        dtype=dtype,
        is_causal=is_causal,
        window_size=window_size,
        softcap=softcap,
        sm_scale=sm_scale,
        varlen=True,
    )
    launch(Q, K, V, O, cu_seqlens=cu_t, seq_lens=seq_lens_t)
    torch.cuda.synchronize()

    refs = []
    for i, s in enumerate(seqlens):
        lo = cu_phys[i]
        q_i = Q[lo : lo + s].transpose(0, 1).unsqueeze(0).contiguous()
        k_i = K[lo : lo + s].transpose(0, 1).unsqueeze(0).contiguous()
        v_i = V_nat[lo : lo + s].transpose(0, 1).unsqueeze(0).contiguous()
        ref_i = _reference_flex_attention_fp32(
            q_i,
            k_i,
            v_i,
            sm_scale=sm_scale,
            is_causal=is_causal,
            window_size=window_size,
            softcap=softcap,
        )
        refs.append(ref_i[0].transpose(0, 1).contiguous())
    ref = torch.cat(refs, dim=0)

    outs = []
    for i, s in enumerate(seqlens):
        lo = cu_phys[i]
        outs.append(O[lo : lo + s].float())
    out = torch.cat(outs, dim=0)
    torch.testing.assert_close(out, ref, rtol=atol_rtol, atol=atol_rtol)


@pytest.mark.parametrize(
    "seqlens,H,Hkv,D,dtype,is_causal,window_size,softcap",
    [
        ([64, 64], 4, 4, 128, "bf16", True, None, None),  # aligned MHA causal
        ([48, 80, 64], 4, 4, 128, "bf16", True, None, None),  # unaligned seqlens
        ([96, 32], 4, 2, 64, "bf16", True, None, None),  # GQA + D64
        ([64, 128], 4, 4, 128, "f16", True, None, None),  # f16 smoke
        ([80, 80], 4, 4, 128, "bf16", True, 32, None),  # SWA
        ([64, 64], 4, 4, 128, "bf16", True, None, 30.0),  # softcap
    ],
)
def test_iluvatar_flex_attention_varlen_v21(monkeypatch, seqlens, H, Hkv, D, dtype, is_causal, window_size, softcap):
    """Packed varlen self-attn matches per-seq dense concat reference."""
    _run_varlen_flex_attn(
        monkeypatch,
        seqlens=seqlens,
        H=H,
        Hkv=Hkv,
        D=D,
        dtype=dtype,
        is_causal=is_causal,
        window_size=window_size,
        softcap=softcap,
    )


def test_iluvatar_flex_attention_varlen_rejects_missing_cu(monkeypatch):
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    mod = _require_flex_attn_module()
    launch = mod.compile_iluvatar_flex_attention(2, 4, 64, 64, 128, dtype="bf16", varlen=True)
    Q = torch.zeros(128, 4, 128, device="cuda", dtype=torch.bfloat16)
    K = torch.zeros(128, 4, 128, device="cuda", dtype=torch.bfloat16)
    V = torch.zeros(4, 128, 128, device="cuda", dtype=torch.bfloat16)
    O = torch.zeros_like(Q)  # noqa: E741
    with pytest.raises(ValueError, match="cu_seqlens"):
        launch(Q, K, V, O)


def test_iluvatar_flex_attention_varlen_rejects_cross_max(monkeypatch):
    mod = _require_flex_attn_module()
    with pytest.raises(ValueError, match="Sq == Skv"):
        mod.compile_iluvatar_flex_attention(2, 4, 64, 128, 128, dtype="bf16", varlen=True)


# --- V2-2 paged KV ------------------------------------------------------------


def _build_paged_kv(
    *,
    B: int,
    Hkv: int,
    D: int,
    seq_lens_kv,
    torch_dtype,
    seed: int = 0,
    page_size: int = 64,
):
    """Build linear K/V pages + block_table; return natural and V-transposed.

    Logical cache layout for both K and V is ``[NumBlocks, page_size, Hkv, D]``.
    Kernel-facing V is per-page transposed to ``[NumBlocks, Hkv, D, page_size]``.
    """
    torch = _require_torch()
    torch.manual_seed(seed)
    seq_lens_kv = [int(s) for s in seq_lens_kv]
    assert len(seq_lens_kv) == B
    max_pages = max((s + page_size - 1) // page_size for s in seq_lens_kv)
    # Allocate enough pages for all sequences (no sharing).
    pages_per_seq = [((s + page_size - 1) // page_size) for s in seq_lens_kv]
    num_blocks = sum(pages_per_seq)
    K_pages = torch.zeros(num_blocks, page_size, Hkv, D, device="cuda", dtype=torch_dtype)
    V_pages_nat = torch.zeros_like(K_pages)
    block_table = torch.full((B, max_pages), -1, device="cuda", dtype=torch.int32)
    cursor = 0
    dense_k = []
    dense_v = []
    for b, slen in enumerate(seq_lens_kv):
        npages = pages_per_seq[b]
        ids = list(range(cursor, cursor + npages))
        cursor += npages
        for i, pid in enumerate(ids):
            block_table[b, i] = pid
            lo = i * page_size
            hi = min(lo + page_size, slen)
            n = hi - lo
            K_pages[pid, :n].copy_(torch.randn(n, Hkv, D, device="cuda", dtype=torch_dtype))
            V_pages_nat[pid, :n].copy_(torch.randn(n, Hkv, D, device="cuda", dtype=torch_dtype))
        # Gather dense [Hkv, Skv, D] then to [B,Hkv,Skv,D] style for ref.
        k_rows = []
        v_rows = []
        for i, pid in enumerate(ids):
            lo = i * page_size
            hi = min(lo + page_size, slen)
            n = hi - lo
            k_rows.append(K_pages[pid, :n])  # [n, Hkv, D]
            v_rows.append(V_pages_nat[pid, :n])
        k_cat = torch.cat(k_rows, dim=0)  # [slen, Hkv, D]
        v_cat = torch.cat(v_rows, dim=0)
        dense_k.append(k_cat.permute(1, 0, 2).contiguous())  # [Hkv, slen, D]
        dense_v.append(v_cat.permute(1, 0, 2).contiguous())
    V_pages = V_pages_nat.permute(0, 2, 3, 1).contiguous()  # [NumBlocks, Hkv, D, page]
    seq_lens_t = torch.tensor(seq_lens_kv, dtype=torch.int32, device="cuda")
    return K_pages, V_pages, V_pages_nat, block_table, seq_lens_t, dense_k, dense_v, max_pages


def _run_paged_flex_attn(
    monkeypatch,
    *,
    B: int,
    H: int,
    Sq: int,
    seq_lens_kv,
    D: int = 128,
    Hkv: int | None = None,
    dtype: str = "bf16",
    is_causal: bool = False,
    window_size: int | None = None,
    softcap: float | None = None,
    seed: int = 0,
):
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    mod = _require_flex_attn_module()

    if Hkv is None:
        Hkv = H
    torch_dtype = {"bf16": torch.bfloat16, "f16": torch.float16}[dtype]
    atol_rtol = 2e-2 if dtype == "bf16" else 1e-2
    Skv = max(seq_lens_kv)
    sm_scale = 1.0 / math.sqrt(D)
    Sq_phys = _phys_seq(Sq, mod.BLOCK_M)

    K_pages, V_pages, _V_nat, block_table, seq_lens_t, dense_k, dense_v, max_pages = _build_paged_kv(
        B=B,
        Hkv=Hkv,
        D=D,
        seq_lens_kv=seq_lens_kv,
        torch_dtype=torch_dtype,
        seed=seed,
        page_size=mod.BLOCK_N,
    )
    assert max_pages == _phys_seq(Skv, mod.BLOCK_N) // mod.BLOCK_N

    torch.manual_seed(seed + 1)
    Q = torch.zeros(B, H, Sq_phys, D, device="cuda", dtype=torch_dtype)
    Q[:, :, :Sq].copy_(torch.randn(B, H, Sq, D, device="cuda", dtype=torch_dtype))
    O = torch.zeros_like(Q)  # noqa: E741

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
        paged=True,
    )
    launch(Q, K_pages, V_pages, O, block_table=block_table, seq_lens_kv=seq_lens_t)
    torch.cuda.synchronize()

    refs = []
    for b in range(B):
        slen = seq_lens_kv[b]
        q_b = Q[b : b + 1, :, :Sq, :].contiguous()
        k_b = dense_k[b].unsqueeze(0)[:, :, :slen, :].contiguous()  # [1,Hkv,slen,D]
        v_b = dense_v[b].unsqueeze(0)[:, :, :slen, :].contiguous()
        ref_b = _reference_flex_attention_fp32(
            q_b,
            k_b,
            v_b,
            sm_scale=sm_scale,
            is_causal=is_causal,
            window_size=window_size,
            softcap=softcap,
        )
        refs.append(ref_b)
    ref = torch.cat(refs, dim=0)
    out = O[:, :, :Sq, :].float()
    torch.testing.assert_close(out, ref, rtol=atol_rtol, atol=atol_rtol)


@pytest.mark.parametrize(
    "B,H,Hkv,Sq,seq_lens_kv,D,dtype,is_causal,window_size,softcap",
    [
        # Decode Sq=1 over 128 KV tokens
        (1, 4, 4, 1, [128], 128, "bf16", True, None, None),
        # Short causal prefill, equal lengths
        (2, 4, 4, 64, [64, 64], 128, "bf16", True, None, None),
        # Cross-length: Sq=64 over Skv=128
        (1, 4, 4, 64, [128], 128, "bf16", True, None, None),
        # Variable seq_lens_kv within batch
        (2, 4, 4, 64, [80, 128], 128, "bf16", True, None, None),
        # GQA + D64
        (1, 4, 2, 64, [64], 64, "bf16", True, None, None),
        # f16 smoke
        (1, 4, 4, 64, [64], 128, "f16", True, None, None),
        # SWA
        (1, 4, 4, 64, [64], 128, "bf16", True, 32, None),
        # softcap
        (1, 4, 4, 64, [64], 128, "bf16", True, None, 30.0),
        # Non-causal
        (1, 4, 4, 64, [96], 128, "bf16", False, None, None),
    ],
)
def test_iluvatar_flex_attention_paged_v22(
    monkeypatch, B, H, Hkv, Sq, seq_lens_kv, D, dtype, is_causal, window_size, softcap
):
    """Paged KV gather matches dense flex reference on gathered K/V."""
    _run_paged_flex_attn(
        monkeypatch,
        B=B,
        H=H,
        Hkv=Hkv,
        Sq=Sq,
        seq_lens_kv=seq_lens_kv,
        D=D,
        dtype=dtype,
        is_causal=is_causal,
        window_size=window_size,
        softcap=softcap,
    )


def test_iluvatar_flex_attention_paged_rejects_varlen_combo():
    mod = _require_flex_attn_module()
    with pytest.raises(ValueError, match="mutually exclusive"):
        mod.compile_iluvatar_flex_attention(1, 4, 64, 64, 128, dtype="bf16", varlen=True, paged=True)


def test_iluvatar_flex_attention_paged_allows_causal_cross_len():
    mod = _require_flex_attn_module()
    # Should not raise (unlike dense causal).
    mod.compile_iluvatar_flex_attention(1, 4, 1, 128, 128, dtype="bf16", is_causal=True, paged=True)


def test_iluvatar_flex_attention_paged_rejects_missing_table(monkeypatch):
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    mod = _require_flex_attn_module()
    launch = mod.compile_iluvatar_flex_attention(1, 4, 64, 64, 128, dtype="bf16", is_causal=True, paged=True)
    Q = torch.zeros(1, 4, 64, 128, device="cuda", dtype=torch.bfloat16)
    K = torch.zeros(1, 64, 4, 128, device="cuda", dtype=torch.bfloat16)
    V = torch.zeros(1, 4, 128, 64, device="cuda", dtype=torch.bfloat16)
    O = torch.zeros_like(Q)  # noqa: E741
    with pytest.raises(ValueError, match="block_table"):
        launch(Q, K, V, O)


# --- V2-3 dispatcher (flydsl_flex_attn_func) ----------------------------------


def _require_flex_attn_interface():
    try:
        from kernels.attention.iluvatar import flex_attn_interface as iface
    except ModuleNotFoundError as exc:
        pytest.skip(f"flex_attn_interface not importable: {exc}")
    return iface


def test_iluvatar_flex_attn_func_dense_matches_compile(monkeypatch):
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    mod = _require_flex_attn_module()
    iface = _require_flex_attn_interface()

    B, H, Sq, Skv, D = 2, 4, 64, 64, 128
    sm_scale = 1.0 / math.sqrt(D)
    torch.manual_seed(0)
    q = torch.randn(B, Sq, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, Skv, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, Skv, H, D, device="cuda", dtype=torch.bfloat16)

    out = iface.flydsl_flex_attn_func(q, k, v, causal=True, sm_scale=sm_scale)
    torch.cuda.synchronize()

    Sq_phys = _phys_seq(Sq, mod.BLOCK_M)
    Skv_phys = _phys_seq(Skv, mod.BLOCK_N)
    Q = torch.zeros(B, H, Sq_phys, D, device="cuda", dtype=torch.bfloat16)
    K = torch.zeros(B, H, Skv_phys, D, device="cuda", dtype=torch.bfloat16)
    Vn = torch.zeros(B, H, Skv_phys, D, device="cuda", dtype=torch.bfloat16)
    Q[:, :, :Sq].copy_(q.permute(0, 2, 1, 3))
    K[:, :, :Skv].copy_(k.permute(0, 2, 1, 3))
    Vn[:, :, :Skv].copy_(v.permute(0, 2, 1, 3))
    Vt = Vn.transpose(-1, -2).contiguous()
    O = torch.zeros_like(Q)  # noqa: E741
    launch = mod.compile_iluvatar_flex_attention(B, H, Sq, Skv, D, dtype="bf16", is_causal=True, sm_scale=sm_scale)
    launch(Q, K, Vt, O)
    torch.cuda.synchronize()
    ref = O[:, :, :Sq].permute(0, 2, 1, 3).contiguous()
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_iluvatar_flex_attn_func_varlen_matches_compile(monkeypatch):
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    mod = _require_flex_attn_module()
    iface = _require_flex_attn_interface()

    seqlens = [64, 64]
    H, Hkv, D = 4, 4, 128
    sm_scale = 1.0 / math.sqrt(D)
    Q, K, V_tn, V_nat, _O_unused, cu_t, seq_lens_t, _ = _pack_varlen_qkv(seqlens, H, Hkv, D, torch.bfloat16, seed=1)
    out = iface.flydsl_flex_attn_func(
        Q,
        K,
        V_nat,
        causal=True,
        sm_scale=sm_scale,
        cu_seqlens=cu_t,
        seq_lens=seq_lens_t,
    )
    torch.cuda.synchronize()

    O_ref = torch.zeros_like(Q)
    launch = mod.compile_iluvatar_flex_attention(
        len(seqlens),
        H,
        max(seqlens),
        max(seqlens),
        D,
        Hkv=Hkv,
        dtype="bf16",
        is_causal=True,
        sm_scale=sm_scale,
        varlen=True,
    )
    launch(Q, K, V_tn, O_ref, cu_seqlens=cu_t, seq_lens=seq_lens_t)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, O_ref, rtol=0, atol=0)


def test_iluvatar_flex_attn_func_paged_matches_compile(monkeypatch):
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    mod = _require_flex_attn_module()
    iface = _require_flex_attn_interface()

    B, H, Hkv, Sq, D = 1, 4, 4, 64, 128
    seq_lens_kv = [64]
    sm_scale = 1.0 / math.sqrt(D)
    K_pages, V_pages, V_nat, block_table, seq_lens_t, _, _, _ = _build_paged_kv(
        B=B,
        Hkv=Hkv,
        D=D,
        seq_lens_kv=seq_lens_kv,
        torch_dtype=torch.bfloat16,
        seed=2,
        page_size=mod.BLOCK_N,
    )
    torch.manual_seed(3)
    q = torch.randn(B, Sq, H, D, device="cuda", dtype=torch.bfloat16)
    out = iface.flydsl_flex_attn_func(
        q,
        K_pages,
        V_nat,
        causal=True,
        sm_scale=sm_scale,
        block_table=block_table,
        seq_lens_kv=seq_lens_t,
    )
    torch.cuda.synchronize()

    Sq_phys = _phys_seq(Sq, mod.BLOCK_M)
    Q = torch.zeros(B, H, Sq_phys, D, device="cuda", dtype=torch.bfloat16)
    Q[:, :, :Sq].copy_(q.permute(0, 2, 1, 3))
    O = torch.zeros_like(Q)  # noqa: E741
    launch = mod.compile_iluvatar_flex_attention(
        B,
        H,
        Sq,
        max(seq_lens_kv),
        D,
        Hkv=Hkv,
        dtype="bf16",
        is_causal=True,
        sm_scale=sm_scale,
        paged=True,
    )
    launch(Q, K_pages, V_pages, O, block_table=block_table, seq_lens_kv=seq_lens_t)
    torch.cuda.synchronize()
    ref = O[:, :, :Sq].permute(0, 2, 1, 3).contiguous()
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_iluvatar_flex_attn_func_rejects_mode_conflict(monkeypatch):
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    iface = _require_flex_attn_interface()
    q = torch.zeros(1, 64, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    cu = torch.zeros(2, dtype=torch.int32, device="cuda")
    sl = torch.zeros(1, dtype=torch.int32, device="cuda")
    bt = torch.zeros(1, 1, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="mutually exclusive"):
        iface.flydsl_flex_attn_func(q, k, v, cu_seqlens=cu, seq_lens=sl, block_table=bt, seq_lens_kv=sl)


def test_iluvatar_flex_attn_func_rejects_partial_paged(monkeypatch):
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    iface = _require_flex_attn_interface()
    q = torch.zeros(1, 64, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros(1, 64, 4, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    bt = torch.zeros(1, 1, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="block_table"):
        iface.flydsl_flex_attn_func(q, k, v, block_table=bt)


# --- V2-4 alibi / score_bias (dense) ------------------------------------------


def _run_dense_alibi_or_bias(
    monkeypatch,
    *,
    mode: str,
    is_causal=True,
    window_size=None,
    softcap=None,
    B=1,
    H=4,
    Sq=64,
    Skv=64,
    D=128,
    dtype="bf16",
):
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    mod = _require_flex_attn_module()
    torch_dtype = {"bf16": torch.bfloat16, "f16": torch.float16}[dtype]
    atol_rtol = 2e-2 if dtype == "bf16" else 1e-2
    sm_scale = 1.0 / math.sqrt(D)
    Sq_phys = _phys_seq(Sq, mod.BLOCK_M)
    Skv_phys = _phys_seq(Skv, mod.BLOCK_N)

    torch.manual_seed(7)
    Q = torch.zeros(B, H, Sq_phys, D, device="cuda", dtype=torch_dtype)
    K = torch.zeros(B, H, Skv_phys, D, device="cuda", dtype=torch_dtype)
    Vn = torch.zeros(B, H, Skv_phys, D, device="cuda", dtype=torch_dtype)
    Q[:, :, :Sq].copy_(torch.randn(B, H, Sq, D, device="cuda", dtype=torch_dtype))
    K[:, :, :Skv].copy_(torch.randn(B, H, Skv, D, device="cuda", dtype=torch_dtype))
    Vn[:, :, :Skv].copy_(torch.randn(B, H, Skv, D, device="cuda", dtype=torch_dtype))
    Vt = Vn.transpose(-1, -2).contiguous()
    O = torch.zeros_like(Q)  # noqa: E741

    alibi = None
    sb = None
    has_alibi = mode == "alibi"
    has_score_bias = mode == "score_bias"
    if has_alibi:
        alibi = torch.rand(H, device="cuda", dtype=torch.float32) * 0.5 + 0.1
    if has_score_bias:
        sb = torch.zeros(B, H, Sq_phys, Skv_phys, device="cuda", dtype=torch.float32)
        sb[:, :, :Sq, :Skv].copy_(torch.randn(B, H, Sq, Skv, device="cuda", dtype=torch.float32) * 0.1)

    launch = mod.compile_iluvatar_flex_attention(
        B,
        H,
        Sq,
        Skv,
        D,
        dtype=dtype,
        is_causal=is_causal,
        window_size=window_size,
        softcap=softcap,
        sm_scale=sm_scale,
        has_alibi=has_alibi,
        has_score_bias=has_score_bias,
    )
    launch(Q, K, Vt, O, alibi_slopes=alibi, score_bias=sb)
    torch.cuda.synchronize()

    ref = _reference_flex_attention_fp32(
        Q[:, :, :Sq],
        K[:, :, :Skv],
        Vn[:, :, :Skv],
        sm_scale=sm_scale,
        is_causal=is_causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi,
        score_bias=None if sb is None else sb[:, :, :Sq, :Skv],
    )
    torch.testing.assert_close(O[:, :, :Sq].float(), ref, rtol=atol_rtol, atol=atol_rtol)


@pytest.mark.parametrize(
    "mode,is_causal,window_size,softcap",
    [
        ("alibi", True, None, None),
        ("alibi", True, 32, None),
        ("alibi", True, None, 30.0),
        ("score_bias", True, None, None),
        ("score_bias", False, None, None),
    ],
)
def test_iluvatar_flex_attention_alibi_score_bias_v24(monkeypatch, mode, is_causal, window_size, softcap):
    _run_dense_alibi_or_bias(
        monkeypatch,
        mode=mode,
        is_causal=is_causal,
        window_size=window_size,
        softcap=softcap,
    )


def test_iluvatar_flex_attention_rejects_alibi_and_bias(monkeypatch):
    mod = _require_flex_attn_module()
    with pytest.raises(ValueError, match="mutually exclusive"):
        mod.compile_iluvatar_flex_attention(1, 4, 64, 64, 128, dtype="bf16", has_alibi=True, has_score_bias=True)


def test_iluvatar_flex_attention_rejects_alibi_with_varlen(monkeypatch):
    mod = _require_flex_attn_module()
    with pytest.raises(ValueError, match="dense-only"):
        mod.compile_iluvatar_flex_attention(2, 4, 64, 64, 128, dtype="bf16", varlen=True, has_alibi=True)


def test_iluvatar_flex_attn_func_alibi_matches_compile(monkeypatch):
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    iface = _require_flex_attn_interface()
    B, H, Sq, Skv, D = 1, 4, 64, 64, 128
    sm_scale = 1.0 / math.sqrt(D)
    torch.manual_seed(11)
    q = torch.randn(B, Sq, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, Skv, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, Skv, H, D, device="cuda", dtype=torch.bfloat16)
    slopes = torch.rand(H, device="cuda", dtype=torch.float32) * 0.4 + 0.05
    out = iface.flydsl_flex_attn_func(q, k, v, causal=True, sm_scale=sm_scale, alibi_slopes=slopes)
    torch.cuda.synchronize()
    ref = _reference_flex_attention_fp32(
        q.permute(0, 2, 1, 3).contiguous(),
        k.permute(0, 2, 1, 3).contiguous(),
        v.permute(0, 2, 1, 3).contiguous(),
        sm_scale=sm_scale,
        is_causal=True,
        alibi_slopes=slopes,
    )
    torch.testing.assert_close(out.permute(0, 2, 1, 3).float(), ref, rtol=2e-2, atol=2e-2)


# --- V2-5 larger head dim (D=256) ---------------------------------------------


def test_iluvatar_flex_attention_kv_stages_smem_choice():
    """Host-only: D=256 must drop to 1-stage; D=64/128 keep 2-stage."""
    from kernels.gemm.iluvatar.mr.common import DEFAULT_SMEM_CAP_BYTES

    mod = _require_flex_attn_module()
    assert mod._choose_kv_stages(64) == 2
    assert mod._choose_kv_stages(128) == 2
    assert mod._choose_kv_stages(256) == 1
    assert mod._flex_attn_smem_bytes(256, 2) > DEFAULT_SMEM_CAP_BYTES
    assert mod._flex_attn_smem_bytes(256, 1) <= DEFAULT_SMEM_CAP_BYTES


@pytest.mark.parametrize("dtype", ["bf16", "f16"])
def test_iluvatar_flex_attention_d256_dense_causal(monkeypatch, dtype):
    out, ref, tol = _run_flex_attn(
        monkeypatch,
        B=1,
        H=4,
        Sq=128,
        Skv=128,
        D=256,
        dtype=dtype,
        is_causal=True,
    )
    torch = _require_torch()
    torch.testing.assert_close(out, ref, rtol=tol, atol=tol)


@pytest.mark.parametrize(
    "is_causal,window_size,softcap,Hkv,mode",
    [
        (True, 32, None, 4, None),  # SWA
        (True, None, 30.0, 4, None),  # softcap
        (True, None, None, 2, None),  # GQA
        (True, None, None, 4, "alibi"),
    ],
)
def test_iluvatar_flex_attention_d256_dense_variant_smoke(monkeypatch, is_causal, window_size, softcap, Hkv, mode):
    if mode == "alibi":
        _run_dense_alibi_or_bias(
            monkeypatch,
            mode="alibi",
            is_causal=is_causal,
            window_size=window_size,
            softcap=softcap,
            B=1,
            H=4,
            Sq=64,
            Skv=64,
            D=256,
            dtype="bf16",
        )
        return
    out, ref, tol = _run_flex_attn(
        monkeypatch,
        B=1,
        H=4,
        Hkv=Hkv,
        Sq=64,
        Skv=64,
        D=256,
        dtype="bf16",
        is_causal=is_causal,
        window_size=window_size,
        softcap=softcap,
    )
    torch = _require_torch()
    torch.testing.assert_close(out, ref, rtol=tol, atol=tol)


def test_iluvatar_flex_attention_d256_varlen_smoke(monkeypatch):
    _run_varlen_flex_attn(
        monkeypatch,
        seqlens=[64, 96],
        H=4,
        D=256,
        dtype="bf16",
        is_causal=True,
    )


def test_iluvatar_flex_attention_d256_paged_smoke(monkeypatch):
    _run_paged_flex_attn(
        monkeypatch,
        B=1,
        H=4,
        Sq=64,
        seq_lens_kv=[128],
        D=256,
        dtype="bf16",
        is_causal=True,
    )


def test_iluvatar_flex_attn_func_d256_dense_matches_compile(monkeypatch):
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    mod = _require_flex_attn_module()
    iface = _require_flex_attn_interface()

    B, H, Sq, Skv, D = 1, 4, 64, 64, 256
    sm_scale = 1.0 / math.sqrt(D)
    torch.manual_seed(3)
    q = torch.randn(B, Sq, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, Skv, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, Skv, H, D, device="cuda", dtype=torch.bfloat16)

    out = iface.flydsl_flex_attn_func(q, k, v, causal=True, sm_scale=sm_scale)
    torch.cuda.synchronize()

    Sq_phys = _phys_seq(Sq, mod.BLOCK_M)
    Skv_phys = _phys_seq(Skv, mod.BLOCK_N)
    Q = torch.zeros(B, H, Sq_phys, D, device="cuda", dtype=torch.bfloat16)
    K = torch.zeros(B, H, Skv_phys, D, device="cuda", dtype=torch.bfloat16)
    Vn = torch.zeros(B, H, Skv_phys, D, device="cuda", dtype=torch.bfloat16)
    Q[:, :, :Sq].copy_(q.permute(0, 2, 1, 3))
    K[:, :, :Skv].copy_(k.permute(0, 2, 1, 3))
    Vn[:, :, :Skv].copy_(v.permute(0, 2, 1, 3))
    Vt = Vn.transpose(-1, -2).contiguous()
    O = torch.zeros_like(Q)  # noqa: E741
    launch = mod.compile_iluvatar_flex_attention(B, H, Sq, Skv, D, dtype="bf16", is_causal=True, sm_scale=sm_scale)
    launch(Q, K, Vt, O)
    torch.cuda.synchronize()
    ref = O[:, :, :Sq].permute(0, 2, 1, 3).contiguous()
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


# --- V2-6 optional tile_config / dense autotune helper ------------------------


def test_iluvatar_flex_attention_rejects_invalid_tile_config():
    mod = _require_flex_attn_module()
    with pytest.raises(ValueError, match=r"block_m/block_n must be in"):
        mod.compile_iluvatar_flex_attention(1, 4, 64, 64, 128, dtype="bf16", tile_config={"block_m": 48})
    with pytest.raises(ValueError, match=r"unexpected keys"):
        mod.compile_iluvatar_flex_attention(1, 4, 64, 64, 128, dtype="bf16", tile_config={"block_m": 64, "foo": 1})
    with pytest.raises(ValueError, match=r"paged path requires block_n"):
        mod.compile_iluvatar_flex_attention(
            1, 4, 64, 64, 128, dtype="bf16", paged=True, tile_config={"block_m": 32, "block_n": 32}
        )


@pytest.mark.parametrize(
    "block_m,block_n",
    [
        (32, 32),
        (32, 64),
        (64, 32),
        (64, 64),
    ],
)
def test_iluvatar_flex_attention_tile_config_dense_smoke(monkeypatch, block_m, block_n):
    """Whitelist tiles vs fp32 reference (default 64x64 included)."""
    out, ref, tol = _run_flex_attn(
        monkeypatch,
        B=1,
        H=4,
        Sq=64,
        Skv=64,
        D=128,
        dtype="bf16",
        is_causal=True,
        tile_config={"block_m": block_m, "block_n": block_n},
        seed=block_m * 100 + block_n,
    )
    torch = _require_torch()
    torch.testing.assert_close(out, ref, rtol=tol, atol=tol)


def test_iluvatar_flex_attn_func_tile_config_matches_compile(monkeypatch):
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    mod = _require_flex_attn_module()
    iface = _require_flex_attn_interface()

    B, H, Sq, Skv, D = 1, 4, 64, 64, 128
    tile = {"block_m": 32, "block_n": 32}
    sm_scale = 1.0 / math.sqrt(D)
    torch.manual_seed(11)
    q = torch.randn(B, Sq, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, Skv, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, Skv, H, D, device="cuda", dtype=torch.bfloat16)

    out = iface.flydsl_flex_attn_func(q, k, v, causal=True, sm_scale=sm_scale, tile_config=tile)
    torch.cuda.synchronize()

    block_m, block_n = tile["block_m"], tile["block_n"]
    Sq_phys = _phys_seq(Sq, block_m)
    Skv_phys = _phys_seq(Skv, block_n)
    Q = torch.zeros(B, H, Sq_phys, D, device="cuda", dtype=torch.bfloat16)
    K = torch.zeros(B, H, Skv_phys, D, device="cuda", dtype=torch.bfloat16)
    Vn = torch.zeros(B, H, Skv_phys, D, device="cuda", dtype=torch.bfloat16)
    Q[:, :, :Sq].copy_(q.permute(0, 2, 1, 3))
    K[:, :, :Skv].copy_(k.permute(0, 2, 1, 3))
    Vn[:, :, :Skv].copy_(v.permute(0, 2, 1, 3))
    Vt = Vn.transpose(-1, -2).contiguous()
    O = torch.zeros_like(Q)  # noqa: E741
    launch = mod.compile_iluvatar_flex_attention(
        B, H, Sq, Skv, D, dtype="bf16", is_causal=True, sm_scale=sm_scale, tile_config=tile
    )
    launch(Q, K, Vt, O)
    torch.cuda.synchronize()
    ref = O[:, :, :Sq].permute(0, 2, 1, 3).contiguous()
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_iluvatar_flex_attention_autotune_tile_helper(monkeypatch):
    """Opt-in helper returns a whitelist config; best is not worse than baseline."""
    torch = _require_torch()
    # Keep disk cache off: CI runners may have an unusable HOME (e.g. "/"), and
    # enabling FLYDSL_RUNTIME_ENABLE_CACHE then tries to mkdir "/.flydsl".
    _configure_iluvatar_env(monkeypatch, enable_cache=False)
    iface = _require_flex_attn_interface()
    from flydsl.autotune import do_bench

    B, H, Sq, Skv, D = 1, 4, 128, 128, 128
    sm_scale = 1.0 / math.sqrt(D)
    torch.manual_seed(21)
    q = torch.randn(B, Sq, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, Skv, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, Skv, H, D, device="cuda", dtype=torch.bfloat16)

    configs = [
        {"block_m": 32, "block_n": 32},
        {"block_m": 64, "block_n": 64},
    ]
    best = iface.autotune_iluvatar_flex_attention_tile(
        q,
        k,
        v,
        causal=True,
        sm_scale=sm_scale,
        configs=configs,
        warmup=2,
        rep=5,
    )
    assert best["block_m"] in (32, 64) and best["block_n"] in (32, 64)

    # Re-measure baseline vs best; by construction best_ms <= baseline_ms.
    def _bench(cfg):
        def run():
            iface.flydsl_flex_attn_func(q, k, v, causal=True, sm_scale=sm_scale, tile_config=cfg)

        run()
        torch.cuda.synchronize()
        return float(do_bench(run, warmup=1, rep=5))

    baseline_ms = _bench({"block_m": 64, "block_n": 64})
    best_ms = _bench(best)
    assert best_ms <= baseline_ms * 1.05 + 1e-6

    out = iface.flydsl_flex_attn_func(q, k, v, causal=True, sm_scale=sm_scale, tile_config=best)
    ref = iface.flydsl_flex_attn_func(
        q, k, v, causal=True, sm_scale=sm_scale, tile_config={"block_m": 64, "block_n": 64}
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


# --- V3-2 dense score_mod -----------------------------------------------------


def test_iluvatar_flex_attention_rejects_raw_score_mod_callable():
    mod = _require_flex_attn_module()

    def raw(score, batch, head, q_idx, kv_idx):
        return score

    with pytest.raises(ValueError, match=r"TracedScoreMod"):
        mod.compile_iluvatar_flex_attention(1, 4, 64, 64, 128, dtype="bf16", score_mod=raw)


def test_iluvatar_flex_attention_rejects_score_mod_varlen_paged():
    mod = _require_flex_attn_module()

    @fx.trace_score_mod
    def identity(score, batch, head, q_idx, kv_idx):
        return score

    with pytest.raises(ValueError, match=r"dense-only"):
        mod.compile_iluvatar_flex_attention(
            1, 4, 64, 64, 128, dtype="bf16", varlen=True, score_mod=identity
        )
    with pytest.raises(ValueError, match=r"dense-only"):
        mod.compile_iluvatar_flex_attention(
            1, 4, 64, 64, 128, dtype="bf16", paged=True, score_mod=identity
        )


def test_iluvatar_flex_attention_dispatcher_rejects_score_mod(monkeypatch):
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    from kernels.attention.iluvatar import flex_attn_interface as iface

    @fx.trace_score_mod
    def identity(score, batch, head, q_idx, kv_idx):
        return score

    q = torch.randn(1, 64, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 64, 4, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 64, 4, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"score_mod"):
        iface.flydsl_flex_attn_func(q, k, v, score_mod=identity)


def _run_flex_attn_score_mod(monkeypatch, *, score_mod, softcap=None, is_causal=True, dtype="bf16"):
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    mod = _require_flex_attn_module()

    B, H, Sq, Skv, D = 1, 4, 64, 64, 128
    torch_dtype = {"bf16": torch.bfloat16, "f16": torch.float16}[dtype]
    tol = 2e-2 if dtype == "bf16" else 1e-2
    sm_scale = 1.0 / math.sqrt(D)
    Sq_phys = _phys_seq(Sq, 64)
    Skv_phys = _phys_seq(Skv, 64)

    torch.manual_seed(0)
    Q = torch.zeros(B, H, Sq_phys, D, device="cuda", dtype=torch_dtype)
    K = torch.zeros(B, H, Skv_phys, D, device="cuda", dtype=torch_dtype)
    V_natural = torch.zeros(B, H, Skv_phys, D, device="cuda", dtype=torch_dtype)
    Q[:, :, :Sq, :].copy_(torch.randn(B, H, Sq, D, device="cuda", dtype=torch_dtype))
    K[:, :, :Skv, :].copy_(torch.randn(B, H, Skv, D, device="cuda", dtype=torch_dtype))
    V_natural[:, :, :Skv, :].copy_(torch.randn(B, H, Skv, D, device="cuda", dtype=torch_dtype))
    V_transposed = V_natural.transpose(-1, -2).contiguous()
    O = torch.zeros(B, H, Sq_phys, D, device="cuda", dtype=torch_dtype)  # noqa: E741

    launch = mod.compile_iluvatar_flex_attention(
        B,
        H,
        Sq,
        Skv,
        D,
        dtype=dtype,
        is_causal=is_causal,
        softcap=softcap,
        sm_scale=sm_scale,
        score_mod=score_mod,
    )
    launch(Q, K, V_transposed, O)
    torch.cuda.synchronize()

    ref = _reference_flex_attention_fp32(
        Q[:, :, :Sq, :],
        K[:, :, :Skv, :],
        V_natural[:, :, :Skv, :],
        sm_scale=sm_scale,
        is_causal=is_causal,
        softcap=softcap,
        score_mod=score_mod,
    ).to(torch_dtype)
    return O[:, :, :Sq, :], ref, tol


def test_iluvatar_flex_attention_score_mod_alibi_like(monkeypatch):
    slope = 0.25

    @fx.trace_score_mod
    def alibi_like(score, batch, head, q_idx, kv_idx):
        return score - slope * (q_idx - kv_idx)

    out, ref, tol = _run_flex_attn_score_mod(monkeypatch, score_mod=alibi_like)
    torch = _require_torch()
    torch.testing.assert_close(out, ref, rtol=tol, atol=tol)


def test_iluvatar_flex_attention_score_mod_where_relu(monkeypatch):
    @fx.trace_score_mod
    def relu_scores(score, batch, head, q_idx, kv_idx):
        return fx.where(score > 0.0, score, 0.0)

    out, ref, tol = _run_flex_attn_score_mod(monkeypatch, score_mod=relu_scores, is_causal=False)
    torch = _require_torch()
    torch.testing.assert_close(out, ref, rtol=tol, atol=tol)


def test_iluvatar_flex_attention_score_mod_with_softcap(monkeypatch):
    slope = 0.1

    @fx.trace_score_mod
    def alibi_like(score, batch, head, q_idx, kv_idx):
        return score - slope * (q_idx - kv_idx)

    out, ref, tol = _run_flex_attn_score_mod(
        monkeypatch, score_mod=alibi_like, softcap=30.0, is_causal=True
    )
    torch = _require_torch()
    torch.testing.assert_close(out, ref, rtol=tol, atol=tol)


# --- V3-3 BlockMask / mask_mod -------------------------------------------------


def _element_visible_ref(q, kv, *, Sq, Skv, is_causal, window_size, mask_mod):
    if q < 0 or kv < 0 or q >= Sq or kv >= Skv:
        return False
    if is_causal and kv > q:
        return False
    if window_size is not None and (q - kv) > int(window_size):
        return False
    if mask_mod is not None and not bool(mask_mod.eval_host(0, 0, q, kv)):
        return False
    return True


def test_create_block_mask_matches_dense_visibility():
    torch = _require_torch()
    mod = _require_flex_attn_module()

    @fx.trace_mask_mod
    def near_band(batch, head, q_idx, kv_idx):
        # |q-kv| <= 64 without FloorDiv (not in V3-1 whitelist).
        return fx.where((q_idx - kv_idx) <= 64, (kv_idx - q_idx) <= 64, False)

    B, H, Sq, Skv = 1, 2, 192, 192
    block_m = block_n = 64
    bm = mod.create_block_mask(
        near_band,
        B,
        H,
        Sq,
        Skv,
        block_m=block_m,
        block_n=block_n,
        is_causal=True,
        device="cpu",
    )
    assert bm.num_q_tiles == 3 and bm.num_kv_tiles == 3
    assert bm.sparsity() > 0.0

    for qi in range(bm.num_q_tiles):
        n = int(bm.kv_num_blocks[qi].item())
        kept = {int(bm.kv_indices[qi, j].item()) for j in range(n)}
        for kj in range(bm.num_kv_tiles):
            q0, q1 = qi * block_m, min((qi + 1) * block_m, _phys_seq(Sq, block_m))
            k0, k1 = kj * block_n, min((kj + 1) * block_n, _phys_seq(Skv, block_n))
            any_vis = False
            all_vis = True
            for q in range(q0, q1):
                for kv in range(k0, k1):
                    vis = _element_visible_ref(
                        q,
                        kv,
                        Sq=Sq,
                        Skv=Skv,
                        is_causal=True,
                        window_size=None,
                        mask_mod=near_band,
                    )
                    any_vis = any_vis or vis
                    all_vis = all_vis and vis
            if any_vis:
                assert kj in kept
                slot = int((bm.kv_indices[qi] == kj).nonzero(as_tuple=False)[0].item())
                expect_full = bool(all_vis) and (q1 <= Sq) and (k1 <= Skv)
                assert int(bm.kv_is_full[qi, slot].item()) == (1 if expect_full else 0)
            else:
                assert kj not in kept


def test_iluvatar_flex_attention_rejects_mask_mod_block_mask_varlen_paged():
    mod = _require_flex_attn_module()

    @fx.trace_mask_mod
    def always(batch, head, q_idx, kv_idx):
        return True

    with pytest.raises(ValueError, match=r"dense-only"):
        mod.compile_iluvatar_flex_attention(
            1, 4, 64, 64, 128, dtype="bf16", varlen=True, mask_mod=always
        )
    with pytest.raises(ValueError, match=r"dense-only"):
        mod.compile_iluvatar_flex_attention(
            1, 4, 64, 64, 128, dtype="bf16", paged=True, has_block_mask=True
        )


def test_iluvatar_flex_attention_dispatcher_rejects_block_mask_mask_mod(monkeypatch):
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    from kernels.attention.iluvatar import create_block_mask
    from kernels.attention.iluvatar import flex_attn_interface as iface

    @fx.trace_mask_mod
    def always(batch, head, q_idx, kv_idx):
        return True

    q = torch.randn(1, 64, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 64, 4, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 64, 4, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"mask_mod|block_mask"):
        iface.flydsl_flex_attn_func(q, k, v, mask_mod=always)
    block_mask = create_block_mask(
        None, 1, 4, 64, 64, block_m=64, block_n=64, is_causal=True, device=q.device
    )
    with pytest.raises(ValueError, match=r"block_mask"):
        iface.flydsl_flex_attn_func(q, k, v, block_mask=block_mask)


def _run_flex_attn_block_mask(
    monkeypatch,
    *,
    mask_mod,
    is_causal=True,
    score_mod=None,
    dtype="bf16",
    Sq=128,
    Skv=128,
):
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    mod = _require_flex_attn_module()

    B, H, D = 1, 4, 128
    torch_dtype = {"bf16": torch.bfloat16, "f16": torch.float16}[dtype]
    tol = 2e-2 if dtype == "bf16" else 1e-2
    sm_scale = 1.0 / math.sqrt(D)
    block_m = block_n = 64
    Sq_phys = _phys_seq(Sq, block_m)
    Skv_phys = _phys_seq(Skv, block_n)

    torch.manual_seed(1)
    Q = torch.zeros(B, H, Sq_phys, D, device="cuda", dtype=torch_dtype)
    K = torch.zeros(B, H, Skv_phys, D, device="cuda", dtype=torch_dtype)
    V_natural = torch.zeros(B, H, Skv_phys, D, device="cuda", dtype=torch_dtype)
    Q[:, :, :Sq, :].copy_(torch.randn(B, H, Sq, D, device="cuda", dtype=torch_dtype))
    K[:, :, :Skv, :].copy_(torch.randn(B, H, Skv, D, device="cuda", dtype=torch_dtype))
    V_natural[:, :, :Skv, :].copy_(torch.randn(B, H, Skv, D, device="cuda", dtype=torch_dtype))
    V_transposed = V_natural.transpose(-1, -2).contiguous()
    O_sparse = torch.zeros(B, H, Sq_phys, D, device="cuda", dtype=torch_dtype)
    O_dense = torch.zeros(B, H, Sq_phys, D, device="cuda", dtype=torch_dtype)

    block_mask = mod.create_block_mask(
        mask_mod,
        B,
        H,
        Sq,
        Skv,
        block_m=block_m,
        block_n=block_n,
        is_causal=is_causal,
        device=Q.device,
    )

    launch_sparse = mod.compile_iluvatar_flex_attention(
        B,
        H,
        Sq,
        Skv,
        D,
        dtype=dtype,
        is_causal=is_causal,
        sm_scale=sm_scale,
        score_mod=score_mod,
        mask_mod=mask_mod,
        has_block_mask=True,
        tile_config={"block_m": block_m, "block_n": block_n},
    )
    launch_dense = mod.compile_iluvatar_flex_attention(
        B,
        H,
        Sq,
        Skv,
        D,
        dtype=dtype,
        is_causal=is_causal,
        sm_scale=sm_scale,
        score_mod=score_mod,
        mask_mod=mask_mod,
        has_block_mask=False,
        tile_config={"block_m": block_m, "block_n": block_n},
    )
    launch_sparse(Q, K, V_transposed, O_sparse, block_mask=block_mask)
    launch_dense(Q, K, V_transposed, O_dense)
    torch.cuda.synchronize()

    ref = _reference_flex_attention_fp32(
        Q[:, :, :Sq, :],
        K[:, :, :Skv, :],
        V_natural[:, :, :Skv, :],
        sm_scale=sm_scale,
        is_causal=is_causal,
        score_mod=score_mod,
        mask_mod=mask_mod,
    )
    return O_sparse[:, :, :Sq, :], O_dense[:, :, :Sq, :], ref, tol, block_mask


def test_iluvatar_flex_attention_block_mask_matches_dense(monkeypatch):
    @fx.trace_mask_mod
    def near_band(batch, head, q_idx, kv_idx):
        return fx.where((q_idx - kv_idx) <= 64, (kv_idx - q_idx) <= 64, False)

    out_s, out_d, ref, tol, bm = _run_flex_attn_block_mask(monkeypatch, mask_mod=near_band)
    assert bm.sparsity() > 0.2
    torch = _require_torch()
    torch.testing.assert_close(out_s, out_d, rtol=tol, atol=tol)
    torch.testing.assert_close(out_s.float(), ref, rtol=tol, atol=tol)


def test_iluvatar_flex_attention_score_mod_with_block_mask(monkeypatch):
    slope = 0.1

    @fx.trace_mask_mod
    def first_two_kv_tiles(batch, head, q_idx, kv_idx):
        return kv_idx < 128

    @fx.trace_score_mod
    def alibi_like(score, batch, head, q_idx, kv_idx):
        return score - slope * (q_idx - kv_idx)

    out_s, out_d, ref, tol, _ = _run_flex_attn_block_mask(
        monkeypatch,
        mask_mod=first_two_kv_tiles,
        score_mod=alibi_like,
        is_causal=False,
        Sq=128,
        Skv=192,
    )
    torch = _require_torch()
    torch.testing.assert_close(out_s, out_d, rtol=tol, atol=tol)
    torch.testing.assert_close(out_s.float(), ref, rtol=tol, atol=tol)


def test_iluvatar_flex_attention_block_mask_sparse_faster(monkeypatch):
    """Loose latency check: high-sparsity BlockMask should beat dense KV loop."""
    _require_perf_enabled()
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    mod = _require_flex_attn_module()

    @fx.trace_mask_mod
    def first_tile_only(batch, head, q_idx, kv_idx):
        return kv_idx < 64

    B, H, Sq, Skv, D = 1, 8, 512, 512, 128
    dtype = "bf16"
    torch_dtype = torch.bfloat16
    sm_scale = 1.0 / math.sqrt(D)
    block_m = block_n = 64
    Sq_phys = _phys_seq(Sq, block_m)
    Skv_phys = _phys_seq(Skv, block_n)

    torch.manual_seed(2)
    Q = torch.randn(B, H, Sq_phys, D, device="cuda", dtype=torch_dtype)
    K = torch.randn(B, H, Skv_phys, D, device="cuda", dtype=torch_dtype)
    V = torch.randn(B, H, Skv_phys, D, device="cuda", dtype=torch_dtype).transpose(-1, -2).contiguous()
    O_s = torch.empty_like(Q)
    O_d = torch.empty_like(Q)

    bm = mod.create_block_mask(
        first_tile_only,
        B,
        H,
        Sq,
        Skv,
        block_m=block_m,
        block_n=block_n,
        is_causal=False,
        device=Q.device,
    )
    assert bm.sparsity() > 0.8

    launch_s = mod.compile_iluvatar_flex_attention(
        B,
        H,
        Sq,
        Skv,
        D,
        dtype=dtype,
        is_causal=False,
        sm_scale=sm_scale,
        mask_mod=first_tile_only,
        has_block_mask=True,
    )
    launch_d = mod.compile_iluvatar_flex_attention(
        B,
        H,
        Sq,
        Skv,
        D,
        dtype=dtype,
        is_causal=False,
        sm_scale=sm_scale,
        mask_mod=first_tile_only,
        has_block_mask=False,
    )

    def run_s():
        launch_s(Q, K, V, O_s, block_mask=bm)

    def run_d():
        launch_d(Q, K, V, O_d)

    us_s = _bench_gpu_us(run_s, warmup=5, iters=20)
    us_d = _bench_gpu_us(run_d, warmup=5, iters=20)
    # Loose: sparse must be at least 10% faster (skip-heavy EMPTY tiles).
    assert us_s < us_d * 0.9, f"sparse={us_s:.1f}us dense={us_d:.1f}us sparsity={bm.sparsity():.3f}"
