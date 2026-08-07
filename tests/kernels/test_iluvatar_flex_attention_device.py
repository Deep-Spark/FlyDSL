# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar flex-attention device tests.

Covers:
* QK^T-only helper vs ``torch.matmul``.
* Non-causal and causal fused forward vs scaled dot-product attention.
* Softcap and sliding-window combinations vs a hand-written fp32 reference.
* f16/bf16, D=64/128, MHA/GQA, cross-attention, and physical-padding tails.
* Packed varlen (``cu_seqlens``) self-attn vs per-seq dense concat.
* Paged KV (``block_table``) vs gather-to-dense flex reference.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

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
    mask = torch.zeros((Sq, Skv), device=S.device, dtype=torch.bool)
    if is_causal:
        # Align q=0 with kv=(Skv-Sq) so decode (Sq=1) and cross-length prefill
        # match the paged kernel's delta = seqlen_kv - Sq.
        delta = Skv - Sq
        mask = mask | (kv_idx > q_idx + delta)
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
    from kernels.attention.iluvatar import compile_iluvatar_flex_attention as exported
    from kernels.attention.iluvatar import flydsl_flex_attn_func as exported_func

    mod = _require_flex_attn_module()
    assert exported is mod.compile_iluvatar_flex_attention
    assert exported_func is not None
    from kernels.attention.iluvatar import flex_attn_interface as iface

    assert exported_func is iface.flydsl_flex_attn_func


# --- Perf (opt-in; consumed by perf-daily-iluvatar) ----------------------------


def test_iluvatar_flex_attention_perf(monkeypatch):
    """Large-shape latency probe for daily trend (plan §3.3, six cases)."""
    _require_perf_enabled()
    torch = _require_torch()
    # The kv loop is unrolled at trace time, so JIT compilation for these shapes
    # costs minutes and grows linearly with Skv. Reuse the disk cache so a daily
    # run pays it once instead of per invocation.
    _configure_iluvatar_env(monkeypatch, enable_cache=True)
    mod = _require_flex_attn_module()

    # warmup must stay >= 1: the first launch triggers JIT compilation, which
    # would otherwise dominate the reported latency.
    warmup = max(1, int(os.environ.get("FLYDSL_ILUVATAR_FLEX_ATTN_PERF_WARMUP", "5")))
    iters = int(os.environ.get("FLYDSL_ILUVATAR_FLEX_ATTN_PERF_ITERS", "10"))

    B, H, Hkv, Sq, Skv, D = 2, 32, 8, 4096, 4096, 128
    cases = (
        ("causal", True, None, None, "bf16"),
        ("causal", True, None, None, "f16"),
        ("causal_swa1024", True, 1024, None, "bf16"),
        ("causal_swa1024", True, 1024, None, "f16"),
        ("causal_softcap30", True, None, 30.0, "bf16"),
        ("causal_softcap30", True, None, 30.0, "f16"),
    )

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
    O = torch.zeros_like(Q)
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

    Q, K, V, V_nat, O, cu_t, seq_lens_t, cu_phys = _pack_varlen_qkv(seqlens, H, Hkv, D, torch_dtype, seed=seed)

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
    O = torch.zeros_like(Q)
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
    O = torch.zeros_like(Q)
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
