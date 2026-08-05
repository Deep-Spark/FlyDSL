# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar flex-attention device tests.

Covers:
* QK^T-only helper vs ``torch.matmul``.
* Non-causal and causal fused forward vs scaled dot-product attention.
* Softcap and sliding-window combinations vs a hand-written fp32 reference.
* f16/bf16, D=64/128, MHA/GQA, cross-attention, and physical-padding tails.
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

    mod = _require_flex_attn_module()
    assert exported is mod.compile_iluvatar_flex_attention


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
