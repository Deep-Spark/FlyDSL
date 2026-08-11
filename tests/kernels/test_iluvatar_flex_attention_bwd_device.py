# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar flex-attention V2-8 backward + return_lse device tests."""

from __future__ import annotations

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
        pytest.skip(f"torch is required: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA-compatible Iluvatar device is not available")
    return torch


def _configure_iluvatar_env(monkeypatch) -> None:
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


def _require_flex_mods():
    try:
        import kernels.attention.iluvatar.flex_attention as fwd
        import kernels.attention.iluvatar.flex_attention_bwd as bwd
    except ModuleNotFoundError as exc:
        pytest.skip(f"flex attention modules not importable: {exc}")
    return fwd, bwd


def _phys_seq(seq: int, block: int) -> int:
    return ((seq + block - 1) // block) * block


def _reference_scores_fp32(Q, K, *, sm_scale, is_causal=False, window_size=None, softcap=None):
    torch = _require_torch()
    S = torch.matmul(Q.float(), K.float().transpose(-1, -2)) * sm_scale
    Sq = S.shape[-2]
    Skv = S.shape[-1]
    q_idx = torch.arange(Sq, device=S.device).view(Sq, 1)
    kv_idx = torch.arange(Skv, device=S.device).view(1, Skv)
    if softcap is not None:
        S = softcap * torch.tanh(S / softcap)
    mask = torch.zeros((Sq, Skv), device=S.device, dtype=torch.bool)
    if is_causal:
        mask = mask | (kv_idx > q_idx)
    if window_size is not None:
        mask = mask | ((q_idx - kv_idx) > window_size)
    S = S.masked_fill(mask, float("-inf"))
    return S


def _reference_lse_fp32(Q, K, *, sm_scale, is_causal=False, window_size=None, softcap=None):
    torch = _require_torch()
    S = _reference_scores_fp32(
        Q, K, sm_scale=sm_scale, is_causal=is_causal, window_size=window_size, softcap=softcap
    )
    return torch.logsumexp(S, dim=-1)


def _reference_fwd_fp32(Q, K, V, *, sm_scale, is_causal=False, window_size=None, softcap=None):
    torch = _require_torch()
    S = _reference_scores_fp32(
        Q, K, sm_scale=sm_scale, is_causal=is_causal, window_size=window_size, softcap=softcap
    )
    P = torch.softmax(S, dim=-1)
    P = torch.nan_to_num(P, nan=0.0)
    return torch.matmul(P, V.float())


def test_return_lse_rejects_varlen(monkeypatch):
    _configure_iluvatar_env(monkeypatch)
    fwd, _ = _require_flex_mods()
    with pytest.raises(ValueError, match="return_lse is dense-only"):
        fwd.compile_iluvatar_flex_attention(1, 1, 64, 64, 64, varlen=True, return_lse=True)


@pytest.mark.parametrize("dtype", ["bf16", "f16"])
def test_return_lse_matches_logsumexp(monkeypatch, dtype):
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    fwd, _ = _require_flex_mods()

    B, H, Sq, Skv, D = 1, 2, 64, 64, 64
    torch_dtype = {"bf16": torch.bfloat16, "f16": torch.float16}[dtype]
    sm_scale = 1.0 / math.sqrt(D)
    Sq_phys = _phys_seq(Sq, fwd.BLOCK_M)
    Skv_phys = _phys_seq(Skv, fwd.BLOCK_N)

    torch.manual_seed(0)
    Q = torch.zeros(B, H, Sq_phys, D, device="cuda", dtype=torch_dtype)
    K = torch.zeros(B, H, Skv_phys, D, device="cuda", dtype=torch_dtype)
    Vn = torch.zeros(B, H, Skv_phys, D, device="cuda", dtype=torch_dtype)
    Q[:, :, :Sq].copy_(torch.randn(B, H, Sq, D, device="cuda", dtype=torch_dtype))
    K[:, :, :Skv].copy_(torch.randn(B, H, Skv, D, device="cuda", dtype=torch_dtype))
    Vn[:, :, :Skv].copy_(torch.randn(B, H, Skv, D, device="cuda", dtype=torch_dtype))
    V = Vn.transpose(-1, -2).contiguous()
    O = torch.empty_like(Q)

    launch = fwd.compile_iluvatar_flex_attention(
        B, H, Sq, Skv, D, dtype=dtype, is_causal=True, sm_scale=sm_scale, return_lse=True
    )
    lse = launch(Q, K, V, O)
    assert lse is not None
    torch.cuda.synchronize()

    ref_lse = _reference_lse_fp32(Q[:, :, :Sq], K[:, :, :Skv], sm_scale=sm_scale, is_causal=True)
    # LSE is relatively stable; use fwd-like tol x2.
    atol = 4e-2 if dtype == "bf16" else 2e-2
    torch.testing.assert_close(lse[:, :, :Sq].float(), ref_lse, rtol=atol, atol=atol)

    # O must still match reference.
    ref_o = _reference_fwd_fp32(
        Q[:, :, :Sq], K[:, :, :Skv], Vn[:, :, :Skv], sm_scale=sm_scale, is_causal=True
    )
    torch.testing.assert_close(O[:, :, :Sq].float(), ref_o, rtol=atol, atol=atol)


def _run_bwd_case(
    monkeypatch,
    *,
    dtype: str,
    is_causal: bool = True,
    window_size=None,
    softcap=None,
    B=1,
    H=2,
    Sq=64,
    Skv=64,
    D=64,
    seed=0,
):
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    fwd, bwd = _require_flex_mods()

    torch_dtype = {"bf16": torch.bfloat16, "f16": torch.float16}[dtype]
    sm_scale = 1.0 / math.sqrt(D)
    Sq_phys = _phys_seq(Sq, fwd.BLOCK_M)
    Skv_phys = _phys_seq(Skv, fwd.BLOCK_N)
    # V2-8: tol = section 3.1 x4
    atol = 8e-2 if dtype == "bf16" else 4e-2

    torch.manual_seed(seed)
    Q = torch.zeros(B, H, Sq_phys, D, device="cuda", dtype=torch_dtype)
    K = torch.zeros(B, H, Skv_phys, D, device="cuda", dtype=torch_dtype)
    Vn = torch.zeros(B, H, Skv_phys, D, device="cuda", dtype=torch_dtype)
    Q[:, :, :Sq].copy_(torch.randn(B, H, Sq, D, device="cuda", dtype=torch_dtype) * 0.5)
    K[:, :, :Skv].copy_(torch.randn(B, H, Skv, D, device="cuda", dtype=torch_dtype) * 0.5)
    Vn[:, :, :Skv].copy_(torch.randn(B, H, Skv, D, device="cuda", dtype=torch_dtype) * 0.5)
    V = Vn.transpose(-1, -2).contiguous()
    O = torch.empty_like(Q)

    launch_fwd = fwd.compile_iluvatar_flex_attention(
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
        return_lse=True,
    )
    lse = launch_fwd(Q, K, V, O)
    torch.cuda.synchronize()

    dO = torch.zeros_like(O)
    dO[:, :, :Sq].copy_(torch.randn(B, H, Sq, D, device="cuda", dtype=torch_dtype) * 0.5)
    dQ = torch.empty_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)

    launch_bwd = bwd.compile_iluvatar_flex_attention_bwd(
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
    )
    launch_bwd(Q, K, V, O, dO, lse, dQ, dK, dV)
    torch.cuda.synchronize()

    # fp32 autograd reference on logical prefix.
    q_ref = Q[:, :, :Sq].float().detach().requires_grad_(True)
    k_ref = K[:, :, :Skv].float().detach().requires_grad_(True)
    v_ref = Vn[:, :, :Skv].float().detach().requires_grad_(True)
    o_ref = _reference_fwd_fp32(
        q_ref,
        k_ref,
        v_ref,
        sm_scale=sm_scale,
        is_causal=is_causal,
        window_size=window_size,
        softcap=softcap,
    )
    o_ref.backward(dO[:, :, :Sq].float())

    torch.testing.assert_close(dQ[:, :, :Sq].float(), q_ref.grad, rtol=atol, atol=atol)
    torch.testing.assert_close(dK[:, :, :Skv].float(), k_ref.grad, rtol=atol, atol=atol)
    # dV kernel layout is transposed vs natural V.
    dVn = dV.transpose(-1, -2).contiguous()
    torch.testing.assert_close(dVn[:, :, :Skv].float(), v_ref.grad, rtol=atol, atol=atol)


@pytest.mark.parametrize("dtype", ["bf16", "f16"])
def test_bwd_causal_main(monkeypatch, dtype):
    _run_bwd_case(monkeypatch, dtype=dtype, is_causal=True, D=64)


def test_bwd_swa_smoke(monkeypatch):
    _run_bwd_case(monkeypatch, dtype="bf16", is_causal=False, window_size=16, Sq=64, Skv=64, D=64)


def test_bwd_softcap_smoke(monkeypatch):
    _run_bwd_case(monkeypatch, dtype="bf16", is_causal=True, softcap=30.0, D=64)


def test_bwd_rejects_non_mha_shape_via_d(monkeypatch):
    _configure_iluvatar_env(monkeypatch)
    _, bwd = _require_flex_mods()
    with pytest.raises(ValueError, match="D must be"):
        bwd.compile_iluvatar_flex_attention_bwd(1, 1, 64, 64, 96, dtype="bf16")
