# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Device tests for Iluvatar MR scaled IGEMM (W8A8 dequant store).

Keeps the matrix small: one 64x64 CTA, two compile keys. Covers the contracts
ixaite will call, not the full pattern x tile table.

* ``dynamic_m`` -- one kernel, several live M (serving launch ABI).
* ``allow_dynamic_m`` short-M on ``tt`` -- k-major A pred + B-first issue.
"""

import os
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.iluvatar_lower]

_REPO_ROOT = Path(__file__).resolve().parents[2]

# 2x2 warps, 2x2 atoms, k_rep=2 -> BM=BN=BK=64. Chunks divide the warp count.
_WARPS_M = 2
_WARPS_N = 2
_WARP_ATOMS_M = 2
_WARP_ATOMS_N = 2
_K_REP = 2
_BM = 64
_BN = 64
_BK = 64


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        pytest.skip(f"torch is required for Iluvatar scaled IGEMM tests: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA-compatible Iluvatar device is not available")
    return torch


def _configure_iluvatar_env(monkeypatch) -> None:
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


def _require_igemm():
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    try:
        from kernels.gemm.iluvatar.common import remap_gemm_tensors
        from kernels.gemm.iluvatar.mr.igemm import compile_iluvatar_mr_igemm
    except ModuleNotFoundError as exc:
        pytest.fail(f"failed to import kernels.gemm.iluvatar.mr.igemm: {exc}")
    return compile_iluvatar_mr_igemm, remap_gemm_tensors


def _reference(torch, A, B, scale_a, scale_b, bias=None):
    acc = A.cpu().to(torch.int32) @ B.cpu().to(torch.int32).T
    out = acc.to(torch.float32)
    out = out * scale_a.cpu().view(-1, 1).to(torch.float32)
    out = out * scale_b.cpu().view(1, -1).to(torch.float32)
    if bias is not None:
        out = out + bias.cpu().view(1, -1).to(torch.float32)
    return out.to(torch.bfloat16)


def _assert_scaled_close(torch, got, expected, *, label: str):
    diff = (got.float() - expected.float()).abs()
    tol = 2e-2 * expected.float().abs().max().clamp(min=1.0) + 1e-3
    ok = bool((diff <= tol).all().item())
    finite_ok = bool(torch.isfinite(got).all().item())
    print(
        f"[check] {label} ok={ok} finite={finite_ok} "
        f"max_abs={float(diff.max()):.3e} n_mismatch={int((diff > tol).sum())}/{got.numel()}"
    )
    if not ok:
        print(f"  C[0,0:4]      = {got[0, 0:4].tolist()}")
        print(f"  expect[0,0:4] = {expected[0, 0:4].tolist()}")
    assert finite_ok, f"{label}: non-finite output"
    assert ok, f"{label}: max_abs={float(diff.max()):.3e}"


def _compile_scaled(*, compile_fn, n, k, major_pattern, allow_dynamic_m=False, dynamic_m=False, m=None, m_max=None):
    m_arg = m_max if dynamic_m else m
    return compile_fn(
        M=m_arg,
        N=n,
        K=k,
        warps_m=_WARPS_M,
        warps_n=_WARPS_N,
        k_rep=_K_REP,
        warp_atoms_m=_WARP_ATOMS_M,
        warp_atoms_n=_WARP_ATOMS_N,
        major_pattern=major_pattern,
        epilogue="scaled_bf16",
        stages=2,
        apply_bias=True,
        allow_dynamic_m=allow_dynamic_m,
        dynamic_m=dynamic_m,
        m_hint=_BM if dynamic_m else None,
    )


def test_iluvatar_mr_igemm_scaled_dynamic_m(monkeypatch):
    """One compiled kernel serves several live M, including a non-multiple of BM."""
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    compile_fn, remap = _require_igemm()

    n, k = _BN, _BK
    m_max = 128
    launcher = _compile_scaled(compile_fn=compile_fn, n=n, k=k, major_pattern="tn", dynamic_m=True, m_max=m_max)
    assert launcher.m_ceil == m_max
    assert launcher.bm == _BM
    stream = torch.cuda.Stream()

    for m in (17, 80):
        torch.manual_seed(m)
        A = torch.randint(-8, 8, (m, k), dtype=torch.int8, device="cuda")
        B = torch.randint(-8, 8, (n, k), dtype=torch.int8, device="cuda")
        scale_a = torch.empty(m, device="cuda", dtype=torch.float32).uniform_(0.01, 0.5)
        scale_b = torch.empty(n, device="cuda", dtype=torch.float32).uniform_(0.01, 0.5)
        bias = torch.empty(n, device="cuda", dtype=torch.float32).uniform_(-1.0, 1.0)
        C = torch.zeros(launcher.m_ceil, n, dtype=torch.bfloat16, device="cuda")
        a_dev, b_dev = remap(A, B, "tn")
        grid_m = (m + launcher.bm - 1) // launcher.bm
        launcher(a_dev, b_dev, scale_a, scale_b, C, bias, m, grid_m, stream=stream)
        torch.cuda.synchronize()
        expected = _reference(torch, A, B, scale_a, scale_b, bias).to("cuda")
        _assert_scaled_close(torch, C[:m], expected, label=f"dynamic_m tn M={m}")


def test_iluvatar_mr_igemm_scaled_allow_dynamic_m_tt(monkeypatch):
    """Short-M on tt: k-major A G2S pred plus N-contiguous B issued first."""
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)
    compile_fn, remap = _require_igemm()

    m, n, k = 40, _BN, _BK
    torch.manual_seed(0)
    A = torch.randint(-8, 8, (m, k), dtype=torch.int8, device="cuda")
    B = torch.randint(-8, 8, (n, k), dtype=torch.int8, device="cuda")
    scale_a = torch.empty(m, device="cuda", dtype=torch.float32).uniform_(0.01, 0.5)
    scale_b = torch.empty(n, device="cuda", dtype=torch.float32).uniform_(0.01, 0.5)
    bias = torch.empty(n, device="cuda", dtype=torch.float32).uniform_(-1.0, 1.0)
    launcher = _compile_scaled(
        compile_fn=compile_fn, n=n, k=k, major_pattern="tt", allow_dynamic_m=True, m=m
    )
    assert launcher.m_ceil == _BM
    C = torch.zeros(launcher.m_ceil, n, dtype=torch.bfloat16, device="cuda")
    a_dev, b_dev = remap(A, B, "tt")
    stream = torch.cuda.Stream()
    launcher(a_dev, b_dev, scale_a, scale_b, C, bias, stream=stream)
    torch.cuda.synchronize()
    expected = _reference(torch, A, B, scale_a, scale_b, bias).to("cuda")
    _assert_scaled_close(torch, C[:m], expected, label=f"allow_dynamic_m tt M={m}")
