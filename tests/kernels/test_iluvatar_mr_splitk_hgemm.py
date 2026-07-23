# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar MR HGEMM Global Split-K device tests.

Covers:

1. ``UniversalAtomicAdd(f16/bf16)`` reduces correctly across many CTAs.
2. ``compile_iluvatar_mr_hgemm_splitk(split_k>1, split_k_mode=...)`` for
   ``serial`` / ``parallel`` / ``atomic`` vs ``A @ B.T``.
3. ``split_k == 1`` stays on the non-split-K ``compile_iluvatar_mr_hgemm`` path.

"""

import os
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.iluvatar_lower]

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _require_imports():
    try:
        import flydsl

        generated_pkg = Path(os.environ.get("FLYDSL_PYTHON_PACKAGES", _REPO_ROOT / "build-fly" / "python_packages"))
        generated_flydsl = generated_pkg / "flydsl"
        if generated_flydsl.is_dir() and str(generated_flydsl) not in flydsl.__path__:
            flydsl.__path__.append(str(generated_flydsl))

        import flydsl.compiler as flyc
        import flydsl.expr as fx
    except ModuleNotFoundError as exc:
        pytest.fail(f"FlyDSL Python package is not importable: {exc}")
    return flyc, fx


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        pytest.skip(f"torch is required for the Iluvatar MR split-K device tests: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA-compatible device is not available")
    return torch


def _set_iluvatar_env(monkeypatch) -> None:
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


# 16 CTAs contend on one output cell; total additions == 256 keeps bf16 exact
# (integers representable through 256), so a lost atomic shows as value < 256.
_ATOMIC_BLOCK_DIM = 16
_ATOMIC_GRID = 16


@pytest.mark.parametrize("dtype_name", ["Float16", "BFloat16"])
def test_universal_atomic_add_multi_cta(dtype_name, monkeypatch):
    """UniversalAtomicAdd on f16/bf16 reduces across CTAs without lost updates."""
    flyc, fx = _require_imports()
    torch = _require_torch()
    _set_iluvatar_env(monkeypatch)

    fx_dtype = getattr(fx, dtype_name)
    torch_dtype = {"Float16": torch.float16, "BFloat16": torch.bfloat16}[dtype_name]

    @flyc.kernel
    def reduce_add_kernel(A: fx.Tensor, Out: fx.Tensor, block_dim: fx.Constexpr[int]):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x

        tA = fx.logical_divide(A, fx.make_layout(block_dim, 1))
        tA = fx.slice(tA, (None, bid))
        tA = fx.logical_divide(tA, fx.make_layout(1, 1))

        load_atom = fx.make_copy_atom(fx.UniversalCopy16b(), fx_dtype)
        atomic_atom = fx.make_copy_atom(fx.UniversalAtomic(fx.AtomicOp.Add, fx_dtype), fx_dtype)

        rA = fx.make_rmem_tensor(1, fx_dtype)
        fx.copy_atom_call(load_atom, fx.slice(tA, (None, tid)), rA)

        tOut = fx.logical_divide(Out, fx.make_layout(1, 1))
        tOut = fx.slice(tOut, (None, fx.Int32(0)))
        tOut = fx.logical_divide(tOut, fx.make_layout(1, 1))
        fx.copy_atom_call(atomic_atom, rA, fx.slice(tOut, (None, fx.Int32(0))))

    @flyc.jit
    def reduce_add(
        A: fx.Tensor, Out, block_dim: fx.Constexpr[int], grid: fx.Constexpr[int], stream: fx.Stream = fx.Stream(None)
    ):
        reduce_add_kernel(A, Out, block_dim).launch(grid=(grid, 1, 1), block=(block_dim, 1, 1), stream=stream)

    n = _ATOMIC_BLOCK_DIM * _ATOMIC_GRID
    a_dev = torch.ones(n, device="cuda", dtype=torch_dtype)
    out_dev = torch.zeros(1, device="cuda", dtype=torch_dtype)
    stream = torch.cuda.Stream()
    tA = flyc.from_torch_tensor(a_dev).mark_layout_dynamic(leading_dim=0, divisibility=1)
    reduce_add(tA, out_dev, _ATOMIC_BLOCK_DIM, _ATOMIC_GRID, stream=stream)
    torch.cuda.synchronize()

    expected = float(a_dev.float().sum().item())
    actual = float(out_dev.float().item())
    assert abs(actual - expected) < 0.5, f"{dtype_name} multi-CTA atomic add: expected {expected}, got {actual}"


# (dtype, M, N, K, split_k, mode); K % (split_k * bk) == 0 with bk = 16 * k_atoms = 32.
# split_k == 1 exercises the non-split-K golden path (hgemm.py).
_SPLITK_CASES = [
    ("Float16", 512, 512, 1024, 1, "serial"),
    ("BFloat16", 512, 512, 1024, 1, "serial"),
    ("Float16", 256, 256, 2048, 2, "serial"),
    ("Float16", 256, 256, 2048, 4, "serial"),
    ("BFloat16", 256, 256, 2048, 2, "serial"),
    ("BFloat16", 256, 256, 2048, 4, "serial"),
    ("Float16", 256, 256, 2048, 2, "parallel"),
    ("Float16", 256, 256, 2048, 4, "parallel"),
    ("BFloat16", 256, 256, 2048, 2, "parallel"),
    ("BFloat16", 256, 256, 2048, 4, "parallel"),
    ("Float16", 256, 256, 2048, 2, "atomic"),
    ("Float16", 256, 256, 2048, 4, "atomic"),
    ("BFloat16", 256, 256, 2048, 2, "atomic"),
    ("BFloat16", 256, 256, 2048, 4, "atomic"),
]


@pytest.mark.parametrize(
    "spec",
    _SPLITK_CASES,
    ids=[f"{c[0]}_{c[1]}x{c[2]}x{c[3]}_spk{c[4]}_{c[5]}" for c in _SPLITK_CASES],
)
def test_mr_splitk_hgemm_device(spec, monkeypatch):
    flyc, fx = _require_imports()
    torch = _require_torch()
    _set_iluvatar_env(monkeypatch)

    from kernels.gemm.iluvatar.common import remap_gemm_tensors
    from kernels.gemm.iluvatar.mr.hgemm import compile_iluvatar_mr_hgemm
    from kernels.gemm.iluvatar.mr.hgemm_splitk import compile_iluvatar_mr_hgemm_splitk

    dtype_name, m, n, k, split_k, split_k_mode = spec
    fx_dtype = getattr(fx, dtype_name)
    torch_dtype = {"Float16": torch.float16, "BFloat16": torch.bfloat16}[dtype_name]
    k_atoms = 2

    torch.manual_seed(0)
    A = torch.randn(m, k, dtype=torch_dtype, device="cuda")
    B = torch.randn(n, k, dtype=torch_dtype, device="cuda")
    # Poison C so a broken/missing zero step surfaces as a failure.
    C = torch.full((m, n), 7777.0, dtype=torch_dtype, device="cuda")

    if split_k > 1:
        launcher = compile_iluvatar_mr_hgemm_splitk(
            M=m,
            N=n,
            K=k,
            k_atoms=k_atoms,
            epilogue="no_c_read",
            major_pattern="tn",
            elem_dtype=fx_dtype,
            split_k=split_k,
            split_k_mode=split_k_mode,
        )
    else:
        launcher = compile_iluvatar_mr_hgemm(
            M=m,
            N=n,
            K=k,
            k_atoms=k_atoms,
            epilogue="no_c_read",
            epilogue_store="shfl",
            major_pattern="tn",
            elem_dtype=fx_dtype,
        )
    a_dev, b_dev = remap_gemm_tensors(A, B, "tn")
    stream = torch.cuda.Stream()
    launcher(a_dev, b_dev, C, stream=stream)
    torch.cuda.synchronize()

    got = C.to(torch.float32)
    expected = A.to(torch.float32) @ B.to(torch.float32).T
    assert torch.isfinite(got).all().item(), f"{spec}: non-finite output"
    cos = torch.nn.functional.cosine_similarity(got.reshape(1, -1), expected.reshape(1, -1)).item()
    # serial/parallel are deterministic with fp32-quality reduce; atomic keeps cosine.
    min_cos = 0.999 if split_k_mode == "atomic" else 0.9999
    if split_k == 1:
        min_cos = 0.999
    assert cos > min_cos, f"{spec}: cosine similarity {cos:.6f} below {min_cos}"


def test_split_k_rejects_read_c_accum(monkeypatch):
    """split_k > 1 must reject the read_c_accum epilogue at compile time."""
    _flyc, fx = _require_imports()
    _set_iluvatar_env(monkeypatch)

    from kernels.gemm.iluvatar.mr.hgemm_splitk import compile_iluvatar_mr_hgemm_splitk

    with pytest.raises(ValueError, match="split_k"):
        compile_iluvatar_mr_hgemm_splitk(
            M=256, N=256, K=2048, k_atoms=2, epilogue="read_c_accum", split_k=4, elem_dtype=fx.Float16
        )


def test_split_k_rejects_unknown_mode(monkeypatch):
    _flyc, fx = _require_imports()
    _set_iluvatar_env(monkeypatch)

    from kernels.gemm.iluvatar.mr.hgemm_splitk import compile_iluvatar_mr_hgemm_splitk

    with pytest.raises(ValueError, match="split_k_mode"):
        compile_iluvatar_mr_hgemm_splitk(
            M=256, N=256, K=2048, k_atoms=2, split_k=4, split_k_mode="bogus", elem_dtype=fx.Float16
        )


def test_split_k_rejects_split_k_one(monkeypatch):
    """The split-K entry point requires split_k > 1; use hgemm.py for split_k == 1."""
    _flyc, fx = _require_imports()
    _set_iluvatar_env(monkeypatch)

    from kernels.gemm.iluvatar.mr.hgemm_splitk import compile_iluvatar_mr_hgemm_splitk

    with pytest.raises(ValueError, match="split_k must be > 1"):
        compile_iluvatar_mr_hgemm_splitk(M=256, N=256, K=2048, k_atoms=2, split_k=1, elem_dtype=fx.Float16)
