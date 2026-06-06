# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Opt-in Iluvatar MR TCU MMA device correctness tests.

One 64-lane warp loads the A / B tiles into register fragments, runs one
``ixdl.mmad`` per tile through ``fx.gemm`` on the ``ixdl.MRMma`` atom, and
writes the accumulator back; the result is checked against ``A @ B.T``.

Set ``FLYDSL_ILUVATAR_RUN_MR_MMA=1`` to run (needs an Iluvatar ivcore11 device).
"""

import os
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l2_device]

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kernels.iluvatar_mr_common import WARP_SIZE

# (name, M, N, K, fx multiplicand dtype, torch dtype, fx acc dtype, torch acc dtype).
_CASES = [
    ("f16_16x16x16", 16, 16, 16, "Float16", "float16", "Float32", "float32"),
    ("bf16_16x16x16", 16, 16, 16, "BFloat16", "bfloat16", "Float32", "float32"),
    ("f32_16x16x16", 16, 16, 16, "Float32", "float32", "Float32", "float32"),
    ("s8_16x16x32", 16, 16, 32, "Int8", "int8", "Int32", "int32"),
]


def _require_enabled() -> None:
    if os.environ.get("FLYDSL_ILUVATAR_RUN_MR_MMA", "").lower() not in {"1", "true", "yes", "on"}:
        pytest.skip("set FLYDSL_ILUVATAR_RUN_MR_MMA=1 to run the Iluvatar MR MMA device tests")


def _require_imports():
    try:
        import flydsl

        generated_pkg = Path(os.environ.get("FLYDSL_PYTHON_PACKAGES", _REPO_ROOT / "build-fly" / "python_packages"))
        generated_flydsl = generated_pkg / "flydsl"
        if generated_flydsl.is_dir() and str(generated_flydsl) not in flydsl.__path__:
            flydsl.__path__.append(str(generated_flydsl))

        import flydsl.compiler as flyc
        import flydsl.expr as fx
        import flydsl.expr.ixdl as ixdl
    except ModuleNotFoundError as exc:
        pytest.fail(f"FlyDSL Python package is not importable: {exc}")
    return flyc, fx, ixdl


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        pytest.skip(f"torch is required for the Iluvatar MR MMA device tests: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA-compatible device is not available")
    return torch


def _make_inputs(torch, *, m, n, k, torch_dtype, acc_torch_dtype):
    if torch_dtype == torch.int8:
        a = torch.randint(-4, 5, (m, k), dtype=torch.int8, device="cuda")
        b = torch.randint(-4, 5, (n, k), dtype=torch.int8, device="cuda")
        # CUDA has no int32 matmul; compute the reference on CPU then move back.
        expected = (a.cpu().to(torch.int32) @ b.cpu().to(torch.int32).T).to("cuda")
    else:
        a = torch.randn(m, k, dtype=torch_dtype, device="cuda")
        b = torch.randn(n, k, dtype=torch_dtype, device="cuda")
        expected = a.float() @ b.float().T
    c = torch.zeros(m, n, dtype=acc_torch_dtype, device="cuda")
    return a, b, c, expected


@pytest.mark.parametrize("spec", _CASES, ids=[c[0] for c in _CASES])
def test_mr_mma_device(spec, monkeypatch):
    _require_enabled()
    flyc, fx, ixdl = _require_imports()
    torch = _require_torch()

    name, m, n, k, fx_dtype_name, torch_dtype_name, fx_acc_name, torch_acc_name = spec

    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)

    fx_dtype = getattr(fx, fx_dtype_name)
    fx_acc = getattr(fx, fx_acc_name)
    torch_dtype = getattr(torch, torch_dtype_name)
    torch_acc = getattr(torch, torch_acc_name)
    ab_bits = {"Float16": 16, "BFloat16": 16, "Float32": 32, "Int8": 8}[fx_dtype_name]
    copy_ab_factory = {16: fx.UniversalCopy16b, 32: fx.UniversalCopy32b, 8: fx.UniversalCopy8b}[ab_bits]
    copy_c_factory = fx.UniversalCopy32b

    @flyc.kernel
    def kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x

        gA = fx.make_view(fx.get_iter(A), fx.make_layout((m, k), (k, 1)))
        gB = fx.make_view(fx.get_iter(B), fx.make_layout((n, k), (k, 1)))
        gC = fx.make_view(fx.get_iter(C), fx.make_layout((m, n), (n, 1)))

        mma_atom = fx.make_mma_atom(ixdl.MRMma(m, n, k, fx_dtype, fx_dtype, fx_acc))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
        thr_mma = tiled_mma.thr_slice(tid)

        copy_atom_ab = fx.make_copy_atom(copy_ab_factory(), fx_dtype)
        copy_atom_c = fx.make_copy_atom(copy_c_factory(), fx_acc)
        tiled_copy_A = fx.make_tiled_copy_A(copy_atom_ab, tiled_mma)
        tiled_copy_B = fx.make_tiled_copy_B(copy_atom_ab, tiled_mma)
        tiled_copy_C = fx.make_tiled_copy_C(copy_atom_c, tiled_mma)

        thr_copy_A = tiled_copy_A.get_slice(tid)
        thr_copy_B = tiled_copy_B.get_slice(tid)
        thr_copy_C = tiled_copy_C.get_slice(tid)

        copy_src_A = thr_copy_A.partition_S(gA)
        copy_src_B = thr_copy_B.partition_S(gB)
        copy_dst_C = thr_copy_C.partition_S(gC)

        frag_A = thr_mma.make_fragment_A(gA)
        frag_B = thr_mma.make_fragment_B(gB)
        frag_C = thr_mma.make_fragment_C(gC)

        copy_frag_A = thr_copy_A.retile(frag_A)
        copy_frag_B = thr_copy_B.retile(frag_B)
        copy_frag_C = thr_copy_C.retile(frag_C)

        fx.copy(copy_atom_ab, copy_src_A, copy_frag_A, pred=None)
        fx.copy(copy_atom_ab, copy_src_B, copy_frag_B, pred=None)

        frag_C.fill(0)
        fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)

        fx.copy(copy_atom_c, copy_frag_C, copy_dst_C, pred=None)

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        kernel(A, B, C).launch(grid=(1, 1, 1), block=(WARP_SIZE, 1, 1), stream=stream)

    a, b, c, expected = _make_inputs(torch, m=m, n=n, k=k, torch_dtype=torch_dtype, acc_torch_dtype=torch_acc)
    launch(a, b, c)
    torch.cuda.synchronize()

    if torch_dtype == torch.int8:
        torch.testing.assert_close(c, expected, rtol=0, atol=0)
    else:
        torch.testing.assert_close(c.float(), expected.float(), rtol=2e-2, atol=2e-2)
