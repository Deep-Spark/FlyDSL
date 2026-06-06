# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Opt-in Iluvatar MR MMA-only device correctness tests.

This file tests the ``MRMma`` instruction boundary only. It does not call G2S,
async-copy, S2R, or ``make_tiled_copy_A/B``. A/B operand fragments are created in
register memory and filled directly with constants; only the accumulator fragment
is copied back to global memory so the result can be checked.

Set ``FLYDSL_ILUVATAR_RUN_MR_MMA=1`` to run (needs an Iluvatar device).

Stage coverage notes
--------------------

* This stage fills A/B register fragments with constants and never touches G2S,
  S2R, or ``major_pattern`` layout. It cannot catch nn/tn operand-layout bugs;
  those require the G2S/S2R chain or full-pipeline tests in
  ``test_iluvatar_mr_hgemm_pipeline_stages.py``.
"""

import os
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l2_device]

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kernels.iluvatar_mr_common import ATOM_M, ATOM_N, WARP_SIZE  # noqa: E402

MMA_MAJOR_PATTERNS = ("nt", "nn", "tn", "tt")

MMA_DTYPE_CASES = [
    {
        "name": "b8",
        "fx_dtype": "Int8",
        "fx_acc": "Int32",
        "torch_acc": "int32",
        "mma_k": 32,
        "a_value": 1,
        "b_value": 2,
    },
    {
        "name": "b16",
        "fx_dtype": "Float16",
        "fx_acc": "Float32",
        "torch_acc": "float32",
        "mma_k": 16,
        "a_value": 1.0,
        "b_value": 2.0,
    },
    {
        "name": "b32",
        "fx_dtype": "Float32",
        "fx_acc": "Float32",
        "torch_acc": "float32",
        "mma_k": 16,
        "a_value": 1.0,
        "b_value": 2.0,
    },
]


def _require_enabled() -> None:
    if os.environ.get("FLYDSL_ILUVATAR_RUN_MR_MMA", "").lower() not in {"1", "true", "yes", "on"}:
        pytest.skip("set FLYDSL_ILUVATAR_RUN_MR_MMA=1 to run Iluvatar MR MMA device tests")


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
        pytest.skip(f"torch is required for Iluvatar MR MMA device tests: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA-compatible Iluvatar device is not available")
    return torch


def _configure_iluvatar_env(monkeypatch) -> None:
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


def _compile_mma_dump_kernel(flyc, fx, ixdl, dtype_case):
    fx_dtype = getattr(fx, dtype_case["fx_dtype"])
    fx_acc = getattr(fx, dtype_case["fx_acc"])
    mma_k = dtype_case["mma_k"]
    a_value = dtype_case["a_value"]
    b_value = dtype_case["b_value"]
    copy_atom_c_factory = fx.UniversalCopy32b

    @flyc.kernel(known_block_size=[WARP_SIZE, 1, 1])
    def mma_dump_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        gA = fx.make_view(fx.get_iter(A), fx.make_layout((ATOM_M, mma_k), (mma_k, 1)))
        gB = fx.make_view(fx.get_iter(B), fx.make_layout((ATOM_N, mma_k), (mma_k, 1)))
        gC = fx.make_view(fx.get_iter(C), fx.make_layout((ATOM_M, ATOM_N), (ATOM_N, 1)))

        mma_atom = fx.make_mma_atom(ixdl.MRMma(ATOM_M, ATOM_N, mma_k, fx_dtype, fx_dtype, fx_acc))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (1, 1, 1)))
        thr_mma = tiled_mma.thr_slice(tid)

        frag_A = thr_mma.make_fragment_A(gA)
        frag_B = thr_mma.make_fragment_B(gB)
        frag_C = thr_mma.make_fragment_C(gC)

        frag_A.fill(a_value)
        frag_B.fill(b_value)
        frag_C.fill(0)
        fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)

        copy_atom_c = fx.make_copy_atom(copy_atom_c_factory(), fx_acc)
        tiled_copy_C = fx.make_tiled_copy_C(copy_atom_c, tiled_mma)
        thr_copy_C = tiled_copy_C.get_slice(tid)
        fx.copy(copy_atom_c, thr_copy_C.retile(frag_C), thr_copy_C.partition_D(gC), pred=None)

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        mma_dump_kernel(A, B, C).launch(grid=(1, 1, 1), block=(WARP_SIZE, 1, 1), stream=stream)

    return launch


@pytest.mark.parametrize("major_pattern", MMA_MAJOR_PATTERNS)
@pytest.mark.parametrize("dtype_case", MMA_DTYPE_CASES, ids=[c["name"] for c in MMA_DTYPE_CASES])
def test_iluvatar_mr_mma_fragment_constants_device(major_pattern, dtype_case, monkeypatch):
    """Run one MR MMA with directly initialized A/B register fragments.

    ``major_pattern`` is intentionally parameterized to keep this stage aligned
    with the G2S/S2R correctness matrix, but it is not used by the kernel:
    major-pattern layout handling belongs to the earlier data-movement tests.

    The expected result is constant because every A element is ``a_value`` and
    every B element is ``b_value``:

        C[m, n] = MMA_K * a_value * b_value

    No A/B global load, shared-memory load, async-copy, or S2R path participates
    in this test.
    """

    _require_enabled()
    flyc, fx, ixdl = _require_imports()
    torch = _require_torch()
    _configure_iluvatar_env(monkeypatch)

    torch_dtype = {"Int8": torch.int8, "Float16": torch.float16, "Float32": torch.float32}[dtype_case["fx_dtype"]]
    torch_acc = getattr(torch, dtype_case["torch_acc"])
    launch = _compile_mma_dump_kernel(flyc, fx, ixdl, dtype_case)
    A = torch.empty((ATOM_M, dtype_case["mma_k"]), device="cuda", dtype=torch_dtype)
    B = torch.empty((ATOM_N, dtype_case["mma_k"]), device="cuda", dtype=torch_dtype)
    C = torch.empty((ATOM_M, ATOM_N), device="cuda", dtype=torch_acc)

    launch(A, B, C)
    torch.cuda.synchronize()

    expected_value = dtype_case["mma_k"] * dtype_case["a_value"] * dtype_case["b_value"]
    expected = torch.full((ATOM_M, ATOM_N), expected_value, device="cuda", dtype=torch_acc)
    if torch_acc == torch.int32:
        torch.testing.assert_close(C, expected, rtol=0, atol=0)
    else:
        torch.testing.assert_close(C, expected, rtol=2e-2, atol=2e-2)
