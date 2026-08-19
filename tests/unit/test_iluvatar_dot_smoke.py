# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Compile-only smoke for Iluvatar ALU int8x4 dot wrappers.

``ixdl.idot4`` / ``udot4`` emit ``llvm.bi.*dot4`` at trace time (no Fly atom).
This compiles a kernel under ``ARCH=ivcore11`` and checks the origin dump.
Results are stored so the speculatable dots are not DCE'd.

``a`` / ``b`` / acc are packed i32, matching ``llvm.bi.idot4`` / ``udot4``.
"""

import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.iluvatar_lower]

_REPO_ROOT = Path(__file__).resolve().parents[2]

_DOT_INTRINSICS = (
    "llvm.bi.idot4",
    "llvm.bi.udot4",
)


def _require_imports():
    import sys

    python_dir = str(_REPO_ROOT / "python")
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)

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


def test_ixdl_dot4_compiles_and_emits_intrinsics(monkeypatch, tmp_path):
    """Kernel using packed-i32 idot4/udot4 compiles; origin IR has llvm.bi.*dot4."""
    flyc, fx, ixdl = _require_imports()
    missing = [name for name in ("idot4", "udot4") if not hasattr(ixdl, name)]
    if missing:
        pytest.fail(f"ALU dot4 API missing from flydsl.expr.ixdl: {', '.join(missing)}")

    try:
        import torch
    except ModuleNotFoundError as exc:
        pytest.skip(f"torch required: {exc}")

    dump_root = tmp_path / "dot_dump"
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", "ivcore11")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(dump_root))

    @flyc.kernel(known_block_size=[64, 1, 1])
    def dot4_smoke_kernel(out_i: fx.Tensor):
        packed_a = fx.Int32(1)
        packed_b = fx.Int32(2)
        acc = fx.Int32(0)
        acc = ixdl.idot4(packed_a, packed_b, acc)
        acc = ixdl.udot4(packed_a, packed_b, acc)
        out_i[0] = acc

    @flyc.jit
    def launch(out_i: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        dot4_smoke_kernel(out_i).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    launch(torch.zeros(1, dtype=torch.int32))

    origin_dumps = list(dump_root.rglob("00_origin.mlir"))
    assert origin_dumps, f"expected FLYDSL_DUMP_IR origin dump under {dump_root}"
    origin = origin_dumps[0].read_text(encoding="utf-8")
    missing_ir = [name for name in _DOT_INTRINSICS if name not in origin]
    assert not missing_ir, f"missing {missing_ir} in origin IR:\n{origin}"
