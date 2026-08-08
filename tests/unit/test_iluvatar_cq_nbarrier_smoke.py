# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Compile-only smoke for CQ named-barrier sync API.

Verifies that ``nbarrier_reach`` / ``nbarrier_wait`` / ``nbarrier_sync`` can be
used inside ``@flyc.kernel``, lower under ``ARCH=ivcore30``, and emit the CQ
``llvm.bi.nbarrier.*`` intrinsics (no ``pipebar``).
"""

import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.iluvatar_lower]

_REPO_ROOT = Path(__file__).resolve().parents[2]


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


def _require_nbarrier_api(ixdl):
    missing = [name for name in ("nbarrier_reach", "nbarrier_wait", "nbarrier_sync") if not hasattr(ixdl, name)]
    if missing:
        pytest.fail(f"CQ named-barrier API missing from flydsl.expr.ixdl: {', '.join(missing)}")


def test_cq_nbarrier_compiles_and_emits_intrinsics(monkeypatch, tmp_path):
    """CQ kernel with named barriers compiles; IR has nbarrier, not pipebar."""
    flyc, fx, ixdl = _require_imports()
    _require_nbarrier_api(ixdl)

    dump_root = tmp_path / "nbarrier_dump"
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", "ivcore30")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(dump_root))

    # Single-warp CTA: tag 0, one participant warp.
    tag_id = 0
    warp_num = 1

    @flyc.kernel(known_block_size=[64, 1, 1])
    def nbarrier_smoke_kernel():
        # Split arrive / wait (CQ analogue of pipebar arrive+wait).
        ixdl.nbarrier_reach(tag_id, warp_num)
        ixdl.nbarrier_wait(tag_id, warp_num)
        # Combined sync (requires named-bar-sync on CQ).
        ixdl.nbarrier_sync(tag_id, warp_num)

    @flyc.jit
    def launch(stream: fx.Stream = fx.Stream(None)):
        nbarrier_smoke_kernel().launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    launch()

    origin_dumps = list(dump_root.rglob("00_origin.mlir"))
    assert origin_dumps, f"expected FLYDSL_DUMP_IR origin dump under {dump_root}"
    origin = origin_dumps[0].read_text(encoding="utf-8")

    assert "llvm.bi.nbarrier.reach" in origin
    assert "llvm.bi.nbarrier.wait" in origin
    assert "llvm.bi.nbarrier.sync" in origin
    assert "llvm.bi.pipebar.req" not in origin
    assert "llvm.bi.pipebar.wait" not in origin
