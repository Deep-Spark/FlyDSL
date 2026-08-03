# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""CQ named-barrier DSL compile smoke (ARCH=ivcore30).

Verifies ``nbarrier_{reach,wait,sync}`` are usable inside ``@flyc.kernel`` and
lower to ``llvm.bi.nbarrier.*`` (not pipebar) under the CQ chip target.
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


def _collect_dump_text(dump_root: Path) -> str:
    texts = []
    for path in sorted(dump_root.rglob("*.mlir")):
        texts.append(path.read_text())
    return "\n".join(texts)


def test_cq_nbarrier_lowers_in_kernel(monkeypatch, tmp_path):
    """COMPILE_ONLY CQ kernel: nbarrier intrinsics present, pipebar absent."""
    flyc, fx, ixdl = _require_imports()

    dump_dir = tmp_path / "nbarrier_dump"
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", "ivcore30")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DUMP_DIR", str(dump_dir))

    @flyc.kernel(known_block_size=[64, 1, 1])
    def cq_nbarrier_smoke():
        # Split reach/wait plus combined sync — CQ double-buffer building blocks.
        ixdl.nbarrier_reach(0, 0)
        ixdl.nbarrier_wait(0, 0)
        ixdl.nbarrier_sync(1, 0)

    @flyc.jit
    def launch(stream: fx.Stream = fx.Stream(None)):
        cq_nbarrier_smoke().launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    launch()

    asm = _collect_dump_text(dump_dir)
    assert asm, f"expected FLYDSL_DUMP_IR output under {dump_dir}"
    assert 'llvm.call_intrinsic "llvm.bi.nbarrier.reach"' in asm
    assert 'llvm.call_intrinsic "llvm.bi.nbarrier.wait"' in asm
    assert 'llvm.call_intrinsic "llvm.bi.nbarrier.sync"' in asm
    assert "llvm.bi.pipebar.req" not in asm
    assert "llvm.bi.pipebar.wait" not in asm
    assert 'chip = "ivcore30"' in asm or "chip=ivcore30" in asm
