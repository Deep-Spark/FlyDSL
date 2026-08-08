# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Unit tests for SME gmem leading-stride 16B alignment (CQ SMEX)."""

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
        from flydsl.expr.ixdl.mr import _check_sme_gmem_leading_stride_align
    except ModuleNotFoundError as exc:
        pytest.fail(f"FlyDSL Python package is not importable: {exc}")
    return flyc, fx, ixdl, _check_sme_gmem_leading_stride_align


def test_sme_gmem_leading_stride_align_helper():
    """Foldable stride_byte must be a multiple of 16B; aligned values pass."""
    _, fx, _, check = _require_imports()

    check(16, fx.Int8)  # 16B
    check(8, fx.Float16)  # 16B
    check(4, fx.Float32)  # 16B
    check(32, fx.Int8)  # 32B

    with pytest.raises(ValueError, match=r"got leading_stride=10 \(20B\)"):
        check(10, fx.Float16)
    with pytest.raises(ValueError, match="16B-aligned"):
        check(24, fx.Int8)
    with pytest.raises(ValueError, match="16B-aligned"):
        check(9, fx.Float16)  # 18B
    # Dynamic / non-int strides are skipped (no raise).
    check(None, fx.Int8)
    check("dynamic", fx.Float16)


def test_make_sme_gmem_tensor_rejects_misaligned_stride(monkeypatch):
    """make_sme_gmem_tensor raises when leading_stride * elem_bytes is not 16B-aligned."""
    flyc, fx, ixdl, _ = _require_imports()

    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", "ivcore30")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

    @flyc.kernel(known_block_size=[64, 1, 1])
    def bad_stride_kernel(src: fx.Tensor):
        # f16 * 10 elems = 20B -- not 16B-aligned.
        ixdl.make_sme_gmem_tensor(src, leading_stride=10)

    @flyc.jit
    def launch_bad(src: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        bad_stride_kernel(src).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    try:
        import torch
    except ModuleNotFoundError as exc:
        pytest.skip(f"torch required: {exc}")

    src = torch.zeros((16, 10), dtype=torch.float16)
    with pytest.raises(ValueError, match="16B-aligned"):
        launch_bad(src)


def test_make_sme_gmem_tensor_accepts_aligned_stride(monkeypatch):
    """Aligned leading_stride (e.g. f16 x 8 = 16B) builds an SME gmem view."""
    flyc, fx, ixdl, _ = _require_imports()

    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", "ivcore30")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

    @flyc.kernel(known_block_size=[64, 1, 1])
    def ok_stride_kernel(src: fx.Tensor):
        sme = ixdl.make_sme_gmem_tensor(src, leading_stride=8)
        # Keep a use so the view is not DCE'd before lowering.
        _ = fx.get_iter(sme)

    @flyc.jit
    def launch_ok(src: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        ok_stride_kernel(src).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    try:
        import torch
    except ModuleNotFoundError as exc:
        pytest.skip(f"torch required: {exc}")

    src = torch.zeros((16, 8), dtype=torch.float16)
    launch_ok(src)
