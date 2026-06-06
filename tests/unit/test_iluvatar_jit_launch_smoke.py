# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Opt-in Iluvatar end-to-end FlyDSL JIT launch smoke."""

import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l2_device]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _require_enabled() -> None:
    if os.environ.get("FLYDSL_ILUVATAR_RUN_JIT_SMOKE", "").lower() not in {"1", "true", "yes", "on"}:
        pytest.skip("set FLYDSL_ILUVATAR_RUN_JIT_SMOKE=1 to run the Iluvatar JIT launch smoke")


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
        pytest.skip(f"torch is required for the Iluvatar JIT launch smoke: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA-compatible device is not available")
    return torch


def _assert_no_runtime_wrapper_errors(captured) -> None:
    assert "failed with" not in captured.err, captured.err
    assert "IX_ERROR" not in captured.err, captured.err


def test_iluvatar_jit_launches_minimal_empty_kernel(monkeypatch, capfd):
    """Launch a no-op FlyDSL kernel through the Iluvatar JIT path."""

    _require_enabled()
    flyc, fx = _require_imports()

    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)

    @flyc.kernel
    def _empty_kernel():
        pass

    @flyc.jit
    def _launch_empty(stream: fx.Stream = fx.Stream(None)):
        _empty_kernel().launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)

    _launch_empty()
    _assert_no_runtime_wrapper_errors(capfd.readouterr())


def test_iluvatar_jit_stores_single_element(monkeypatch, capfd):
    """Launch a scalar store kernel and verify one device element."""

    _require_enabled()
    flyc, fx = _require_imports()
    torch = _require_torch()

    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)

    @flyc.kernel
    def _store_one(out: fx.Tensor):
        out[0] = fx.Int32(7)

    @flyc.jit
    def _launch_store_one(out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        _store_one(out).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)

    out = torch.zeros(1, device="cuda", dtype=torch.int32)
    _launch_store_one(out)
    torch.cuda.synchronize()
    _assert_no_runtime_wrapper_errors(capfd.readouterr())

    assert out.cpu().item() == 7
