# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar JIT launch across two CUDA contexts.

Not in device must-pass: that job exposes one GPU and fails on skipped cases.
The full device suite still collects this file via test_iluvatar_*.py.
"""

import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l2_device, pytest.mark.iluvatar_lower]

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


def test_iluvatar_jit_compiles_and_runs_on_non_zero_device(monkeypatch, capfd):
    """A TP worker on local_rank=1 compiles and launches in that context only."""

    flyc, fx = _require_imports()
    torch = _require_torch()
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA-compatible devices")

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

    with torch.cuda.device(1):
        out = torch.zeros(1, device="cuda:1", dtype=torch.int32)
        flyc.compile(_launch_store_one, out, fx.Stream(None))
        torch.cuda.synchronize()
    _assert_no_runtime_wrapper_errors(capfd.readouterr())
    assert out.cpu().item() == 7


def test_iluvatar_jit_reuses_compiled_kernel_across_devices(monkeypatch, capfd):
    """One flyc.compile artifact must reload the cubin in each CUDA context."""

    flyc, fx = _require_imports()
    torch = _require_torch()
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA-compatible devices")

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

    with torch.cuda.device(0):
        out0 = torch.zeros(1, device="cuda:0", dtype=torch.int32)
        compiled = flyc.compile(_launch_store_one, out0, fx.Stream(None))
        torch.cuda.synchronize()
    _assert_no_runtime_wrapper_errors(capfd.readouterr())
    assert out0.cpu().item() == 7

    with torch.cuda.device(1):
        out1 = torch.zeros(1, device="cuda:1", dtype=torch.int32)
        compiled(out1, fx.Stream(None))
        torch.cuda.synchronize()
    _assert_no_runtime_wrapper_errors(capfd.readouterr())
    assert out1.cpu().item() == 7
    assert out0.cpu().item() == 7

    for step in range(1, 4):
        for device, out in ((0, out0), (1, out1)):
            out.zero_()
            with torch.cuda.device(device):
                compiled(out, fx.Stream(None))
                torch.cuda.synchronize()
            _assert_no_runtime_wrapper_errors(capfd.readouterr())
            assert out.cpu().item() == 7, f"device {device} failed on step {step}"
