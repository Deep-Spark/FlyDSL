# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar JIT runtime library resolution."""

import importlib
import sys
import types
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l0_backend_agnostic]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
_COMPILER_DIR = _PYTHON_DIR / "flydsl" / "compiler"


def _load_jit_executor(monkeypatch):
    """Import jit_executor without importing flydsl.compiler package exports."""
    monkeypatch.syspath_prepend(str(_PYTHON_DIR))
    for name in list(sys.modules):
        if (
            name == "flydsl.compiler"
            or name == "flydsl.compiler.jit_executor"
            or name.startswith("flydsl.compiler.backends")
        ):
            monkeypatch.delitem(sys.modules, name, raising=False)

    compiler_pkg = types.ModuleType("flydsl.compiler")
    compiler_pkg.__path__ = [str(_COMPILER_DIR)]
    monkeypatch.setitem(sys.modules, "flydsl.compiler", compiler_pkg)

    # _resolve_runtime_libs does not touch MLIR, but jit_executor imports these
    # modules at import time for the full ExecutionEngine path.
    mlir_pkg = types.ModuleType("flydsl._mlir")
    mlir_pkg.ir = types.SimpleNamespace(Context=object, Module=object, Type=object, Value=object)
    execution_engine_pkg = types.ModuleType("flydsl._mlir.execution_engine")
    execution_engine_pkg.ExecutionEngine = object
    monkeypatch.setitem(sys.modules, "flydsl._mlir", mlir_pkg)
    monkeypatch.setitem(sys.modules, "flydsl._mlir.execution_engine", execution_engine_pkg)

    return importlib.import_module("flydsl.compiler.jit_executor")


def test_iluvatar_jit_runtime_libraries_resolve_from_backend(monkeypatch, tmp_path):
    mlir_libs_dir = tmp_path / "_mlir" / "_mlir_libs"
    mlir_libs_dir.mkdir(parents=True)
    iluvatar_runtime = mlir_libs_dir / "libfly_iluvatar_jit_runtime.so"
    c_runner_utils = mlir_libs_dir / "libmlir_c_runner_utils.so"
    iluvatar_runtime.touch()
    c_runner_utils.touch()

    jit_executor = _load_jit_executor(monkeypatch)
    backends = importlib.import_module("flydsl.compiler.backends")

    monkeypatch.setattr(jit_executor, "__file__", str(tmp_path / "compiler" / "jit_executor.py"))
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("ARCH", "ivcore11")
    backends._make_backend.cache_clear()
    jit_executor._resolve_runtime_libs.cache_clear()

    try:
        assert [Path(path) for path in jit_executor._resolve_runtime_libs()] == [
            iluvatar_runtime,
            c_runner_utils,
        ]
    finally:
        jit_executor._resolve_runtime_libs.cache_clear()
        backends._make_backend.cache_clear()
