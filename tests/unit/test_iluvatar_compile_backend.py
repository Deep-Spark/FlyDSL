# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar compile backend behavior."""

import importlib
import sys
import types
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l0_backend_agnostic]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
_COMPILER_DIR = _REPO_ROOT / "python" / "flydsl" / "compiler"


def _load_backends(monkeypatch):
    """Import flydsl.compiler.backends without importing JIT-only compiler exports."""
    monkeypatch.syspath_prepend(str(_PYTHON_DIR))
    for name in list(sys.modules):
        if name == "flydsl.compiler" or name.startswith("flydsl.compiler.backends"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    compiler_pkg = types.ModuleType("flydsl.compiler")
    compiler_pkg.__path__ = [str(_COMPILER_DIR)]
    monkeypatch.setitem(sys.modules, "flydsl.compiler", compiler_pkg)
    return importlib.import_module("flydsl.compiler.backends")


def test_iluvatar_backend_explicit_arch(monkeypatch):
    backends = _load_backends(monkeypatch)
    backend = backends.get_backend("iluvatar", arch="ivcore11")

    assert backend.target.backend == "iluvatar"
    assert backend.target.arch == "ivcore11"
    assert backend.target.warp_size == 64
    assert backend.gpu_module_targets() == ['#ixdl.target<chip = "ivcore11">']


def test_iluvatar_backend_detects_arch_from_env(monkeypatch):
    backends = _load_backends(monkeypatch)
    monkeypatch.setenv("ARCH", "ivcore30")

    backend = backends.get_backend("iluvatar")

    assert backend.target.backend == "iluvatar"
    assert backend.target.arch == "ivcore30"
    assert backend.target.warp_size == 64


def test_iluvatar_backend_defaults_to_ivcore11(monkeypatch):
    backends = _load_backends(monkeypatch)
    monkeypatch.delenv("ARCH", raising=False)

    backend = backends.get_backend("iluvatar")

    assert backend.target.arch == "ivcore11"
    assert backend.target.warp_size == 64


def test_iluvatar_pipeline_uses_ixdl_attach_and_binary_codegen(monkeypatch):
    backends = _load_backends(monkeypatch)
    backend = backends.get_backend("iluvatar", arch="ivcore11")

    fragments = backend.pipeline_fragments(compile_hints={})

    assert "convert-fly-to-ixdl" in fragments
    assert any("convert-gpu-to-ixdl" in fragment for fragment in fragments)
    assert "ixdl-attach-target{O=2 chip=ivcore11 triple=bi-iluvatar-ilurt}" in fragments
    assert "gpu-module-to-binary{format=fatbin}" in fragments
    assert backend.gpu_module_targets() == ['#ixdl.target<chip = "ivcore11">']
    assert fragments.index("ixdl-attach-target{O=2 chip=ivcore11 triple=bi-iluvatar-ilurt}") < fragments.index(
        "gpu-module-to-binary{format=fatbin}"
    )
    assert not any("rocdl-attach-target" in fragment for fragment in fragments)
    assert not any("fly-rocdl-cluster-attr" in fragment for fragment in fragments)
    assert not any("runtime" in fragment.lower() for fragment in fragments)


def test_iluvatar_runtime_metadata_uses_iluvatar_libraries(monkeypatch):
    backends = _load_backends(monkeypatch)
    backend = backends.get_backend("iluvatar", arch="ivcore11")

    native_patterns = backend.native_lib_patterns()
    runtime_libs = backend.jit_runtime_lib_basenames()

    assert "_mlirDialectsFly*.so" in native_patterns
    assert "libFly*.so" in native_patterns
    assert "_mlirRegisterEverything*.so" in native_patterns
    assert "libfly_iluvatar_jit_runtime.so" in native_patterns
    assert "libfly_jit_runtime.so" not in native_patterns
    assert "libmlir_rocm_runtime.so" not in native_patterns
    assert runtime_libs == [
        "libfly_iluvatar_jit_runtime.so",
        "libmlir_c_runner_utils.so",
    ]


def test_rocm_default_runtime_metadata_stays_rocm(monkeypatch):
    backends = _load_backends(monkeypatch)
    monkeypatch.delenv("FLYDSL_COMPILE_BACKEND", raising=False)

    backend = backends.get_backend(arch="gfx942")

    assert backend.target.backend == "rocm"
    assert "libfly_jit_runtime.so" in backend.native_lib_patterns()
    assert "libmlir_rocm_runtime.so" in backend.native_lib_patterns()
    assert "libfly_iluvatar_jit_runtime.so" not in backend.native_lib_patterns()
    assert backend.jit_runtime_lib_basenames()[0] == "libfly_jit_runtime.so"


def test_rocm_default_pipeline_does_not_use_ixdl(monkeypatch):
    backends = _load_backends(monkeypatch)
    monkeypatch.delenv("FLYDSL_COMPILE_BACKEND", raising=False)

    backend = backends.get_backend(arch="gfx942")
    fragments = backend.pipeline_fragments(compile_hints={})

    assert backend.target.backend == "rocm"
    assert "convert-fly-to-ixdl" not in fragments
    assert "convert-fly-to-rocdl" in fragments
