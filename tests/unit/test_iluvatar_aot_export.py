# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Iluvatar AOT export: ``flyc.compile(...).export_to_c(...)``.

The exported object embeds the Iluvatar AOT runtime archive, so a deployed
binary links against the CUDA driver and libc only -- no FlyDSL runtime. The
device test links one with the system C compiler, loads it with ctypes and
launches the kernel through the packed C ABI.
"""

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.iluvatar_lower]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


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
        pytest.skip(f"torch is required for the Iluvatar AOT launch test: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA-compatible device is not available")
    return torch


def _use_iluvatar(monkeypatch):
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


def _axpy_launcher(flyc, fx):
    @flyc.kernel
    def axpy_f32(X: fx.Pointer, Y: fx.Pointer, n: fx.Int32, alpha: fx.Float32):
        idx = fx.block_idx.x * fx.block_dim.x + fx.thread_idx.x
        if idx < n:
            Y[idx] = alpha * X[idx] + Y[idx]

    @flyc.jit
    def axpy(X: fx.Pointer, Y: fx.Pointer, n: fx.Int32, alpha: fx.Float32):
        axpy_f32(X, Y, n, alpha).launch(grid=((n + 255) // 256, 1, 1), block=(256, 1, 1))

    return axpy


@pytest.mark.l1a_compile_no_target_dialect
def test_iluvatar_aot_runtime_archive_is_built(monkeypatch):
    """The backend names an archive that the build actually produced."""
    _require_imports()
    _use_iluvatar(monkeypatch)

    from flydsl.compiler.backends import _get_backend_class
    from flydsl.runtime._libraries import _find_aot_runtime_archive

    config = _get_backend_class().aot_runtime_config()
    assert config.archive_basename == "libfly_iluvatar_aot_runtime.a"
    assert config.runtime_libraries == ("libcuda.so",)
    assert "-lcuda" in config.linker_flags
    assert not any("amdhip64" in flag for flag in config.linker_flags)

    archive = _find_aot_runtime_archive("iluvatar")
    assert archive is not None and Path(archive).is_file()


@pytest.mark.l2_device
@pytest.mark.skipif(_CC is None, reason="requires a C compiler to link the exported object")
def test_iluvatar_aot_export_links_and_launches(monkeypatch, tmp_path):
    """Export, link, load and launch, then check the result on device."""
    flyc, fx = _require_imports()
    torch = _require_torch()
    _use_iluvatar(monkeypatch)

    from flydsl.compiler.backends import _get_backend_class
    from flydsl.utils._elf import _dynamic_info

    def ptr(value=0):
        return flyc.from_c_void_p(fx.Float32, value)

    n = 1024
    compiled = flyc.compile(_axpy_launcher(flyc, fx), ptr(), ptr(), n, 1.0)
    assert compiled.export_to_c(tmp_path, "ilu_axpy", "ilu_axpy") is None
    obj = tmp_path / "ilu_axpy.o"
    assert obj.stat().st_size > 0

    shared = tmp_path / "libilu_axpy.so"
    flags = list(_get_backend_class().aot_runtime_config().linker_flags)
    subprocess.run([_CC, "-shared", "-o", str(shared), str(obj), *flags], check=True)

    needed = _dynamic_info(shared).needed
    assert any(name.startswith("libcuda.so") for name in needed), needed
    assert not any("fly_iluvatar_jit_runtime" in name for name in needed), needed
    # The embedded runtime must not drag a C++ runtime into the deployment.
    assert not any(name.startswith("libstdc++") for name in needed), needed

    lib = ctypes.CDLL(str(shared))
    init, load, unload = lib.ilu_axpy__module_init, lib.ilu_axpy__module_load, lib.ilu_axpy__module_unload
    init.restype = unload.restype = ctypes.c_int32
    load.argtypes, load.restype = [ctypes.c_int32], ctypes.c_int32
    entry = lib.ilu_axpy
    entry.argtypes, entry.restype = [ctypes.c_void_p], ctypes.c_int32

    X = torch.arange(n, dtype=torch.float32, device="cuda")
    Y = torch.ones(n, dtype=torch.float32, device="cuda")
    expected = 2.0 * X + Y

    assert init() == 0
    assert load(torch.cuda.current_device()) == 0
    try:
        # The exporter appends an implicit stream slot when the launcher
        # declares no stream argument.
        slots = [
            ctypes.c_void_p(X.data_ptr()),
            ctypes.c_void_p(Y.data_ptr()),
            ctypes.c_int32(n),
            ctypes.c_float(2.0),
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
        ]
        assert len(slots) == len(compiled._get_aot_compiled()._abi)
        args = (ctypes.c_void_p * len(slots))(*[ctypes.addressof(slot) for slot in slots])

        assert entry(ctypes.cast(args, ctypes.c_void_p)) == 0
        torch.cuda.synchronize()
        assert torch.allclose(Y, expected)

        # A second call reuses the cached function handle.
        assert entry(ctypes.cast(args, ctypes.c_void_p)) == 0
        torch.cuda.synchronize()
        assert torch.allclose(Y, 2.0 * X + expected)
    finally:
        assert unload() == 0
    # Unload is idempotent, and the module can be initialized and loaded again.
    assert unload() == 0
    assert init() == 0
    assert load(torch.cuda.current_device()) == 0
    assert unload() == 0
