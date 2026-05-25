# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Opt-in smoke tests for the Iluvatar JIT runtime library."""

import os
from ctypes import CDLL, c_char_p, c_int32, c_size_t, c_void_p, create_string_buffer
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l2_device]


def _required_path_from_env(name: str) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        pytest.skip(f"{name} is not set")

    path = Path(value)
    if not path.is_file():
        pytest.fail(f"{name} does not point to a file: {path}")
    return path


def _required_value_from_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        pytest.skip(f"{name} is not set")
    return value


def _load_runtime_and_blob():
    runtime_path = _required_path_from_env("FLYDSL_ILUVATAR_JIT_RUNTIME_LIB")
    blob_path = _required_path_from_env("FLYDSL_ILUVATAR_SMOKE_BLOB")

    runtime = CDLL(str(runtime_path))
    runtime.mgpuModuleLoad.argtypes = [c_void_p, c_size_t]
    runtime.mgpuModuleLoad.restype = c_void_p
    runtime.mgpuModuleUnload.argtypes = [c_void_p]
    runtime.mgpuModuleUnload.restype = None

    blob = blob_path.read_bytes()
    if not blob:
        pytest.fail(f"FLYDSL_ILUVATAR_SMOKE_BLOB is empty: {blob_path}")

    return runtime, create_string_buffer(blob), len(blob)


def test_iluvatar_runtime_loads_and_unloads_module():
    runtime, buffer, blob_size = _load_runtime_and_blob()

    module = runtime.mgpuModuleLoad(buffer, blob_size)
    assert module, "mgpuModuleLoad returned a null module handle"

    runtime.mgpuModuleUnload(module)


def test_iluvatar_runtime_gets_function():
    kernel_name = _required_value_from_env("FLYDSL_ILUVATAR_SMOKE_KERNEL")
    runtime, buffer, blob_size = _load_runtime_and_blob()
    runtime.mgpuModuleGetFunction.argtypes = [c_void_p, c_char_p]
    runtime.mgpuModuleGetFunction.restype = c_void_p

    module = runtime.mgpuModuleLoad(buffer, blob_size)
    assert module, "mgpuModuleLoad returned a null module handle"
    try:
        function = runtime.mgpuModuleGetFunction(module, kernel_name.encode())
        assert function, f"mgpuModuleGetFunction returned null for {kernel_name!r}"
    finally:
        runtime.mgpuModuleUnload(module)


def test_iluvatar_runtime_launches_noarg_kernel():
    kernel_name = _required_value_from_env("FLYDSL_ILUVATAR_LAUNCH_KERNEL")
    runtime, buffer, blob_size = _load_runtime_and_blob()
    runtime.mgpuModuleGetFunction.argtypes = [c_void_p, c_char_p]
    runtime.mgpuModuleGetFunction.restype = c_void_p
    runtime.mgpuStreamCreate.argtypes = []
    runtime.mgpuStreamCreate.restype = c_void_p
    runtime.mgpuStreamSynchronize.argtypes = [c_void_p]
    runtime.mgpuStreamSynchronize.restype = None
    runtime.mgpuStreamDestroy.argtypes = [c_void_p]
    runtime.mgpuStreamDestroy.restype = None
    runtime.mgpuLaunchKernel.argtypes = [
        c_void_p,
        c_size_t,
        c_size_t,
        c_size_t,
        c_size_t,
        c_size_t,
        c_size_t,
        c_int32,
        c_void_p,
        c_void_p,
        c_void_p,
        c_size_t,
    ]
    runtime.mgpuLaunchKernel.restype = None

    module = runtime.mgpuModuleLoad(buffer, blob_size)
    assert module, "mgpuModuleLoad returned a null module handle"
    stream = None
    try:
        function = runtime.mgpuModuleGetFunction(module, kernel_name.encode())
        assert function, f"mgpuModuleGetFunction returned null for {kernel_name!r}"
        stream = runtime.mgpuStreamCreate()
        assert stream, "mgpuStreamCreate returned a null stream handle"

        runtime.mgpuLaunchKernel(
            function,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            stream,
            None,
            None,
            0,
        )
        runtime.mgpuStreamSynchronize(stream)
    finally:
        if stream:
            runtime.mgpuStreamDestroy(stream)
        runtime.mgpuModuleUnload(module)
