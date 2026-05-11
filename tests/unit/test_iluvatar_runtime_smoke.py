# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Opt-in smoke tests for the Iluvatar JIT runtime library."""

import os
from ctypes import CDLL, c_size_t, c_void_p, create_string_buffer
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


def test_iluvatar_runtime_loads_and_unloads_module():
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

    buffer = create_string_buffer(blob)
    module = runtime.mgpuModuleLoad(buffer, len(blob))
    assert module, "mgpuModuleLoad returned a null module handle"

    runtime.mgpuModuleUnload(module)
