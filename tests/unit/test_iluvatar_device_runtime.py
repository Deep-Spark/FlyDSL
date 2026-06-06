# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar device runtime pairing."""

import pytest

import flydsl.runtime.device_runtime as dr

pytestmark = [pytest.mark.l0_backend_agnostic]


@pytest.fixture(autouse=True)
def _reset_device_runtime_singleton():
    """Each test starts without a cached DeviceRuntime instance."""
    dr._instance = None
    dr._runtime_cls_override = None
    dr._EXTRA_MAPPINGS.clear()
    yield
    dr._instance = None
    dr._runtime_cls_override = None
    dr._EXTRA_MAPPINGS.clear()


def test_iluvatar_runtime_kind_matches_compile_backend(monkeypatch):
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    rt = dr.get_device_runtime()
    assert rt.kind == "iluvatar"
    assert isinstance(rt, dr.IluvatarDeviceRuntime)
    dr.ensure_compile_runtime_compatible("iluvatar", runtime=rt)


def test_iluvatar_pairing_from_env_no_singleton(monkeypatch):
    """Iluvatar pairing check should not instantiate the runtime."""
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    dr.ensure_compile_runtime_pairing_from_env("iluvatar")
    assert dr._instance is None


def test_iluvatar_runtime_kind_mismatch_raises(monkeypatch):
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "rocm")
    with pytest.raises(RuntimeError, match="requires device runtime kind"):
        dr.get_device_runtime()
