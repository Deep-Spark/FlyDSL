# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

import pytest

from flydsl._mlir import ir
from flydsl.compiler.diagnostics import location_chain
from flydsl.compiler.kernel_function import func_def_location
from flydsl.expr import meta

pytestmark = [pytest.mark.l0_backend_agnostic]


def _capture_from_helper():
    return meta.capture_user_location()


@pytest.mark.parametrize("value", [None, "0"])
def test_disabled_source_locations_skip_stack_inspection(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("FLYDSL_DEBUG_ENABLE_DEBUG_INFO", raising=False)
    else:
        monkeypatch.setenv("FLYDSL_DEBUG_ENABLE_DEBUG_INFO", value)

    def unexpected_inspection(*args, **kwargs):
        raise AssertionError("debug info is disabled; frames must not be inspected")

    with ir.Context(), ir.Location.unknown():
        with monkeypatch.context() as patch:
            patch.setattr(meta.inspect, "currentframe", unexpected_inspection)
            patch.setattr(meta.inspect, "getframeinfo", unexpected_inspection)
            assert str(meta.capture_user_location()) == "loc(unknown)"

        chain = location_chain(func_def_location(_capture_from_helper))
        assert len(chain) == 1
        assert chain[0].filename == __file__
        assert chain[0].line == _capture_from_helper.__code__.co_firstlineno


def test_enabled_source_locations_preserve_call_chain(monkeypatch):
    monkeypatch.setenv("FLYDSL_DEBUG_ENABLE_DEBUG_INFO", "1")
    monkeypatch.setenv("FLYDSL_DEBUG_MAX_LOC_DEPTH", "5")

    with ir.Context(), meta.tracing_context(test_enabled_source_locations_preserve_call_chain):
        chain = location_chain(_capture_from_helper())

    assert len(chain) == 2
    assert all(frame.filename == __file__ for frame in chain)
    assert chain[0].line == _capture_from_helper.__code__.co_firstlineno + 1
    assert chain[-1].line > test_enabled_source_locations_preserve_call_chain.__code__.co_firstlineno
