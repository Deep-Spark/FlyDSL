# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Release-wheel IR/ISA dump guard."""

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _REPO_ROOT / "python"
# conftest prepends build-fly/python_packages; prefer in-tree sources here.
sys.path.insert(0, str(_PYTHON_DIR))
for _name in list(sys.modules):
    if _name == "flydsl" or _name.startswith("flydsl."):
        del sys.modules[_name]

from flydsl.utils.release_guard import (  # noqa: E402
    assert_ir_dump_allowed,
    clear_ir_dump_allowed_cache,
    ir_dump_allowed,
)

pytestmark = [pytest.mark.l0_backend_agnostic]


@pytest.fixture(autouse=True)
def _clear_guard_cache():
    clear_ir_dump_allowed_cache()
    yield
    clear_ir_dump_allowed_cache()


def test_ir_dump_allowed_with_explicit_override(monkeypatch):
    monkeypatch.setenv("FLYDSL_ALLOW_IR_DUMP", "1")
    monkeypatch.setattr("flydsl.utils.release_guard._running_from_source_tree", lambda: False)
    assert ir_dump_allowed() is True
    assert_ir_dump_allowed(feature="FLYDSL_DUMP_IR")


def test_ir_dump_denied_for_packaged_wheel_install(monkeypatch):
    monkeypatch.delenv("FLYDSL_ALLOW_IR_DUMP", raising=False)
    monkeypatch.setattr("flydsl.utils.release_guard._running_from_source_tree", lambda: False)

    class _FakeDist:
        def read_text(self, name):
            raise FileNotFoundError(name)

    def _fake_distribution(name):
        assert name == "flydsl"
        return _FakeDist()

    monkeypatch.setattr("importlib.metadata.distribution", _fake_distribution)
    assert ir_dump_allowed() is False
    with pytest.raises(RuntimeError, match="disabled for packaged FlyDSL installs"):
        assert_ir_dump_allowed(feature="FLYDSL_DUMP_IR")


def test_ir_dump_allowed_for_editable_install(monkeypatch):
    monkeypatch.delenv("FLYDSL_ALLOW_IR_DUMP", raising=False)
    monkeypatch.setattr("flydsl.utils.release_guard._running_from_source_tree", lambda: False)

    class _FakeDist:
        def read_text(self, name):
            assert name == "direct_url.json"
            return json.dumps({"dir_info": {"editable": True}, "url": "file:///tmp/FlyDSL"})

    monkeypatch.setattr("importlib.metadata.distribution", lambda name: _FakeDist())
    assert ir_dump_allowed() is True


def test_ir_dump_allowed_from_source_tree(monkeypatch):
    monkeypatch.delenv("FLYDSL_ALLOW_IR_DUMP", raising=False)
    assert ir_dump_allowed() is True
