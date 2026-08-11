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

from flydsl.utils.env import DebugEnvManager  # noqa: E402
from flydsl.utils.release_guard import (  # noqa: E402
    assert_ir_dump_allowed,
    assert_isa_format_allowed,
    assert_passmanager_parse_allowed,
    authorize_pass_pipeline,
    clear_ir_dump_allowed_cache,
    ir_dump_allowed,
    is_packaged_install,
)

pytestmark = [pytest.mark.l0_backend_agnostic]


@pytest.fixture(autouse=True)
def _clear_guard_cache():
    clear_ir_dump_allowed_cache()
    yield
    clear_ir_dump_allowed_cache()


def _fake_packaged_dist():
    class _FakeDist:
        def read_text(self, name):
            raise FileNotFoundError(name)

    return _FakeDist()


def test_ir_dump_denied_even_with_allow_env(monkeypatch):
    monkeypatch.setenv("FLYDSL_ALLOW_IR_DUMP", "1")
    monkeypatch.setattr("flydsl.utils.release_guard._running_from_source_tree", lambda: False)
    monkeypatch.setattr("importlib.metadata.distribution", lambda name: _fake_packaged_dist())
    assert is_packaged_install() is True
    assert ir_dump_allowed() is False
    with pytest.raises(RuntimeError, match="disabled for packaged FlyDSL installs"):
        assert_ir_dump_allowed(feature="FLYDSL_DUMP_IR")


def test_ir_dump_denied_for_packaged_wheel_install(monkeypatch):
    monkeypatch.delenv("FLYDSL_ALLOW_IR_DUMP", raising=False)
    monkeypatch.setattr("flydsl.utils.release_guard._running_from_source_tree", lambda: False)
    monkeypatch.setattr("importlib.metadata.distribution", lambda name: _fake_packaged_dist())
    assert ir_dump_allowed() is False
    with pytest.raises(RuntimeError, match="disabled for packaged FlyDSL installs"):
        assert_ir_dump_allowed(feature="FLYDSL_DUMP_IR")


def test_packaged_debug_env_forced_false(monkeypatch):
    monkeypatch.setenv("FLYDSL_DUMP_IR", "1")
    monkeypatch.setenv("FLYDSL_DEBUG_DUMP_ASM", "1")
    monkeypatch.setenv("FLYDSL_DEBUG_PRINT_ORIGIN_IR", "1")
    monkeypatch.setenv("FLYDSL_DEBUG_PRINT_AFTER_ALL", "1")
    monkeypatch.setattr("flydsl.utils.release_guard._running_from_source_tree", lambda: False)
    monkeypatch.setattr("importlib.metadata.distribution", lambda name: _fake_packaged_dist())
    dbg = DebugEnvManager()
    assert dbg.dump_ir is False
    assert dbg.dump_asm is False
    assert dbg.print_origin_ir is False
    assert dbg.print_after_all is False


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


def test_dump_support_false_blocks_even_when_guard_allows(monkeypatch):
    from flydsl.utils.dump_support import require_dump_support

    monkeypatch.setattr("flydsl.utils.dump_support.DUMP_SUPPORT", False)
    monkeypatch.setattr("flydsl.utils.release_guard.ir_dump_allowed", lambda: True)
    with pytest.raises(RuntimeError, match="DUMP_SUPPORT=False"):
        require_dump_support(feature="FLYDSL_DUMP_IR")


def test_official_dump_write_sites_share_packaged_guard(monkeypatch):
    """Official dump writers all call assert_ir_dump_allowed before writing."""
    monkeypatch.setattr("flydsl.utils.release_guard._running_from_source_tree", lambda: False)
    monkeypatch.setattr("importlib.metadata.distribution", lambda name: _fake_packaged_dist())
    for feature in (
        "FLYDSL_DUMP_IR",
        "FLYDSL_DUMP_IR (ISA dump)",
        "FLYDSL_DUMP_IR (LLVM IR dump)",
        "FLYDSL_DUMP_IR (external LLVM dump)",
    ):
        with pytest.raises(RuntimeError, match="disabled for packaged FlyDSL installs"):
            assert_ir_dump_allowed(feature=feature)


def test_passmanager_parse_denied_without_authorization(monkeypatch):
    monkeypatch.setattr("flydsl.utils.release_guard._running_from_source_tree", lambda: False)
    monkeypatch.setattr("importlib.metadata.distribution", lambda name: _fake_packaged_dist())
    with pytest.raises(RuntimeError, match="PassManager.parse is disabled"):
        assert_passmanager_parse_allowed()
    with authorize_pass_pipeline():
        assert_passmanager_parse_allowed()


def test_format_isa_denied_for_packaged_or_stripped(monkeypatch):
    monkeypatch.setattr("flydsl.utils.release_guard._running_from_source_tree", lambda: False)
    monkeypatch.setattr("importlib.metadata.distribution", lambda name: _fake_packaged_dist())
    with pytest.raises(RuntimeError, match="format=isa"):
        assert_isa_format_allowed()

    clear_ir_dump_allowed_cache()
    monkeypatch.setattr("flydsl.utils.release_guard._running_from_source_tree", lambda: True)
    monkeypatch.setattr("flydsl.utils.dump_support.DUMP_SUPPORT", False)
    with pytest.raises(RuntimeError, match="format=isa"):
        assert_isa_format_allowed()
