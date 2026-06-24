# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Unit tests for scripts/ci/should_run_device.py policy decisions."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l0_backend_agnostic]


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "ci" / "should_run_device.py"
    spec = importlib.util.spec_from_file_location("should_run_device", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_config():
    return {
        "device": {
            "force_label": "run-device-ci",
            "required_skip_message": "Skipped by policy: no matched paths and no run-device-ci label.",
            "path_filters": ["kernels/**", "python/flydsl/compiler/**"],
        }
    }


def test_workflow_dispatch_force_run_true_always_runs():
    module = _load_module()
    should_run, reason, changed, matched = module._decide(  # pylint: disable=protected-access
        event_name="workflow_dispatch",
        event_payload={"inputs": {"force_run": True}},
        config=_base_config(),
        repository="deepspark/FlyDSL",
        token="dummy",
    )
    assert should_run is True
    assert "force_run=true" in reason
    assert changed == []
    assert matched == []


def test_workflow_dispatch_force_run_false_skips():
    module = _load_module()
    should_run, reason, _, _ = module._decide(  # pylint: disable=protected-access
        event_name="workflow_dispatch",
        event_payload={"inputs": {"force_run": False}},
        config=_base_config(),
        repository="deepspark/FlyDSL",
        token="dummy",
    )
    assert should_run is False
    assert "force_run=false" in reason


def test_workflow_run_label_forces_run(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        module,
        "_fetch_pull_request",
        lambda *_args, **_kwargs: {"head": {"repo": {"fork": False}}},
    )
    monkeypatch.setattr(module, "_fetch_pull_request_labels", lambda *_args, **_kwargs: ["run-device-ci"])
    monkeypatch.setattr(module, "_fetch_pull_request_files", lambda *_args, **_kwargs: ["docs/readme.md"])

    should_run, reason, _, _ = module._decide(  # pylint: disable=protected-access
        event_name="workflow_run",
        event_payload={"workflow_run": {"pull_requests": [{"number": 123}]}},
        config=_base_config(),
        repository="deepspark/FlyDSL",
        token="dummy",
    )
    assert should_run is True
    assert "label" in reason


def test_workflow_run_fork_pr_skips(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        module,
        "_fetch_pull_request",
        lambda *_args, **_kwargs: {"head": {"repo": {"fork": True}}},
    )
    monkeypatch.setattr(module, "_fetch_pull_request_labels", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(module, "_fetch_pull_request_files", lambda *_args, **_kwargs: ["kernels/x.py"])

    should_run, reason, _, _ = module._decide(  # pylint: disable=protected-access
        event_name="workflow_run",
        event_payload={"workflow_run": {"pull_requests": [{"number": 7}]}},
        config=_base_config(),
        repository="deepspark/FlyDSL",
        token="dummy",
    )
    assert should_run is False
    assert "fork pull_request" in reason


def test_workflow_run_path_match_triggers(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(
        module,
        "_fetch_pull_request",
        lambda *_args, **_kwargs: {"head": {"repo": {"fork": False}}},
    )
    monkeypatch.setattr(module, "_fetch_pull_request_labels", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        module,
        "_fetch_pull_request_files",
        lambda *_args, **_kwargs: ["python/flydsl/compiler/foo.py", "docs/readme.md"],
    )

    should_run, reason, changed, matched = module._decide(  # pylint: disable=protected-access
        event_name="workflow_run",
        event_payload={"workflow_run": {"pull_requests": [{"number": 9}]}},
        config=_base_config(),
        repository="deepspark/FlyDSL",
        token="dummy",
    )
    assert should_run is True
    assert "path filter" in reason
    assert "python/flydsl/compiler/foo.py" in changed
    assert "python/flydsl/compiler/foo.py" in matched
