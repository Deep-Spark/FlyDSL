# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Unit tests for ``pytest_perf`` handler perf_config materialization."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l0_backend_agnostic]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_pytest_perf():
    scripts_dir = _REPO_ROOT / ".github" / "scripts"
    scripts_path = str(scripts_dir)
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    # Drop a stale partial import from a previous failed load in the same process.
    for name in list(sys.modules):
        if name == "perf_parsers" or name.startswith("perf_parsers."):
            del sys.modules[name]
    from perf_parsers import pytest_perf as loaded  # type: ignore

    return loaded


def test_env_with_perf_config_writes_file_and_sets_env(tmp_path):
    mod = _load_pytest_perf()
    cfg = {
        "shape": {"B": 1, "H": 2, "Hkv": 1, "Sq": 64, "Skv": 64, "D": 64},
        "cases": [
            {
                "name": "causal",
                "is_causal": True,
                "window_size": None,
                "softcap": None,
                "dtype": "bf16",
            }
        ],
    }
    env = mod.env_with_perf_config(
        {"PATH": "/usr/bin", mod.PERF_CONFIG_ENV: "/stale/path.json"},
        repo_root=tmp_path,
        case_id="iluvatar_flex_attention",
        params={"perf_config": cfg},
    )
    # Env must stay repo-relative so docker (-w /workspace) can open the file.
    rel = env[mod.PERF_CONFIG_ENV]
    assert rel == ".perf/iluvatar_flex_attention.perf_config.json"
    path = tmp_path / rel
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == cfg


def test_repo_relative_perf_config_path_falls_back_outside_repo(tmp_path):
    mod = _load_pytest_perf()
    outside = Path("/tmp/not-under-repo.json")
    assert mod.repo_relative_perf_config_path(tmp_path, outside) == str(outside.resolve())


def test_env_with_perf_config_clears_env_without_mapping(tmp_path):
    mod = _load_pytest_perf()
    env = mod.env_with_perf_config(
        {mod.PERF_CONFIG_ENV: "/stale/path.json", "KEEP": "1"},
        repo_root=tmp_path,
        case_id="other",
        params={"extra_args": ["-k", "x"]},
    )
    assert mod.PERF_CONFIG_ENV not in env
    assert env["KEEP"] == "1"
    assert not (tmp_path / ".perf").exists()


def test_write_perf_config_file_sanitizes_case_id(tmp_path):
    mod = _load_pytest_perf()
    path = mod.write_perf_config_file(tmp_path, "a/b:c", {"warmup": 1})
    assert path.name == "a_b_c.perf_config.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"warmup": 1}
