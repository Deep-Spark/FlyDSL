# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Unit tests for flex-attention perf_config load / parse helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l0_backend_agnostic]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_flex_perf_helpers():
    """Load only the helpers from the device test module (no GPU required)."""
    path = _REPO_ROOT / "tests" / "kernels" / "test_iluvatar_flex_attention_device.py"
    spec = importlib.util.spec_from_file_location("flex_attn_device_test_helpers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Avoid collecting the module as a pytest test file via this alternate name.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_flex_attn_perf_config_fallback(monkeypatch):
    mod = _load_flex_perf_helpers()
    monkeypatch.delenv(mod._FLEX_ATTN_PERF_CONFIG_ENV, raising=False)
    cfg = mod._load_flex_attn_perf_config()
    assert cfg["shape"]["Sq"] == 4096
    assert len(cfg["cases"]) == 6
    keys = [f"{c['name']}.{c['dtype']}" for c in cfg["cases"]]
    assert keys == [
        "causal.bf16",
        "causal.f16",
        "causal_swa1024.bf16",
        "causal_swa1024.f16",
        "causal_softcap30.bf16",
        "causal_softcap30.f16",
    ]


def test_load_flex_attn_perf_config_from_path(monkeypatch, tmp_path):
    mod = _load_flex_perf_helpers()
    payload = {
        "shape": {"B": 1, "H": 4, "Hkv": 2, "Sq": 128, "Skv": 128, "D": 64},
        "warmup": 2,
        "iters": 3,
        "cases": [
            {
                "name": "causal",
                "is_causal": True,
                "window_size": None,
                "softcap": None,
                "dtype": "f16",
            }
        ],
    }
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv(mod._FLEX_ATTN_PERF_CONFIG_ENV, str(path))
    cfg = mod._load_flex_attn_perf_config()
    B, H, Hkv, Sq, Skv, D, cases = mod._parse_flex_attn_perf_cases(cfg)
    assert (B, H, Hkv, Sq, Skv, D) == (1, 4, 2, 128, 128, 64)
    assert cases == [("causal", True, None, None, "f16")]
    assert int(cfg["warmup"]) == 2
    assert int(cfg["iters"]) == 3
