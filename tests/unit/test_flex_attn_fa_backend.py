# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Unit tests for flex-attention FA fast-path eligibility."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l0_backend_agnostic]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_fa_backend():
    path = _REPO_ROOT / "kernels" / "attention" / "iluvatar" / "flex_attn_fa_backend.py"
    spec = importlib.util.spec_from_file_location("flex_attn_fa_backend_isolated", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _eligible(mod, **overrides):
    kwargs = dict(
        mode="dense",
        dtype="bf16",
        head_dim=128,
        causal=True,
        sq=64,
        skv=64,
        window_size=None,
        softcap=None,
        alibi_slopes=None,
        score_bias=None,
        score_mod=None,
        mask_mod=None,
        block_mask=None,
        block_masks=None,
        tile_config_explicit=False,
        varlen_tight=True,
        env_enabled=True,
    )
    kwargs.update(overrides)
    return mod.is_flex_fa_fastpath_eligible(**kwargs)


def test_flex_fa_fastpath_vanilla_eligible():
    mod = _load_fa_backend()
    assert _eligible(mod)
    assert _eligible(mod, head_dim=256)
    assert _eligible(mod, mode="varlen", sq=0, skv=0)
    assert _eligible(mod, mode="paged", sq=1, skv=128, causal=True)


def test_flex_fa_fastpath_rejects_non_subset():
    mod = _load_fa_backend()
    assert not _eligible(mod, env_enabled=False)
    assert not _eligible(mod, tile_config_explicit=True)
    assert not _eligible(mod, dtype="f16")
    assert not _eligible(mod, head_dim=64)
    assert not _eligible(mod, window_size=1024)
    assert not _eligible(mod, softcap=30.0)
    assert not _eligible(mod, score_mod=object())
    assert not _eligible(mod, mask_mod=object())
    assert not _eligible(mod, block_mask=object())
    assert not _eligible(mod, alibi_slopes=object())
    assert not _eligible(mod, score_bias=object())
    assert not _eligible(mod, causal=True, sq=64, skv=128)
    assert not _eligible(mod, mode="varlen", varlen_tight=False)


def test_flex_fa_fastpath_env_default_on(monkeypatch):
    mod = _load_fa_backend()
    monkeypatch.delenv("FLYDSL_FLEX_FA_FASTPATH", raising=False)
    assert mod.flex_fa_fastpath_enabled() is True
    monkeypatch.setenv("FLYDSL_FLEX_FA_FASTPATH", "0")
    assert mod.flex_fa_fastpath_enabled() is False
    monkeypatch.setenv("FLYDSL_FLEX_FA_FASTPATH", "false")
    assert mod.flex_fa_fastpath_enabled() is False
    monkeypatch.setenv("FLYDSL_FLEX_FA_FASTPATH", "1")
    assert mod.flex_fa_fastpath_enabled() is True
