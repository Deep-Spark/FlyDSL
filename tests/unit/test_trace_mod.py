# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Unit tests for ``flydsl.expr.trace_mod`` (V3-1 tracing)."""

from __future__ import annotations

import math

import pytest

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler import jit_function

pytestmark = [pytest.mark.l0_backend_agnostic]


def test_trace_score_mod_eval_host_alibi():
    slope = 0.25

    @fx.trace_score_mod
    def alibi(score, batch, head, q_idx, kv_idx):
        return score - slope * (q_idx - kv_idx)

    assert isinstance(alibi, fx.TracedScoreMod)
    assert alibi.fingerprint
    got = alibi.eval_host(1.0, 0, 0, 10, 3)
    assert got == pytest.approx(1.0 - slope * (10 - 3))


def test_trace_mask_mod_eval_host_causal():
    @fx.trace_mask_mod
    def causal(batch, head, q_idx, kv_idx):
        return kv_idx <= q_idx

    assert isinstance(causal, fx.TracedMaskMod)
    assert causal.eval_host(0, 0, 5, 5) is True
    assert causal.eval_host(0, 0, 5, 6) is False


def test_where_host_and_score_mod():
    assert fx.where(True, 1.5, -9.0) == 1.5
    assert fx.where(False, 1.5, -9.0) == -9.0

    @fx.trace_score_mod
    def capped(score, batch, head, q_idx, kv_idx):
        return fx.where(score > 0.0, score, 0.0)

    assert capped.eval_host(2.0, 0, 0, 0, 0) == pytest.approx(2.0)
    assert capped.eval_host(-1.0, 0, 0, 0, 0) == pytest.approx(0.0)


def test_tanh_host_via_fx():
    @fx.trace_score_mod
    def soft(score, batch, head, q_idx, kv_idx):
        return fx.tanh(score / 30.0) * 30.0

    assert soft.eval_host(3.0, 0, 0, 0, 0) == pytest.approx(math.tanh(3.0 / 30.0) * 30.0)


def test_fingerprint_changes_with_body_or_closure():
    a = 1.0

    @fx.trace_score_mod
    def m1(score, batch, head, q_idx, kv_idx):
        return score + a

    @fx.trace_score_mod
    def m2(score, batch, head, q_idx, kv_idx):
        return score + a + 1.0

    b = 2.0

    @fx.trace_score_mod
    def m3(score, batch, head, q_idx, kv_idx):
        return score + b

    assert m1.fingerprint != m2.fingerprint
    assert m1.fingerprint != m3.fingerprint

    @fx.trace_score_mod
    def m1b(score, batch, head, q_idx, kv_idx):
        return score + a

    assert m1.fingerprint == m1b.fingerprint


def test_rejects_bad_signature():
    with pytest.raises(ValueError, match="parameter names"):

        @fx.trace_score_mod
        def bad(score, b, h, q, k):
            return score


def test_rejects_control_flow():
    with pytest.raises(ValueError, match="control-flow|ternary|If"):

        @fx.trace_score_mod
        def bad(score, batch, head, q_idx, kv_idx):
            if score > 0:
                return score
            return 0.0


def test_rejects_python_ternary():
    with pytest.raises(ValueError, match="ternary|where"):

        @fx.trace_score_mod
        def bad(score, batch, head, q_idx, kv_idx):
            return score if score > 0 else 0.0


def test_rejects_non_whitelist_call():
    with pytest.raises(ValueError, match="whitelist"):

        @fx.trace_score_mod
        def bad(score, batch, head, q_idx, kv_idx):
            return fx.exp(score)


def test_rejects_non_scalar_closure():
    table = [1.0, 2.0]

    with pytest.raises(ValueError, match="closure"):

        @fx.trace_score_mod
        def bad(score, batch, head, q_idx, kv_idx):
            return score + table[0]


def test_rejects_mask_wrong_arity():
    with pytest.raises(ValueError, match="expects 4"):

        @fx.trace_mask_mod
        def bad(batch, head, q_idx):
            return True


@pytest.fixture
def frontend_only_jit(monkeypatch):
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "rocm")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "rocm")
    monkeypatch.setenv("ARCH", "gfx942")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setattr(jit_function, "_flydsl_key", lambda: "test-flydsl-key")

    def compile_noop(cls, module, **_kwargs):
        return module

    monkeypatch.setattr(jit_function.MlirCompiler, "compile", classmethod(compile_noop))


def test_apply_inlines_in_non_attn_kernel(frontend_only_jit):
    """Non-attn consumer: ``mod.apply`` must be usable inside ``@flyc.kernel``."""

    @fx.trace_score_mod
    def alibi(score, batch, head, q_idx, kv_idx):
        return score - 0.5 * (q_idx - kv_idx)

    @flyc.kernel
    def consumer(score: fx.Float32, q_idx: fx.Int32, kv_idx: fx.Int32):
        out = alibi.apply(score, fx.Int32(0), fx.Int32(0), q_idx, kv_idx)
        fx.printf("trace_mod_consumer out={}", out)

    @flyc.jit
    def launch(
        score: fx.Float32,
        q_idx: fx.Int32,
        kv_idx: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        consumer(score, q_idx, kv_idx).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)

    # Force frontend compile / trace (MlirCompiler.compile patched to noop).
    launch(2.0, 8, 3)

    assert alibi.eval_host(2.0, 0, 0, 8, 3) == pytest.approx(2.0 - 0.5 * (8 - 3))
