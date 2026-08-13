# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Frontend coverage: memref subscript stores vs Tensor rebinding in dynamic CF."""

import pytest
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler import jit_function
from flydsl.expr import range_constexpr

pytestmark = [pytest.mark.l1b_target_dialect, pytest.mark.rocm_lower]


def _scf_op_headers(text, op_name):
    """Return printed ``scf.{op}`` op lines, skipping yield/condition."""
    skip = ("scf.yield", "scf.condition")
    headers = []
    for line in text.splitlines():
        stripped = line.strip()
        if op_name not in stripped or any(token in stripped for token in skip):
            continue
        headers.append(stripped)
    assert headers, f"no {op_name} op header in:\n{text}"
    return headers


def _trace_kernel(monkeypatch, launch, *args, cache_key):
    captured = []

    def capture_compile(cls, module, **_kwargs):
        captured.append(str(module))
        return module

    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.setattr(jit_function, "_flydsl_key", lambda: cache_key)
    monkeypatch.setattr(jit_function.MlirCompiler, "compile", classmethod(capture_compile))
    launch(*args)
    assert captured, "kernel tracing produced no module"
    return "\n".join(captured)


def test_subscript_store_is_not_control_flow_state(monkeypatch):
    """A memref element store must not become an scf result or iter_arg."""
    size = 16

    @flyc.kernel
    def kernel(Out: fx.Tensor, n: fx.Int32):
        tid = fx.thread_idx.x
        scratch = fx.SharedAllocator().allocate(fx.Array[fx.Int32, size]).peek()
        ptr = scratch.ptr
        values = scratch.view(fx.make_layout(size, 1))

        total = fx.Int32(0)
        for i in range(n):
            if tid == fx.Int32(0):
                # Exercise all three reference-semantics opt-ins directly:
                # Array, Pointer, and Tensor.
                scratch[i] = fx.Int32(i)
                ptr[i] = fx.Int32(i)
                values[i] = fx.Int32(i) + fx.Int32(1)
            total = total + fx.Int32(1)

        checksum = total
        for i in range_constexpr(size):
            checksum = checksum + values[i]
        Out[0] = checksum

    @flyc.jit
    def launch(Out: fx.Tensor, n: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        kernel(Out, n).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream.value)

    out = torch.empty(1, dtype=torch.int32)
    text = _trace_kernel(monkeypatch, launch, out, fx.Int32(size), cache_key="test-subscript-store-key")
    headers = _scf_op_headers(text, "scf.if") + _scf_op_headers(text, "scf.for")
    assert all("memref" not in header for header in headers)
    assert all("!fly.ptr" not in header for header in headers)


def test_tensor_rebinding_remains_control_flow_state(monkeypatch):
    """A real Tensor name assignment must remain an scf result."""

    @flyc.kernel
    def kernel(Out: fx.Tensor, flag: fx.Int32):
        buf = fx.make_rmem_tensor(4, fx.Int32)
        replacement = fx.make_rmem_tensor(4, fx.Int32)
        if flag > fx.Int32(0):
            buf = replacement
        Out[0] = buf[0]

    @flyc.jit
    def launch(Out: fx.Tensor, flag: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        kernel(Out, flag).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream.value)

    out = torch.empty(1, dtype=torch.int32)
    text = _trace_kernel(monkeypatch, launch, out, fx.Int32(1), cache_key="test-tensor-rebind-key")
    assert any("memref" in header for header in _scf_op_headers(text, "scf.if"))
