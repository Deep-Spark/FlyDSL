#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""MLIR-level tests: storage handles are shared, not SSA-carried, unless rebound."""

import pytest

import flydsl.expr as fx
from flydsl._mlir.dialects import arith, func
from flydsl._mlir.ir import Context, FunctionType, InsertionPoint, IntegerType, Location, Module
from flydsl.compiler.ast_rewriter import (
    CanonicalizeWhile,
    InsertEmptyYieldForSCFFor,
    ReplaceIfWithDispatch,
)
from flydsl.expr.numeric import Int32


def _i32(v):
    return Int32(arith.ConstantOp(IntegerType.get_signless(32), v).result)


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


def test_if_tensor_element_store_is_not_scf_result():
    """``buf[i] = v`` mutates storage; ``buf`` must not become an scf.if result."""
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with Location.unknown():
            module = Module.create()
            i1 = IntegerType.get_signless(1)
            with InsertionPoint(module.body):
                f = func.FuncOp("if_tensor_store", FunctionType.get([i1], []))
                entry = f.add_entry_block()
                with InsertionPoint(entry):
                    cond = entry.arguments[0]
                    buf = fx.make_rmem_tensor(4, fx.Int32)
                    acc = _i32(0)

                    def then_branch(names, branch_buf, branch_acc):
                        assert branch_buf is buf
                        branch_buf[0] = _i32(7)
                        return {"buf": branch_buf, "acc": _i32(1)}

                    def else_branch(names, branch_buf, branch_acc):
                        assert branch_buf is buf
                        return {"buf": branch_buf, "acc": branch_acc}

                    out_buf, out_acc = ReplaceIfWithDispatch.scf_if_dispatch(
                        cond,
                        then_branch,
                        else_branch,
                        result_names=("buf", "acc"),
                        result_values=(buf, acc),
                    )
                    assert out_buf is buf
                    assert isinstance(out_acc, Int32)
                    func.ReturnOp([])

            text = str(module)
            assert module.operation.verify()
            headers = _scf_op_headers(text, "scf.if")
            assert any("-> (i32)" in header for header in headers)
            assert all("memref" not in header for header in headers)


def test_for_tensor_element_store_is_not_iter_arg():
    """``buf[i] = v`` inside a dynamic for must not create a memref iter_arg."""
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with Location.unknown():
            module = Module.create()
            with InsertionPoint(module.body):
                f = func.FuncOp("for_tensor_store", FunctionType.get([], []))
                entry = f.add_entry_block()
                with InsertionPoint(entry):
                    buf = fx.make_rmem_tensor(4, fx.Int32)
                    acc = _i32(0)

                    def body(iv, names, branch_buf, branch_acc):
                        assert branch_buf is buf
                        branch_buf[0] = _i32(3)
                        return {"buf": branch_buf, "acc": branch_acc + _i32(1)}

                    out_buf, out_acc = InsertEmptyYieldForSCFFor.scf_for_dispatch(
                        _i32(0),
                        _i32(2),
                        _i32(1),
                        body,
                        result_names=("buf", "acc"),
                        result_values=(buf, acc),
                    )
                    assert out_buf is buf
                    assert isinstance(out_acc, Int32)
                    func.ReturnOp([])

            text = str(module)
            assert module.operation.verify()
            headers = _scf_op_headers(text, "scf.for")
            assert any("iter_args(" in header and "-> (i32)" in header for header in headers)
            assert all("memref" not in header for header in headers)


def test_if_rebound_tensor_remains_an_scf_result():
    """Reference semantics must not hide a real ``buf = replacement`` assignment."""
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with Location.unknown():
            module = Module.create()
            i1 = IntegerType.get_signless(1)
            with InsertionPoint(module.body):
                f = func.FuncOp("if_tensor_rebind", FunctionType.get([i1], []))
                entry = f.add_entry_block()
                with InsertionPoint(entry):
                    cond = entry.arguments[0]
                    buf = fx.make_rmem_tensor(4, fx.Int32)
                    replacement = fx.make_rmem_tensor(4, fx.Int32)

                    out = ReplaceIfWithDispatch.scf_if_dispatch(
                        cond,
                        lambda names, branch_buf: {"buf": replacement},
                        lambda names, branch_buf: {"buf": branch_buf},
                        result_names=("buf",),
                        result_values=(buf,),
                        rebound_names=("buf",),
                    )
                    assert out is not buf
                    func.ReturnOp([])

            text = str(module)
            assert module.operation.verify()
            headers = _scf_op_headers(text, "scf.if")
            assert any("memref" in header for header in headers)


def test_if_mutation_only_tensor_needs_no_scf_results():
    """Exercise the storage-only path where there are no SSA-carried values."""
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with Location.unknown():
            module = Module.create()
            i1 = IntegerType.get_signless(1)
            with InsertionPoint(module.body):
                f = func.FuncOp("if_tensor_store_only", FunctionType.get([i1], []))
                entry = f.add_entry_block()
                with InsertionPoint(entry):
                    cond = entry.arguments[0]
                    buf = fx.make_rmem_tensor(4, fx.Int32)

                    def then_branch(names, branch_buf):
                        branch_buf[0] = _i32(7)
                        return {"buf": branch_buf}

                    out = ReplaceIfWithDispatch.scf_if_dispatch(
                        cond,
                        then_branch,
                        result_names=("buf",),
                        result_values=(buf,),
                    )
                    assert out is buf
                    func.ReturnOp([])

            text = str(module)
            assert module.operation.verify()
            headers = _scf_op_headers(text, "scf.if")
            assert all("->" not in header for header in headers)


def test_while_tensor_element_store_is_not_loop_carried():
    """Storage filtering applies to while loops without dropping scalar state."""
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with Location.unknown():
            module = Module.create()
            with InsertionPoint(module.body):
                f = func.FuncOp("while_tensor_store", FunctionType.get([], []))
                entry = f.add_entry_block()
                with InsertionPoint(entry):
                    buf = fx.make_rmem_tensor(4, fx.Int32)
                    count = _i32(0)

                    def before(names, branch_buf, branch_count):
                        return branch_count < _i32(2)

                    def after(names, branch_buf, branch_count):
                        branch_buf[0] = branch_count
                        return {"buf": branch_buf, "count": branch_count + _i32(1)}

                    out_buf, out_count = CanonicalizeWhile.scf_while_dispatch(
                        before,
                        after,
                        result_names=("buf", "count"),
                        result_values=(buf, count),
                        rebound_names=("count",),
                    )
                    assert out_buf is buf
                    assert isinstance(out_count, Int32)
                    func.ReturnOp([])

            text = str(module)
            assert module.operation.verify()
            headers = _scf_op_headers(text, "scf.while")
            assert any("i32" in header for header in headers)
            assert all("memref" not in header for header in headers)


def test_while_mutation_only_tensor_needs_no_iter_args():
    """A while that only stores through a tensor must not carry that tensor."""
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with Location.unknown():
            module = Module.create()
            with InsertionPoint(module.body):
                f = func.FuncOp("while_tensor_store_only", FunctionType.get([], []))
                entry = f.add_entry_block()
                with InsertionPoint(entry):
                    buf = fx.make_rmem_tensor(4, fx.Int32)

                    def before(names, branch_buf):
                        return _i32(0) > _i32(1)

                    def after(names, branch_buf):
                        assert branch_buf is buf
                        branch_buf[0] = _i32(9)
                        return {"buf": branch_buf}

                    out = CanonicalizeWhile.scf_while_dispatch(
                        before,
                        after,
                        result_names=("buf",),
                        result_values=(buf,),
                    )
                    assert out is buf
                    func.ReturnOp([])

            text = str(module)
            assert module.operation.verify()
            headers = _scf_op_headers(text, "scf.while")
            assert all("memref" not in header for header in headers)
            assert any("() -> ()" in header for header in headers)


def test_if_dispatch_rejects_undeclared_rebinding():
    """A direct caller that omits ``rebound_names`` must fail, not lose the binding."""
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with Location.unknown():
            module = Module.create()
            i1 = IntegerType.get_signless(1)
            with InsertionPoint(module.body):
                f = func.FuncOp("if_undeclared_rebind", FunctionType.get([i1], []))
                entry = f.add_entry_block()
                with InsertionPoint(entry):
                    cond = entry.arguments[0]
                    buf = fx.make_rmem_tensor(4, fx.Int32)
                    replacement = fx.make_rmem_tensor(4, fx.Int32)

                    with pytest.raises(TypeError, match="rebound_names"):
                        ReplaceIfWithDispatch.scf_if_dispatch(
                            cond,
                            lambda names, branch_buf: {"buf": replacement},
                            lambda names, branch_buf: {"buf": branch_buf},
                            result_names=("buf",),
                            result_values=(buf,),
                        )
                    func.ReturnOp([])


def test_for_dispatch_rejects_undeclared_rebinding():
    """The same guard applies on the loop path that still carries scalar state."""
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with Location.unknown():
            module = Module.create()
            with InsertionPoint(module.body):
                f = func.FuncOp("for_undeclared_rebind", FunctionType.get([], []))
                entry = f.add_entry_block()
                with InsertionPoint(entry):
                    buf = fx.make_rmem_tensor(4, fx.Int32)
                    replacement = fx.make_rmem_tensor(4, fx.Int32)
                    acc = _i32(0)

                    def body(iv, names, branch_buf, branch_acc):
                        return {"buf": replacement, "acc": branch_acc}

                    with pytest.raises(TypeError, match="rebound_names"):
                        InsertEmptyYieldForSCFFor.scf_for_dispatch(
                            _i32(0),
                            _i32(2),
                            _i32(1),
                            body,
                            result_names=("buf", "acc"),
                            result_values=(buf, acc),
                        )
                    func.ReturnOp([])


def test_while_dispatch_rejects_undeclared_rebinding():
    """The while body must not silently replace a shared storage handle."""
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with Location.unknown():
            module = Module.create()
            with InsertionPoint(module.body):
                f = func.FuncOp("while_undeclared_rebind", FunctionType.get([], []))
                entry = f.add_entry_block()
                with InsertionPoint(entry):
                    buf = fx.make_rmem_tensor(4, fx.Int32)
                    replacement = fx.make_rmem_tensor(4, fx.Int32)

                    def before(names, branch_buf):
                        return _i32(0) < _i32(1)

                    def after(names, branch_buf):
                        return {"buf": replacement}

                    with pytest.raises(TypeError, match="rebound_names"):
                        CanonicalizeWhile.scf_while_dispatch(
                            before,
                            after,
                            result_names=("buf",),
                            result_values=(buf,),
                        )
                    func.ReturnOp([])
