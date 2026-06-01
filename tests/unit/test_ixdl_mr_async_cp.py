# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FlyIXDL MRAsyncCp Python end-to-end (compile-only / IR check).

Drives the Python high-level API (``flydsl.expr.ixdl``) to build a
``copy_atom_call`` (``#fly_ixdl.sme_gmem`` global -> shared) plus both
synchronization schemes, then runs ``convert-fly-to-ixdl`` and checks that
the NoSwizzle path lowers to ``ixdl.cp_async.16x16.b32.row`` and that the
sync primitives appear in the IR.

Skipped unless the Iluvatar (FlyIXDL) Python extension is built; no GPU is
required (hardware acceptance on ivcore11 is a separate follow-up).
"""

import pytest

pytestmark = [pytest.mark.l1b_target_dialect]

# The FlyIXDL extension only exists in Iluvatar-enabled builds.
pytest.importorskip("flydsl._mlir._mlir_libs._mlirDialectsFlyIXDL")

import flydsl._mlir.ir as ir  # noqa: E402
import flydsl.expr as fx  # noqa: E402
import flydsl.expr.ixdl as ixdl  # noqa: E402
from flydsl._mlir.dialects import func  # noqa: E402
from flydsl._mlir.passmanager import PassManager  # noqa: E402


def test_copy_atom_call_lowers_to_cp_async_with_sync():
    with ir.Context(), ir.Location.unknown():
        m = ir.Module.create()
        src_ty = ir.Type.parse("!fly.memref<f32, #fly_ixdl.sme_gmem, 1:1>")
        dst_ty = ir.Type.parse("!fly.memref<f32, shared, 1:1>")
        with ir.InsertionPoint(m.body):
            f = func.FuncOp("py_mr_async_cp", ([src_ty, dst_ty], []))
            blk = f.add_entry_block()
            with ir.InsertionPoint(blk):
                src, dst = blk.arguments
                atom = fx.make_copy_atom(ixdl.MRAsyncCpNoSwizzle(), fx.Float32)
                fx.copy_atom_call(atom, src, dst)
                # Scheme A: CUDA-style commit / wait group.
                ixdl.cp_async_commit_group()
                ixdl.cp_async_wait_group(0)
                # Scheme B: CUTLASS multi-stage pipeline.
                ixdl.sl_waitmem(0)
                ixdl.sl_pipebar_arrive()
                ixdl.sl_pipebar_wait()
                func.ReturnOp([])

        PassManager.parse("builtin.module(convert-fly-to-ixdl)").run(m.operation)
        asm = str(m)

    assert "ixdl.cp_async.16x16.b32.row" in asm
    assert "vector<4xi32>" in asm
    for intr in (
        "llvm.bi.cp.async.commit.group",
        "llvm.bi.cp.async.wait.group",
        "llvm.bi.sl.waitcnt",
        "llvm.bi.pipebar.req",
        "llvm.bi.pipebar.wait",
    ):
        assert intr in asm, f"missing sync intrinsic {intr}"


@pytest.mark.parametrize(
    "factory,dtype,elem_ir,expected_op",
    [
        (ixdl.MRAsyncCpCol, fx.Int8, "i8", "ixdl.cp_async.16x64.b8.col"),
        (ixdl.MRAsyncCpCol, fx.Float16, "f16", "ixdl.cp_async.16x32.b16.col"),
        (ixdl.MRAsyncCpCol, fx.Float32, "f32", "ixdl.cp_async.16x16.b32.col"),
        (ixdl.MRAsyncCpRow8b, fx.Int8, "i8", "ixdl.cp_async.16x64.b8.row"),
        (ixdl.MRAsyncCpRow16b, fx.Float16, "f16", "ixdl.cp_async.16x32.b16.row"),
    ],
)
def test_copy_atom_call_lowers_all_supported_swizzles(factory, dtype, elem_ir, expected_op):
    with ir.Context(), ir.Location.unknown():
        m = ir.Module.create()
        src_ty = ir.Type.parse(f"!fly.memref<{elem_ir}, #fly_ixdl.sme_gmem, 1:1>")
        dst_ty = ir.Type.parse(f"!fly.memref<{elem_ir}, shared, 1:1>")
        with ir.InsertionPoint(m.body):
            f = func.FuncOp("py_mr_async_cp_swizzle", ([src_ty, dst_ty], []))
            blk = f.add_entry_block()
            with ir.InsertionPoint(blk):
                src, dst = blk.arguments
                atom = fx.make_copy_atom(factory(), dtype)
                fx.copy_atom_call(atom, src, dst)
                func.ReturnOp([])

        PassManager.parse("builtin.module(convert-fly-to-ixdl)").run(m.operation)
        asm = str(m)

    assert expected_op in asm


@pytest.mark.parametrize(
    "factory,dtype,expected",
    [
        (ixdl.MRAsyncCpNoSwizzle, fx.Float32, "S<0,0,0>"),
        (ixdl.MRAsyncCpCol, fx.Int8, "S<2,1,4>"),
        (ixdl.MRAsyncCpRow8b, fx.Int8, "MS<2,3,2>"),
        (ixdl.MRAsyncCpRow16b, fx.Float16, "S<1,3,2>"),
    ],
)
def test_copy_atom_dst_layout_models_supported_swizzles(factory, dtype, expected):
    with ir.Context():
        atom = fx.make_copy_atom(factory(), dtype)
        assert expected in str(atom.type.tv_layout_dst)
        assert expected in str(atom.type.tv_layout_ref)


def test_make_sme_gmem_tensor_emits_sme_gmem_make_ptr():
    with ir.Context(), ir.Location.unknown():
        m = ir.Module.create()
        g_ty = ir.Type.parse("!fly.memref<f32, global, (16, 16) : (16, 1)>")
        with ir.InsertionPoint(m.body):
            f = func.FuncOp("py_make_sme_tensor", ([g_ty], []))
            blk = f.add_entry_block()
            with ir.InsertionPoint(blk):
                (g,) = blk.arguments
                view = ixdl.make_sme_gmem_tensor(g)
                assert "#fly_ixdl.sme_gmem" in str(view.type)
                func.ReturnOp([])
        asm = str(m)

    assert "fly.make_ptr" in asm
    assert "#fly_ixdl.sme_gmem" in asm


@pytest.mark.parametrize(
    "factory,swizzle",
    [
        (ixdl.MRAsyncCpNoSwizzle, 0),
        (ixdl.MRAsyncCpCol, 1),
        (ixdl.MRAsyncCpRow8b, 2),
        (ixdl.MRAsyncCpRow16b, 3),
    ],
)
def test_mr_async_cp_factories(factory, swizzle):
    with ir.Context(), ir.Location.unknown():
        assert str(factory()) == f"!fly_ixdl.mr.async_copy<swizzle = {swizzle}>"
        assert str(ixdl.MRAsyncCp(swizzle)) == f"!fly_ixdl.mr.async_copy<swizzle = {swizzle}>"
