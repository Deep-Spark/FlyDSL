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
from flydsl._mlir.dialects import arith, fly, func  # noqa: E402
from flydsl._mlir.dialects.fly import IntTupleType, ModSwizzleType, SwizzleType  # noqa: E402
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
                # Scheme A: commit / wait group.
                ixdl.cp_async_commit_group()
                ixdl.cp_async_wait_group(0)
                # Scheme B: multi-stage pipeline.
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
    "factory,elem_ir,dtype,err",
    [
        # Each MRAsyncCp swizzle binds to a fixed element width; a mismatched
        # dtype must be rejected by CopyOpMRAsyncCpType::emitAtomCall.
        (ixdl.MRAsyncCpNoSwizzle, "i8", fx.Int8, "NoSwizzle requires valBits = 32, got 8"),
        (ixdl.MRAsyncCpRow8b, "f16", fx.Float16, "Row8b requires valBits = 8, got 16"),
        (ixdl.MRAsyncCpRow16b, "i8", fx.Int8, "Row16b requires valBits = 16, got 8"),
    ],
)
def test_copy_atom_call_rejects_dtype_swizzle_mismatch(factory, elem_ir, dtype, err):
    with ir.Context() as ctx, ir.Location.unknown():
        m = ir.Module.create()
        src_ty = ir.Type.parse(f"!fly.memref<{elem_ir}, #fly_ixdl.sme_gmem, 1:1>")
        dst_ty = ir.Type.parse(f"!fly.memref<{elem_ir}, shared, 1:1>")
        with ir.InsertionPoint(m.body):
            f = func.FuncOp("py_mr_async_cp_mismatch", ([src_ty, dst_ty], []))
            with ir.InsertionPoint(f.add_entry_block()):
                src, dst = f.entry_block.arguments
                atom = fx.make_copy_atom(factory(), dtype)
                fx.copy_atom_call(atom, src, dst)
                func.ReturnOp([])
        with pytest.raises(ir.MLIRError, match=err):
            PassManager.parse("builtin.module(convert-fly-to-ixdl)").run(m.operation)


@pytest.mark.parametrize(
    "factory,dtype",
    [
        (ixdl.MRAsyncCpNoSwizzle, fx.Float32),
        (ixdl.MRAsyncCpCol, fx.Int8),
        (ixdl.MRAsyncCpRow8b, fx.Int8),
        (ixdl.MRAsyncCpRow16b, fx.Float16),
    ],
)
def test_copy_atom_layout_keeps_atom_footprint(factory, dtype):
    with ir.Context():
        atom = fx.make_copy_atom(factory(), dtype)
        assert str(atom.type.tv_layout_dst) == str(atom.type.tv_layout_src)
        assert str(atom.type.tv_layout_ref) == str(atom.type.tv_layout_dst)


@pytest.mark.parametrize(
    "dtype,footprint",
    [
        # CopyOpMRAsyncCpType bit footprint is (1,8192):(0,1) (one logical
        # thread owns the whole 8192-bit SME tile); the value-granularity
        # tv_layout is that recast by the element width.
        (fx.Int8, "(1,1024):(0,1)"),
        (fx.Float16, "(1,512):(0,1)"),
        (fx.Float32, "(1,256):(0,1)"),
    ],
)
def test_copy_atom_thr_val_layout_footprint(dtype, footprint):
    with ir.Context():
        atom = fx.make_copy_atom(ixdl.MRAsyncCpNoSwizzle(), dtype)
        assert str(atom.type.tv_layout_src) == f"!fly.layout<{footprint}>"


# make_sme_shared_layout returns a value(element)-granular layout: the physical
# layout is assembled at byte granularity (the SME swizzle is byte-granular) and
# recast byte->element by a uniform value reinterpret. The byte-granular swizzle
# bases are Col S<2,4,4>, Row8b MS<2,6,2>, Row16b S<1,7,2> (NoSwizzle trivial);
# recasting by elem_bits/8 gives the per-dtype element-granular swizzle. These
# values were verified against the physical SME write on ivcore11 (design doc
# flydsl-mr-async-cp-tiledcopy-alignment.html section 9.1).
@pytest.mark.parametrize(
    "swizzle,dtype,swz_tag",
    [
        (ixdl.SMESwizzle.NoSwizzle, fx.Float32, "!fly.layout"),
        (ixdl.SMESwizzle.Col, fx.Int8, "S<2,4,4>"),
        (ixdl.SMESwizzle.Col, fx.Float16, "S<2,3,4>"),
        (ixdl.SMESwizzle.Col, fx.Float32, "S<2,2,4>"),
        (ixdl.SMESwizzle.Row16b, fx.Float16, "S<1,6,2>"),
        (ixdl.SMESwizzle.Row8b, fx.Int8, "MS<2,6,2>"),
    ],
)
def test_make_sme_shared_layout_byte_granular_swizzle(swizzle, dtype, swz_tag):
    # make_sme_shared_layout emits fly.make_layout / fly.static ops, so it needs
    # an active Location + InsertionPoint; a bare Context detaches the ops and
    # aborts with "operation destroyed but still has uses".
    with ir.Context() as ctx, ir.Location.unknown(ctx):
        m = ir.Module.create()
        with ir.InsertionPoint(m.body):
            f = func.FuncOp("py_make_sme_shared_layout", ([], []))
            with ir.InsertionPoint(f.add_entry_block()):
                layout = ixdl.make_sme_shared_layout(swizzle, dtype, major=ixdl.SMEMajor.K)
                layout_ty = str(layout.type)
                func.ReturnOp([])
        assert swz_tag in layout_ty
        if swizzle == ixdl.SMESwizzle.NoSwizzle:
            assert "composed_layout" not in layout_ty


@pytest.mark.parametrize(
    "swizzle,dtype",
    [
        (ixdl.SMESwizzle.NoSwizzle, fx.Int8),
        (ixdl.SMESwizzle.Row8b, fx.Float16),
        (ixdl.SMESwizzle.Row16b, fx.Float32),
    ],
)
def test_make_sme_shared_layout_rejects_unsupported_dtype(swizzle, dtype):
    with ir.Context():
        with pytest.raises(ValueError):
            ixdl.make_sme_shared_layout(swizzle, dtype)


def _static_int_tuple(spec):
    return fly.static(IntTupleType.get(spec))


def _make_swizzle_inner(kind, mask, base, shift):
    if kind == "MS":
        return fly.static(ModSwizzleType.get(mask, base, shift))
    return fly.static(SwizzleType.get(mask, base, shift))


@pytest.mark.parametrize(
    "inner,coord,expected",
    [
        # ModSwizzle MS<2,3,2> applies an additive wrap-around (with carry)
        # within the low (mask+base)=5 bits. Composed with the identity outer
        # layout 128:1, crd2idx(c) == applyModSwizzle(c). 120 carries from bit4.
        (("MS", 2, 3, 2), 120, 112),
        (("MS", 2, 3, 2), 96, 120),
        (("MS", 2, 3, 2), 64, 80),
        # The XOR Swizzle S<2,3,2> with the same params gives 120 ^ 24 = 96,
        # which differs from ModSwizzle's 112 -> proves MS adds (with carry)
        # whereas S xors. This is exactly why the SME Row8b shared layout needs
        # ModSwizzle rather than plain Swizzle.
        (("S", 2, 3, 2), 120, 96),
        # Trivial ModSwizzle (mask=0) is the identity mapping.
        (("MS", 0, 3, 2), 120, 120),
    ],
)
def test_mod_swizzle_crd2idx_semantics(inner, coord, expected):
    pipeline = "builtin.module(fly-canonicalize,fly-layout-lowering,fly-canonicalize)"
    with ir.Context() as ctx, ir.Location.unknown(ctx):
        module = ir.Module.create()
        idx = ir.IndexType.get()
        with ir.InsertionPoint(module.body):
            f = func.FuncOp("py_mod_swizzle_crd2idx", ir.FunctionType.get([], [idx]))
            with ir.InsertionPoint(f.add_entry_block()):
                inner_val = _make_swizzle_inner(*inner)
                outer = fly.make_layout(_static_int_tuple(128), stride=_static_int_tuple(1))
                cl = fx.make_composed_layout(inner_val, _static_int_tuple(0), outer)
                folded = fx.crd2idx(_static_int_tuple(coord), cl)
                func.ReturnOp([arith.IndexCastOp(idx, fly.get_scalar(folded)).result])
        PassManager.parse(pipeline, ctx).run(module.operation)
        func_op = list(module.body.operations)[0]
        ret_op = list(func_op.entry_block.operations)[-1]
        actual = int(ret_op.operands[0].owner.attributes["value"])
    assert actual == expected


def test_crd2idx_accepts_standalone_mod_swizzle():
    # The fly.crd2idx op accepts a ModSwizzle directly (no composed layout); the
    # static result folds via applyModSwizzle (120 -> 112, additive wrap-around).
    # The raw dialect op is used here because the fx.crd2idx helper guards out
    # all standalone swizzle types at the Python profile-check layer.
    pipeline = "builtin.module(fly-canonicalize,fly-layout-lowering,fly-canonicalize)"
    with ir.Context() as ctx, ir.Location.unknown(ctx):
        module = ir.Module.create()
        idx = ir.IndexType.get()
        with ir.InsertionPoint(module.body):
            f = func.FuncOp("py_standalone_mod_swizzle", ir.FunctionType.get([], [idx]))
            with ir.InsertionPoint(f.add_entry_block()):
                ms = fly.static(ModSwizzleType.get(2, 3, 2))
                folded = fly.crd2idx(_static_int_tuple(120), ms)
                func.ReturnOp([arith.IndexCastOp(idx, fly.get_scalar(folded)).result])
        PassManager.parse(pipeline, ctx).run(module.operation)
        func_op = list(module.body.operations)[0]
        ret_op = list(func_op.entry_block.operations)[-1]
        actual = int(ret_op.operands[0].owner.attributes["value"])
    assert actual == 112


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


@pytest.mark.parametrize("swizzle", [-1, 4, 99])
def test_mr_async_cp_type_rejects_invalid_swizzle(swizzle):
    # CopyOpMRAsyncCpType::verify accepts only the four SME swizzle
    # states (0..3). Parsing goes through getChecked, so an out-of-range value
    # raises rather than aborting (unlike the asserting .get() path).
    with ir.Context(), ir.Location.unknown():
        with pytest.raises(ir.MLIRError, match=f"unsupported smeSwizzle = {swizzle}"):
            ir.Type.parse(f"!fly_ixdl.mr.async_copy<swizzle = {swizzle}>")
