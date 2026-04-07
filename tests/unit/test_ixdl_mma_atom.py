#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from flydsl._mlir.ir import FunctionType, InsertionPoint
from flydsl._mlir.dialects import func
import flydsl.expr as fx


def test_ixdl_mmad_type_properties(ctx):
    atom_ty = fx.ixdl.MMAD(16, 16, 16, fx.Float16, elem_type_acc=fx.Float32)

    assert str(atom_ty) == "!fly.ixdl_mmad<16x16x16, (f16, f16) -> f32>"
    assert str(atom_ty.shape_mnk) == "!fly.int_tuple<(16,16,16)>"
    assert str(atom_ty.thr_layout) == "!fly.layout<64:1>"
    assert str(atom_ty.tv_layout_a) == "!fly.layout<((16,2,2),(2,2)):((16,8,1),(4,2))>"
    assert str(atom_ty.tv_layout_b) == "!fly.layout<((16,4),4):((16,4),1)>"
    assert str(atom_ty.tv_layout_c) == "!fly.layout<((16,4),4):((16,4),1)>"


def test_backend_aware_mma_helper_ixdl(ctx):
    atom_ty = fx.MMA(16, 16, 16, fx.Float16, elem_type_acc=fx.Float32, backend="ixdl")
    assert str(atom_ty) == "!fly.ixdl_mmad<16x16x16, (f16, f16) -> f32>"


def test_make_mma_atom_accepts_ixdl_type(ctx):
    with InsertionPoint(ctx.module.body):
        f = func.FuncOp("test_ixdl_make_mma_atom", FunctionType.get([], []))
        with InsertionPoint(f.add_entry_block()):
            atom = fx.make_mma_atom(fx.ixdl.MMAD(16, 16, 16, fx.Float16, elem_type_acc=fx.Float32))
            func.ReturnOp([])

    ir_text = str(ctx.module)
    assert "fly.make_mma_atom" in ir_text
    assert "!fly.ixdl_mmad<16x16x16, (f16, f16) -> f32>" in ir_text
    assert "test_ixdl_make_mma_atom" in ir_text
    assert atom.atom_ty is not None


def test_make_tiled_mma_accepts_ixdl_atom(ctx):
    with InsertionPoint(ctx.module.body):
        f = func.FuncOp("test_ixdl_make_tiled_mma", FunctionType.get([], []))
        with InsertionPoint(f.add_entry_block()):
            atom = fx.make_mma_atom(fx.ixdl.MMAD(16, 16, 16, fx.Float16, elem_type_acc=fx.Float32))
            tiled = fx.make_tiled_mma(atom, fx.make_layout((1, 1, 1), (1, 1, 1)))
            func.ReturnOp([])

    ir_text = str(ctx.module)
    assert "!fly.tiled_mma<!fly.ixdl_mmad<16x16x16, (f16, f16) -> f32>, <(1,1,1):(1,1,1)>>" in ir_text
    assert "test_ixdl_make_tiled_mma" in ir_text

    assert str(tiled.tile_size_mnk.type) == "!fly.int_tuple<(16,16,16)>"
    assert str(tiled.thr_layout_vmnk.type) == "!fly.layout<(64,1,1,1):(1,0,0,0)>"
    assert str(tiled.tiled_tv_layout_A.type) == "!fly.layout<((16,2,2),((2,2),(1,1))):((16,8,1),((4,2),(0,0)))>"
    assert str(tiled.tiled_tv_layout_B.type) == "!fly.layout<((16,4),(4,(1,1))):((16,4),(1,(0,0)))>"
    assert str(tiled.tiled_tv_layout_C.type) == "!fly.layout<((16,4),(4,(1,1))):((16,4),(1,(0,0)))>"
