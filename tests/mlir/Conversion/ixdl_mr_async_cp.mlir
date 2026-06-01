// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-ixdl | FileCheck %s
// RUN: %fly-opt %s --convert-fly-to-ixdl --cse | FileCheck %s --check-prefix=CPASYNC
// RUN: %fly-opt %s | FileCheck %s --check-prefix=ROUNDTRIP

// FlyIXDL MRAsyncCp PR1: sme_gmem fat-struct lowering + type round-trip.
// FlyIXDL MRAsyncCp PR2: copy_atom_call (sme_gmem -> shared) -> ixdl.cp_async.*.

// === make_ptr (#fly_ixdl.sme_gmem) -> SmeGmemFatPtr { ptr<1>, i32, i32 } ===

// CHECK-LABEL: @test_make_ptr_sme_gmem
// CHECK-SAME: (%[[BASE:.*]]: !llvm.ptr<1>, %[[STRIDE:.*]]: i32)
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : i32
// CHECK-DAG: %[[SB:.*]] = arith.muli %[[STRIDE]], %[[C4]] : i32
// CHECK-DAG: %[[U:.*]] = llvm.mlir.undef : !llvm.struct<(ptr<1>, i32, i32)>
// CHECK: %[[I0:.*]] = llvm.insertvalue %[[BASE]], %[[U]][0] : !llvm.struct<(ptr<1>, i32, i32)>
// CHECK: %[[I1:.*]] = llvm.insertvalue %[[SB]], %[[I0]][1] : !llvm.struct<(ptr<1>, i32, i32)>
// CHECK: llvm.insertvalue %{{.*}}, %[[I1]][2] : !llvm.struct<(ptr<1>, i32, i32)>
func.func @test_make_ptr_sme_gmem(%base: !fly.ptr<f32, global>, %stride: i32) {
  %p = fly.make_ptr(%base, %stride)
      : (!fly.ptr<f32, global>, i32) -> !fly.ptr<f32, #fly_ixdl.sme_gmem>
  return
}

// === add_offset (#fly_ixdl.sme_gmem) updates the byte_offset field [2] ===

// CHECK-LABEL: @test_add_offset_sme_gmem
// CHECK: %[[OFF:.*]] = llvm.extractvalue %{{.*}}[2] : !llvm.struct<(ptr<1>, i32, i32)>
// CHECK: %[[NEW:.*]] = arith.addi %[[OFF]], %{{.*}} : i32
// CHECK: llvm.insertvalue %[[NEW]], %{{.*}}[2] : !llvm.struct<(ptr<1>, i32, i32)>
func.func @test_add_offset_sme_gmem(%base: !fly.ptr<f32, global>, %stride: i32) {
  %p = fly.make_ptr(%base, %stride)
      : (!fly.ptr<f32, global>, i32) -> !fly.ptr<f32, #fly_ixdl.sme_gmem>
  %off = fly.static : !fly.int_tuple<4>
  %p2 = fly.add_offset(%p, %off)
      : (!fly.ptr<f32, #fly_ixdl.sme_gmem>, !fly.int_tuple<4>) -> !fly.ptr<f32, #fly_ixdl.sme_gmem>
  return
}

// === !fly_ixdl.mr.async_copy<swizzle = N> parse/print round-trip (no lowering) ===

// ROUNDTRIP-LABEL: @test_mr_async_cp_type
// ROUNDTRIP-SAME: !fly_ixdl.mr.async_copy<swizzle = 0>
func.func @test_mr_async_cp_type(
    %atom: !fly.copy_atom<!fly_ixdl.mr.async_copy<swizzle = 0>, 128>) {
  return
}

// === end-to-end: copy_atom_call (sme_gmem -> shared) -> ixdl.cp_async.* ===
//
// NoSwizzle (swizzle = 0) prototype path. sOffset is the shared pointer cast to
// i32; gBase is the v4i32 SmeDescriptor packed from the SmeGmemFatPtr; gOffset and
// kop are 0 (tile byte offset is folded into the descriptor pointer; CacheAll).

// CPASYNC-LABEL: @test_mr_async_cp_call
// CPASYNC-SAME: (%[[SRC:.*]]: !llvm.struct<(ptr<1>, i32, i32)>, %[[DST:.*]]: !llvm.ptr<3>)
// CPASYNC: %[[SOFF:.*]] = llvm.ptrtoint %[[DST]] : !llvm.ptr<3> to i32
// CPASYNC: ixdl.cp_async.16x16.b32.row %[[SOFF]], %{{.*}}, %{{.*}}, %{{.*}} : vector<4xi32> -> i32
func.func @test_mr_async_cp_call(
    %src: !fly.memref<f32, #fly_ixdl.sme_gmem, 1:1>,
    %dst: !fly.memref<f32, shared, 1:1>) {
  %atom = fly.make_copy_atom {valBits = 32 : i32}
      : !fly.copy_atom<!fly_ixdl.mr.async_copy<swizzle = 0>, 32>
  fly.copy_atom_call(%atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.mr.async_copy<swizzle = 0>, 32>,
         !fly.memref<f32, #fly_ixdl.sme_gmem, 1:1>,
         !fly.memref<f32, shared, 1:1>) -> ()
  return
}

// CPASYNC-LABEL: @test_mr_async_cp_col_b8
// CPASYNC: ixdl.cp_async.16x64.b8.col
func.func @test_mr_async_cp_col_b8(
    %src: !fly.memref<i8, #fly_ixdl.sme_gmem, 1:1>,
    %dst: !fly.memref<i8, shared, 1:1>) {
  %atom = fly.make_copy_atom {valBits = 8 : i32}
      : !fly.copy_atom<!fly_ixdl.mr.async_copy<swizzle = 1>, 8>
  fly.copy_atom_call(%atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.mr.async_copy<swizzle = 1>, 8>,
         !fly.memref<i8, #fly_ixdl.sme_gmem, 1:1>,
         !fly.memref<i8, shared, 1:1>) -> ()
  return
}

// CPASYNC-LABEL: @test_mr_async_cp_col_b16
// CPASYNC: ixdl.cp_async.16x32.b16.col
func.func @test_mr_async_cp_col_b16(
    %src: !fly.memref<f16, #fly_ixdl.sme_gmem, 1:1>,
    %dst: !fly.memref<f16, shared, 1:1>) {
  %atom = fly.make_copy_atom {valBits = 16 : i32}
      : !fly.copy_atom<!fly_ixdl.mr.async_copy<swizzle = 1>, 16>
  fly.copy_atom_call(%atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.mr.async_copy<swizzle = 1>, 16>,
         !fly.memref<f16, #fly_ixdl.sme_gmem, 1:1>,
         !fly.memref<f16, shared, 1:1>) -> ()
  return
}

// CPASYNC-LABEL: @test_mr_async_cp_col_b32
// CPASYNC: ixdl.cp_async.16x16.b32.col
func.func @test_mr_async_cp_col_b32(
    %src: !fly.memref<f32, #fly_ixdl.sme_gmem, 1:1>,
    %dst: !fly.memref<f32, shared, 1:1>) {
  %atom = fly.make_copy_atom {valBits = 32 : i32}
      : !fly.copy_atom<!fly_ixdl.mr.async_copy<swizzle = 1>, 32>
  fly.copy_atom_call(%atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.mr.async_copy<swizzle = 1>, 32>,
         !fly.memref<f32, #fly_ixdl.sme_gmem, 1:1>,
         !fly.memref<f32, shared, 1:1>) -> ()
  return
}

// CPASYNC-LABEL: @test_mr_async_cp_row8b
// CPASYNC: ixdl.cp_async.16x64.b8.row
func.func @test_mr_async_cp_row8b(
    %src: !fly.memref<i8, #fly_ixdl.sme_gmem, 1:1>,
    %dst: !fly.memref<i8, shared, 1:1>) {
  %atom = fly.make_copy_atom {valBits = 8 : i32}
      : !fly.copy_atom<!fly_ixdl.mr.async_copy<swizzle = 2>, 8>
  fly.copy_atom_call(%atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.mr.async_copy<swizzle = 2>, 8>,
         !fly.memref<i8, #fly_ixdl.sme_gmem, 1:1>,
         !fly.memref<i8, shared, 1:1>) -> ()
  return
}

// CPASYNC-LABEL: @test_mr_async_cp_row16b
// CPASYNC: ixdl.cp_async.16x32.b16.row
func.func @test_mr_async_cp_row16b(
    %src: !fly.memref<f16, #fly_ixdl.sme_gmem, 1:1>,
    %dst: !fly.memref<f16, shared, 1:1>) {
  %atom = fly.make_copy_atom {valBits = 16 : i32}
      : !fly.copy_atom<!fly_ixdl.mr.async_copy<swizzle = 3>, 16>
  fly.copy_atom_call(%atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.mr.async_copy<swizzle = 3>, 16>,
         !fly.memref<f16, #fly_ixdl.sme_gmem, 1:1>,
         !fly.memref<f16, shared, 1:1>) -> ()
  return
}
