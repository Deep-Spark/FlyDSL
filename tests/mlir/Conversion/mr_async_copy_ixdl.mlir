// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-rewrite-func-signature --fly-canonicalize --fly-layout-lowering --convert-fly-to-ixdl | FileCheck %s
// MR StatefulAsyncCopy (global -> shared): all 16 ivcore11 SME shapes

// -----

// CHECK-LABEL: @test_mr_async_copy_type
// CHECK-SAME: (%{{.*}}: !llvm.struct<(i32, i32, i32)>)
func.func @test_mr_async_copy_type(
    %atom: !fly.copy_atom<!fly_ixdl.mr.async_copy.16x64.b8.row, 8192>) {
  return
}

// -----

// CHECK-LABEL: @test_make_copy_atom_mr_default
func.func @test_make_copy_atom_mr_default(
    %src: !fly.memref<i8, global, 1:1>,
    %dst: !fly.memref<i8, shared, 1:1>) {
  // CHECK-DAG: %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.struct<(i32, i32, i32)>
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : i32
  // CHECK: %[[S1:.*]] = llvm.insertvalue %[[C0]], %[[UNDEF]][0]
  // CHECK: %[[S2:.*]] = llvm.insertvalue %[[C0]], %[[S1]][1]
  // CHECK: %[[ATOM:.*]] = llvm.insertvalue %[[C0]], %[[S2]][2]
  %atom = fly.make_copy_atom {valBits = 8192 : i32}
      : !fly.copy_atom<!fly_ixdl.mr.async_copy.16x64.b8.row, 8192>
  // CHECK: ixdl.cp_async.16x64.b8.row
  fly.copy_atom_call(%atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.mr.async_copy.16x64.b8.row, 8192>,
         !fly.memref<i8, global, 1:1>, !fly.memref<i8, shared, 1:1>) -> ()
  return
}

// -----

// CHECK-LABEL: @test_mr_async_copy_set_soffset
// CHECK-SAME: (%[[ATOM:.*]]: !llvm.struct<(i32, i32, i32)>, %[[SOFF:.*]]: i32
func.func @test_mr_async_copy_set_soffset(
    %atom: !fly.copy_atom<!fly_ixdl.mr.async_copy.16x64.b8.row, 8192>,
    %soff: i32,
    %src: !fly.memref<i8, global, 1:1>,
    %dst: !fly.memref<i8, shared, 1:1>) {
  // CHECK: %[[NEW_ATOM:.*]] = llvm.insertvalue %[[SOFF]], %[[ATOM]][0]
  %new_atom = fly.atom.set_value(%atom, "soffset", %soff)
      : (!fly.copy_atom<!fly_ixdl.mr.async_copy.16x64.b8.row, 8192>, i32)
      -> !fly.copy_atom<!fly_ixdl.mr.async_copy.16x64.b8.row, 8192>
  // CHECK: llvm.extractvalue %[[NEW_ATOM]][0]
  // CHECK: ixdl.cp_async.16x64.b8.row
  fly.copy_atom_call(%new_atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.mr.async_copy.16x64.b8.row, 8192>,
         !fly.memref<i8, global, 1:1>, !fly.memref<i8, shared, 1:1>) -> ()
  return
}

// -----

// CHECK-LABEL: @test_mr_async_copy_set_stride_byte
// CHECK-SAME: (%[[ATOM:.*]]: !llvm.struct<(i32, i32, i32)>, %[[STRIDE:.*]]: i32
func.func @test_mr_async_copy_set_stride_byte(
    %atom: !fly.copy_atom<!fly_ixdl.mr.async_copy.16x64.b8.row, 8192>,
    %stride_byte: i32,
    %src: !fly.memref<i8, global, (16,64):(64,1)>,
    %dst: !fly.memref<i8, shared, 1:1>) {
  // CHECK: %[[NEW_ATOM:.*]] = llvm.insertvalue %[[STRIDE]], %[[ATOM]][2]
  %new_atom = fly.atom.set_value(%atom, "stride_byte", %stride_byte)
      : (!fly.copy_atom<!fly_ixdl.mr.async_copy.16x64.b8.row, 8192>, i32)
      -> !fly.copy_atom<!fly_ixdl.mr.async_copy.16x64.b8.row, 8192>
  // CHECK: llvm.extractvalue %[[NEW_ATOM]][2]
  // CHECK: arith.cmpi ne
  // CHECK: arith.select
  // CHECK: ixdl.cp_async.16x64.b8.row
  fly.copy_atom_call(%new_atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.mr.async_copy.16x64.b8.row, 8192>,
         !fly.memref<i8, global, (16,64):(64,1)>, !fly.memref<i8, shared, 1:1>) -> ()
  return
}

// -----

// CHECK-LABEL: @test_mr_cp_async_commit_wait
func.func @test_mr_cp_async_commit_wait() {
  // CHECK: ixdl.cp.async.commit.group
  fly_ixdl.cp_async_commit_group
  // CHECK: ixdl.cp.async.wait.group 0
  fly_ixdl.cp_async_wait_group 0
  return
}

// ----- i8 SME shapes (4 shapes)

// CHECK-LABEL: @test_mr_async_4x64_b8_row
// CHECK: ixdl.cp_async.4x64.b8.row
func.func @test_mr_async_4x64_b8_row(
    %src: !fly.memref<i8, global, 1:1>, %dst: !fly.memref<i8, shared, 1:1>) {
  %a = fly.make_copy_atom {valBits = 2048 : i32}
      : !fly.copy_atom<!fly_ixdl.mr.async_copy.4x64.b8.row, 2048>
  fly.copy_atom_call(%a, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.mr.async_copy.4x64.b8.row, 2048>,
         !fly.memref<i8, global, 1:1>, !fly.memref<i8, shared, 1:1>) -> ()
  return
}

// CHECK-LABEL: @test_mr_async_4x64_b8_row_signed_to_signless
// CHECK: ixdl.cp_async.4x64.b8.row
func.func @test_mr_async_4x64_b8_row_signed_to_signless(
    %src: !fly.memref<si8, global, (4,64):(64,1)>,
    %dst: !fly.memref<i8, shared, (4,64):(64,1)>) {
  %a = fly.make_copy_atom {valBits = 2048 : i32}
      : !fly.copy_atom<!fly_ixdl.mr.async_copy.4x64.b8.row, 2048>
  fly.copy_atom_call(%a, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.mr.async_copy.4x64.b8.row, 2048>,
         !fly.memref<si8, global, (4,64):(64,1)>, !fly.memref<i8, shared, (4,64):(64,1)>) -> ()
  return
}

// CHECK-LABEL: @test_mr_async_4x64_b8_col
// CHECK: ixdl.cp_async.4x64.b8.col
func.func @test_mr_async_4x64_b8_col(
    %src: !fly.memref<i8, global, 1:1>, %dst: !fly.memref<i8, shared, 1:1>) {
  %a = fly.make_copy_atom {valBits = 2048 : i32}
      : !fly.copy_atom<!fly_ixdl.mr.async_copy.4x64.b8.col, 2048>
  fly.copy_atom_call(%a, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.mr.async_copy.4x64.b8.col, 2048>,
         !fly.memref<i8, global, 1:1>, !fly.memref<i8, shared, 1:1>) -> ()
  return
}

// CHECK-LABEL: @test_mr_async_16x64_b8_row
// CHECK: ixdl.cp_async.16x64.b8.row
func.func @test_mr_async_16x64_b8_row(
    %src: !fly.memref<i8, global, 1:1>, %dst: !fly.memref<i8, shared, 1:1>) {
  %a = fly.make_copy_atom {valBits = 8192 : i32}
      : !fly.copy_atom<!fly_ixdl.mr.async_copy.16x64.b8.row, 8192>
  fly.copy_atom_call(%a, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.mr.async_copy.16x64.b8.row, 8192>,
         !fly.memref<i8, global, 1:1>, !fly.memref<i8, shared, 1:1>) -> ()
  return
}

// CHECK-LABEL: @test_mr_async_16x64_b8_col
// CHECK: ixdl.cp_async.16x64.b8.col
func.func @test_mr_async_16x64_b8_col(
    %src: !fly.memref<i8, global, 1:1>, %dst: !fly.memref<i8, shared, 1:1>) {
  %a = fly.make_copy_atom {valBits = 8192 : i32}
      : !fly.copy_atom<!fly_ixdl.mr.async_copy.16x64.b8.col, 8192>
  fly.copy_atom_call(%a, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.mr.async_copy.16x64.b8.col, 8192>,
         !fly.memref<i8, global, 1:1>, !fly.memref<i8, shared, 1:1>) -> ()
  return
}

// ----- f16 SME shapes (4 shapes)

// CHECK-LABEL: @test_mr_async_4x32_b16_row
// CHECK: ixdl.cp_async.4x32.b16.row
func.func @test_mr_async_4x32_b16_row(
    %src: !fly.memref<f16, global, 1:1>, %dst: !fly.memref<f16, shared, 1:1>) {
  %a = fly.make_copy_atom {valBits = 2048 : i32}
      : !fly.copy_atom<!fly_ixdl.mr.async_copy.4x32.b16.row, 2048>
  fly.copy_atom_call(%a, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.mr.async_copy.4x32.b16.row, 2048>,
         !fly.memref<f16, global, 1:1>, !fly.memref<f16, shared, 1:1>) -> ()
  return
}

// CHECK-LABEL: @test_mr_async_4x32_b16_row_signed_to_signless
// CHECK: ixdl.cp_async.4x32.b16.row
func.func @test_mr_async_4x32_b16_row_signed_to_signless(
    %src: !fly.memref<si16, global, (4,32):(32,1)>,
    %dst: !fly.memref<i16, shared, (4,32):(32,1)>) {
  %a = fly.make_copy_atom {valBits = 2048 : i32}
      : !fly.copy_atom<!fly_ixdl.mr.async_copy.4x32.b16.row, 2048>
  fly.copy_atom_call(%a, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.mr.async_copy.4x32.b16.row, 2048>,
         !fly.memref<si16, global, (4,32):(32,1)>, !fly.memref<i16, shared, (4,32):(32,1)>) -> ()
  return
}

// CHECK-LABEL: @test_mr_async_4x32_b16_row_float_to_int
// CHECK: ixdl.cp_async.4x32.b16.row
func.func @test_mr_async_4x32_b16_row_float_to_int(
    %src: !fly.memref<f16, global, (4,32):(32,1)>,
    %dst: !fly.memref<i16, shared, (4,32):(32,1)>) {
  %a = fly.make_copy_atom {valBits = 2048 : i32}
      : !fly.copy_atom<!fly_ixdl.mr.async_copy.4x32.b16.row, 2048>
  fly.copy_atom_call(%a, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.mr.async_copy.4x32.b16.row, 2048>,
         !fly.memref<f16, global, (4,32):(32,1)>, !fly.memref<i16, shared, (4,32):(32,1)>) -> ()
  return
}

// CHECK-LABEL: @test_mr_async_4x32_b16_col
// CHECK: ixdl.cp_async.4x32.b16.col
func.func @test_mr_async_4x32_b16_col(
    %src: !fly.memref<f16, global, 1:1>, %dst: !fly.memref<f16, shared, 1:1>) {
  %a = fly.make_copy_atom {valBits = 2048 : i32}
      : !fly.copy_atom<!fly_ixdl.mr.async_copy.4x32.b16.col, 2048>
  fly.copy_atom_call(%a, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.mr.async_copy.4x32.b16.col, 2048>,
         !fly.memref<f16, global, 1:1>, !fly.memref<f16, shared, 1:1>) -> ()
  return
}

// CHECK-LABEL: @test_mr_async_16x32_b16_row
// CHECK: ixdl.cp_async.16x32.b16.row
func.func @test_mr_async_16x32_b16_row(
    %src: !fly.memref<f16, global, 1:1>, %dst: !fly.memref<f16, shared, 1:1>) {
  %a = fly.make_copy_atom {valBits = 8192 : i32}
      : !fly.copy_atom<!fly_ixdl.mr.async_copy.16x32.b16.row, 8192>
  fly.copy_atom_call(%a, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.mr.async_copy.16x32.b16.row, 8192>,
         !fly.memref<f16, global, 1:1>, !fly.memref<f16, shared, 1:1>) -> ()
  return
}

// CHECK-LABEL: @test_mr_async_16x32_b16_col
// CHECK: ixdl.cp_async.16x32.b16.col
func.func @test_mr_async_16x32_b16_col(
    %src: !fly.memref<f16, global, 1:1>, %dst: !fly.memref<f16, shared, 1:1>) {
  %a = fly.make_copy_atom {valBits = 8192 : i32}
      : !fly.copy_atom<!fly_ixdl.mr.async_copy.16x32.b16.col, 8192>
  fly.copy_atom_call(%a, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.mr.async_copy.16x32.b16.col, 8192>,
         !fly.memref<f16, global, 1:1>, !fly.memref<f16, shared, 1:1>) -> ()
  return
}

// ----- f32 linear b64 shapes (4 shapes)

// CHECK-LABEL: @test_mr_async_1x1b64
// CHECK: ixdl.cp_async.1x1b64
func.func @test_mr_async_1x1b64(
    %src: !fly.memref<f32, global, 1:1>, %dst: !fly.memref<f32, shared, 1:1>) {
  %a = fly.make_copy_atom {valBits = 512 : i32}
      : !fly.copy_atom<!fly_ixdl.mr.async_copy.1x1b64, 512>
  fly.copy_atom_call(%a, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.mr.async_copy.1x1b64, 512>,
         !fly.memref<f32, global, 1:1>, !fly.memref<f32, shared, 1:1>) -> ()
  return
}

// CHECK-LABEL: @test_mr_async_1x4b64
// CHECK: ixdl.cp_async.1x4b64
func.func @test_mr_async_1x4b64(
    %src: !fly.memref<f32, global, 1:1>, %dst: !fly.memref<f32, shared, 1:1>) {
  %a = fly.make_copy_atom {valBits = 2048 : i32}
      : !fly.copy_atom<!fly_ixdl.mr.async_copy.1x4b64, 2048>
  fly.copy_atom_call(%a, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.mr.async_copy.1x4b64, 2048>,
         !fly.memref<f32, global, 1:1>, !fly.memref<f32, shared, 1:1>) -> ()
  return
}

// CHECK-LABEL: @test_mr_async_1x8b64
// CHECK: ixdl.cp_async.1x8b64
func.func @test_mr_async_1x8b64(
    %src: !fly.memref<f32, global, 1:1>, %dst: !fly.memref<f32, shared, 1:1>) {
  %a = fly.make_copy_atom {valBits = 4096 : i32}
      : !fly.copy_atom<!fly_ixdl.mr.async_copy.1x8b64, 4096>
  fly.copy_atom_call(%a, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.mr.async_copy.1x8b64, 4096>,
         !fly.memref<f32, global, 1:1>, !fly.memref<f32, shared, 1:1>) -> ()
  return
}

// CHECK-LABEL: @test_mr_async_1x16b64
// CHECK: ixdl.cp_async.1x16b64
func.func @test_mr_async_1x16b64(
    %src: !fly.memref<f32, global, 1:1>, %dst: !fly.memref<f32, shared, 1:1>) {
  %a = fly.make_copy_atom {valBits = 8192 : i32}
      : !fly.copy_atom<!fly_ixdl.mr.async_copy.1x16b64, 8192>
  fly.copy_atom_call(%a, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.mr.async_copy.1x16b64, 8192>,
         !fly.memref<f32, global, 1:1>, !fly.memref<f32, shared, 1:1>) -> ()
  return
}

// ----- f32 SME shapes (4 shapes)

// CHECK-LABEL: @test_mr_async_4x16_b32_row
// CHECK: ixdl.cp_async.4x16.b32.row
func.func @test_mr_async_4x16_b32_row(
    %src: !fly.memref<f32, global, 1:1>, %dst: !fly.memref<f32, shared, 1:1>) {
  %a = fly.make_copy_atom {valBits = 2048 : i32}
      : !fly.copy_atom<!fly_ixdl.mr.async_copy.4x16.b32.row, 2048>
  fly.copy_atom_call(%a, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.mr.async_copy.4x16.b32.row, 2048>,
         !fly.memref<f32, global, 1:1>, !fly.memref<f32, shared, 1:1>) -> ()
  return
}

// CHECK-LABEL: @test_mr_async_8x16_b32_row
// CHECK: ixdl.cp_async.8x16.b32.row
func.func @test_mr_async_8x16_b32_row(
    %src: !fly.memref<f32, global, 1:1>, %dst: !fly.memref<f32, shared, 1:1>) {
  %a = fly.make_copy_atom {valBits = 4096 : i32}
      : !fly.copy_atom<!fly_ixdl.mr.async_copy.8x16.b32.row, 4096>
  fly.copy_atom_call(%a, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.mr.async_copy.8x16.b32.row, 4096>,
         !fly.memref<f32, global, 1:1>, !fly.memref<f32, shared, 1:1>) -> ()
  return
}

// CHECK-LABEL: @test_mr_async_16x16_b32_row
// CHECK: ixdl.cp_async.16x16.b32.row
func.func @test_mr_async_16x16_b32_row(
    %src: !fly.memref<f32, global, 1:1>, %dst: !fly.memref<f32, shared, 1:1>) {
  %a = fly.make_copy_atom {valBits = 8192 : i32}
      : !fly.copy_atom<!fly_ixdl.mr.async_copy.16x16.b32.row, 8192>
  fly.copy_atom_call(%a, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.mr.async_copy.16x16.b32.row, 8192>,
         !fly.memref<f32, global, 1:1>, !fly.memref<f32, shared, 1:1>) -> ()
  return
}

// CHECK-LABEL: @test_mr_async_16x16_b32_row_signed_to_signless
// CHECK: ixdl.cp_async.16x16.b32.row
func.func @test_mr_async_16x16_b32_row_signed_to_signless(
    %src: !fly.memref<si32, global, (16,16):(16,1)>,
    %dst: !fly.memref<i32, shared, (16,16):(16,1)>) {
  %a = fly.make_copy_atom {valBits = 8192 : i32}
      : !fly.copy_atom<!fly_ixdl.mr.async_copy.16x16.b32.row, 8192>
  fly.copy_atom_call(%a, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.mr.async_copy.16x16.b32.row, 8192>,
         !fly.memref<si32, global, (16,16):(16,1)>, !fly.memref<i32, shared, (16,16):(16,1)>) -> ()
  return
}

// CHECK-LABEL: @test_mr_async_16x16_b32_row_float_to_int
// CHECK: ixdl.cp_async.16x16.b32.row
func.func @test_mr_async_16x16_b32_row_float_to_int(
    %src: !fly.memref<f32, global, (16,16):(16,1)>,
    %dst: !fly.memref<i32, shared, (16,16):(16,1)>) {
  %a = fly.make_copy_atom {valBits = 8192 : i32}
      : !fly.copy_atom<!fly_ixdl.mr.async_copy.16x16.b32.row, 8192>
  fly.copy_atom_call(%a, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.mr.async_copy.16x16.b32.row, 8192>,
         !fly.memref<f32, global, (16,16):(16,1)>, !fly.memref<i32, shared, (16,16):(16,1)>) -> ()
  return
}

// CHECK-LABEL: @test_mr_async_16x16_b32_col
// CHECK: ixdl.cp_async.16x16.b32.col
func.func @test_mr_async_16x16_b32_col(
    %src: !fly.memref<f32, global, 1:1>, %dst: !fly.memref<f32, shared, 1:1>) {
  %a = fly.make_copy_atom {valBits = 8192 : i32}
      : !fly.copy_atom<!fly_ixdl.mr.async_copy.16x16.b32.col, 8192>
  fly.copy_atom_call(%a, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.mr.async_copy.16x16.b32.col, 8192>,
         !fly.memref<f32, global, 1:1>, !fly.memref<f32, shared, 1:1>) -> ()
  return
}
