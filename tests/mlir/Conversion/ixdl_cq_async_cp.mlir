// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-ixdl | FileCheck %s
// RUN: %fly-opt %s | FileCheck %s --check-prefix=ROUNDTRIP

// CQAsyncCp models enhanced-SME global-to-shared copies only. Shared-to-register
// matrix loads (`loadn16`/`loadn64` → `ixdl.mtx_loadn_*`) use `cq.mtx_loadn`
// (SmexMtx path; see ixdl_cq_mtx_loadn.mlir).

// ROUNDTRIP-LABEL: @test_cq_async_cp_type
// ROUNDTRIP-SAME: !fly_ixdl.cq.async_copy<64, 64, transpose = 0>
func.func @test_cq_async_cp_type(
    %atom: !fly.copy_atom<!fly_ixdl.cq.async_copy<64, 64, transpose = 0>, 8>) {
  return
}

// CHECK-LABEL: @test_cq_async_cp_b8
// CHECK: %[[NEG1:.*]] = arith.constant -1 : i32
// CHECK: llvm.insertelement %[[NEG1]], %{{.*}}[%{{.*}} : i32] : vector<4xi32>
// CHECK: %[[KOP:.*]] = arith.constant 1 : i32
// CHECK: ixdl.cp_async.64x64.b8.row %{{.*}}, %{{.*}}, %{{.*}}, %[[KOP]]
func.func @test_cq_async_cp_b8(
    %src: !fly.memref<i8, #fly_ixdl.sme_gmem, 1:1>,
    %dst: !fly.memref<i8, shared, 1:1>) {
  %atom = fly.make_copy_atom {valBits = 8 : i32}
      : !fly.copy_atom<!fly_ixdl.cq.async_copy<64, 64, transpose = 0>, 8>
  fly.copy_atom_call(%atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.cq.async_copy<64, 64, transpose = 0>, 8>,
         !fly.memref<i8, #fly_ixdl.sme_gmem, 1:1>,
         !fly.memref<i8, shared, 1:1>) -> ()
  return
}

// CHECK-LABEL: @test_cq_async_cp_b16
// CHECK: ixdl.cp_async.64x32.b16.row
func.func @test_cq_async_cp_b16(
    %src: !fly.memref<f16, #fly_ixdl.sme_gmem, 1:1>,
    %dst: !fly.memref<f16, shared, 1:1>) {
  %atom = fly.make_copy_atom {valBits = 16 : i32}
      : !fly.copy_atom<!fly_ixdl.cq.async_copy<64, 32, transpose = 0>, 16>
  fly.copy_atom_call(%atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.cq.async_copy<64, 32, transpose = 0>, 16>,
         !fly.memref<f16, #fly_ixdl.sme_gmem, 1:1>,
         !fly.memref<f16, shared, 1:1>) -> ()
  return
}

// CHECK-LABEL: @test_cq_async_cp_b32_1x64b64
// CHECK: ixdl.cp_async.1x64b64
func.func @test_cq_async_cp_b32_1x64b64(
    %src: !fly.memref<f32, #fly_ixdl.sme_gmem, 1:1>,
    %dst: !fly.memref<f32, shared, 1:1>) {
  %atom = fly.make_copy_atom {valBits = 32 : i32}
      : !fly.copy_atom<!fly_ixdl.cq.async_copy<1, 1024, transpose = 0>, 32>
  fly.copy_atom_call(%atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.cq.async_copy<1, 1024, transpose = 0>, 32>,
         !fly.memref<f32, #fly_ixdl.sme_gmem, 1:1>,
         !fly.memref<f32, shared, 1:1>) -> ()
  return
}

// CHECK-LABEL: @test_cq_async_cp_b32_row
// CHECK: ixdl.cp_async.64x16.b32.row
func.func @test_cq_async_cp_b32_row(
    %src: !fly.memref<f32, #fly_ixdl.sme_gmem, 1:1>,
    %dst: !fly.memref<f32, shared, 1:1>) {
  %atom = fly.make_copy_atom {valBits = 32 : i32}
      : !fly.copy_atom<!fly_ixdl.cq.async_copy<64, 16, transpose = 0>, 32>
  fly.copy_atom_call(%atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.cq.async_copy<64, 16, transpose = 0>, 32>,
         !fly.memref<f32, #fly_ixdl.sme_gmem, 1:1>,
         !fly.memref<f32, shared, 1:1>) -> ()
  return
}

// CHECK-LABEL: @test_cq_async_cp_b32_col
// CHECK: ixdl.cp_async.64x16.b32.col
func.func @test_cq_async_cp_b32_col(
    %src: !fly.memref<f32, #fly_ixdl.sme_gmem, 1:1>,
    %dst: !fly.memref<f32, shared, 1:1>) {
  %atom = fly.make_copy_atom {valBits = 32 : i32}
      : !fly.copy_atom<!fly_ixdl.cq.async_copy<64, 16, transpose = 1>, 32>
  fly.copy_atom_call(%atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.cq.async_copy<64, 16, transpose = 1>, 32>,
         !fly.memref<f32, #fly_ixdl.sme_gmem, 1:1>,
         !fly.memref<f32, shared, 1:1>) -> ()
  return
}
