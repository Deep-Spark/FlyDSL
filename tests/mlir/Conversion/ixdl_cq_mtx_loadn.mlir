// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-ixdl | FileCheck %s
// RUN: %fly-opt %s | FileCheck %s --check-prefix=ROUNDTRIP

// CQMtxLoadn: SmexMtx shared->register matrix load (loadn16/loadn64).
// Incompatible with LegacySme ldmatrix / byte swizzle on the same buffer;
// swizzle policy is Bypass. Lowers to ixdl.mtx_loadn_*.

// ROUNDTRIP-LABEL: @test_cq_mtx_loadn_type
// ROUNDTRIP-SAME: !fly_ixdl.cq.mtx_loadn<pattern = 0, dir = 0, b = 16, x2 = 1>
func.func @test_cq_mtx_loadn_type(
    %atom: !fly.copy_atom<!fly_ixdl.cq.mtx_loadn<pattern = 0, dir = 0, b = 16, x2 = 1>, 16>) {
  return
}

// === loadn16 row b16 x2 (A / f16 fragment) ===

// CHECK-LABEL: @test_cq_mtx_loadn_b16_rowx2
// CHECK-SAME: (%[[SRC:.*]]: !llvm.ptr<3>, %[[DST:.*]]: !llvm.ptr<5>)
// CHECK: %[[LD:.*]] = ixdl.mtx_loadn_b16_rowx2 %[[SRC]]
// CHECK-SAME: -> vector<2xi32>
// CHECK: %[[BC:.*]] = llvm.bitcast %[[LD]] : vector<2xi32> to vector<4xf16>
// CHECK: llvm.store %[[BC]], %[[DST]] : vector<4xf16>, !llvm.ptr<5>
func.func @test_cq_mtx_loadn_b16_rowx2(
    %src: !fly.memref<f16, shared, 1:1>,
    %dst: !fly.memref<f16, register, 4:1>) {
  %atom = fly.make_copy_atom {valBits = 16 : i32}
      : !fly.copy_atom<!fly_ixdl.cq.mtx_loadn<pattern = 0, dir = 0, b = 16, x2 = 1>, 16>
  fly.copy_atom_call(%atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.cq.mtx_loadn<pattern = 0, dir = 0, b = 16, x2 = 1>, 16>,
         !fly.memref<f16, shared, 1:1>,
         !fly.memref<f16, register, 4:1>) -> ()
  return
}

// === loadn64 col b16 x2 (B / f16 fragment) ===

// CHECK-LABEL: @test_cq_mtx_loadn_b16_colx2
// CHECK: ixdl.mtx_loadn_colx2
// CHECK-SAME: -> vector<2xi32>
// CHECK: llvm.bitcast {{.*}} : vector<2xi32> to vector<4xf16>
func.func @test_cq_mtx_loadn_b16_colx2(
    %src: !fly.memref<f16, shared, 1:1>,
    %dst: !fly.memref<f16, register, 4:1>) {
  %atom = fly.make_copy_atom {valBits = 16 : i32}
      : !fly.copy_atom<!fly_ixdl.cq.mtx_loadn<pattern = 1, dir = 1, b = 16, x2 = 1>, 16>
  fly.copy_atom_call(%atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.cq.mtx_loadn<pattern = 1, dir = 1, b = 16, x2 = 1>, 16>,
         !fly.memref<f16, shared, 1:1>,
         !fly.memref<f16, register, 4:1>) -> ()
  return
}

// === loadn16 row b8 x2 (A / i8 fragment) ===

// CHECK-LABEL: @test_cq_mtx_loadn_b8_rowx2
// CHECK: %[[LD:.*]] = ixdl.mtx_loadn_b8_rowx2
// CHECK-SAME: -> vector<2xi32>
// CHECK: %[[BC:.*]] = llvm.bitcast %[[LD]] : vector<2xi32> to vector<8xi8>
// CHECK: llvm.store %[[BC]]
func.func @test_cq_mtx_loadn_b8_rowx2(
    %src: !fly.memref<i8, shared, 1:1>,
    %dst: !fly.memref<i8, register, 8:1>) {
  %atom = fly.make_copy_atom {valBits = 8 : i32}
      : !fly.copy_atom<!fly_ixdl.cq.mtx_loadn<pattern = 0, dir = 0, b = 8, x2 = 1>, 8>
  fly.copy_atom_call(%atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.cq.mtx_loadn<pattern = 0, dir = 0, b = 8, x2 = 1>, 8>,
         !fly.memref<i8, shared, 1:1>,
         !fly.memref<i8, register, 8:1>) -> ()
  return
}

// === loadn64 col b8 x2 ===

// CHECK-LABEL: @test_cq_mtx_loadn_b8_colx2
// CHECK: ixdl.mtx_loadn_colx2
// CHECK-SAME: -> vector<2xi32>
func.func @test_cq_mtx_loadn_b8_colx2(
    %src: !fly.memref<i8, shared, 1:1>,
    %dst: !fly.memref<i8, register, 8:1>) {
  %atom = fly.make_copy_atom {valBits = 8 : i32}
      : !fly.copy_atom<!fly_ixdl.cq.mtx_loadn<pattern = 1, dir = 1, b = 8, x2 = 1>, 8>
  fly.copy_atom_call(%atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.cq.mtx_loadn<pattern = 1, dir = 1, b = 8, x2 = 1>, 8>,
         !fly.memref<i8, shared, 1:1>,
         !fly.memref<i8, register, 8:1>) -> ()
  return
}

// === non-x2 (32b) row b16 ===

// CHECK-LABEL: @test_cq_mtx_loadn_b16_row
// CHECK: %[[LD:.*]] = ixdl.mtx_loadn_b16_row
// CHECK-SAME: -> i32
// CHECK: %[[BC:.*]] = llvm.bitcast %[[LD]] : i32 to vector<2xf16>
// CHECK: llvm.store %[[BC]]
func.func @test_cq_mtx_loadn_b16_row(
    %src: !fly.memref<f16, shared, 1:1>,
    %dst: !fly.memref<f16, register, 2:1>) {
  %atom = fly.make_copy_atom {valBits = 16 : i32}
      : !fly.copy_atom<!fly_ixdl.cq.mtx_loadn<pattern = 0, dir = 0, b = 16, x2 = 0>, 16>
  fly.copy_atom_call(%atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.cq.mtx_loadn<pattern = 0, dir = 0, b = 16, x2 = 0>, 16>,
         !fly.memref<f16, shared, 1:1>,
         !fly.memref<f16, register, 2:1>) -> ()
  return
}
