// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-ixdl | FileCheck %s
// RUN: %fly-opt %s | FileCheck %s --check-prefix=ROUNDTRIP

// CQ SmexMtx S2R uses one x2 matrix load per atom call (64 bits/lane). The
// shared pointer is warp-uniform and already selects the tile base plus EmPart.
// loadn16 vs loadn64 on the atom type only pairs with 16-row vs 64-row SmexMtx
// G2S; both lower to the same ixdl.mtx_loadn_*x2 opcode. A 64-row footprint
// needs multiple atom calls with different EmPart / slot bases.

// ROUNDTRIP-LABEL: @test_cq_mtx_loadn_type
// ROUNDTRIP-SAME: !fly_ixdl.cq.mtx_loadn<loadn16, row, 16>
func.func @test_cq_mtx_loadn_type(
    %atom: !fly.copy_atom<!fly_ixdl.cq.mtx_loadn<loadn16, row, 16>, 16>) {
  return
}

// CHECK-LABEL: @test_loadn16_b16_row
// CHECK-SAME: (%[[SRC:.*]]: !llvm.ptr<3>, %[[DST:.*]]: !llvm.ptr<5>)
// CHECK: %[[PACKED:.*]] = ixdl.mtx_loadn_b16_rowx2 %[[SRC]] : <3> -> vector<2xi32>
// CHECK: %[[FRAG:.*]] = llvm.bitcast %[[PACKED]] : vector<2xi32> to vector<4xf16>
// CHECK: llvm.store %[[FRAG]], %[[DST]] : vector<4xf16>, !llvm.ptr<5>
func.func @test_loadn16_b16_row(
    %src: !fly.memref<f16, shared, 1:1>,
    %dst: !fly.memref<f16, register, 4:1>) {
  %atom = fly.make_copy_atom {valBits = 16 : i32}
      : !fly.copy_atom<!fly_ixdl.cq.mtx_loadn<loadn16, row, 16>, 16>
  fly.copy_atom_call(%atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.cq.mtx_loadn<loadn16, row, 16>, 16>,
         !fly.memref<f16, shared, 1:1>, !fly.memref<f16, register, 4:1>) -> ()
  return
}

// CHECK-LABEL: @test_loadn16_b16_col
// CHECK: ixdl.mtx_loadn_colx2 %{{.*}} : <3> -> vector<2xi32>
func.func @test_loadn16_b16_col(
    %src: !fly.memref<f16, shared, 1:1>,
    %dst: !fly.memref<f16, register, 4:1>) {
  %atom = fly.make_copy_atom {valBits = 16 : i32}
      : !fly.copy_atom<!fly_ixdl.cq.mtx_loadn<loadn16, col, 16>, 16>
  fly.copy_atom_call(%atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.cq.mtx_loadn<loadn16, col, 16>, 16>,
         !fly.memref<f16, shared, 1:1>, !fly.memref<f16, register, 4:1>) -> ()
  return
}

// CHECK-LABEL: @test_loadn64_b8_row
// CHECK: ixdl.mtx_loadn_b8_rowx2 %{{.*}} : <3> -> vector<2xi32>
func.func @test_loadn64_b8_row(
    %src: !fly.memref<i8, shared, 1:1>,
    %dst: !fly.memref<i8, register, 8:1>) {
  %atom = fly.make_copy_atom {valBits = 8 : i32}
      : !fly.copy_atom<!fly_ixdl.cq.mtx_loadn<loadn64, row, 8>, 8>
  fly.copy_atom_call(%atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.cq.mtx_loadn<loadn64, row, 8>, 8>,
         !fly.memref<i8, shared, 1:1>, !fly.memref<i8, register, 8:1>) -> ()
  return
}

// CHECK-LABEL: @test_loadn64_b8_col
// CHECK: ixdl.mtx_loadn_colx2 %{{.*}} : <3> -> vector<2xi32>
func.func @test_loadn64_b8_col(
    %src: !fly.memref<i8, shared, 1:1>,
    %dst: !fly.memref<i8, register, 8:1>) {
  %atom = fly.make_copy_atom {valBits = 8 : i32}
      : !fly.copy_atom<!fly_ixdl.cq.mtx_loadn<loadn64, col, 8>, 8>
  fly.copy_atom_call(%atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.cq.mtx_loadn<loadn64, col, 8>, 8>,
         !fly.memref<i8, shared, 1:1>, !fly.memref<i8, register, 8:1>) -> ()
  return
}

// The row and column copy results use the same vector<4xf16> fragments consumed
// by the 16x16x16 CQ MMA atom.
// CHECK-LABEL: @test_loadn16_s2r_mma
// CHECK: %[[A:.*]] = ixdl.mtx_loadn_b16_rowx2
// CHECK: %[[B:.*]] = ixdl.mtx_loadn_colx2
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xf16>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xf16>
// CHECK: ixdl.mmad A[%[[AV]]] B[%[[BV]]]
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 16, k = 16>
func.func @test_loadn16_s2r_mma(
    %smemA: !fly.memref<f16, shared, 1:1>,
    %smemB: !fly.memref<f16, shared, 1:1>,
    %fragA: !fly.memref<f16, register, 4:1>,
    %fragB: !fly.memref<f16, register, 4:1>,
    %d: !fly.memref<f32, register, 4:1>,
    %c: !fly.memref<f32, register, 4:1>) {
  %copyA = fly.make_copy_atom {valBits = 16 : i32}
      : !fly.copy_atom<!fly_ixdl.cq.mtx_loadn<loadn16, row, 16>, 16>
  %copyB = fly.make_copy_atom {valBits = 16 : i32}
      : !fly.copy_atom<!fly_ixdl.cq.mtx_loadn<loadn16, col, 16>, 16>
  fly.copy_atom_call(%copyA, %smemA, %fragA)
      : (!fly.copy_atom<!fly_ixdl.cq.mtx_loadn<loadn16, row, 16>, 16>,
         !fly.memref<f16, shared, 1:1>, !fly.memref<f16, register, 4:1>) -> ()
  fly.copy_atom_call(%copyB, %smemB, %fragB)
      : (!fly.copy_atom<!fly_ixdl.cq.mtx_loadn<loadn16, col, 16>, 16>,
         !fly.memref<f16, shared, 1:1>, !fly.memref<f16, register, 4:1>) -> ()
  %mma = fly.make_mma_atom
      : !fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 16, (f16, f16) -> f32>>
  fly.mma_atom_call(%mma, %d, %fragA, %fragB, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 16, (f16, f16) -> f32>>,
         !fly.memref<f32, register, 4:1>, !fly.memref<f16, register, 4:1>,
         !fly.memref<f16, register, 4:1>, !fly.memref<f32, register, 4:1>) -> ()
  return
}
