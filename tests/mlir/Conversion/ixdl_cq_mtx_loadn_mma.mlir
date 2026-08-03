// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-ixdl | FileCheck %s

// Combined lit: CQMtxLoadn S2R (A/B) + CQMma. Layouts are sized so a single
// loadn x2 fills the base 16x16x16 f16 fragment consumed by mmad.

// CHECK-LABEL: @test_cq_s2r_mma_f16
// CHECK: %[[A:.*]] = ixdl.mtx_loadn_b16_rowx2
// CHECK-SAME: -> vector<2xi32>
// CHECK: %[[AB:.*]] = llvm.bitcast %[[A]] : vector<2xi32> to vector<4xf16>
// CHECK: llvm.store %[[AB]], %[[APTR:.*]] : vector<4xf16>, !llvm.ptr<5>
// CHECK: %[[B:.*]] = ixdl.mtx_loadn_colx2
// CHECK-SAME: -> vector<2xi32>
// CHECK: %[[BB:.*]] = llvm.bitcast %[[B]] : vector<2xi32> to vector<4xf16>
// CHECK: llvm.store %[[BB]], %[[BPTR:.*]] : vector<4xf16>, !llvm.ptr<5>
// CHECK: %[[AV:.*]] = llvm.load %[[APTR]] : !llvm.ptr<5> -> vector<4xf16>
// CHECK: %[[BV:.*]] = llvm.load %[[BPTR]] : !llvm.ptr<5> -> vector<4xf16>
// CHECK: %[[CV:.*]] = llvm.load {{.*}} : !llvm.ptr<5> -> vector<4xf32>
// CHECK: %[[R:.*]] = ixdl.mmad A[%[[AV]]] B[%[[BV]]] C[%[[CV]]]
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 16, k = 16>
// CHECK: llvm.store %[[R]], {{.*}} : vector<4xf32>, !llvm.ptr<5>
func.func @test_cq_s2r_mma_f16(
    %smem_a: !fly.memref<f16, shared, 1:1>,
    %smem_b: !fly.memref<f16, shared, 1:1>,
    %frag_a: !fly.memref<f16, register, 4:1>,
    %frag_b: !fly.memref<f16, register, 4:1>,
    %frag_c: !fly.memref<f32, register, 4:1>,
    %frag_d: !fly.memref<f32, register, 4:1>) {
  %copy_a = fly.make_copy_atom {valBits = 16 : i32}
      : !fly.copy_atom<!fly_ixdl.cq.mtx_loadn<pattern = 0, dir = 0, b = 16, x2 = 1>, 16>
  fly.copy_atom_call(%copy_a, %smem_a, %frag_a)
      : (!fly.copy_atom<!fly_ixdl.cq.mtx_loadn<pattern = 0, dir = 0, b = 16, x2 = 1>, 16>,
         !fly.memref<f16, shared, 1:1>,
         !fly.memref<f16, register, 4:1>) -> ()

  %copy_b = fly.make_copy_atom {valBits = 16 : i32}
      : !fly.copy_atom<!fly_ixdl.cq.mtx_loadn<pattern = 0, dir = 1, b = 16, x2 = 1>, 16>
  fly.copy_atom_call(%copy_b, %smem_b, %frag_b)
      : (!fly.copy_atom<!fly_ixdl.cq.mtx_loadn<pattern = 0, dir = 1, b = 16, x2 = 1>, 16>,
         !fly.memref<f16, shared, 1:1>,
         !fly.memref<f16, register, 4:1>) -> ()

  %mma = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 16, (f16, f16) -> f32>>
  fly.mma_atom_call(%mma, %frag_d, %frag_a, %frag_b, %frag_c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 16, (f16, f16) -> f32>>,
         !fly.memref<f32, register, 4:1>, !fly.memref<f16, register, 4:1>,
         !fly.memref<f16, register, 4:1>, !fly.memref<f32, register, 4:1>) -> ()
  return
}
