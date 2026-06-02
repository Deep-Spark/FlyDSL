// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-rewrite-func-signature --fly-canonicalize --fly-layout-lowering --convert-fly-to-ixdl | FileCheck %s

// MR MMAD atom call lowering:
//   fly.mma_atom_call -> ixdl.mmad
//   Loads A/B as vector<4xf16>, C as vector<4xf32>, stores result to D.

// CHECK-LABEL: @mmad
// CHECK-SAME: (%[[D:.*]]: !llvm.ptr<5>, %[[A:.*]]: !llvm.ptr<5>, %[[B:.*]]: !llvm.ptr<5>, %[[C:.*]]: !llvm.ptr<5>)
func.func @mmad(
    %d: !fly.memref<f32, register, 4:1>,
    %a: !fly.memref<f16, register, 4:1>,
    %b: !fly.memref<f16, register, 4:1>,
    %c: !fly.memref<f32, register, 4:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.mr.mmad<16x16x16, (f16, f16) -> f32>>
  // CHECK: %[[AV:.*]] = llvm.load %[[A]] : !llvm.ptr<5> -> vector<4xf16>
  // CHECK: %[[BV:.*]] = llvm.load %[[B]] : !llvm.ptr<5> -> vector<4xf16>
  // CHECK: %[[CV:.*]] = llvm.load %[[C]] : !llvm.ptr<5> -> vector<4xf32>
  // CHECK: %[[RES:.*]] = ixdl.mmad A[%[[AV]]] B[%[[BV]]] C[%[[CV]]]
  // CHECK-SAME: layoutA = #ixdl.mmad_layout<row>
  // CHECK-SAME: layoutB = #ixdl.mmad_layout<col>
  // CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f16>
  // CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f16>
  // CHECK-SAME: shape = #ixdl.shape<m = 16, n = 16, k = 16>
  // CHECK-SAME: (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
  // CHECK: llvm.store %[[RES]], %[[D]] : vector<4xf32>, !llvm.ptr<5>
  fly.mma_atom_call(%atom, %d, %a, %b, %c) : (!fly.mma_atom<!fly_ixdl.mr.mmad<16x16x16, (f16, f16) -> f32>>, !fly.memref<f32, register, 4:1>, !fly.memref<f16, register, 4:1>, !fly.memref<f16, register, 4:1>, !fly.memref<f32, register, 4:1>) -> ()
  return
}
