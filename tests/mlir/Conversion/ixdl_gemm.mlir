// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-layout-lowering --convert-fly-to-rocdl | FileCheck %s

// Minimal IXDL GEMM lowering test:
//   fly.gemm -> fly.mma_atom_call -> ixdl.mmad

// CHECK-LABEL: @test_ixdl_gemm
// CHECK-SAME: (%[[D:.*]]: !llvm.ptr<5>, %[[A:.*]]: !llvm.ptr<5>, %[[B:.*]]: !llvm.ptr<5>, %[[C:.*]]: !llvm.ptr<5>)
func.func @test_ixdl_gemm(
    %d: !fly.memref<f32, register, 4:1>,
    %a: !fly.memref<f16, register, 4:1>,
    %b: !fly.memref<f16, register, 4:1>,
    %c: !fly.memref<f32, register, 4:1>) {
  %atom = fly.make_mma_atom : !fly.ixdl_mmad<16x16x16, (f16, f16) -> f32>
  // CHECK: %[[A_VAL:.*]] = llvm.load %[[A]] : !llvm.ptr<5> -> vector<4xf16>
  // CHECK: %[[B_VAL:.*]] = llvm.load %[[B]] : !llvm.ptr<5> -> vector<4xf16>
  // CHECK: %[[C_VAL:.*]] = llvm.load %[[C]] : !llvm.ptr<5> -> vector<4xf32>
  // CHECK: %[[RES:.*]] = ixdl.mmad A[%[[A_VAL]]]  B[%[[B_VAL]]]  C[%[[C_VAL]]]
  // CHECK-SAME: {layoutA = #ixdl.mmad_layout<row>, layoutB = #ixdl.mmad_layout<col>
  // CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f16>, multiplicandBType = #ixdl.mmad_type<f16>
  // CHECK-SAME: shape = #ixdl.shape<m = 16, n = 16, k = 16>}
  // CHECK: llvm.store %[[RES]], %[[D]] : vector<4xf32>, !llvm.ptr<5>
  fly.gemm(%atom, %d, %a, %b, %c) : (!fly.ixdl_mmad<16x16x16, (f16, f16) -> f32>, !fly.memref<f32, register, 4:1>, !fly.memref<f16, register, 4:1>, !fly.memref<f16, register, 4:1>, !fly.memref<f32, register, 4:1>) -> ()
  return
}
