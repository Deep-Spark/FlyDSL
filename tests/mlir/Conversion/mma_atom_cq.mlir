// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-cq-opt %s --fly-rewrite-func-signature --fly-canonicalize --fly-layout-lowering --convert-fly-to-cq | FileCheck %s

// CQ placeholder matmul atom: fly.mma_atom_call -> vector.broadcast + arith on vectors

// CHECK-LABEL: @test_mma_atom_call_cq
// CHECK-SAME: (%[[D:.*]]: !llvm.ptr<5>, %[[A:.*]]: !llvm.ptr<5>, %[[B:.*]]: !llvm.ptr<5>, %[[C:.*]]: !llvm.ptr<5>)
func.func @test_mma_atom_call_cq(
    %d: !fly.memref<f32, register, 4:1>,
    %a: !fly.memref<f32, register, 1:1>,
    %b: !fly.memref<f32, register, 1:1>,
    %c: !fly.memref<f32, register, 4:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_cq.matmul_f32<16x16x4, (f32, f32) -> f32>>
  // CHECK: %[[A_VAL:.*]] = llvm.load %[[A]] : !llvm.ptr<5> -> f32
  // CHECK: %[[B_VAL:.*]] = llvm.load %[[B]] : !llvm.ptr<5> -> f32
  // CHECK: %[[C_VAL:.*]] = llvm.load %[[C]] : !llvm.ptr<5> -> vector<4xf32>
  // CHECK: vector.broadcast %[[A_VAL]]
  // CHECK: vector.broadcast %[[B_VAL]]
  // CHECK: arith.mulf
  // CHECK: arith.addf
  // CHECK: llvm.store {{.*}}, %[[D]] : vector<4xf32>, !llvm.ptr<5>
  fly.mma_atom_call(%atom, %d, %a, %b, %c) : (!fly.mma_atom<!fly_cq.matmul_f32<16x16x4, (f32, f32) -> f32>>, !fly.memref<f32, register, 4:1>, !fly.memref<f32, register, 1:1>, !fly.memref<f32, register, 1:1>, !fly.memref<f32, register, 4:1>) -> ()
  return
}
