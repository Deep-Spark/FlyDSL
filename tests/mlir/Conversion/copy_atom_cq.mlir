// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-cq-opt %s --fly-rewrite-func-signature --fly-canonicalize --fly-layout-lowering --convert-fly-to-cq | FileCheck %s

// CQ placeholder scalar_mem copy: fly.copy_atom_call -> llvm.load + llvm.store (register memrefs)

// CHECK-LABEL: @test_copy_scalar_mem_cq
func.func @test_copy_scalar_mem_cq(
    %src: !fly.memref<f32, register, 1:1>,
    %dst: !fly.memref<f32, register, 1:1>) {
  // Stateless atom: make_copy_atom lowers to llvm.mlir.undef of empty struct
  // CHECK: %[[ATOM:.*]] = llvm.mlir.undef : !llvm.struct<()>
  %atom = fly.make_copy_atom {valBits = 32 : i32} : !fly.copy_atom<!fly_cq.scalar_mem<32>, 32>
  // CHECK: %[[V:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> f32
  // CHECK: llvm.store %[[V]], %{{.*}} : f32, !llvm.ptr<5>
  fly.copy_atom_call(%atom, %src, %dst) : (!fly.copy_atom<!fly_cq.scalar_mem<32>, 32>, !fly.memref<f32, register, 1:1>, !fly.memref<f32, register, 1:1>) -> ()
  return
}
