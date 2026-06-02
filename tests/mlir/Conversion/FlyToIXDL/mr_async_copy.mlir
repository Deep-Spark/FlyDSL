// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-rewrite-func-signature --fly-canonicalize --fly-layout-lowering --convert-fly-to-ixdl | FileCheck %s

// MR stateful async G2S copy lowering (Global -> Shared SME cp.async):
//   fly.copy_atom_call -> ixdl.cp.async with a vector<4xi32> descriptor
//   (addr_lo, addr_hi, -1, stride_bytes).

// CHECK-LABEL: @async
// CHECK-SAME: (%[[ATOM:.*]]: !llvm.struct<(i32, i32)>, %[[SRC:.*]]: !llvm.ptr<1>, %[[DST:.*]]: !llvm.ptr<3>)
func.func @async(
    %atom: !fly.copy_atom<!fly_ixdl.mr.async_copy.16x64.b8.row<128>, 128>,
    %src: !fly.memref<f16, global, 32:1>,
    %dst: !fly.memref<f16, shared, 32:1>) {
  // CHECK: %[[IMM:.*]] = llvm.extractvalue %[[ATOM]][1]
  // CHECK: %[[DA:.*]] = llvm.ptrtoint %[[DST]] : !llvm.ptr<3> to i64
  // CHECK: %[[DA32:.*]] = arith.trunci %[[DA]] : i64 to i32
  // CHECK: %[[SOFF:.*]] = arith.addi %[[DA32]], %[[IMM]] : i32
  // CHECK: %[[GA:.*]] = llvm.ptrtoint %[[SRC]] : !llvm.ptr<1> to i64
  // CHECK: %[[POISON:.*]] = llvm.mlir.poison : vector<4xi32>
  // CHECK: vector.insert
  // CHECK: %[[DESC:.*]] = vector.insert %{{.*}}, %{{.*}} [%c3{{.*}}] : i32 into vector<4xi32>
  // f16 row-major operand lowers to rowxfb16: shape [16, 32], elementSize 16.
  // CHECK: ixdl.cp.async %[[SOFF]], %[[DESC]], %{{.*}}, %{{.*}}, [16, 32], 16, false : vector<4xi32> -> i32
  fly.copy_atom_call(%atom, %src, %dst) : (!fly.copy_atom<!fly_ixdl.mr.async_copy.16x64.b8.row<128>, 128>, !fly.memref<f16, global, 32:1>, !fly.memref<f16, shared, 32:1>) -> ()
  return
}
