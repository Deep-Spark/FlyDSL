// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-rewrite-func-signature --fly-canonicalize --fly-layout-lowering --convert-fly-to-ixdl | FileCheck %s

// MR SLBLoad atom call lowering (Shared -> Register fragment):
//   fly.copy_atom_call -> llvm.load (no hardware ldmatrix on MR).

// CHECK-LABEL: @slb
// CHECK-SAME: (%[[DST:.*]]: !llvm.ptr<5>, %[[SRC:.*]]: !llvm.ptr<3>)
func.func @slb(
    %dst: !fly.memref<f16, register, 4:1>,
    %src: !fly.memref<f16, shared, 4:1>) {
  %atom = fly.make_copy_atom {valBits = 16 : i32} : !fly.copy_atom<!fly_ixdl.mr.slb_load<16>, 16>
  // CHECK: %[[V:.*]] = llvm.load %[[SRC]] : !llvm.ptr<3> -> vector<4xf16>
  // CHECK: llvm.store %[[V]], %[[DST]] : vector<4xf16>, !llvm.ptr<5>
  fly.copy_atom_call(%atom, %src, %dst) : (!fly.copy_atom<!fly_ixdl.mr.slb_load<16>, 16>, !fly.memref<f16, shared, 4:1>, !fly.memref<f16, register, 4:1>) -> ()
  return
}
