// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-ixdl | FileCheck %s

// `convert-fly-to-ixdl` lowers both the SME global-memory pointer and the
// `scf.for` carrying it to `!llvm.struct<(ptr<1>, i32, i32)>`.

// CHECK-LABEL: func.func @carry_sme_pointer(
// CHECK-SAME: %[[CURSOR:.*]]: !llvm.struct<(ptr<1>, i32, i32)>, %[[STEPS:.*]]: i32)
// CHECK: %[[RESULT:.*]] = scf.for
// CHECK-SAME: iter_args(%[[PTR:.*]] = %[[CURSOR]])
// CHECK-SAME: -> (!llvm.struct<(ptr<1>, i32, i32)>)
// CHECK: llvm.extractvalue %[[PTR]][2] : !llvm.struct<(ptr<1>, i32, i32)>
// CHECK: scf.yield {{.*}} : !llvm.struct<(ptr<1>, i32, i32)>
func.func @carry_sme_pointer(
    %cursor: !fly.ptr<f32, #fly_ixdl.sme_gmem>, %steps: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %offset = fly.static : !fly.int_tuple<256>
  %result = scf.for %i = %c0 to %steps step %c1 iter_args(%ptr = %cursor)
      -> (!fly.ptr<f32, #fly_ixdl.sme_gmem>) : i32 {
    %next = fly.add_offset(%ptr, %offset)
        : (!fly.ptr<f32, #fly_ixdl.sme_gmem>, !fly.int_tuple<256>)
          -> !fly.ptr<f32, #fly_ixdl.sme_gmem>
    gpu.barrier
    scf.yield %next : !fly.ptr<f32, #fly_ixdl.sme_gmem>
  }
  return
}
