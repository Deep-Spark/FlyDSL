// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: not %fly-opt %s --convert-fly-to-ixdl 2>&1 | FileCheck %s

// `convert-fly-to-ixdl` lowers an SME global-memory pointer to
// `!llvm.struct<(ptr<1>, i32, i32)>`.  The `scf.for` carried type must be
// converted at the same time; otherwise the body result requires an
// unrealized conversion cast back to the original `!fly.ptr` type.

// CHECK: failed to legalize unresolved materialization
// CHECK-SAME: !llvm.struct<(ptr<1>, i32, i32)>
// CHECK-SAME: !fly.ptr<f32, #fly_ixdl.sme_gmem>
// CHECK: scf.for
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
    scf.yield %next : !fly.ptr<f32, #fly_ixdl.sme_gmem>
  }
  return
}
