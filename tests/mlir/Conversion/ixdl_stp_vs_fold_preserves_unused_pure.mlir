// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-ixdl | FileCheck %s

// Pre-commit baseline for the stp.vs ptr-roundtrip fold.
//
// This records CURRENT (pre-fix) behavior. The greedy stp.vs fold seeds every
// op into the worklist and DCEs trivially-dead Pure results as a side effect,
// so unused add_offset / make_ptr lowerings vanish and only `return` remains.
// The follow-up fix scopes the fold to stp.vs calls and flips the FIXME checks
// below to assert the lowerings survive.

// === unused Pure: global f32 add_offset -> byte GEP (no consumer) ===
// FIXME(pre-fix): the byte-scaled GEP is DCE'd away.

// CHECK-LABEL: @unused_add_offset_global_b32
// CHECK-NOT:     arith.muli
// CHECK-NOT:     llvm.getelementptr
func.func @unused_add_offset_global_b32(%p: !fly.ptr<f32, global>, %dyn: i32) {
  %off = fly.make_int_tuple(%dyn) : (i32) -> !fly.int_tuple<?>
  %p2 = fly.add_offset(%p, %off)
      : (!fly.ptr<f32, global>, !fly.int_tuple<?>) -> !fly.ptr<f32, global>
  return
}

// === unused Pure: static shared make_ptr -> addressof (no consumer) ===
// FIXME(pre-fix): the addressof is DCE'd away.

// CHECK-LABEL: @unused_static_shared_make_ptr
// CHECK-NOT:     llvm.mlir.addressof
gpu.module @m {
  func.func @unused_static_shared_make_ptr() {
    %a = fly.make_ptr() {dictAttrs = {allocBytes = 512 : i64, allocAlign = 16 : i64}} : () -> !fly.ptr<i8, shared>
    return
  }
}

// === stp.vs fold still fires (unchanged by the fix) ===

// CHECK-LABEL: @stp_vs_b32_folded
// CHECK-SAME:  (%[[P:.*]]: !llvm.ptr<1>, %[[VAL:.*]]: i32, %[[VOFFSET:.*]]: i32, %[[SOFFSET:.*]]: i32)
// CHECK-NOT:     llvm.ptrtoint
// CHECK-NOT:     llvm.inttoptr
// CHECK:         %[[KOP:.*]] = arith.constant 0 : i32
// CHECK:         llvm.call_intrinsic "llvm.bi.stp.vs.i32"(%[[VAL]], %[[P]], %[[VOFFSET]], %[[SOFFSET]], %[[KOP]]) : (i32, !llvm.ptr<1>, i32, i32, i32) -> ()
func.func @stp_vs_b32_folded(%p: !fly.ptr<i32, global>, %val: i32, %voffset: i32, %soffset: i32) {
  %addr = fly.ptrtoint(%p) : (!fly.ptr<i32, global>) -> i64
  %lp = llvm.inttoptr %addr : i64 to !llvm.ptr<1>
  %kop = arith.constant 0 : i32
  llvm.call_intrinsic "llvm.bi.stp.vs.i32"(%val, %lp, %voffset, %soffset, %kop) : (i32, !llvm.ptr<1>, i32, i32, i32) -> ()
  return
}
