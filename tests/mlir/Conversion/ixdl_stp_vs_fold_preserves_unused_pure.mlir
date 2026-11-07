// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-ixdl | FileCheck %s

// Regression for foldStpVsPtrRoundtrips.
//
// The stp.vs inttoptr(ptrtoint) peephole must not whole-module DCE unused Pure
// results. The old greedy fold seeded every op into the worklist and erased
// trivially-dead Pure ops as a side effect. The scoped fold now only rewrites
// stp.vs calls, so unused add_offset / make_ptr lowerings survive while the
// stp.vs base still folds.

// === unused Pure: global f32 add_offset -> byte GEP (no consumer) ===
// The byte-scaled GEP must survive even though its result has no consumer.

// CHECK-LABEL: @unused_add_offset_global_b32
// CHECK:         %[[EB:.*]] = arith.constant 4 : i32
// CHECK:         %[[BYTES:.*]] = arith.muli %{{.*}}, %[[EB]] : i32
// CHECK:         llvm.getelementptr %{{.*}}[%[[BYTES]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i8
func.func @unused_add_offset_global_b32(%p: !fly.ptr<f32, global>, %dyn: i32) {
  %off = fly.make_int_tuple(%dyn) : (i32) -> !fly.int_tuple<?>
  %p2 = fly.add_offset(%p, %off)
      : (!fly.ptr<f32, global>, !fly.int_tuple<?>) -> !fly.ptr<f32, global>
  return
}

// === unused Pure: static shared make_ptr -> addressof (no consumer) ===
// The addressof must survive even though its result has no consumer.

// CHECK-LABEL: @unused_static_shared_make_ptr
// CHECK:         llvm.mlir.addressof @__shared_alloc_0 : !llvm.ptr<3>
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
