// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-ixdl | FileCheck %s

// FoldStpVsPtrRoundtrip.
//
// Why the round-trip exists: tracing has ``!fly.ptr``, but ``llvm.bi.stp.vs*``
// via ``llvm.call_intrinsic`` needs ``!llvm.ptr``. The Python helpers cross
// with ``fly.ptrtoint`` / ``llvm.inttoptr`` instead of an unrealized cast
// (which deadlocks dialect conversion). After convert-fly-to-ixdl rewrites
// ``fly.ptr`` to ``llvm.ptr``, the round-trip is folded so ISel sees a clean
// descriptor base. Covers i8 / i16 / i32 / i64 / v4i32 and their ``.pred`` forms.

// CHECK-LABEL: @stp_vs_b8
// CHECK-SAME: (%[[P:.*]]: !llvm.ptr<1>, %[[VAL:.*]]: i8, %[[VOFFSET:.*]]: i32, %[[SOFFSET:.*]]: i32)
// CHECK-NOT: llvm.ptrtoint
// CHECK-NOT: llvm.inttoptr
// CHECK: %[[KOP:.*]] = arith.constant 0 : i32
// CHECK: llvm.call_intrinsic "llvm.bi.stp.vs.i8"(%[[VAL]], %[[P]], %[[VOFFSET]], %[[SOFFSET]], %[[KOP]]) : (i8, !llvm.ptr<1>, i32, i32, i32) -> ()
func.func @stp_vs_b8(%p: !fly.ptr<i8, global>, %val: i8, %voffset: i32, %soffset: i32) {
  %addr = fly.ptrtoint(%p) : (!fly.ptr<i8, global>) -> i64
  %lp = llvm.inttoptr %addr : i64 to !llvm.ptr<1>
  %kop = arith.constant 0 : i32
  llvm.call_intrinsic "llvm.bi.stp.vs.i8"(%val, %lp, %voffset, %soffset, %kop) : (i8, !llvm.ptr<1>, i32, i32, i32) -> ()
  return
}

// CHECK-LABEL: @stp_vs_b16
// CHECK-SAME: (%[[P:.*]]: !llvm.ptr<1>, %[[VAL:.*]]: i16, %[[VOFFSET:.*]]: i32, %[[SOFFSET:.*]]: i32)
// CHECK-NOT: llvm.ptrtoint
// CHECK-NOT: llvm.inttoptr
// CHECK: %[[KOP:.*]] = arith.constant 0 : i32
// CHECK: llvm.call_intrinsic "llvm.bi.stp.vs.i16"(%[[VAL]], %[[P]], %[[VOFFSET]], %[[SOFFSET]], %[[KOP]]) : (i16, !llvm.ptr<1>, i32, i32, i32) -> ()
func.func @stp_vs_b16(%p: !fly.ptr<i16, global>, %val: i16, %voffset: i32, %soffset: i32) {
  %addr = fly.ptrtoint(%p) : (!fly.ptr<i16, global>) -> i64
  %lp = llvm.inttoptr %addr : i64 to !llvm.ptr<1>
  %kop = arith.constant 0 : i32
  llvm.call_intrinsic "llvm.bi.stp.vs.i16"(%val, %lp, %voffset, %soffset, %kop) : (i16, !llvm.ptr<1>, i32, i32, i32) -> ()
  return
}

// CHECK-LABEL: @stp_vs_b32
// CHECK-SAME: (%[[P:.*]]: !llvm.ptr<1>, %[[VAL:.*]]: i32, %[[VOFFSET:.*]]: i32, %[[SOFFSET:.*]]: i32)
// CHECK-NOT: llvm.ptrtoint
// CHECK-NOT: llvm.inttoptr
// CHECK: %[[KOP:.*]] = arith.constant 0 : i32
// CHECK: llvm.call_intrinsic "llvm.bi.stp.vs.i32"(%[[VAL]], %[[P]], %[[VOFFSET]], %[[SOFFSET]], %[[KOP]]) : (i32, !llvm.ptr<1>, i32, i32, i32) -> ()
func.func @stp_vs_b32(%p: !fly.ptr<i32, global>, %val: i32, %voffset: i32, %soffset: i32) {
  %addr = fly.ptrtoint(%p) : (!fly.ptr<i32, global>) -> i64
  %lp = llvm.inttoptr %addr : i64 to !llvm.ptr<1>
  %kop = arith.constant 0 : i32
  llvm.call_intrinsic "llvm.bi.stp.vs.i32"(%val, %lp, %voffset, %soffset, %kop) : (i32, !llvm.ptr<1>, i32, i32, i32) -> ()
  return
}

// CHECK-LABEL: @stp_vs_b64
// CHECK-SAME: (%[[P:.*]]: !llvm.ptr<1>, %[[VAL:.*]]: i64, %[[VOFFSET:.*]]: i32, %[[SOFFSET:.*]]: i32)
// CHECK-NOT: llvm.ptrtoint
// CHECK-NOT: llvm.inttoptr
// CHECK: %[[KOP:.*]] = arith.constant 0 : i32
// CHECK: llvm.call_intrinsic "llvm.bi.stp.vs.i64"(%[[VAL]], %[[P]], %[[VOFFSET]], %[[SOFFSET]], %[[KOP]]) : (i64, !llvm.ptr<1>, i32, i32, i32) -> ()
func.func @stp_vs_b64(%p: !fly.ptr<i64, global>, %val: i64, %voffset: i32, %soffset: i32) {
  %addr = fly.ptrtoint(%p) : (!fly.ptr<i64, global>) -> i64
  %lp = llvm.inttoptr %addr : i64 to !llvm.ptr<1>
  %kop = arith.constant 0 : i32
  llvm.call_intrinsic "llvm.bi.stp.vs.i64"(%val, %lp, %voffset, %soffset, %kop) : (i64, !llvm.ptr<1>, i32, i32, i32) -> ()
  return
}

// CHECK-LABEL: @stp_vs_b128
// CHECK-SAME: (%[[P:.*]]: !llvm.ptr<1>, %[[VAL:.*]]: vector<4xi32>, %[[VOFFSET:.*]]: i32, %[[SOFFSET:.*]]: i32)
// CHECK-NOT: llvm.ptrtoint
// CHECK-NOT: llvm.inttoptr
// CHECK: %[[KOP:.*]] = arith.constant 0 : i32
// CHECK: llvm.call_intrinsic "llvm.bi.stp.vs.v4i32"(%[[VAL]], %[[P]], %[[VOFFSET]], %[[SOFFSET]], %[[KOP]]) : (vector<4xi32>, !llvm.ptr<1>, i32, i32, i32) -> ()
func.func @stp_vs_b128(%p: !fly.ptr<i32, global>, %val: vector<4xi32>, %voffset: i32, %soffset: i32) {
  %addr = fly.ptrtoint(%p) : (!fly.ptr<i32, global>) -> i64
  %lp = llvm.inttoptr %addr : i64 to !llvm.ptr<1>
  %kop = arith.constant 0 : i32
  llvm.call_intrinsic "llvm.bi.stp.vs.v4i32"(%val, %lp, %voffset, %soffset, %kop) : (vector<4xi32>, !llvm.ptr<1>, i32, i32, i32) -> ()
  return
}

// CHECK-LABEL: @stp_vs_pred_b8
// CHECK-SAME: (%[[P:.*]]: !llvm.ptr<1>, %[[VAL:.*]]: i8, %[[VOFFSET:.*]]: i32, %[[SOFFSET:.*]]: i32, %[[PRED:.*]]: i1)
// CHECK-NOT: llvm.ptrtoint
// CHECK-NOT: llvm.inttoptr
// CHECK: %[[KOP:.*]] = arith.constant 0 : i32
// CHECK: llvm.call_intrinsic "llvm.bi.stp.vs.pred.i8"(%[[VAL]], %[[P]], %[[VOFFSET]], %[[SOFFSET]], %[[KOP]], %[[PRED]]) : (i8, !llvm.ptr<1>, i32, i32, i32, i1) -> ()
func.func @stp_vs_pred_b8(%p: !fly.ptr<i8, global>, %val: i8, %voffset: i32, %soffset: i32, %pred: i1) {
  %addr = fly.ptrtoint(%p) : (!fly.ptr<i8, global>) -> i64
  %lp = llvm.inttoptr %addr : i64 to !llvm.ptr<1>
  %kop = arith.constant 0 : i32
  llvm.call_intrinsic "llvm.bi.stp.vs.pred.i8"(%val, %lp, %voffset, %soffset, %kop, %pred) : (i8, !llvm.ptr<1>, i32, i32, i32, i1) -> ()
  return
}

// CHECK-LABEL: @stp_vs_pred_b16
// CHECK-SAME: (%[[P:.*]]: !llvm.ptr<1>, %[[VAL:.*]]: i16, %[[VOFFSET:.*]]: i32, %[[SOFFSET:.*]]: i32, %[[PRED:.*]]: i1)
// CHECK-NOT: llvm.ptrtoint
// CHECK-NOT: llvm.inttoptr
// CHECK: %[[KOP:.*]] = arith.constant 0 : i32
// CHECK: llvm.call_intrinsic "llvm.bi.stp.vs.pred.i16"(%[[VAL]], %[[P]], %[[VOFFSET]], %[[SOFFSET]], %[[KOP]], %[[PRED]]) : (i16, !llvm.ptr<1>, i32, i32, i32, i1) -> ()
func.func @stp_vs_pred_b16(%p: !fly.ptr<i16, global>, %val: i16, %voffset: i32, %soffset: i32, %pred: i1) {
  %addr = fly.ptrtoint(%p) : (!fly.ptr<i16, global>) -> i64
  %lp = llvm.inttoptr %addr : i64 to !llvm.ptr<1>
  %kop = arith.constant 0 : i32
  llvm.call_intrinsic "llvm.bi.stp.vs.pred.i16"(%val, %lp, %voffset, %soffset, %kop, %pred) : (i16, !llvm.ptr<1>, i32, i32, i32, i1) -> ()
  return
}

// CHECK-LABEL: @stp_vs_pred_b32
// CHECK-SAME: (%[[P:.*]]: !llvm.ptr<1>, %[[VAL:.*]]: i32, %[[VOFFSET:.*]]: i32, %[[SOFFSET:.*]]: i32, %[[PRED:.*]]: i1)
// CHECK-NOT: llvm.ptrtoint
// CHECK-NOT: llvm.inttoptr
// CHECK: %[[KOP:.*]] = arith.constant 0 : i32
// CHECK: llvm.call_intrinsic "llvm.bi.stp.vs.pred.i32"(%[[VAL]], %[[P]], %[[VOFFSET]], %[[SOFFSET]], %[[KOP]], %[[PRED]]) : (i32, !llvm.ptr<1>, i32, i32, i32, i1) -> ()
func.func @stp_vs_pred_b32(%p: !fly.ptr<i32, global>, %val: i32, %voffset: i32, %soffset: i32, %pred: i1) {
  %addr = fly.ptrtoint(%p) : (!fly.ptr<i32, global>) -> i64
  %lp = llvm.inttoptr %addr : i64 to !llvm.ptr<1>
  %kop = arith.constant 0 : i32
  llvm.call_intrinsic "llvm.bi.stp.vs.pred.i32"(%val, %lp, %voffset, %soffset, %kop, %pred) : (i32, !llvm.ptr<1>, i32, i32, i32, i1) -> ()
  return
}

// CHECK-LABEL: @stp_vs_pred_b64
// CHECK-SAME: (%[[P:.*]]: !llvm.ptr<1>, %[[VAL:.*]]: i64, %[[VOFFSET:.*]]: i32, %[[SOFFSET:.*]]: i32, %[[PRED:.*]]: i1)
// CHECK-NOT: llvm.ptrtoint
// CHECK-NOT: llvm.inttoptr
// CHECK: %[[KOP:.*]] = arith.constant 0 : i32
// CHECK: llvm.call_intrinsic "llvm.bi.stp.vs.pred.i64"(%[[VAL]], %[[P]], %[[VOFFSET]], %[[SOFFSET]], %[[KOP]], %[[PRED]]) : (i64, !llvm.ptr<1>, i32, i32, i32, i1) -> ()
func.func @stp_vs_pred_b64(%p: !fly.ptr<i64, global>, %val: i64, %voffset: i32, %soffset: i32, %pred: i1) {
  %addr = fly.ptrtoint(%p) : (!fly.ptr<i64, global>) -> i64
  %lp = llvm.inttoptr %addr : i64 to !llvm.ptr<1>
  %kop = arith.constant 0 : i32
  llvm.call_intrinsic "llvm.bi.stp.vs.pred.i64"(%val, %lp, %voffset, %soffset, %kop, %pred) : (i64, !llvm.ptr<1>, i32, i32, i32, i1) -> ()
  return
}

// CHECK-LABEL: @stp_vs_pred_b128
// CHECK-SAME: (%[[P:.*]]: !llvm.ptr<1>, %[[VAL:.*]]: vector<4xi32>, %[[VOFFSET:.*]]: i32, %[[SOFFSET:.*]]: i32, %[[PRED:.*]]: i1)
// CHECK-NOT: llvm.ptrtoint
// CHECK-NOT: llvm.inttoptr
// CHECK: %[[KOP:.*]] = arith.constant 0 : i32
// CHECK: llvm.call_intrinsic "llvm.bi.stp.vs.pred.v4i32"(%[[VAL]], %[[P]], %[[VOFFSET]], %[[SOFFSET]], %[[KOP]], %[[PRED]]) : (vector<4xi32>, !llvm.ptr<1>, i32, i32, i32, i1) -> ()
func.func @stp_vs_pred_b128(%p: !fly.ptr<i32, global>, %val: vector<4xi32>, %voffset: i32, %soffset: i32, %pred: i1) {
  %addr = fly.ptrtoint(%p) : (!fly.ptr<i32, global>) -> i64
  %lp = llvm.inttoptr %addr : i64 to !llvm.ptr<1>
  %kop = arith.constant 0 : i32
  llvm.call_intrinsic "llvm.bi.stp.vs.pred.v4i32"(%val, %lp, %voffset, %soffset, %kop, %pred) : (vector<4xi32>, !llvm.ptr<1>, i32, i32, i32, i1) -> ()
  return
}
