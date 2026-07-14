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

// CHECK-LABEL: @stp_vs_i8
// CHECK-SAME: (%[[P:.*]]: !llvm.ptr<1>, %[[VAL:.*]]: i8, %[[WCO:.*]]: i32, %[[WSO:.*]]: i32)
// CHECK-NOT: llvm.ptrtoint
// CHECK-NOT: llvm.inttoptr
// CHECK: %[[KOP:.*]] = arith.constant 0 : i32
// CHECK: llvm.call_intrinsic "llvm.bi.stp.vs.i8"(%[[VAL]], %[[P]], %[[WCO]], %[[WSO]], %[[KOP]]) : (i8, !llvm.ptr<1>, i32, i32, i32) -> ()
func.func @stp_vs_i8(%p: !fly.ptr<i8, global>, %val: i8, %wco: i32, %wso: i32) {
  %addr = fly.ptrtoint(%p) : (!fly.ptr<i8, global>) -> i64
  %lp = llvm.inttoptr %addr : i64 to !llvm.ptr<1>
  %kop = arith.constant 0 : i32
  llvm.call_intrinsic "llvm.bi.stp.vs.i8"(%val, %lp, %wco, %wso, %kop) : (i8, !llvm.ptr<1>, i32, i32, i32) -> ()
  return
}

// CHECK-LABEL: @stp_vs_i16
// CHECK-SAME: (%[[P:.*]]: !llvm.ptr<1>, %[[VAL:.*]]: i16, %[[WCO:.*]]: i32, %[[WSO:.*]]: i32)
// CHECK-NOT: llvm.ptrtoint
// CHECK-NOT: llvm.inttoptr
// CHECK: %[[KOP:.*]] = arith.constant 0 : i32
// CHECK: llvm.call_intrinsic "llvm.bi.stp.vs.i16"(%[[VAL]], %[[P]], %[[WCO]], %[[WSO]], %[[KOP]]) : (i16, !llvm.ptr<1>, i32, i32, i32) -> ()
func.func @stp_vs_i16(%p: !fly.ptr<i16, global>, %val: i16, %wco: i32, %wso: i32) {
  %addr = fly.ptrtoint(%p) : (!fly.ptr<i16, global>) -> i64
  %lp = llvm.inttoptr %addr : i64 to !llvm.ptr<1>
  %kop = arith.constant 0 : i32
  llvm.call_intrinsic "llvm.bi.stp.vs.i16"(%val, %lp, %wco, %wso, %kop) : (i16, !llvm.ptr<1>, i32, i32, i32) -> ()
  return
}

// CHECK-LABEL: @stp_vs_i32
// CHECK-SAME: (%[[P:.*]]: !llvm.ptr<1>, %[[VAL:.*]]: i32, %[[WCO:.*]]: i32, %[[WSO:.*]]: i32)
// CHECK-NOT: llvm.ptrtoint
// CHECK-NOT: llvm.inttoptr
// CHECK: %[[KOP:.*]] = arith.constant 0 : i32
// CHECK: llvm.call_intrinsic "llvm.bi.stp.vs.i32"(%[[VAL]], %[[P]], %[[WCO]], %[[WSO]], %[[KOP]]) : (i32, !llvm.ptr<1>, i32, i32, i32) -> ()
func.func @stp_vs_i32(%p: !fly.ptr<i32, global>, %val: i32, %wco: i32, %wso: i32) {
  %addr = fly.ptrtoint(%p) : (!fly.ptr<i32, global>) -> i64
  %lp = llvm.inttoptr %addr : i64 to !llvm.ptr<1>
  %kop = arith.constant 0 : i32
  llvm.call_intrinsic "llvm.bi.stp.vs.i32"(%val, %lp, %wco, %wso, %kop) : (i32, !llvm.ptr<1>, i32, i32, i32) -> ()
  return
}

// CHECK-LABEL: @stp_vs_i64
// CHECK-SAME: (%[[P:.*]]: !llvm.ptr<1>, %[[VAL:.*]]: i64, %[[WCO:.*]]: i32, %[[WSO:.*]]: i32)
// CHECK-NOT: llvm.ptrtoint
// CHECK-NOT: llvm.inttoptr
// CHECK: %[[KOP:.*]] = arith.constant 0 : i32
// CHECK: llvm.call_intrinsic "llvm.bi.stp.vs.i64"(%[[VAL]], %[[P]], %[[WCO]], %[[WSO]], %[[KOP]]) : (i64, !llvm.ptr<1>, i32, i32, i32) -> ()
func.func @stp_vs_i64(%p: !fly.ptr<i64, global>, %val: i64, %wco: i32, %wso: i32) {
  %addr = fly.ptrtoint(%p) : (!fly.ptr<i64, global>) -> i64
  %lp = llvm.inttoptr %addr : i64 to !llvm.ptr<1>
  %kop = arith.constant 0 : i32
  llvm.call_intrinsic "llvm.bi.stp.vs.i64"(%val, %lp, %wco, %wso, %kop) : (i64, !llvm.ptr<1>, i32, i32, i32) -> ()
  return
}

// CHECK-LABEL: @stp_vs_v4i32
// CHECK-SAME: (%[[P:.*]]: !llvm.ptr<1>, %[[VAL:.*]]: vector<4xi32>, %[[WCO:.*]]: i32, %[[WSO:.*]]: i32)
// CHECK-NOT: llvm.ptrtoint
// CHECK-NOT: llvm.inttoptr
// CHECK: %[[KOP:.*]] = arith.constant 0 : i32
// CHECK: llvm.call_intrinsic "llvm.bi.stp.vs.v4i32"(%[[VAL]], %[[P]], %[[WCO]], %[[WSO]], %[[KOP]]) : (vector<4xi32>, !llvm.ptr<1>, i32, i32, i32) -> ()
func.func @stp_vs_v4i32(%p: !fly.ptr<i32, global>, %val: vector<4xi32>, %wco: i32, %wso: i32) {
  %addr = fly.ptrtoint(%p) : (!fly.ptr<i32, global>) -> i64
  %lp = llvm.inttoptr %addr : i64 to !llvm.ptr<1>
  %kop = arith.constant 0 : i32
  llvm.call_intrinsic "llvm.bi.stp.vs.v4i32"(%val, %lp, %wco, %wso, %kop) : (vector<4xi32>, !llvm.ptr<1>, i32, i32, i32) -> ()
  return
}

// CHECK-LABEL: @stp_vs_pred_i8
// CHECK-SAME: (%[[P:.*]]: !llvm.ptr<1>, %[[VAL:.*]]: i8, %[[WCO:.*]]: i32, %[[WSO:.*]]: i32, %[[PRED:.*]]: i1)
// CHECK-NOT: llvm.ptrtoint
// CHECK-NOT: llvm.inttoptr
// CHECK: %[[KOP:.*]] = arith.constant 0 : i32
// CHECK: llvm.call_intrinsic "llvm.bi.stp.vs.pred.i8"(%[[VAL]], %[[P]], %[[WCO]], %[[WSO]], %[[KOP]], %[[PRED]]) : (i8, !llvm.ptr<1>, i32, i32, i32, i1) -> ()
func.func @stp_vs_pred_i8(%p: !fly.ptr<i8, global>, %val: i8, %wco: i32, %wso: i32, %pred: i1) {
  %addr = fly.ptrtoint(%p) : (!fly.ptr<i8, global>) -> i64
  %lp = llvm.inttoptr %addr : i64 to !llvm.ptr<1>
  %kop = arith.constant 0 : i32
  llvm.call_intrinsic "llvm.bi.stp.vs.pred.i8"(%val, %lp, %wco, %wso, %kop, %pred) : (i8, !llvm.ptr<1>, i32, i32, i32, i1) -> ()
  return
}

// CHECK-LABEL: @stp_vs_pred_i16
// CHECK-SAME: (%[[P:.*]]: !llvm.ptr<1>, %[[VAL:.*]]: i16, %[[WCO:.*]]: i32, %[[WSO:.*]]: i32, %[[PRED:.*]]: i1)
// CHECK-NOT: llvm.ptrtoint
// CHECK-NOT: llvm.inttoptr
// CHECK: %[[KOP:.*]] = arith.constant 0 : i32
// CHECK: llvm.call_intrinsic "llvm.bi.stp.vs.pred.i16"(%[[VAL]], %[[P]], %[[WCO]], %[[WSO]], %[[KOP]], %[[PRED]]) : (i16, !llvm.ptr<1>, i32, i32, i32, i1) -> ()
func.func @stp_vs_pred_i16(%p: !fly.ptr<i16, global>, %val: i16, %wco: i32, %wso: i32, %pred: i1) {
  %addr = fly.ptrtoint(%p) : (!fly.ptr<i16, global>) -> i64
  %lp = llvm.inttoptr %addr : i64 to !llvm.ptr<1>
  %kop = arith.constant 0 : i32
  llvm.call_intrinsic "llvm.bi.stp.vs.pred.i16"(%val, %lp, %wco, %wso, %kop, %pred) : (i16, !llvm.ptr<1>, i32, i32, i32, i1) -> ()
  return
}

// CHECK-LABEL: @stp_vs_pred_i32
// CHECK-SAME: (%[[P:.*]]: !llvm.ptr<1>, %[[VAL:.*]]: i32, %[[WCO:.*]]: i32, %[[WSO:.*]]: i32, %[[PRED:.*]]: i1)
// CHECK-NOT: llvm.ptrtoint
// CHECK-NOT: llvm.inttoptr
// CHECK: %[[KOP:.*]] = arith.constant 0 : i32
// CHECK: llvm.call_intrinsic "llvm.bi.stp.vs.pred.i32"(%[[VAL]], %[[P]], %[[WCO]], %[[WSO]], %[[KOP]], %[[PRED]]) : (i32, !llvm.ptr<1>, i32, i32, i32, i1) -> ()
func.func @stp_vs_pred_i32(%p: !fly.ptr<i32, global>, %val: i32, %wco: i32, %wso: i32, %pred: i1) {
  %addr = fly.ptrtoint(%p) : (!fly.ptr<i32, global>) -> i64
  %lp = llvm.inttoptr %addr : i64 to !llvm.ptr<1>
  %kop = arith.constant 0 : i32
  llvm.call_intrinsic "llvm.bi.stp.vs.pred.i32"(%val, %lp, %wco, %wso, %kop, %pred) : (i32, !llvm.ptr<1>, i32, i32, i32, i1) -> ()
  return
}

// CHECK-LABEL: @stp_vs_pred_i64
// CHECK-SAME: (%[[P:.*]]: !llvm.ptr<1>, %[[VAL:.*]]: i64, %[[WCO:.*]]: i32, %[[WSO:.*]]: i32, %[[PRED:.*]]: i1)
// CHECK-NOT: llvm.ptrtoint
// CHECK-NOT: llvm.inttoptr
// CHECK: %[[KOP:.*]] = arith.constant 0 : i32
// CHECK: llvm.call_intrinsic "llvm.bi.stp.vs.pred.i64"(%[[VAL]], %[[P]], %[[WCO]], %[[WSO]], %[[KOP]], %[[PRED]]) : (i64, !llvm.ptr<1>, i32, i32, i32, i1) -> ()
func.func @stp_vs_pred_i64(%p: !fly.ptr<i64, global>, %val: i64, %wco: i32, %wso: i32, %pred: i1) {
  %addr = fly.ptrtoint(%p) : (!fly.ptr<i64, global>) -> i64
  %lp = llvm.inttoptr %addr : i64 to !llvm.ptr<1>
  %kop = arith.constant 0 : i32
  llvm.call_intrinsic "llvm.bi.stp.vs.pred.i64"(%val, %lp, %wco, %wso, %kop, %pred) : (i64, !llvm.ptr<1>, i32, i32, i32, i1) -> ()
  return
}

// CHECK-LABEL: @stp_vs_pred_v4i32
// CHECK-SAME: (%[[P:.*]]: !llvm.ptr<1>, %[[VAL:.*]]: vector<4xi32>, %[[WCO:.*]]: i32, %[[WSO:.*]]: i32, %[[PRED:.*]]: i1)
// CHECK-NOT: llvm.ptrtoint
// CHECK-NOT: llvm.inttoptr
// CHECK: %[[KOP:.*]] = arith.constant 0 : i32
// CHECK: llvm.call_intrinsic "llvm.bi.stp.vs.pred.v4i32"(%[[VAL]], %[[P]], %[[WCO]], %[[WSO]], %[[KOP]], %[[PRED]]) : (vector<4xi32>, !llvm.ptr<1>, i32, i32, i32, i1) -> ()
func.func @stp_vs_pred_v4i32(%p: !fly.ptr<i32, global>, %val: vector<4xi32>, %wco: i32, %wso: i32, %pred: i1) {
  %addr = fly.ptrtoint(%p) : (!fly.ptr<i32, global>) -> i64
  %lp = llvm.inttoptr %addr : i64 to !llvm.ptr<1>
  %kop = arith.constant 0 : i32
  llvm.call_intrinsic "llvm.bi.stp.vs.pred.v4i32"(%val, %lp, %wco, %wso, %kop, %pred) : (vector<4xi32>, !llvm.ptr<1>, i32, i32, i32, i1) -> ()
  return
}
