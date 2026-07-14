// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-ixdl | FileCheck %s

// FoldStpVsPtrRoundtrip.
//
// Why the round-trip exists: tracing has ``!fly.ptr``, but ``llvm.bi.stp.vs.i32``
// via ``llvm.call_intrinsic`` needs ``!llvm.ptr``. The Python helper crosses
// with ``fly.ptrtoint`` → ``llvm.inttoptr`` instead of an unrealized cast
// (which deadlocks dialect conversion). After convert-fly-to-ixdl rewrites
// ``fly.ptr`` → ``llvm.ptr``, the round-trip is folded so ISel sees a clean
// descriptor base.
//
// Input mirrors ``stp_vs_i32``; the ``stp.vs`` pointer arg must be the
// converted global base (no inttoptr/ptrtoint).

// CHECK-LABEL: @stp_vs_ptr_roundtrip
// CHECK-SAME: (%[[P:.*]]: !llvm.ptr<1>, %[[VAL:.*]]: i32, %[[WCO:.*]]: i32, %[[WSO:.*]]: i32)
// CHECK-NOT: llvm.ptrtoint
// CHECK-NOT: llvm.inttoptr
// CHECK: %[[KOP:.*]] = arith.constant 0 : i32
// CHECK: llvm.call_intrinsic "llvm.bi.stp.vs.i32"(%[[VAL]], %[[P]], %[[WCO]], %[[WSO]], %[[KOP]]) : (i32, !llvm.ptr<1>, i32, i32, i32) -> ()
func.func @stp_vs_ptr_roundtrip(%p: !fly.ptr<i32, global>, %val: i32, %wco: i32, %wso: i32) {
  %addr = fly.ptrtoint(%p) : (!fly.ptr<i32, global>) -> i64
  %lp = llvm.inttoptr %addr : i64 to !llvm.ptr<1>
  %kop = arith.constant 0 : i32
  llvm.call_intrinsic "llvm.bi.stp.vs.i32"(%val, %lp, %wco, %wso, %kop) : (i32, !llvm.ptr<1>, i32, i32, i32) -> ()
  return
}
