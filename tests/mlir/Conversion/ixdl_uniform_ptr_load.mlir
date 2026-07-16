// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-ixdl | FileCheck %s

// A uniform global load uses a readfirstlane-normalized address and is marked
// invariant so the Iluvatar backend can select an sl_lsa_load instruction.

// CHECK-LABEL: @uniform_ptr_load
// CHECK-SAME: (%[[PTR:.*]]: !llvm.ptr<1>)
// CHECK: %[[ADDR:.*]] = llvm.ptrtoint %[[PTR]] : !llvm.ptr<1> to i64
// CHECK: %[[UNIFORM:.*]] = llvm.call_intrinsic "llvm.bi.readfirstlane.i64"(%[[ADDR]]) : (i64) -> i64
// CHECK: %[[UPTR:.*]] = llvm.inttoptr %[[UNIFORM]] : i64 to !llvm.ptr<1>
// CHECK: %[[VALUE:.*]] = llvm.load %[[UPTR]] invariant : !llvm.ptr<1> -> i32
// CHECK: return %[[VALUE]] : i32
func.func @uniform_ptr_load(%ptr: !fly.ptr<i32, global>) -> i32 {
  %value = fly.ptr.load(%ptr) {uniform = true} : (!fly.ptr<i32, global>) -> i32
  return %value : i32
}
