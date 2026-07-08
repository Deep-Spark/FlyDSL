// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-ixdl | FileCheck %s

// Static shared-memory make_ptr lowering.
//
// A shared-addrspace `fly.make_ptr` carrying {allocBytes, allocAlign} lowers to
// a freshly named `@__shared_alloc_<n>` shared-memory global of `[allocBytes x i8]`
// in addrspace(3) (alignment = allocAlign) plus an `llvm.mlir.addressof` producing
// an `!llvm.ptr<3>`, mirroring the static-shared global path in FlyToROCDL.
// Distinct make_ptr ops get distinct, uniqued globals; size and alignment come
// straight from the dict attrs. Globals are inserted at the module start, so
// CHECK-DAG tolerates their emission order.

// CHECK-DAG: llvm.mlir.global external @__shared_alloc_0() {addr_space = 3 : i32, alignment = 16 : i64, dso_local} : !llvm.array<512 x i8>
// CHECK-DAG: llvm.mlir.global external @__shared_alloc_1() {addr_space = 3 : i32, alignment = 32 : i64, dso_local} : !llvm.array<1024 x i8>

// CHECK-LABEL: @static_shared
// CHECK: llvm.mlir.addressof @__shared_alloc_0 : !llvm.ptr<3>
// CHECK: llvm.mlir.addressof @__shared_alloc_1 : !llvm.ptr<3>
gpu.module @m {
  func.func @static_shared() {
    %a = fly.make_ptr() {dictAttrs = {allocBytes = 512 : i64, allocAlign = 16 : i64}} : () -> !fly.ptr<i8, shared>
    %b = fly.make_ptr() {dictAttrs = {allocBytes = 1024 : i64, allocAlign = 32 : i64}} : () -> !fly.ptr<i8, shared>
    return
  }
}
