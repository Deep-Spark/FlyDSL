// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-ixdl | FileCheck %s

// add_offset (fly.ptr GEP) byte-addressing optimization for global memory.
//
// On Iluvatar an element-typed GEP into global memory lowers to a *scaled*,
// sign-extended index. The backend has no zoom encoding for a scaled signed
// moffset, so it cannot fold the index into the descriptor store and falls back
// to 64-bit absolute addressing (ml_lsa_store_a64_*), costing an addo/addc per
// store. Rewriting the GEP to an i8 (byte) element type with an explicit
// byte-offset keeps the index a plain sign-extended i32, which the backend
// matches to the cheaper descriptor store (ml_lsa_store_*_U).
//
// Before/after in one pass run: the rewrite only fires for a >8-bit *global*
// add_offset. Every function below starts from the same `fly.add_offset(%p,
// %off)`; compare the two f32 cases to read the effect directly:
//
//   BEFORE (element-typed GEP, -> ml_lsa_store_a64_*): @add_offset_shared_b32
//           llvm.getelementptr %p[%idx] ..., f32          (raw index, no scale)
//   AFTER  (byte i8 GEP,       -> ml_lsa_store_*_U):    @add_offset_global_b32
//           %b = arith.muli %idx, 4; llvm.getelementptr %p[%b] ..., i8
//
// The only difference between those two inputs is the address space, and the
// pattern rewrites only the global one -- so @add_offset_shared_b32 is exactly
// the IR @add_offset_global_b32 would emit without this optimization. The i8
// global case (already byte addressed) is unaffected and stays element-typed.

// === AFTER (optimized): global f32 (b32) -> index * 4, i8-typed GEP ===

// CHECK-LABEL: @add_offset_global_b32
// CHECK-SAME: (%[[P:.*]]: !llvm.ptr<1>, %[[IDX:.*]]: i32)
// CHECK: %[[EB:.*]] = arith.constant 4 : i32
// CHECK: %[[BYTES:.*]] = arith.muli %[[IDX]], %[[EB]] : i32
// CHECK: llvm.getelementptr %[[P]][%[[BYTES]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i8
func.func @add_offset_global_b32(%p: !fly.ptr<f32, global>, %dyn: i32) -> f32 {
  %off = fly.make_int_tuple(%dyn) : (i32) -> !fly.int_tuple<?>
  %p2 = fly.add_offset(%p, %off)
      : (!fly.ptr<f32, global>, !fly.int_tuple<?>) -> !fly.ptr<f32, global>
  %value = fly.ptr.load(%p2) : (!fly.ptr<f32, global>) -> f32
  return %value : f32
}

// === AFTER (optimized): global f16 (b16) -> index * 2, i8-typed GEP (scale generalizes) ===

// CHECK-LABEL: @add_offset_global_b16
// CHECK-SAME: (%[[P:.*]]: !llvm.ptr<1>, %[[IDX:.*]]: i32)
// CHECK: %[[EB:.*]] = arith.constant 2 : i32
// CHECK: %[[BYTES:.*]] = arith.muli %[[IDX]], %[[EB]] : i32
// CHECK: llvm.getelementptr %[[P]][%[[BYTES]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i8
func.func @add_offset_global_b16(%p: !fly.ptr<f16, global>, %dyn: i32) -> f16 {
  %off = fly.make_int_tuple(%dyn) : (i32) -> !fly.int_tuple<?>
  %p2 = fly.add_offset(%p, %off)
      : (!fly.ptr<f16, global>, !fly.int_tuple<?>) -> !fly.ptr<f16, global>
  %value = fly.ptr.load(%p2) : (!fly.ptr<f16, global>) -> f16
  return %value : f16
}

// === Large global offsets preserve i64 through byte scaling ===

// CHECK-LABEL: @add_offset_global_b16_i64
// CHECK-SAME: (%[[P:.*]]: !llvm.ptr<1>, %[[IDX:.*]]: i64)
// CHECK: %[[EB:.*]] = arith.constant 2 : i64
// CHECK: %[[BYTES:.*]] = arith.muli %[[IDX]], %[[EB]] : i64
// CHECK: llvm.getelementptr %[[P]][%[[BYTES]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
func.func @add_offset_global_b16_i64(%p: !fly.ptr<f16, global>, %dyn: i64) -> f16 {
  %off = fly.make_int_tuple(%dyn) : (i64) -> !fly.int_tuple<?>
  %p2 = fly.add_offset(%p, %off)
      : (!fly.ptr<f16, global>, !fly.int_tuple<?>) -> !fly.ptr<f16, global>
  %value = fly.ptr.load(%p2) : (!fly.ptr<f16, global>) -> f16
  return %value : f16
}

// === N/A (already byte-addressed): global i8 -> no scale, i8-typed GEP ===

// CHECK-LABEL: @add_offset_global_b8
// CHECK-SAME: (%[[P:.*]]: !llvm.ptr<1>, %[[IDX:.*]]: i32)
// CHECK-NOT: arith.muli
// CHECK: llvm.getelementptr %[[P]][%[[IDX]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i8
func.func @add_offset_global_b8(%p: !fly.ptr<i8, global>, %dyn: i32) -> i8 {
  %off = fly.make_int_tuple(%dyn) : (i32) -> !fly.int_tuple<?>
  %p2 = fly.add_offset(%p, %off)
      : (!fly.ptr<i8, global>, !fly.int_tuple<?>) -> !fly.ptr<i8, global>
  %value = fly.ptr.load(%p2) : (!fly.ptr<i8, global>) -> i8
  return %value : i8
}

// === BEFORE (baseline): same f32 add_offset, shared space -> element-typed GEP, no scale ===
// This is the exact form @add_offset_global_b32 would emit without the rewrite
// (element GEP -> ml_lsa_store_a64_* on a global store).

// CHECK-LABEL: @add_offset_shared_b32
// CHECK-SAME: (%[[P:.*]]: !llvm.ptr<3>, %[[IDX:.*]]: i32)
// CHECK-NOT: arith.muli
// CHECK: llvm.getelementptr %[[P]][%[[IDX]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
func.func @add_offset_shared_b32(%p: !fly.ptr<f32, shared>, %dyn: i32) -> f32 {
  %off = fly.make_int_tuple(%dyn) : (i32) -> !fly.int_tuple<?>
  %p2 = fly.add_offset(%p, %off)
      : (!fly.ptr<f32, shared>, !fly.int_tuple<?>) -> !fly.ptr<f32, shared>
  %value = fly.ptr.load(%p2) : (!fly.ptr<f32, shared>) -> f32
  return %value : f32
}
