// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-ixdl | FileCheck %s
// RUN: %fly-opt %s | FileCheck %s --check-prefix=ROUNDTRIP

// FlyIXDL MR TCU MMA: fly.mma_atom_call -> ixdl.mmad.
// Loads A/B/C fragments from register pointers, builds ixdl.mmad (D = A*B + C),
// stores the result vector back to D. Mirrors CuTe ivcore11 MMA_Traits.

// === !fly_ixdl.mr.mma<...> parse/print round-trip (no lowering) ===

// ROUNDTRIP-LABEL: @test_mr_mma_type
// ROUNDTRIP-SAME: !fly_ixdl.mr.mma<16, 16, 16, (f16, f16) -> f32>
func.func @test_mr_mma_type(
    %atom: !fly.mma_atom<!fly_ixdl.mr.mma<16, 16, 16, (f16, f16) -> f32>>) {
  return
}

// === f16: D[4xf32] = A[4xf16] * B[4xf16] + C[4xf32], 16x16x16 ===

// CHECK-LABEL: @test_mr_mma_f16
// CHECK-SAME: (%[[D:.*]]: !llvm.ptr<5>, %[[A:.*]]: !llvm.ptr<5>, %[[B:.*]]: !llvm.ptr<5>, %[[C:.*]]: !llvm.ptr<5>)
// CHECK: %[[AV:.*]] = llvm.load %[[A]] : !llvm.ptr<5> -> vector<4xf16>
// CHECK: %[[BV:.*]] = llvm.load %[[B]] : !llvm.ptr<5> -> vector<4xf16>
// CHECK: %[[CV:.*]] = llvm.load %[[C]] : !llvm.ptr<5> -> vector<4xf32>
// CHECK: %[[R:.*]] = ixdl.mmad A[%[[AV]]] B[%[[BV]]] C[%[[CV]]]
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f16>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f16>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 16, k = 16>
// CHECK-SAME: (vector<4xf16>, vector<4xf16>, vector<4xf32>) -> vector<4xf32>
// CHECK: llvm.store %[[R]], %[[D]] : vector<4xf32>, !llvm.ptr<5>
func.func @test_mr_mma_f16(
    %d: !fly.memref<f32, register, 4:1>,
    %a: !fly.memref<f16, register, 4:1>,
    %b: !fly.memref<f16, register, 4:1>,
    %c: !fly.memref<f32, register, 4:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.mr.mma<16, 16, 16, (f16, f16) -> f32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.mr.mma<16, 16, 16, (f16, f16) -> f32>>,
         !fly.memref<f32, register, 4:1>, !fly.memref<f16, register, 4:1>,
         !fly.memref<f16, register, 4:1>, !fly.memref<f32, register, 4:1>) -> ()
  return
}

// === bf16 ===

// CHECK-LABEL: @test_mr_mma_bf16
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<bf16>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 16, k = 16>
// CHECK-SAME: (vector<4xbf16>, vector<4xbf16>, vector<4xf32>) -> vector<4xf32>
func.func @test_mr_mma_bf16(
    %d: !fly.memref<f32, register, 4:1>,
    %a: !fly.memref<bf16, register, 4:1>,
    %b: !fly.memref<bf16, register, 4:1>,
    %c: !fly.memref<f32, register, 4:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.mr.mma<16, 16, 16, (bf16, bf16) -> f32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.mr.mma<16, 16, 16, (bf16, bf16) -> f32>>,
         !fly.memref<f32, register, 4:1>, !fly.memref<bf16, register, 4:1>,
         !fly.memref<bf16, register, 4:1>, !fly.memref<f32, register, 4:1>) -> ()
  return
}

// === f32 (MR/QS only) ===

// CHECK-LABEL: @test_mr_mma_f32
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f32>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 16, k = 16>
// CHECK-SAME: (vector<4xf32>, vector<4xf32>, vector<4xf32>) -> vector<4xf32>
func.func @test_mr_mma_f32(
    %d: !fly.memref<f32, register, 4:1>,
    %a: !fly.memref<f32, register, 4:1>,
    %b: !fly.memref<f32, register, 4:1>,
    %c: !fly.memref<f32, register, 4:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.mr.mma<16, 16, 16, (f32, f32) -> f32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.mr.mma<16, 16, 16, (f32, f32) -> f32>>,
         !fly.memref<f32, register, 4:1>, !fly.memref<f32, register, 4:1>,
         !fly.memref<f32, register, 4:1>, !fly.memref<f32, register, 4:1>) -> ()
  return
}

// === int8: 16x16x32, A/B = 8xi8 per lane, acc = 4xi32 ===

// CHECK-LABEL: @test_mr_mma_s8
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xi8>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xi8>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xi32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<s8>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<s8>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 16, k = 32>
// CHECK-SAME: (vector<8xi8>, vector<8xi8>, vector<4xi32>) -> vector<4xi32>
func.func @test_mr_mma_s8(
    %d: !fly.memref<i32, register, 4:1>,
    %a: !fly.memref<i8, register, 8:1>,
    %b: !fly.memref<i8, register, 8:1>,
    %c: !fly.memref<i32, register, 4:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.mr.mma<16, 16, 32, (i8, i8) -> i32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.mr.mma<16, 16, 32, (i8, i8) -> i32>>,
         !fly.memref<i32, register, 4:1>, !fly.memref<i8, register, 8:1>,
         !fly.memref<i8, register, 8:1>, !fly.memref<i32, register, 4:1>) -> ()
  return
}
