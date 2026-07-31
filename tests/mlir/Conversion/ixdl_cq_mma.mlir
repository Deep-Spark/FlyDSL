// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-ixdl | FileCheck %s

// FlyIXDL CQ TCU MMA: fly.mma_atom_call -> ixdl.mmad.
// Dtypes: f16/bf16->f32 (K=16); s8/u8->i32 (K=32); f8E4M3/f8E5M2->f32|f16 (K=32).
// (M,N) in {16x16, 32x32, 16x64, 64x16}; long-mtx keeps K. No f32 multiplicand.

// === f16 base: D[4xf32] = A[4xf16] * B[4xf16] + C[4xf32], 16x16x16 ===

// CHECK-LABEL: @test_cq_mma_f16
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
func.func @test_cq_mma_f16(
    %d: !fly.memref<f32, register, 4:1>,
    %a: !fly.memref<f16, register, 4:1>,
    %b: !fly.memref<f16, register, 4:1>,
    %c: !fly.memref<f32, register, 4:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 16, (f16, f16) -> f32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 16, (f16, f16) -> f32>>,
         !fly.memref<f32, register, 4:1>, !fly.memref<f16, register, 4:1>,
         !fly.memref<f16, register, 4:1>, !fly.memref<f32, register, 4:1>) -> ()
  return
}

// === bf16 base ===

// CHECK-LABEL: @test_cq_mma_bf16
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<bf16>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 16, k = 16>
// CHECK-SAME: (vector<4xbf16>, vector<4xbf16>, vector<4xf32>) -> vector<4xf32>
func.func @test_cq_mma_bf16(
    %d: !fly.memref<f32, register, 4:1>,
    %a: !fly.memref<bf16, register, 4:1>,
    %b: !fly.memref<bf16, register, 4:1>,
    %c: !fly.memref<f32, register, 4:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 16, (bf16, bf16) -> f32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 16, (bf16, bf16) -> f32>>,
         !fly.memref<f32, register, 4:1>, !fly.memref<bf16, register, 4:1>,
         !fly.memref<bf16, register, 4:1>, !fly.memref<f32, register, 4:1>) -> ()
  return
}

// === int8 / s8 base: 16x16x32 ===

// CHECK-LABEL: @test_cq_mma_s8
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xi8>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xi8>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xi32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<s8>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<s8>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 16, k = 32>
// CHECK-SAME: (vector<8xi8>, vector<8xi8>, vector<4xi32>) -> vector<4xi32>
func.func @test_cq_mma_s8(
    %d: !fly.memref<i32, register, 4:1>,
    %a: !fly.memref<i8, register, 8:1>,
    %b: !fly.memref<i8, register, 8:1>,
    %c: !fly.memref<i32, register, 4:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (i8, i8) -> i32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (i8, i8) -> i32>>,
         !fly.memref<i32, register, 4:1>, !fly.memref<i8, register, 8:1>,
         !fly.memref<i8, register, 8:1>, !fly.memref<i32, register, 4:1>) -> ()
  return
}

// === FeatureLongMtx f16: 32x32x16 ===

// CHECK-LABEL: @test_cq_mma_f16_32x32
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf16>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf16>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: shape = #ixdl.shape<m = 32, n = 32, k = 16>
// CHECK-SAME: (vector<8xf16>, vector<8xf16>, vector<16xf32>) -> vector<16xf32>
func.func @test_cq_mma_f16_32x32(
    %d: !fly.memref<f32, register, 16:1>,
    %a: !fly.memref<f16, register, 8:1>,
    %b: !fly.memref<f16, register, 8:1>,
    %c: !fly.memref<f32, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 16, (f16, f16) -> f32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 16, (f16, f16) -> f32>>,
         !fly.memref<f32, register, 16:1>, !fly.memref<f16, register, 8:1>,
         !fly.memref<f16, register, 8:1>, !fly.memref<f32, register, 16:1>) -> ()
  return
}

// === FeatureLongMtx f16: 16x64x16 (asymmetric A/B) ===

// CHECK-LABEL: @test_cq_mma_f16_16x64
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xf16>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf16>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 64, k = 16>
// CHECK-SAME: (vector<4xf16>, vector<16xf16>, vector<16xf32>) -> vector<16xf32>
func.func @test_cq_mma_f16_16x64(
    %d: !fly.memref<f32, register, 16:1>,
    %a: !fly.memref<f16, register, 4:1>,
    %b: !fly.memref<f16, register, 16:1>,
    %c: !fly.memref<f32, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 16, (f16, f16) -> f32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 16, (f16, f16) -> f32>>,
         !fly.memref<f32, register, 16:1>, !fly.memref<f16, register, 4:1>,
         !fly.memref<f16, register, 16:1>, !fly.memref<f32, register, 16:1>) -> ()
  return
}

// === FeatureLongMtx f16: 64x16x16 (asymmetric A/B) ===

// CHECK-LABEL: @test_cq_mma_f16_64x16
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf16>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xf16>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: shape = #ixdl.shape<m = 64, n = 16, k = 16>
// CHECK-SAME: (vector<16xf16>, vector<4xf16>, vector<16xf32>) -> vector<16xf32>
func.func @test_cq_mma_f16_64x16(
    %d: !fly.memref<f32, register, 16:1>,
    %a: !fly.memref<f16, register, 16:1>,
    %b: !fly.memref<f16, register, 4:1>,
    %c: !fly.memref<f32, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 16, (f16, f16) -> f32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 16, (f16, f16) -> f32>>,
         !fly.memref<f32, register, 16:1>, !fly.memref<f16, register, 16:1>,
         !fly.memref<f16, register, 4:1>, !fly.memref<f32, register, 16:1>) -> ()
  return
}

// === FeatureLongMtx bf16: 32x32x16 ===

// CHECK-LABEL: @test_cq_mma_bf16_32x32
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<bf16>
// CHECK-SAME: shape = #ixdl.shape<m = 32, n = 32, k = 16>
// CHECK-SAME: (vector<8xbf16>, vector<8xbf16>, vector<16xf32>) -> vector<16xf32>
func.func @test_cq_mma_bf16_32x32(
    %d: !fly.memref<f32, register, 16:1>,
    %a: !fly.memref<bf16, register, 8:1>,
    %b: !fly.memref<bf16, register, 8:1>,
    %c: !fly.memref<f32, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 16, (bf16, bf16) -> f32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 16, (bf16, bf16) -> f32>>,
         !fly.memref<f32, register, 16:1>, !fly.memref<bf16, register, 8:1>,
         !fly.memref<bf16, register, 8:1>, !fly.memref<f32, register, 16:1>) -> ()
  return
}

// === FeatureLongMtx s8: 32x32x32 ===

// CHECK-LABEL: @test_cq_mma_s8_32x32
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xi8>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xi8>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xi32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<s8>
// CHECK-SAME: shape = #ixdl.shape<m = 32, n = 32, k = 32>
// CHECK-SAME: (vector<16xi8>, vector<16xi8>, vector<16xi32>) -> vector<16xi32>
func.func @test_cq_mma_s8_32x32(
    %d: !fly.memref<i32, register, 16:1>,
    %a: !fly.memref<i8, register, 16:1>,
    %b: !fly.memref<i8, register, 16:1>,
    %c: !fly.memref<i32, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 32, (i8, i8) -> i32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 32, (i8, i8) -> i32>>,
         !fly.memref<i32, register, 16:1>, !fly.memref<i8, register, 16:1>,
         !fly.memref<i8, register, 16:1>, !fly.memref<i32, register, 16:1>) -> ()
  return
}

// === FeatureLongMtx s8: 16x64x32 ===

// CHECK-LABEL: @test_cq_mma_s8_16x64
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xi8>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<32xi8>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xi32>
// CHECK: ixdl.mmad
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 64, k = 32>
// CHECK-SAME: (vector<8xi8>, vector<32xi8>, vector<16xi32>) -> vector<16xi32>
func.func @test_cq_mma_s8_16x64(
    %d: !fly.memref<i32, register, 16:1>,
    %a: !fly.memref<i8, register, 8:1>,
    %b: !fly.memref<i8, register, 32:1>,
    %c: !fly.memref<i32, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (i8, i8) -> i32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (i8, i8) -> i32>>,
         !fly.memref<i32, register, 16:1>, !fly.memref<i8, register, 8:1>,
         !fly.memref<i8, register, 32:1>, !fly.memref<i32, register, 16:1>) -> ()
  return
}

// === FeatureLongMtx s8: 64x16x32 ===

// CHECK-LABEL: @test_cq_mma_s8_64x16
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<32xi8>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xi8>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xi32>
// CHECK: ixdl.mmad
// CHECK-SAME: shape = #ixdl.shape<m = 64, n = 16, k = 32>
// CHECK-SAME: (vector<32xi8>, vector<8xi8>, vector<16xi32>) -> vector<16xi32>
func.func @test_cq_mma_s8_64x16(
    %d: !fly.memref<i32, register, 16:1>,
    %a: !fly.memref<i8, register, 32:1>,
    %b: !fly.memref<i8, register, 8:1>,
    %c: !fly.memref<i32, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (i8, i8) -> i32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (i8, i8) -> i32>>,
         !fly.memref<i32, register, 16:1>, !fly.memref<i8, register, 32:1>,
         !fly.memref<i8, register, 8:1>, !fly.memref<i32, register, 16:1>) -> ()
  return
}

// === u8 base: 16x16x32 ===

// CHECK-LABEL: @test_cq_mma_u8
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xui8>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xui8>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xi32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<u8>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<u8>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 16, k = 32>
// CHECK-SAME: (vector<8xui8>, vector<8xui8>, vector<4xi32>) -> vector<4xi32>
func.func @test_cq_mma_u8(
    %d: !fly.memref<i32, register, 4:1>,
    %a: !fly.memref<ui8, register, 8:1>,
    %b: !fly.memref<ui8, register, 8:1>,
    %c: !fly.memref<i32, register, 4:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (ui8, ui8) -> i32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (ui8, ui8) -> i32>>,
         !fly.memref<i32, register, 4:1>, !fly.memref<ui8, register, 8:1>,
         !fly.memref<ui8, register, 8:1>, !fly.memref<i32, register, 4:1>) -> ()
  return
}

// === FeatureLongMtx u8: 32x32x32 ===

// CHECK-LABEL: @test_cq_mma_u8_32x32
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<u8>
// CHECK-SAME: shape = #ixdl.shape<m = 32, n = 32, k = 32>
// CHECK-SAME: (vector<16xui8>, vector<16xui8>, vector<16xi32>) -> vector<16xi32>
func.func @test_cq_mma_u8_32x32(
    %d: !fly.memref<i32, register, 16:1>,
    %a: !fly.memref<ui8, register, 16:1>,
    %b: !fly.memref<ui8, register, 16:1>,
    %c: !fly.memref<i32, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 32, (ui8, ui8) -> i32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 32, (ui8, ui8) -> i32>>,
         !fly.memref<i32, register, 16:1>, !fly.memref<ui8, register, 16:1>,
         !fly.memref<ui8, register, 16:1>, !fly.memref<i32, register, 16:1>) -> ()
  return
}

// === FP8 e4m3->f32 base: 16x16x32 ===

// CHECK-LABEL: @test_cq_mma_f8e4m3_f32
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E4M3>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E4M3>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 16, k = 32>
// CHECK-SAME: (vector<8xf8E4M3>, vector<8xf8E4M3>, vector<4xf32>) -> vector<4xf32>
func.func @test_cq_mma_f8e4m3_f32(
    %d: !fly.memref<f32, register, 4:1>,
    %a: !fly.memref<f8E4M3, register, 8:1>,
    %b: !fly.memref<f8E4M3, register, 8:1>,
    %c: !fly.memref<f32, register, 4:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (f8E4M3, f8E4M3) -> f32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (f8E4M3, f8E4M3) -> f32>>,
         !fly.memref<f32, register, 4:1>, !fly.memref<f8E4M3, register, 8:1>,
         !fly.memref<f8E4M3, register, 8:1>, !fly.memref<f32, register, 4:1>) -> ()
  return
}

// === FP8 mixed e4m3*e5m2->f32 long-mtx 32x32 ===

// CHECK-LABEL: @test_cq_mma_f8_mixed_32x32
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: shape = #ixdl.shape<m = 32, n = 32, k = 32>
// CHECK-SAME: (vector<16xf8E4M3>, vector<16xf8E5M2>, vector<16xf32>) -> vector<16xf32>
func.func @test_cq_mma_f8_mixed_32x32(
    %d: !fly.memref<f32, register, 16:1>,
    %a: !fly.memref<f8E4M3, register, 16:1>,
    %b: !fly.memref<f8E5M2, register, 16:1>,
    %c: !fly.memref<f32, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 32, (f8E4M3, f8E5M2) -> f32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 32, (f8E4M3, f8E5M2) -> f32>>,
         !fly.memref<f32, register, 16:1>, !fly.memref<f8E4M3, register, 16:1>,
         !fly.memref<f8E5M2, register, 16:1>, !fly.memref<f32, register, 16:1>) -> ()
  return
}

// === FP8 e5m2->f16 base ===

// CHECK-LABEL: @test_cq_mma_f8e5m2_f16
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E5M2>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E5M2>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xf16>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 16, k = 32>
// CHECK-SAME: (vector<8xf8E5M2>, vector<8xf8E5M2>, vector<4xf16>) -> vector<4xf16>
func.func @test_cq_mma_f8e5m2_f16(
    %d: !fly.memref<f16, register, 4:1>,
    %a: !fly.memref<f8E5M2, register, 8:1>,
    %b: !fly.memref<f8E5M2, register, 8:1>,
    %c: !fly.memref<f16, register, 4:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (f8E5M2, f8E5M2) -> f16>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (f8E5M2, f8E5M2) -> f16>>,
         !fly.memref<f16, register, 4:1>, !fly.memref<f8E5M2, register, 8:1>,
         !fly.memref<f8E5M2, register, 8:1>, !fly.memref<f16, register, 4:1>) -> ()
  return
}

// === FP8 mixed e4m3*e5m2->f16 long-mtx 32x32 ===

// CHECK-LABEL: @test_cq_mma_f8_mixed_f16_32x32
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf8E4M3>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf8E5M2>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf16>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: shape = #ixdl.shape<m = 32, n = 32, k = 32>
// CHECK-SAME: (vector<16xf8E4M3>, vector<16xf8E5M2>, vector<16xf16>) -> vector<16xf16>
func.func @test_cq_mma_f8_mixed_f16_32x32(
    %d: !fly.memref<f16, register, 16:1>,
    %a: !fly.memref<f8E4M3, register, 16:1>,
    %b: !fly.memref<f8E5M2, register, 16:1>,
    %c: !fly.memref<f16, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 32, (f8E4M3, f8E5M2) -> f16>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 32, (f8E4M3, f8E5M2) -> f16>>,
         !fly.memref<f16, register, 16:1>, !fly.memref<f8E4M3, register, 16:1>,
         !fly.memref<f8E5M2, register, 16:1>, !fly.memref<f16, register, 16:1>) -> ()
  return
}

// === FP8 e4m3->f16 long-mtx 16x64 ===

// CHECK-LABEL: @test_cq_mma_f8e4m3_f16_16x64
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E4M3>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<32xf8E4M3>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf16>
// CHECK: ixdl.mmad
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 64, k = 32>
// CHECK-SAME: (vector<8xf8E4M3>, vector<32xf8E4M3>, vector<16xf16>) -> vector<16xf16>
func.func @test_cq_mma_f8e4m3_f16_16x64(
    %d: !fly.memref<f16, register, 16:1>,
    %a: !fly.memref<f8E4M3, register, 8:1>,
    %b: !fly.memref<f8E4M3, register, 32:1>,
    %c: !fly.memref<f16, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (f8E4M3, f8E4M3) -> f16>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (f8E4M3, f8E4M3) -> f16>>,
         !fly.memref<f16, register, 16:1>, !fly.memref<f8E4M3, register, 8:1>,
         !fly.memref<f8E4M3, register, 32:1>, !fly.memref<f16, register, 16:1>) -> ()
  return
}

// === FP8 e5m2->f16 long-mtx 64x16 ===

// CHECK-LABEL: @test_cq_mma_f8e5m2_f16_64x16
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<32xf8E5M2>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E5M2>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf16>
// CHECK: ixdl.mmad
// CHECK-SAME: shape = #ixdl.shape<m = 64, n = 16, k = 32>
// CHECK-SAME: (vector<32xf8E5M2>, vector<8xf8E5M2>, vector<16xf16>) -> vector<16xf16>
func.func @test_cq_mma_f8e5m2_f16_64x16(
    %d: !fly.memref<f16, register, 16:1>,
    %a: !fly.memref<f8E5M2, register, 32:1>,
    %b: !fly.memref<f8E5M2, register, 8:1>,
    %c: !fly.memref<f16, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (f8E5M2, f8E5M2) -> f16>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (f8E5M2, f8E5M2) -> f16>>,
         !fly.memref<f16, register, 16:1>, !fly.memref<f8E5M2, register, 32:1>,
         !fly.memref<f8E5M2, register, 8:1>, !fly.memref<f16, register, 16:1>) -> ()
  return
}

// === Remaining bf16/u8 long-mtx combinations ===

// CHECK-LABEL: @test_cq_mma_bf16_u8_asymmetric
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<bf16>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<bf16>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 64, k = 16>
// CHECK-SAME: (vector<4xbf16>, vector<16xbf16>, vector<16xf32>) -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<bf16>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<bf16>
// CHECK-SAME: shape = #ixdl.shape<m = 64, n = 16, k = 16>
// CHECK-SAME: (vector<16xbf16>, vector<4xbf16>, vector<16xf32>) -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<u8>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<u8>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 64, k = 32>
// CHECK-SAME: (vector<8xui8>, vector<32xui8>, vector<16xi32>) -> vector<16xi32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<u8>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<u8>
// CHECK-SAME: shape = #ixdl.shape<m = 64, n = 16, k = 32>
// CHECK-SAME: (vector<32xui8>, vector<8xui8>, vector<16xi32>) -> vector<16xi32>
func.func @test_cq_mma_bf16_u8_asymmetric(
    %df: !fly.memref<f32, register, 16:1>,
    %cf: !fly.memref<f32, register, 16:1>,
    %ab16: !fly.memref<bf16, register, 4:1>,
    %bb64: !fly.memref<bf16, register, 16:1>,
    %ab64: !fly.memref<bf16, register, 16:1>,
    %bb16: !fly.memref<bf16, register, 4:1>,
    %di: !fly.memref<i32, register, 16:1>,
    %ci: !fly.memref<i32, register, 16:1>,
    %au16: !fly.memref<ui8, register, 8:1>,
    %bu64: !fly.memref<ui8, register, 32:1>,
    %au64: !fly.memref<ui8, register, 32:1>,
    %bu16: !fly.memref<ui8, register, 8:1>) {
  %bf16_16x64 = fly.make_mma_atom
      : !fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 16, (bf16, bf16) -> f32>>
  fly.mma_atom_call(%bf16_16x64, %df, %ab16, %bb64, %cf)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 16, (bf16, bf16) -> f32>>,
         !fly.memref<f32, register, 16:1>, !fly.memref<bf16, register, 4:1>,
         !fly.memref<bf16, register, 16:1>, !fly.memref<f32, register, 16:1>) -> ()
  %bf16_64x16 = fly.make_mma_atom
      : !fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 16, (bf16, bf16) -> f32>>
  fly.mma_atom_call(%bf16_64x16, %df, %ab64, %bb16, %cf)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 16, (bf16, bf16) -> f32>>,
         !fly.memref<f32, register, 16:1>, !fly.memref<bf16, register, 16:1>,
         !fly.memref<bf16, register, 4:1>, !fly.memref<f32, register, 16:1>) -> ()
  %u8_16x64 = fly.make_mma_atom
      : !fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (ui8, ui8) -> i32>>
  fly.mma_atom_call(%u8_16x64, %di, %au16, %bu64, %ci)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (ui8, ui8) -> i32>>,
         !fly.memref<i32, register, 16:1>, !fly.memref<ui8, register, 8:1>,
         !fly.memref<ui8, register, 32:1>, !fly.memref<i32, register, 16:1>) -> ()
  %u8_64x16 = fly.make_mma_atom
      : !fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (ui8, ui8) -> i32>>
  fly.mma_atom_call(%u8_64x16, %di, %au64, %bu16, %ci)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (ui8, ui8) -> i32>>,
         !fly.memref<i32, register, 16:1>, !fly.memref<ui8, register, 32:1>,
         !fly.memref<ui8, register, 8:1>, !fly.memref<i32, register, 16:1>) -> ()
  return
}

// === Remaining FP8 base combinations ===

// CHECK-LABEL: @test_cq_mma_fp8_remaining_16x16
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 16, k = 32>
// CHECK-SAME: (vector<8xf8E4M3>, vector<8xf8E5M2>, vector<4xf32>) -> vector<4xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 16, k = 32>
// CHECK-SAME: (vector<8xf8E5M2>, vector<8xf8E4M3>, vector<4xf32>) -> vector<4xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 16, k = 32>
// CHECK-SAME: (vector<8xf8E5M2>, vector<8xf8E5M2>, vector<4xf32>) -> vector<4xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 16, k = 32>
// CHECK-SAME: (vector<8xf8E4M3>, vector<8xf8E4M3>, vector<4xf16>) -> vector<4xf16>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 16, k = 32>
// CHECK-SAME: (vector<8xf8E4M3>, vector<8xf8E5M2>, vector<4xf16>) -> vector<4xf16>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 16, k = 32>
// CHECK-SAME: (vector<8xf8E5M2>, vector<8xf8E4M3>, vector<4xf16>) -> vector<4xf16>
func.func @test_cq_mma_fp8_remaining_16x16(
    %d32: !fly.memref<f32, register, 4:1>,
    %c32: !fly.memref<f32, register, 4:1>,
    %d16: !fly.memref<f16, register, 4:1>,
    %c16: !fly.memref<f16, register, 4:1>,
    %a4: !fly.memref<f8E4M3, register, 8:1>,
    %a5: !fly.memref<f8E5M2, register, 8:1>,
    %b4: !fly.memref<f8E4M3, register, 8:1>,
    %b5: !fly.memref<f8E5M2, register, 8:1>) {
  %a0 = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (f8E4M3, f8E5M2) -> f32>>
  fly.mma_atom_call(%a0, %d32, %a4, %b5, %c32)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (f8E4M3, f8E5M2) -> f32>>,
         !fly.memref<f32, register, 4:1>, !fly.memref<f8E4M3, register, 8:1>,
         !fly.memref<f8E5M2, register, 8:1>, !fly.memref<f32, register, 4:1>) -> ()
  %a1 = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (f8E5M2, f8E4M3) -> f32>>
  fly.mma_atom_call(%a1, %d32, %a5, %b4, %c32)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (f8E5M2, f8E4M3) -> f32>>,
         !fly.memref<f32, register, 4:1>, !fly.memref<f8E5M2, register, 8:1>,
         !fly.memref<f8E4M3, register, 8:1>, !fly.memref<f32, register, 4:1>) -> ()
  %a2 = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (f8E5M2, f8E5M2) -> f32>>
  fly.mma_atom_call(%a2, %d32, %a5, %b5, %c32)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (f8E5M2, f8E5M2) -> f32>>,
         !fly.memref<f32, register, 4:1>, !fly.memref<f8E5M2, register, 8:1>,
         !fly.memref<f8E5M2, register, 8:1>, !fly.memref<f32, register, 4:1>) -> ()
  %a3 = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (f8E4M3, f8E4M3) -> f16>>
  fly.mma_atom_call(%a3, %d16, %a4, %b4, %c16)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (f8E4M3, f8E4M3) -> f16>>,
         !fly.memref<f16, register, 4:1>, !fly.memref<f8E4M3, register, 8:1>,
         !fly.memref<f8E4M3, register, 8:1>, !fly.memref<f16, register, 4:1>) -> ()
  %a4v = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (f8E4M3, f8E5M2) -> f16>>
  fly.mma_atom_call(%a4v, %d16, %a4, %b5, %c16)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (f8E4M3, f8E5M2) -> f16>>,
         !fly.memref<f16, register, 4:1>, !fly.memref<f8E4M3, register, 8:1>,
         !fly.memref<f8E5M2, register, 8:1>, !fly.memref<f16, register, 4:1>) -> ()
  %a5v = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (f8E5M2, f8E4M3) -> f16>>
  fly.mma_atom_call(%a5v, %d16, %a5, %b4, %c16)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (f8E5M2, f8E4M3) -> f16>>,
         !fly.memref<f16, register, 4:1>, !fly.memref<f8E5M2, register, 8:1>,
         !fly.memref<f8E4M3, register, 8:1>, !fly.memref<f16, register, 4:1>) -> ()
  return
}

// === Remaining FP8 32x32 combinations ===

// CHECK-LABEL: @test_cq_mma_fp8_remaining_32x32
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: shape = #ixdl.shape<m = 32, n = 32, k = 32>
// CHECK-SAME: -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: shape = #ixdl.shape<m = 32, n = 32, k = 32>
// CHECK-SAME: -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: shape = #ixdl.shape<m = 32, n = 32, k = 32>
// CHECK-SAME: -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: shape = #ixdl.shape<m = 32, n = 32, k = 32>
// CHECK-SAME: -> vector<16xf16>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: shape = #ixdl.shape<m = 32, n = 32, k = 32>
// CHECK-SAME: -> vector<16xf16>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: shape = #ixdl.shape<m = 32, n = 32, k = 32>
// CHECK-SAME: -> vector<16xf16>
func.func @test_cq_mma_fp8_remaining_32x32(
    %d32: !fly.memref<f32, register, 16:1>,
    %c32: !fly.memref<f32, register, 16:1>,
    %d16: !fly.memref<f16, register, 16:1>,
    %c16: !fly.memref<f16, register, 16:1>,
    %a4: !fly.memref<f8E4M3, register, 16:1>,
    %a5: !fly.memref<f8E5M2, register, 16:1>,
    %b4: !fly.memref<f8E4M3, register, 16:1>,
    %b5: !fly.memref<f8E5M2, register, 16:1>) {
  %a0 = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 32, (f8E4M3, f8E4M3) -> f32>>
  fly.mma_atom_call(%a0, %d32, %a4, %b4, %c32)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 32, (f8E4M3, f8E4M3) -> f32>>,
         !fly.memref<f32, register, 16:1>, !fly.memref<f8E4M3, register, 16:1>,
         !fly.memref<f8E4M3, register, 16:1>, !fly.memref<f32, register, 16:1>) -> ()
  %a1 = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 32, (f8E5M2, f8E4M3) -> f32>>
  fly.mma_atom_call(%a1, %d32, %a5, %b4, %c32)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 32, (f8E5M2, f8E4M3) -> f32>>,
         !fly.memref<f32, register, 16:1>, !fly.memref<f8E5M2, register, 16:1>,
         !fly.memref<f8E4M3, register, 16:1>, !fly.memref<f32, register, 16:1>) -> ()
  %a2 = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 32, (f8E5M2, f8E5M2) -> f32>>
  fly.mma_atom_call(%a2, %d32, %a5, %b5, %c32)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 32, (f8E5M2, f8E5M2) -> f32>>,
         !fly.memref<f32, register, 16:1>, !fly.memref<f8E5M2, register, 16:1>,
         !fly.memref<f8E5M2, register, 16:1>, !fly.memref<f32, register, 16:1>) -> ()
  %a3 = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 32, (f8E4M3, f8E4M3) -> f16>>
  fly.mma_atom_call(%a3, %d16, %a4, %b4, %c16)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 32, (f8E4M3, f8E4M3) -> f16>>,
         !fly.memref<f16, register, 16:1>, !fly.memref<f8E4M3, register, 16:1>,
         !fly.memref<f8E4M3, register, 16:1>, !fly.memref<f16, register, 16:1>) -> ()
  %a4v = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 32, (f8E5M2, f8E4M3) -> f16>>
  fly.mma_atom_call(%a4v, %d16, %a5, %b4, %c16)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 32, (f8E5M2, f8E4M3) -> f16>>,
         !fly.memref<f16, register, 16:1>, !fly.memref<f8E5M2, register, 16:1>,
         !fly.memref<f8E4M3, register, 16:1>, !fly.memref<f16, register, 16:1>) -> ()
  %a5v = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 32, (f8E5M2, f8E5M2) -> f16>>
  fly.mma_atom_call(%a5v, %d16, %a5, %b5, %c16)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 32, (f8E5M2, f8E5M2) -> f16>>,
         !fly.memref<f16, register, 16:1>, !fly.memref<f8E5M2, register, 16:1>,
         !fly.memref<f8E5M2, register, 16:1>, !fly.memref<f16, register, 16:1>) -> ()
  return
}

// === Remaining FP8 16x64 combinations ===

// CHECK-LABEL: @test_cq_mma_fp8_remaining_16x64
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 64, k = 32>
// CHECK-SAME: -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 64, k = 32>
// CHECK-SAME: -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 64, k = 32>
// CHECK-SAME: -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 64, k = 32>
// CHECK-SAME: -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 64, k = 32>
// CHECK-SAME: -> vector<16xf16>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 64, k = 32>
// CHECK-SAME: -> vector<16xf16>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 64, k = 32>
// CHECK-SAME: -> vector<16xf16>
func.func @test_cq_mma_fp8_remaining_16x64(
    %d32: !fly.memref<f32, register, 16:1>,
    %c32: !fly.memref<f32, register, 16:1>,
    %d16: !fly.memref<f16, register, 16:1>,
    %c16: !fly.memref<f16, register, 16:1>,
    %a4: !fly.memref<f8E4M3, register, 8:1>,
    %a5: !fly.memref<f8E5M2, register, 8:1>,
    %b4: !fly.memref<f8E4M3, register, 32:1>,
    %b5: !fly.memref<f8E5M2, register, 32:1>) {
  %a0 = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (f8E4M3, f8E4M3) -> f32>>
  fly.mma_atom_call(%a0, %d32, %a4, %b4, %c32)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (f8E4M3, f8E4M3) -> f32>>,
         !fly.memref<f32, register, 16:1>, !fly.memref<f8E4M3, register, 8:1>,
         !fly.memref<f8E4M3, register, 32:1>, !fly.memref<f32, register, 16:1>) -> ()
  %a1 = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (f8E4M3, f8E5M2) -> f32>>
  fly.mma_atom_call(%a1, %d32, %a4, %b5, %c32)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (f8E4M3, f8E5M2) -> f32>>,
         !fly.memref<f32, register, 16:1>, !fly.memref<f8E4M3, register, 8:1>,
         !fly.memref<f8E5M2, register, 32:1>, !fly.memref<f32, register, 16:1>) -> ()
  %a2 = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (f8E5M2, f8E4M3) -> f32>>
  fly.mma_atom_call(%a2, %d32, %a5, %b4, %c32)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (f8E5M2, f8E4M3) -> f32>>,
         !fly.memref<f32, register, 16:1>, !fly.memref<f8E5M2, register, 8:1>,
         !fly.memref<f8E4M3, register, 32:1>, !fly.memref<f32, register, 16:1>) -> ()
  %a3 = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (f8E5M2, f8E5M2) -> f32>>
  fly.mma_atom_call(%a3, %d32, %a5, %b5, %c32)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (f8E5M2, f8E5M2) -> f32>>,
         !fly.memref<f32, register, 16:1>, !fly.memref<f8E5M2, register, 8:1>,
         !fly.memref<f8E5M2, register, 32:1>, !fly.memref<f32, register, 16:1>) -> ()
  %a4v = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (f8E4M3, f8E5M2) -> f16>>
  fly.mma_atom_call(%a4v, %d16, %a4, %b5, %c16)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (f8E4M3, f8E5M2) -> f16>>,
         !fly.memref<f16, register, 16:1>, !fly.memref<f8E4M3, register, 8:1>,
         !fly.memref<f8E5M2, register, 32:1>, !fly.memref<f16, register, 16:1>) -> ()
  %a5v = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (f8E5M2, f8E4M3) -> f16>>
  fly.mma_atom_call(%a5v, %d16, %a5, %b4, %c16)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (f8E5M2, f8E4M3) -> f16>>,
         !fly.memref<f16, register, 16:1>, !fly.memref<f8E5M2, register, 8:1>,
         !fly.memref<f8E4M3, register, 32:1>, !fly.memref<f16, register, 16:1>) -> ()
  %a6 = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (f8E5M2, f8E5M2) -> f16>>
  fly.mma_atom_call(%a6, %d16, %a5, %b5, %c16)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (f8E5M2, f8E5M2) -> f16>>,
         !fly.memref<f16, register, 16:1>, !fly.memref<f8E5M2, register, 8:1>,
         !fly.memref<f8E5M2, register, 32:1>, !fly.memref<f16, register, 16:1>) -> ()
  return
}

// === Remaining FP8 64x16 combinations ===

// CHECK-LABEL: @test_cq_mma_fp8_remaining_64x16
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: shape = #ixdl.shape<m = 64, n = 16, k = 32>
// CHECK-SAME: -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: shape = #ixdl.shape<m = 64, n = 16, k = 32>
// CHECK-SAME: -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: shape = #ixdl.shape<m = 64, n = 16, k = 32>
// CHECK-SAME: -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: shape = #ixdl.shape<m = 64, n = 16, k = 32>
// CHECK-SAME: -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: shape = #ixdl.shape<m = 64, n = 16, k = 32>
// CHECK-SAME: -> vector<16xf16>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: shape = #ixdl.shape<m = 64, n = 16, k = 32>
// CHECK-SAME: -> vector<16xf16>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: shape = #ixdl.shape<m = 64, n = 16, k = 32>
// CHECK-SAME: -> vector<16xf16>
func.func @test_cq_mma_fp8_remaining_64x16(
    %d32: !fly.memref<f32, register, 16:1>,
    %c32: !fly.memref<f32, register, 16:1>,
    %d16: !fly.memref<f16, register, 16:1>,
    %c16: !fly.memref<f16, register, 16:1>,
    %a4: !fly.memref<f8E4M3, register, 32:1>,
    %a5: !fly.memref<f8E5M2, register, 32:1>,
    %b4: !fly.memref<f8E4M3, register, 8:1>,
    %b5: !fly.memref<f8E5M2, register, 8:1>) {
  %a0 = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (f8E4M3, f8E4M3) -> f32>>
  fly.mma_atom_call(%a0, %d32, %a4, %b4, %c32)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (f8E4M3, f8E4M3) -> f32>>,
         !fly.memref<f32, register, 16:1>, !fly.memref<f8E4M3, register, 32:1>,
         !fly.memref<f8E4M3, register, 8:1>, !fly.memref<f32, register, 16:1>) -> ()
  %a1 = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (f8E4M3, f8E5M2) -> f32>>
  fly.mma_atom_call(%a1, %d32, %a4, %b5, %c32)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (f8E4M3, f8E5M2) -> f32>>,
         !fly.memref<f32, register, 16:1>, !fly.memref<f8E4M3, register, 32:1>,
         !fly.memref<f8E5M2, register, 8:1>, !fly.memref<f32, register, 16:1>) -> ()
  %a2 = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (f8E5M2, f8E4M3) -> f32>>
  fly.mma_atom_call(%a2, %d32, %a5, %b4, %c32)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (f8E5M2, f8E4M3) -> f32>>,
         !fly.memref<f32, register, 16:1>, !fly.memref<f8E5M2, register, 32:1>,
         !fly.memref<f8E4M3, register, 8:1>, !fly.memref<f32, register, 16:1>) -> ()
  %a3 = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (f8E5M2, f8E5M2) -> f32>>
  fly.mma_atom_call(%a3, %d32, %a5, %b5, %c32)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (f8E5M2, f8E5M2) -> f32>>,
         !fly.memref<f32, register, 16:1>, !fly.memref<f8E5M2, register, 32:1>,
         !fly.memref<f8E5M2, register, 8:1>, !fly.memref<f32, register, 16:1>) -> ()
  %a4v = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (f8E4M3, f8E4M3) -> f16>>
  fly.mma_atom_call(%a4v, %d16, %a4, %b4, %c16)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (f8E4M3, f8E4M3) -> f16>>,
         !fly.memref<f16, register, 16:1>, !fly.memref<f8E4M3, register, 32:1>,
         !fly.memref<f8E4M3, register, 8:1>, !fly.memref<f16, register, 16:1>) -> ()
  %a5v = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (f8E4M3, f8E5M2) -> f16>>
  fly.mma_atom_call(%a5v, %d16, %a4, %b5, %c16)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (f8E4M3, f8E5M2) -> f16>>,
         !fly.memref<f16, register, 16:1>, !fly.memref<f8E4M3, register, 32:1>,
         !fly.memref<f8E5M2, register, 8:1>, !fly.memref<f16, register, 16:1>) -> ()
  %a6 = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (f8E5M2, f8E4M3) -> f16>>
  fly.mma_atom_call(%a6, %d16, %a5, %b4, %c16)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (f8E5M2, f8E4M3) -> f16>>,
         !fly.memref<f16, register, 16:1>, !fly.memref<f8E5M2, register, 32:1>,
         !fly.memref<f8E4M3, register, 8:1>, !fly.memref<f16, register, 16:1>) -> ()
  return
}
