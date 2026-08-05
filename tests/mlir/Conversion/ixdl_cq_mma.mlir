// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-ixdl | FileCheck %s

// FlyIXDL CQ TCU MMA: fly.mma_atom_call -> ixdl.mmad.
// Grouped by dtype; each group covers all legal (M,N) shapes.
//   f16/bf16 -> f32, K=16
//   i8/ui8   -> i32, K=32 (IXDL mmad s8/u8)
//   FP8      -> f32|f16, K=32; A/B in {f8E4M3, f8E5M2} (matched or mixed)
// (M,N) in {16x16, 32x32, 16x64, 64x16}; long-mtx keeps K.


// ========================================================================
// f16 x f16 -> f32  (K=16)
// ========================================================================


// --- 16x16x16 ---

// CHECK-LABEL: @test_cq_mma_f16_16x16
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
func.func @test_cq_mma_f16_16x16(
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

// --- 32x32x16 ---

// CHECK-LABEL: @test_cq_mma_f16_32x32
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf16>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf16>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f16>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f16>
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

// --- 16x64x16 ---

// CHECK-LABEL: @test_cq_mma_f16_16x64
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xf16>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf16>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f16>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f16>
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

// --- 64x16x16 ---

// CHECK-LABEL: @test_cq_mma_f16_64x16
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf16>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xf16>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f16>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f16>
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

// ========================================================================
// bf16 x bf16 -> f32  (K=16)
// ========================================================================


// --- 16x16x16 ---

// CHECK-LABEL: @test_cq_mma_bf16_16x16
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xbf16>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xbf16>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<bf16>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<bf16>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 16, k = 16>
// CHECK-SAME: (vector<4xbf16>, vector<4xbf16>, vector<4xf32>) -> vector<4xf32>
func.func @test_cq_mma_bf16_16x16(
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

// --- 32x32x16 ---

// CHECK-LABEL: @test_cq_mma_bf16_32x32
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xbf16>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xbf16>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<bf16>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<bf16>
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

// --- 16x64x16 ---

// CHECK-LABEL: @test_cq_mma_bf16_16x64
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xbf16>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xbf16>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<bf16>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<bf16>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 64, k = 16>
// CHECK-SAME: (vector<4xbf16>, vector<16xbf16>, vector<16xf32>) -> vector<16xf32>
func.func @test_cq_mma_bf16_16x64(
    %d: !fly.memref<f32, register, 16:1>,
    %a: !fly.memref<bf16, register, 4:1>,
    %b: !fly.memref<bf16, register, 16:1>,
    %c: !fly.memref<f32, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 16, (bf16, bf16) -> f32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 16, (bf16, bf16) -> f32>>,
         !fly.memref<f32, register, 16:1>, !fly.memref<bf16, register, 4:1>,
         !fly.memref<bf16, register, 16:1>, !fly.memref<f32, register, 16:1>) -> ()
  return
}

// --- 64x16x16 ---

// CHECK-LABEL: @test_cq_mma_bf16_64x16
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xbf16>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xbf16>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<bf16>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<bf16>
// CHECK-SAME: shape = #ixdl.shape<m = 64, n = 16, k = 16>
// CHECK-SAME: (vector<16xbf16>, vector<4xbf16>, vector<16xf32>) -> vector<16xf32>
func.func @test_cq_mma_bf16_64x16(
    %d: !fly.memref<f32, register, 16:1>,
    %a: !fly.memref<bf16, register, 16:1>,
    %b: !fly.memref<bf16, register, 4:1>,
    %c: !fly.memref<f32, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 16, (bf16, bf16) -> f32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 16, (bf16, bf16) -> f32>>,
         !fly.memref<f32, register, 16:1>, !fly.memref<bf16, register, 16:1>,
         !fly.memref<bf16, register, 4:1>, !fly.memref<f32, register, 16:1>) -> ()
  return
}

// ========================================================================
// i8 x i8 -> i32  (K=32; IXDL mmad s8)
// ========================================================================


// --- 16x16x32 ---

// CHECK-LABEL: @test_cq_mma_s8_16x16
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xi8>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xi8>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xi32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<s8>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<s8>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 16, k = 32>
// CHECK-SAME: (vector<8xi8>, vector<8xi8>, vector<4xi32>) -> vector<4xi32>
func.func @test_cq_mma_s8_16x16(
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

// --- 32x32x32 ---

// CHECK-LABEL: @test_cq_mma_s8_32x32
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xi8>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xi8>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xi32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<s8>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<s8>
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

// --- 16x64x32 ---

// CHECK-LABEL: @test_cq_mma_s8_16x64
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xi8>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<32xi8>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xi32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<s8>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<s8>
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

// --- 64x16x32 ---

// CHECK-LABEL: @test_cq_mma_s8_64x16
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<32xi8>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xi8>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xi32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<s8>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<s8>
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

// ========================================================================
// ui8 x ui8 -> i32  (K=32; IXDL mmad u8)
// ========================================================================


// --- 16x16x32 ---

// CHECK-LABEL: @test_cq_mma_u8_16x16
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xui8>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xui8>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xi32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<u8>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<u8>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 16, k = 32>
// CHECK-SAME: (vector<8xui8>, vector<8xui8>, vector<4xi32>) -> vector<4xi32>
func.func @test_cq_mma_u8_16x16(
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

// --- 32x32x32 ---

// CHECK-LABEL: @test_cq_mma_u8_32x32
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xui8>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xui8>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xi32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<u8>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<u8>
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

// --- 16x64x32 ---

// CHECK-LABEL: @test_cq_mma_u8_16x64
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xui8>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<32xui8>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xi32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<u8>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<u8>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 64, k = 32>
// CHECK-SAME: (vector<8xui8>, vector<32xui8>, vector<16xi32>) -> vector<16xi32>
func.func @test_cq_mma_u8_16x64(
    %d: !fly.memref<i32, register, 16:1>,
    %a: !fly.memref<ui8, register, 8:1>,
    %b: !fly.memref<ui8, register, 32:1>,
    %c: !fly.memref<i32, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (ui8, ui8) -> i32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (ui8, ui8) -> i32>>,
         !fly.memref<i32, register, 16:1>, !fly.memref<ui8, register, 8:1>,
         !fly.memref<ui8, register, 32:1>, !fly.memref<i32, register, 16:1>) -> ()
  return
}

// --- 64x16x32 ---

// CHECK-LABEL: @test_cq_mma_u8_64x16
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<32xui8>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xui8>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xi32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<u8>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<u8>
// CHECK-SAME: shape = #ixdl.shape<m = 64, n = 16, k = 32>
// CHECK-SAME: (vector<32xui8>, vector<8xui8>, vector<16xi32>) -> vector<16xi32>
func.func @test_cq_mma_u8_64x16(
    %d: !fly.memref<i32, register, 16:1>,
    %a: !fly.memref<ui8, register, 32:1>,
    %b: !fly.memref<ui8, register, 8:1>,
    %c: !fly.memref<i32, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (ui8, ui8) -> i32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (ui8, ui8) -> i32>>,
         !fly.memref<i32, register, 16:1>, !fly.memref<ui8, register, 32:1>,
         !fly.memref<ui8, register, 8:1>, !fly.memref<i32, register, 16:1>) -> ()
  return
}

// ========================================================================
// FP8 f8E4M3 x f8E4M3 -> f32  (K=32)
// ========================================================================


// --- 16x16x32 ---

// CHECK-LABEL: @test_cq_mma_f8_e4m3_e4m3_f32_16x16
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E4M3>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E4M3>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 16, k = 32>
// CHECK-SAME: (vector<8xf8E4M3>, vector<8xf8E4M3>, vector<4xf32>) -> vector<4xf32>
func.func @test_cq_mma_f8_e4m3_e4m3_f32_16x16(
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

// --- 32x32x32 ---

// CHECK-LABEL: @test_cq_mma_f8_e4m3_e4m3_f32_32x32
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf8E4M3>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf8E4M3>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: shape = #ixdl.shape<m = 32, n = 32, k = 32>
// CHECK-SAME: (vector<16xf8E4M3>, vector<16xf8E4M3>, vector<16xf32>) -> vector<16xf32>
func.func @test_cq_mma_f8_e4m3_e4m3_f32_32x32(
    %d: !fly.memref<f32, register, 16:1>,
    %a: !fly.memref<f8E4M3, register, 16:1>,
    %b: !fly.memref<f8E4M3, register, 16:1>,
    %c: !fly.memref<f32, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 32, (f8E4M3, f8E4M3) -> f32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 32, (f8E4M3, f8E4M3) -> f32>>,
         !fly.memref<f32, register, 16:1>, !fly.memref<f8E4M3, register, 16:1>,
         !fly.memref<f8E4M3, register, 16:1>, !fly.memref<f32, register, 16:1>) -> ()
  return
}

// --- 16x64x32 ---

// CHECK-LABEL: @test_cq_mma_f8_e4m3_e4m3_f32_16x64
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E4M3>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<32xf8E4M3>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 64, k = 32>
// CHECK-SAME: (vector<8xf8E4M3>, vector<32xf8E4M3>, vector<16xf32>) -> vector<16xf32>
func.func @test_cq_mma_f8_e4m3_e4m3_f32_16x64(
    %d: !fly.memref<f32, register, 16:1>,
    %a: !fly.memref<f8E4M3, register, 8:1>,
    %b: !fly.memref<f8E4M3, register, 32:1>,
    %c: !fly.memref<f32, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (f8E4M3, f8E4M3) -> f32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (f8E4M3, f8E4M3) -> f32>>,
         !fly.memref<f32, register, 16:1>, !fly.memref<f8E4M3, register, 8:1>,
         !fly.memref<f8E4M3, register, 32:1>, !fly.memref<f32, register, 16:1>) -> ()
  return
}

// --- 64x16x32 ---

// CHECK-LABEL: @test_cq_mma_f8_e4m3_e4m3_f32_64x16
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<32xf8E4M3>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E4M3>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: shape = #ixdl.shape<m = 64, n = 16, k = 32>
// CHECK-SAME: (vector<32xf8E4M3>, vector<8xf8E4M3>, vector<16xf32>) -> vector<16xf32>
func.func @test_cq_mma_f8_e4m3_e4m3_f32_64x16(
    %d: !fly.memref<f32, register, 16:1>,
    %a: !fly.memref<f8E4M3, register, 32:1>,
    %b: !fly.memref<f8E4M3, register, 8:1>,
    %c: !fly.memref<f32, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (f8E4M3, f8E4M3) -> f32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (f8E4M3, f8E4M3) -> f32>>,
         !fly.memref<f32, register, 16:1>, !fly.memref<f8E4M3, register, 32:1>,
         !fly.memref<f8E4M3, register, 8:1>, !fly.memref<f32, register, 16:1>) -> ()
  return
}

// ========================================================================
// FP8 f8E4M3 x f8E5M2 -> f32  (K=32)
// ========================================================================


// --- 16x16x32 ---

// CHECK-LABEL: @test_cq_mma_f8_e4m3_e5m2_f32_16x16
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E4M3>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E5M2>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 16, k = 32>
// CHECK-SAME: (vector<8xf8E4M3>, vector<8xf8E5M2>, vector<4xf32>) -> vector<4xf32>
func.func @test_cq_mma_f8_e4m3_e5m2_f32_16x16(
    %d: !fly.memref<f32, register, 4:1>,
    %a: !fly.memref<f8E4M3, register, 8:1>,
    %b: !fly.memref<f8E5M2, register, 8:1>,
    %c: !fly.memref<f32, register, 4:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (f8E4M3, f8E5M2) -> f32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (f8E4M3, f8E5M2) -> f32>>,
         !fly.memref<f32, register, 4:1>, !fly.memref<f8E4M3, register, 8:1>,
         !fly.memref<f8E5M2, register, 8:1>, !fly.memref<f32, register, 4:1>) -> ()
  return
}

// --- 32x32x32 ---

// CHECK-LABEL: @test_cq_mma_f8_e4m3_e5m2_f32_32x32
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf8E4M3>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf8E5M2>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: shape = #ixdl.shape<m = 32, n = 32, k = 32>
// CHECK-SAME: (vector<16xf8E4M3>, vector<16xf8E5M2>, vector<16xf32>) -> vector<16xf32>
func.func @test_cq_mma_f8_e4m3_e5m2_f32_32x32(
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

// --- 16x64x32 ---

// CHECK-LABEL: @test_cq_mma_f8_e4m3_e5m2_f32_16x64
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E4M3>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<32xf8E5M2>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 64, k = 32>
// CHECK-SAME: (vector<8xf8E4M3>, vector<32xf8E5M2>, vector<16xf32>) -> vector<16xf32>
func.func @test_cq_mma_f8_e4m3_e5m2_f32_16x64(
    %d: !fly.memref<f32, register, 16:1>,
    %a: !fly.memref<f8E4M3, register, 8:1>,
    %b: !fly.memref<f8E5M2, register, 32:1>,
    %c: !fly.memref<f32, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (f8E4M3, f8E5M2) -> f32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (f8E4M3, f8E5M2) -> f32>>,
         !fly.memref<f32, register, 16:1>, !fly.memref<f8E4M3, register, 8:1>,
         !fly.memref<f8E5M2, register, 32:1>, !fly.memref<f32, register, 16:1>) -> ()
  return
}

// --- 64x16x32 ---

// CHECK-LABEL: @test_cq_mma_f8_e4m3_e5m2_f32_64x16
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<32xf8E4M3>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E5M2>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: shape = #ixdl.shape<m = 64, n = 16, k = 32>
// CHECK-SAME: (vector<32xf8E4M3>, vector<8xf8E5M2>, vector<16xf32>) -> vector<16xf32>
func.func @test_cq_mma_f8_e4m3_e5m2_f32_64x16(
    %d: !fly.memref<f32, register, 16:1>,
    %a: !fly.memref<f8E4M3, register, 32:1>,
    %b: !fly.memref<f8E5M2, register, 8:1>,
    %c: !fly.memref<f32, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (f8E4M3, f8E5M2) -> f32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (f8E4M3, f8E5M2) -> f32>>,
         !fly.memref<f32, register, 16:1>, !fly.memref<f8E4M3, register, 32:1>,
         !fly.memref<f8E5M2, register, 8:1>, !fly.memref<f32, register, 16:1>) -> ()
  return
}

// ========================================================================
// FP8 f8E5M2 x f8E4M3 -> f32  (K=32)
// ========================================================================


// --- 16x16x32 ---

// CHECK-LABEL: @test_cq_mma_f8_e5m2_e4m3_f32_16x16
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E5M2>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E4M3>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 16, k = 32>
// CHECK-SAME: (vector<8xf8E5M2>, vector<8xf8E4M3>, vector<4xf32>) -> vector<4xf32>
func.func @test_cq_mma_f8_e5m2_e4m3_f32_16x16(
    %d: !fly.memref<f32, register, 4:1>,
    %a: !fly.memref<f8E5M2, register, 8:1>,
    %b: !fly.memref<f8E4M3, register, 8:1>,
    %c: !fly.memref<f32, register, 4:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (f8E5M2, f8E4M3) -> f32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (f8E5M2, f8E4M3) -> f32>>,
         !fly.memref<f32, register, 4:1>, !fly.memref<f8E5M2, register, 8:1>,
         !fly.memref<f8E4M3, register, 8:1>, !fly.memref<f32, register, 4:1>) -> ()
  return
}

// --- 32x32x32 ---

// CHECK-LABEL: @test_cq_mma_f8_e5m2_e4m3_f32_32x32
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf8E5M2>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf8E4M3>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: shape = #ixdl.shape<m = 32, n = 32, k = 32>
// CHECK-SAME: (vector<16xf8E5M2>, vector<16xf8E4M3>, vector<16xf32>) -> vector<16xf32>
func.func @test_cq_mma_f8_e5m2_e4m3_f32_32x32(
    %d: !fly.memref<f32, register, 16:1>,
    %a: !fly.memref<f8E5M2, register, 16:1>,
    %b: !fly.memref<f8E4M3, register, 16:1>,
    %c: !fly.memref<f32, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 32, (f8E5M2, f8E4M3) -> f32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 32, (f8E5M2, f8E4M3) -> f32>>,
         !fly.memref<f32, register, 16:1>, !fly.memref<f8E5M2, register, 16:1>,
         !fly.memref<f8E4M3, register, 16:1>, !fly.memref<f32, register, 16:1>) -> ()
  return
}

// --- 16x64x32 ---

// CHECK-LABEL: @test_cq_mma_f8_e5m2_e4m3_f32_16x64
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E5M2>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<32xf8E4M3>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 64, k = 32>
// CHECK-SAME: (vector<8xf8E5M2>, vector<32xf8E4M3>, vector<16xf32>) -> vector<16xf32>
func.func @test_cq_mma_f8_e5m2_e4m3_f32_16x64(
    %d: !fly.memref<f32, register, 16:1>,
    %a: !fly.memref<f8E5M2, register, 8:1>,
    %b: !fly.memref<f8E4M3, register, 32:1>,
    %c: !fly.memref<f32, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (f8E5M2, f8E4M3) -> f32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (f8E5M2, f8E4M3) -> f32>>,
         !fly.memref<f32, register, 16:1>, !fly.memref<f8E5M2, register, 8:1>,
         !fly.memref<f8E4M3, register, 32:1>, !fly.memref<f32, register, 16:1>) -> ()
  return
}

// --- 64x16x32 ---

// CHECK-LABEL: @test_cq_mma_f8_e5m2_e4m3_f32_64x16
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<32xf8E5M2>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E4M3>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: shape = #ixdl.shape<m = 64, n = 16, k = 32>
// CHECK-SAME: (vector<32xf8E5M2>, vector<8xf8E4M3>, vector<16xf32>) -> vector<16xf32>
func.func @test_cq_mma_f8_e5m2_e4m3_f32_64x16(
    %d: !fly.memref<f32, register, 16:1>,
    %a: !fly.memref<f8E5M2, register, 32:1>,
    %b: !fly.memref<f8E4M3, register, 8:1>,
    %c: !fly.memref<f32, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (f8E5M2, f8E4M3) -> f32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (f8E5M2, f8E4M3) -> f32>>,
         !fly.memref<f32, register, 16:1>, !fly.memref<f8E5M2, register, 32:1>,
         !fly.memref<f8E4M3, register, 8:1>, !fly.memref<f32, register, 16:1>) -> ()
  return
}

// ========================================================================
// FP8 f8E5M2 x f8E5M2 -> f32  (K=32)
// ========================================================================


// --- 16x16x32 ---

// CHECK-LABEL: @test_cq_mma_f8_e5m2_e5m2_f32_16x16
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E5M2>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E5M2>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 16, k = 32>
// CHECK-SAME: (vector<8xf8E5M2>, vector<8xf8E5M2>, vector<4xf32>) -> vector<4xf32>
func.func @test_cq_mma_f8_e5m2_e5m2_f32_16x16(
    %d: !fly.memref<f32, register, 4:1>,
    %a: !fly.memref<f8E5M2, register, 8:1>,
    %b: !fly.memref<f8E5M2, register, 8:1>,
    %c: !fly.memref<f32, register, 4:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (f8E5M2, f8E5M2) -> f32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (f8E5M2, f8E5M2) -> f32>>,
         !fly.memref<f32, register, 4:1>, !fly.memref<f8E5M2, register, 8:1>,
         !fly.memref<f8E5M2, register, 8:1>, !fly.memref<f32, register, 4:1>) -> ()
  return
}

// --- 32x32x32 ---

// CHECK-LABEL: @test_cq_mma_f8_e5m2_e5m2_f32_32x32
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf8E5M2>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf8E5M2>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: shape = #ixdl.shape<m = 32, n = 32, k = 32>
// CHECK-SAME: (vector<16xf8E5M2>, vector<16xf8E5M2>, vector<16xf32>) -> vector<16xf32>
func.func @test_cq_mma_f8_e5m2_e5m2_f32_32x32(
    %d: !fly.memref<f32, register, 16:1>,
    %a: !fly.memref<f8E5M2, register, 16:1>,
    %b: !fly.memref<f8E5M2, register, 16:1>,
    %c: !fly.memref<f32, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 32, (f8E5M2, f8E5M2) -> f32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 32, (f8E5M2, f8E5M2) -> f32>>,
         !fly.memref<f32, register, 16:1>, !fly.memref<f8E5M2, register, 16:1>,
         !fly.memref<f8E5M2, register, 16:1>, !fly.memref<f32, register, 16:1>) -> ()
  return
}

// --- 16x64x32 ---

// CHECK-LABEL: @test_cq_mma_f8_e5m2_e5m2_f32_16x64
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E5M2>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<32xf8E5M2>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 64, k = 32>
// CHECK-SAME: (vector<8xf8E5M2>, vector<32xf8E5M2>, vector<16xf32>) -> vector<16xf32>
func.func @test_cq_mma_f8_e5m2_e5m2_f32_16x64(
    %d: !fly.memref<f32, register, 16:1>,
    %a: !fly.memref<f8E5M2, register, 8:1>,
    %b: !fly.memref<f8E5M2, register, 32:1>,
    %c: !fly.memref<f32, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (f8E5M2, f8E5M2) -> f32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (f8E5M2, f8E5M2) -> f32>>,
         !fly.memref<f32, register, 16:1>, !fly.memref<f8E5M2, register, 8:1>,
         !fly.memref<f8E5M2, register, 32:1>, !fly.memref<f32, register, 16:1>) -> ()
  return
}

// --- 64x16x32 ---

// CHECK-LABEL: @test_cq_mma_f8_e5m2_e5m2_f32_64x16
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<32xf8E5M2>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E5M2>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf32>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: shape = #ixdl.shape<m = 64, n = 16, k = 32>
// CHECK-SAME: (vector<32xf8E5M2>, vector<8xf8E5M2>, vector<16xf32>) -> vector<16xf32>
func.func @test_cq_mma_f8_e5m2_e5m2_f32_64x16(
    %d: !fly.memref<f32, register, 16:1>,
    %a: !fly.memref<f8E5M2, register, 32:1>,
    %b: !fly.memref<f8E5M2, register, 8:1>,
    %c: !fly.memref<f32, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (f8E5M2, f8E5M2) -> f32>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (f8E5M2, f8E5M2) -> f32>>,
         !fly.memref<f32, register, 16:1>, !fly.memref<f8E5M2, register, 32:1>,
         !fly.memref<f8E5M2, register, 8:1>, !fly.memref<f32, register, 16:1>) -> ()
  return
}

// ========================================================================
// FP8 f8E4M3 x f8E4M3 -> f16  (K=32)
// ========================================================================


// --- 16x16x32 ---

// CHECK-LABEL: @test_cq_mma_f8_e4m3_e4m3_f16_16x16
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E4M3>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E4M3>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xf16>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 16, k = 32>
// CHECK-SAME: (vector<8xf8E4M3>, vector<8xf8E4M3>, vector<4xf16>) -> vector<4xf16>
func.func @test_cq_mma_f8_e4m3_e4m3_f16_16x16(
    %d: !fly.memref<f16, register, 4:1>,
    %a: !fly.memref<f8E4M3, register, 8:1>,
    %b: !fly.memref<f8E4M3, register, 8:1>,
    %c: !fly.memref<f16, register, 4:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (f8E4M3, f8E4M3) -> f16>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (f8E4M3, f8E4M3) -> f16>>,
         !fly.memref<f16, register, 4:1>, !fly.memref<f8E4M3, register, 8:1>,
         !fly.memref<f8E4M3, register, 8:1>, !fly.memref<f16, register, 4:1>) -> ()
  return
}

// --- 32x32x32 ---

// CHECK-LABEL: @test_cq_mma_f8_e4m3_e4m3_f16_32x32
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf8E4M3>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf8E4M3>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf16>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: shape = #ixdl.shape<m = 32, n = 32, k = 32>
// CHECK-SAME: (vector<16xf8E4M3>, vector<16xf8E4M3>, vector<16xf16>) -> vector<16xf16>
func.func @test_cq_mma_f8_e4m3_e4m3_f16_32x32(
    %d: !fly.memref<f16, register, 16:1>,
    %a: !fly.memref<f8E4M3, register, 16:1>,
    %b: !fly.memref<f8E4M3, register, 16:1>,
    %c: !fly.memref<f16, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 32, (f8E4M3, f8E4M3) -> f16>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 32, (f8E4M3, f8E4M3) -> f16>>,
         !fly.memref<f16, register, 16:1>, !fly.memref<f8E4M3, register, 16:1>,
         !fly.memref<f8E4M3, register, 16:1>, !fly.memref<f16, register, 16:1>) -> ()
  return
}

// --- 16x64x32 ---

// CHECK-LABEL: @test_cq_mma_f8_e4m3_e4m3_f16_16x64
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E4M3>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<32xf8E4M3>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf16>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 64, k = 32>
// CHECK-SAME: (vector<8xf8E4M3>, vector<32xf8E4M3>, vector<16xf16>) -> vector<16xf16>
func.func @test_cq_mma_f8_e4m3_e4m3_f16_16x64(
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

// --- 64x16x32 ---

// CHECK-LABEL: @test_cq_mma_f8_e4m3_e4m3_f16_64x16
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<32xf8E4M3>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E4M3>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf16>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: shape = #ixdl.shape<m = 64, n = 16, k = 32>
// CHECK-SAME: (vector<32xf8E4M3>, vector<8xf8E4M3>, vector<16xf16>) -> vector<16xf16>
func.func @test_cq_mma_f8_e4m3_e4m3_f16_64x16(
    %d: !fly.memref<f16, register, 16:1>,
    %a: !fly.memref<f8E4M3, register, 32:1>,
    %b: !fly.memref<f8E4M3, register, 8:1>,
    %c: !fly.memref<f16, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (f8E4M3, f8E4M3) -> f16>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (f8E4M3, f8E4M3) -> f16>>,
         !fly.memref<f16, register, 16:1>, !fly.memref<f8E4M3, register, 32:1>,
         !fly.memref<f8E4M3, register, 8:1>, !fly.memref<f16, register, 16:1>) -> ()
  return
}

// ========================================================================
// FP8 f8E4M3 x f8E5M2 -> f16  (K=32)
// ========================================================================


// --- 16x16x32 ---

// CHECK-LABEL: @test_cq_mma_f8_e4m3_e5m2_f16_16x16
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E4M3>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E5M2>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xf16>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 16, k = 32>
// CHECK-SAME: (vector<8xf8E4M3>, vector<8xf8E5M2>, vector<4xf16>) -> vector<4xf16>
func.func @test_cq_mma_f8_e4m3_e5m2_f16_16x16(
    %d: !fly.memref<f16, register, 4:1>,
    %a: !fly.memref<f8E4M3, register, 8:1>,
    %b: !fly.memref<f8E5M2, register, 8:1>,
    %c: !fly.memref<f16, register, 4:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (f8E4M3, f8E5M2) -> f16>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (f8E4M3, f8E5M2) -> f16>>,
         !fly.memref<f16, register, 4:1>, !fly.memref<f8E4M3, register, 8:1>,
         !fly.memref<f8E5M2, register, 8:1>, !fly.memref<f16, register, 4:1>) -> ()
  return
}

// --- 32x32x32 ---

// CHECK-LABEL: @test_cq_mma_f8_e4m3_e5m2_f16_32x32
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf8E4M3>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf8E5M2>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf16>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: shape = #ixdl.shape<m = 32, n = 32, k = 32>
// CHECK-SAME: (vector<16xf8E4M3>, vector<16xf8E5M2>, vector<16xf16>) -> vector<16xf16>
func.func @test_cq_mma_f8_e4m3_e5m2_f16_32x32(
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

// --- 16x64x32 ---

// CHECK-LABEL: @test_cq_mma_f8_e4m3_e5m2_f16_16x64
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E4M3>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<32xf8E5M2>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf16>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 64, k = 32>
// CHECK-SAME: (vector<8xf8E4M3>, vector<32xf8E5M2>, vector<16xf16>) -> vector<16xf16>
func.func @test_cq_mma_f8_e4m3_e5m2_f16_16x64(
    %d: !fly.memref<f16, register, 16:1>,
    %a: !fly.memref<f8E4M3, register, 8:1>,
    %b: !fly.memref<f8E5M2, register, 32:1>,
    %c: !fly.memref<f16, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (f8E4M3, f8E5M2) -> f16>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (f8E4M3, f8E5M2) -> f16>>,
         !fly.memref<f16, register, 16:1>, !fly.memref<f8E4M3, register, 8:1>,
         !fly.memref<f8E5M2, register, 32:1>, !fly.memref<f16, register, 16:1>) -> ()
  return
}

// --- 64x16x32 ---

// CHECK-LABEL: @test_cq_mma_f8_e4m3_e5m2_f16_64x16
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<32xf8E4M3>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E5M2>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf16>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: shape = #ixdl.shape<m = 64, n = 16, k = 32>
// CHECK-SAME: (vector<32xf8E4M3>, vector<8xf8E5M2>, vector<16xf16>) -> vector<16xf16>
func.func @test_cq_mma_f8_e4m3_e5m2_f16_64x16(
    %d: !fly.memref<f16, register, 16:1>,
    %a: !fly.memref<f8E4M3, register, 32:1>,
    %b: !fly.memref<f8E5M2, register, 8:1>,
    %c: !fly.memref<f16, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (f8E4M3, f8E5M2) -> f16>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (f8E4M3, f8E5M2) -> f16>>,
         !fly.memref<f16, register, 16:1>, !fly.memref<f8E4M3, register, 32:1>,
         !fly.memref<f8E5M2, register, 8:1>, !fly.memref<f16, register, 16:1>) -> ()
  return
}

// ========================================================================
// FP8 f8E5M2 x f8E4M3 -> f16  (K=32)
// ========================================================================


// --- 16x16x32 ---

// CHECK-LABEL: @test_cq_mma_f8_e5m2_e4m3_f16_16x16
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E5M2>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E4M3>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xf16>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 16, k = 32>
// CHECK-SAME: (vector<8xf8E5M2>, vector<8xf8E4M3>, vector<4xf16>) -> vector<4xf16>
func.func @test_cq_mma_f8_e5m2_e4m3_f16_16x16(
    %d: !fly.memref<f16, register, 4:1>,
    %a: !fly.memref<f8E5M2, register, 8:1>,
    %b: !fly.memref<f8E4M3, register, 8:1>,
    %c: !fly.memref<f16, register, 4:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (f8E5M2, f8E4M3) -> f16>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (f8E5M2, f8E4M3) -> f16>>,
         !fly.memref<f16, register, 4:1>, !fly.memref<f8E5M2, register, 8:1>,
         !fly.memref<f8E4M3, register, 8:1>, !fly.memref<f16, register, 4:1>) -> ()
  return
}

// --- 32x32x32 ---

// CHECK-LABEL: @test_cq_mma_f8_e5m2_e4m3_f16_32x32
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf8E5M2>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf8E4M3>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf16>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: shape = #ixdl.shape<m = 32, n = 32, k = 32>
// CHECK-SAME: (vector<16xf8E5M2>, vector<16xf8E4M3>, vector<16xf16>) -> vector<16xf16>
func.func @test_cq_mma_f8_e5m2_e4m3_f16_32x32(
    %d: !fly.memref<f16, register, 16:1>,
    %a: !fly.memref<f8E5M2, register, 16:1>,
    %b: !fly.memref<f8E4M3, register, 16:1>,
    %c: !fly.memref<f16, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 32, (f8E5M2, f8E4M3) -> f16>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 32, (f8E5M2, f8E4M3) -> f16>>,
         !fly.memref<f16, register, 16:1>, !fly.memref<f8E5M2, register, 16:1>,
         !fly.memref<f8E4M3, register, 16:1>, !fly.memref<f16, register, 16:1>) -> ()
  return
}

// --- 16x64x32 ---

// CHECK-LABEL: @test_cq_mma_f8_e5m2_e4m3_f16_16x64
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E5M2>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<32xf8E4M3>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf16>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 64, k = 32>
// CHECK-SAME: (vector<8xf8E5M2>, vector<32xf8E4M3>, vector<16xf16>) -> vector<16xf16>
func.func @test_cq_mma_f8_e5m2_e4m3_f16_16x64(
    %d: !fly.memref<f16, register, 16:1>,
    %a: !fly.memref<f8E5M2, register, 8:1>,
    %b: !fly.memref<f8E4M3, register, 32:1>,
    %c: !fly.memref<f16, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (f8E5M2, f8E4M3) -> f16>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (f8E5M2, f8E4M3) -> f16>>,
         !fly.memref<f16, register, 16:1>, !fly.memref<f8E5M2, register, 8:1>,
         !fly.memref<f8E4M3, register, 32:1>, !fly.memref<f16, register, 16:1>) -> ()
  return
}

// --- 64x16x32 ---

// CHECK-LABEL: @test_cq_mma_f8_e5m2_e4m3_f16_64x16
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<32xf8E5M2>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E4M3>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf16>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e4m3>
// CHECK-SAME: shape = #ixdl.shape<m = 64, n = 16, k = 32>
// CHECK-SAME: (vector<32xf8E5M2>, vector<8xf8E4M3>, vector<16xf16>) -> vector<16xf16>
func.func @test_cq_mma_f8_e5m2_e4m3_f16_64x16(
    %d: !fly.memref<f16, register, 16:1>,
    %a: !fly.memref<f8E5M2, register, 32:1>,
    %b: !fly.memref<f8E4M3, register, 8:1>,
    %c: !fly.memref<f16, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (f8E5M2, f8E4M3) -> f16>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<64, 16, 32, (f8E5M2, f8E4M3) -> f16>>,
         !fly.memref<f16, register, 16:1>, !fly.memref<f8E5M2, register, 32:1>,
         !fly.memref<f8E4M3, register, 8:1>, !fly.memref<f16, register, 16:1>) -> ()
  return
}

// ========================================================================
// FP8 f8E5M2 x f8E5M2 -> f16  (K=32)
// ========================================================================


// --- 16x16x32 ---

// CHECK-LABEL: @test_cq_mma_f8_e5m2_e5m2_f16_16x16
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E5M2>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E5M2>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<4xf16>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 16, k = 32>
// CHECK-SAME: (vector<8xf8E5M2>, vector<8xf8E5M2>, vector<4xf16>) -> vector<4xf16>
func.func @test_cq_mma_f8_e5m2_e5m2_f16_16x16(
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

// --- 32x32x32 ---

// CHECK-LABEL: @test_cq_mma_f8_e5m2_e5m2_f16_32x32
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf8E5M2>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf8E5M2>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf16>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: shape = #ixdl.shape<m = 32, n = 32, k = 32>
// CHECK-SAME: (vector<16xf8E5M2>, vector<16xf8E5M2>, vector<16xf16>) -> vector<16xf16>
func.func @test_cq_mma_f8_e5m2_e5m2_f16_32x32(
    %d: !fly.memref<f16, register, 16:1>,
    %a: !fly.memref<f8E5M2, register, 16:1>,
    %b: !fly.memref<f8E5M2, register, 16:1>,
    %c: !fly.memref<f16, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 32, (f8E5M2, f8E5M2) -> f16>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<32, 32, 32, (f8E5M2, f8E5M2) -> f16>>,
         !fly.memref<f16, register, 16:1>, !fly.memref<f8E5M2, register, 16:1>,
         !fly.memref<f8E5M2, register, 16:1>, !fly.memref<f16, register, 16:1>) -> ()
  return
}

// --- 16x64x32 ---

// CHECK-LABEL: @test_cq_mma_f8_e5m2_e5m2_f16_16x64
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E5M2>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<32xf8E5M2>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf16>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: shape = #ixdl.shape<m = 16, n = 64, k = 32>
// CHECK-SAME: (vector<8xf8E5M2>, vector<32xf8E5M2>, vector<16xf16>) -> vector<16xf16>
func.func @test_cq_mma_f8_e5m2_e5m2_f16_16x64(
    %d: !fly.memref<f16, register, 16:1>,
    %a: !fly.memref<f8E5M2, register, 8:1>,
    %b: !fly.memref<f8E5M2, register, 32:1>,
    %c: !fly.memref<f16, register, 16:1>) {
  %atom = fly.make_mma_atom : !fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (f8E5M2, f8E5M2) -> f16>>
  fly.mma_atom_call(%atom, %d, %a, %b, %c)
      : (!fly.mma_atom<!fly_ixdl.cq.mma<16, 64, 32, (f8E5M2, f8E5M2) -> f16>>,
         !fly.memref<f16, register, 16:1>, !fly.memref<f8E5M2, register, 8:1>,
         !fly.memref<f8E5M2, register, 32:1>, !fly.memref<f16, register, 16:1>) -> ()
  return
}

// --- 64x16x32 ---

// CHECK-LABEL: @test_cq_mma_f8_e5m2_e5m2_f16_64x16
// CHECK: %[[AV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<32xf8E5M2>
// CHECK: %[[BV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<8xf8E5M2>
// CHECK: %[[CV:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> vector<16xf16>
// CHECK: ixdl.mmad
// CHECK-SAME: multiplicandAType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: multiplicandBType = #ixdl.mmad_type<f8e5m2>
// CHECK-SAME: shape = #ixdl.shape<m = 64, n = 16, k = 32>
// CHECK-SAME: (vector<32xf8E5M2>, vector<8xf8E5M2>, vector<16xf16>) -> vector<16xf16>
func.func @test_cq_mma_f8_e5m2_e5m2_f16_64x16(
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
