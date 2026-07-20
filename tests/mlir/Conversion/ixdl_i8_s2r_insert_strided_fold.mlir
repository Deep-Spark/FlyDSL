// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-ixdl | FileCheck %s

// i8 S2R peep-holes in convert-fly-to-ixdl:
// 1) two i32→v4i8 bitcasts packed via vector.insert_strided_slice into
//    vector<8xi8> fold to a permute-free vector<2xi32> bitcast
// 2) shared→reg UniversalCopy of vector<4xi8> rewrites to i32 load + bitcast
//
// Without (1), convert-vector-to-llvm emits a byte-granular shufflevector that
// ISel lowers to an expensive byte-permute.

// CHECK-LABEL: @fold_i8_insert_strided_concat
// CHECK-SAME:  (%[[LO:.*]]: i32, %[[HI:.*]]: i32)
// CHECK-NOT:     vector.insert_strided_slice
// CHECK-DAG:     %[[POISON:.*]] = llvm.mlir.poison : vector<2xi32>
// CHECK-DAG:     %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-DAG:     %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:         %[[LOINS:.*]] = llvm.insertelement %[[LO]], %[[POISON]][%[[C0]] : i32] : vector<2xi32>
// CHECK:         %[[HIINS:.*]] = llvm.insertelement %[[HI]], %[[LOINS]][%[[C1]] : i32] : vector<2xi32>
// CHECK:         %[[OUT:.*]] = llvm.bitcast %[[HIINS]] : vector<2xi32> to vector<8xi8>
// CHECK:         return %[[OUT]] : vector<8xi8>
func.func @fold_i8_insert_strided_concat(%lo: i32, %hi: i32) -> vector<8xi8> {
  %lo4 = llvm.bitcast %lo : i32 to vector<4xi8>
  %hi4 = llvm.bitcast %hi : i32 to vector<4xi8>
  %dest = arith.constant dense<0> : vector<8xi8>
  %mid = vector.insert_strided_slice %lo4, %dest {offsets = [0], strides = [1]}
      : vector<4xi8> into vector<8xi8>
  %out = vector.insert_strided_slice %hi4, %mid {offsets = [4], strides = [1]}
      : vector<4xi8> into vector<8xi8>
  return %out : vector<8xi8>
}

// CHECK-LABEL: @shared_i8_s2r_i32_load
// CHECK:         %[[LD:.*]] = llvm.load %{{.*}} : !llvm.ptr<3> -> i32
// CHECK:         %[[BC:.*]] = llvm.bitcast %[[LD]] : i32 to vector<4xi8>
// CHECK:         return %[[BC]] : vector<4xi8>
func.func @shared_i8_s2r_i32_load(%src: !fly.memref<i8, shared, 4:1>) -> vector<4xi8> {
  %atom = fly.make_copy_atom {valBits = 32 : i32} : !fly.copy_atom<!fly.universal_copy<32>, 32>
  %v = fly.copy_atom_call_ssa(%atom, %src) {operandSegmentSizes = array<i32: 1, 1, 0, 0>}
      : (!fly.copy_atom<!fly.universal_copy<32>, 32>, !fly.memref<i8, shared, 4:1>) -> vector<4xi8>
  return %v : vector<4xi8>
}

// Only the 32-bit i8 S2R atom is widened; other i8 vector widths stay as-is.
// CHECK-LABEL: @shared_i8x8_load_untouched
// CHECK:         llvm.load %{{.*}} : !llvm.ptr<3> -> vector<8xi8>
// CHECK-NOT:     llvm.bitcast
func.func @shared_i8x8_load_untouched(%src: !llvm.ptr<3>) -> vector<8xi8> {
  %v = llvm.load %src : !llvm.ptr<3> -> vector<8xi8>
  return %v : vector<8xi8>
}
