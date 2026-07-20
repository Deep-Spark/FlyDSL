// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-ixdl | FileCheck %s

// i8 S2R packing through convert-fly-to-ixdl:
// 1) two i32→v4i8 bitcasts packed via vector.insert_strided_slice into
//    vector<8xi8> stay as an insert_strided_slice chain
// 2) shared→reg UniversalCopy of vector<4xi8> stays a byte-granular
//    vector i8 load

// CHECK-LABEL: @fold_i8_insert_strided_concat
// CHECK-SAME:  (%[[LO:.*]]: i32, %[[HI:.*]]: i32)
// CHECK:         %[[LO4:.*]] = llvm.bitcast %[[LO]] : i32 to vector<4xi8>
// CHECK:         %[[HI4:.*]] = llvm.bitcast %[[HI]] : i32 to vector<4xi8>
// CHECK:         vector.insert_strided_slice %[[LO4]], %{{.*}} {offsets = [0], strides = [1]}
// CHECK:         vector.insert_strided_slice %[[HI4]], %{{.*}} {offsets = [4], strides = [1]}
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
// CHECK:         %[[LD:.*]] = llvm.load %{{.*}} : !llvm.ptr<3> -> vector<4xi8>
// CHECK:         return %[[LD]] : vector<4xi8>
func.func @shared_i8_s2r_i32_load(%src: !fly.memref<i8, shared, 4:1>) -> vector<4xi8> {
  %atom = fly.make_copy_atom {valBits = 32 : i32} : !fly.copy_atom<!fly.universal_copy<32>, 32>
  %v = fly.copy_atom_call_ssa(%atom, %src) {operandSegmentSizes = array<i32: 1, 1, 0, 0>}
      : (!fly.copy_atom<!fly.universal_copy<32>, 32>, !fly.memref<i8, shared, 4:1>) -> vector<4xi8>
  return %v : vector<4xi8>
}
