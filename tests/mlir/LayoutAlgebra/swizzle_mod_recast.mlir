// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --split-input-file | FileCheck %s

// Bit-level <-> element-level SwizzleMod recast for the SME `rowxfb8` atom.
//
// cute defines the rowxfb8 SmemAtom at BIT granularity with a `smem_ptr_flag`
// (1-bit) and `Swizzle_Mod<2,6,2>`, and recovers the per-type element atom via
// the flag-preserving `upcast<sizeof_bits<T>>` (the swizzle survives unchanged).
//
// FlyDSL has NO `smem_ptr_flag` specialization: `recast_layout` (and the
// underlying `layoutUpcast`) applies the GENERIC cute recast, which rescales the
// swizzle base by log2(factor): SwizzleMod<B,M,S> -> SwizzleMod<B, M-log2(N), S>.
//
// Therefore the FlyDSL canonical *bit-level* rowxfb8 swizzle must be written as
// `Swizzle_Mod<2,9,2>` (NOT <2,6,2>), so that recasting a 1-bit element layout
// to an 8-bit (int8) element layout yields the element-space `Swizzle_Mod<2,6,2>`
// used by the Row8b SmemAtom (see swizzle_mod_crd2idx.mlir):
//   base: 9 - log2(8) = 9 - 3 = 6.

// CHECK-LABEL: @rowxfb8_bit_to_int8_flat
func.func @rowxfb8_bit_to_int8_flat(%d: i32) -> i32 {
  %s = fly.static : !fly.swizzle_mod<SM<2,9,2>>
  %off = fly.static : !fly.int_tuple<0>
  %sh = fly.static : !fly.int_tuple<8192>
  %st = fly.static : !fly.int_tuple<1>
  %outer = fly.make_layout(%sh, %st)
      : (!fly.int_tuple<8192>, !fly.int_tuple<1>) -> !fly.layout<8192:1>
  %cl = fly.make_composed_layout(%s, %off, %outer)
      : (!fly.swizzle_mod<SM<2,9,2>>, !fly.int_tuple<0>, !fly.layout<8192:1>)
      -> !fly.composed_layout<SM<2,9,2> o 0 o 8192:1>
  // 1-bit -> 8-bit (int8): base 9 -> 6, span 8192 bits -> 1024 elements.
  // CHECK: fly.recast_layout{{.*}} -> !fly.composed_layout<SM<2,6,2> o 0 o 1024:1>
  %rc = fly.recast_layout(%cl) {new_type_bits = 8 : i32, old_type_bits = 1 : i32}
      : (!fly.composed_layout<SM<2,9,2> o 0 o 8192:1>)
      -> !fly.composed_layout<SM<2,6,2> o 0 o 1024:1>
  return %d : i32
}

// -----

// Full cute `Layout_SME_I_16x512b_MN_ROWXFB8_Atom_Bits` shape, transcribed with
// the FlyDSL no-flag canonical swizzle SM<2,9,2>. Recasting 1-bit -> 8-bit gives
// the element-space int8 Row8b atom (inner stride-1/size-8 collapses to size-1).

// CHECK-LABEL: @rowxfb8_bit_to_int8_atom
func.func @rowxfb8_bit_to_int8_atom(%d: i32) -> i32 {
  %s = fly.static : !fly.swizzle_mod<SM<2,9,2>>
  %off = fly.static : !fly.int_tuple<0>
  %sh = fly.static : !fly.int_tuple<((8,4,4,4),(4,4))>
  %st = fly.static : !fly.int_tuple<((1,32,128,2048),(8,512))>
  %outer = fly.make_layout(%sh, %st)
      : (!fly.int_tuple<((8,4,4,4),(4,4))>, !fly.int_tuple<((1,32,128,2048),(8,512))>)
      -> !fly.layout<((8, 4, 4, 4), (4, 4)) : ((1, 32, 128, 2048), (8, 512))>
  %cl = fly.make_composed_layout(%s, %off, %outer)
      : (!fly.swizzle_mod<SM<2,9,2>>, !fly.int_tuple<0>,
         !fly.layout<((8, 4, 4, 4), (4, 4)) : ((1, 32, 128, 2048), (8, 512))>)
      -> !fly.composed_layout<SM<2,9,2> o 0 o ((8, 4, 4, 4), (4, 4)) : ((1, 32, 128, 2048), (8, 512))>
  // CHECK: fly.recast_layout{{.*}} -> !fly.composed_layout<SM<2,6,2> o 0 o ((1,4,4,4),(4,4)):((1,4,16,256),(1,64))>
  %rc = fly.recast_layout(%cl) {new_type_bits = 8 : i32, old_type_bits = 1 : i32}
      : (!fly.composed_layout<SM<2,9,2> o 0 o ((8, 4, 4, 4), (4, 4)) : ((1, 32, 128, 2048), (8, 512))>)
      -> !fly.composed_layout<SM<2,6,2> o 0 o ((1, 4, 4, 4), (4, 4)) : ((1, 4, 16, 256), (1, 64))>
  return %d : i32
}

// -----

// Reverse direction (downcast int8 element -> 1-bit): base 6 -> 9 confirms the
// bit-level canonical form is SM<2,9,2>.

// CHECK-LABEL: @rowxfb8_int8_to_bit_flat
func.func @rowxfb8_int8_to_bit_flat(%d: i32) -> i32 {
  %s = fly.static : !fly.swizzle_mod<SM<2,6,2>>
  %off = fly.static : !fly.int_tuple<0>
  %sh = fly.static : !fly.int_tuple<1024>
  %st = fly.static : !fly.int_tuple<1>
  %outer = fly.make_layout(%sh, %st)
      : (!fly.int_tuple<1024>, !fly.int_tuple<1>) -> !fly.layout<1024:1>
  %cl = fly.make_composed_layout(%s, %off, %outer)
      : (!fly.swizzle_mod<SM<2,6,2>>, !fly.int_tuple<0>, !fly.layout<1024:1>)
      -> !fly.composed_layout<SM<2,6,2> o 0 o 1024:1>
  // 8-bit (int8) -> 1-bit: base 6 -> 9, span 1024 elements -> 8192 bits.
  // CHECK: fly.recast_layout{{.*}} -> !fly.composed_layout<SM<2,9,2> o 0 o 8192:1>
  %rc = fly.recast_layout(%cl) {new_type_bits = 1 : i32, old_type_bits = 8 : i32}
      : (!fly.composed_layout<SM<2,6,2> o 0 o 1024:1>)
      -> !fly.composed_layout<SM<2,9,2> o 0 o 8192:1>
  return %d : i32
}
