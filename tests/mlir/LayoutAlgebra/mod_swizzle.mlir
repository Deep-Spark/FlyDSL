// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s | FileCheck %s

// Tests for the ModSwizzle attribute/type (MS<mask,base,shift>) and its use as
// the inner slot of ComposedLayoutAttr. Unlike the XOR-based Swizzle (S<...>),
// ModSwizzle applies an additive wrap-around within the low (mask+base) bits:
//
//   yyy     = ((1<<mask)-1) << (base + max(0,shift))
//   zb      = (1 << (mask+base)) - 1            // low-bit window
//   shifted = shift>=0 ? (v & yyy) >> shift : (v & yyy) << -shift
//   result  = (v & ~zb) | ((v + shifted) & zb)  // ADD, can carry
//
// Covers:
//   * Standalone ModSwizzle materialized via fly.static.
//   * Trivial ModSwizzle (mask=0) round-trip.
//   * ComposedLayout whose inner is a ModSwizzle.
//   * Extractors composed_get_inner / composed_get_offset / composed_get_outer.
//   * Static crd2idx folding showing the additive-carry semantics, and a
//     side-by-side contrast against the XOR Swizzle that proves they differ.
//   * Trivial ModSwizzle behaves as identity under crd2idx.

// -----

// CHECK-LABEL: @test_mod_swizzle_static
func.func @test_mod_swizzle_static() -> !fly.mod_swizzle<MS<2,3,2>> {
  // CHECK: fly.static : !fly.mod_swizzle<MS<2,3,2>>
  %ms = fly.static : !fly.mod_swizzle<MS<2,3,2>>
  return %ms : !fly.mod_swizzle<MS<2,3,2>>
}

// CHECK-LABEL: @test_mod_swizzle_trivial
func.func @test_mod_swizzle_trivial() -> !fly.mod_swizzle<MS<0,0,0>> {
  // CHECK: fly.static : !fly.mod_swizzle<MS<0,0,0>>
  %ms = fly.static : !fly.mod_swizzle<MS<0,0,0>>
  return %ms : !fly.mod_swizzle<MS<0,0,0>>
}

// CHECK-LABEL: @test_composed_layout_with_mod_swizzle_inner
func.func @test_composed_layout_with_mod_swizzle_inner()
    -> !fly.composed_layout<MS<2,3,2> o 0 o 128:1> {
  %ms = fly.static : !fly.mod_swizzle<MS<2,3,2>>
  %off = fly.static : !fly.int_tuple<0>
  %s = fly.static : !fly.int_tuple<128>
  %d = fly.static : !fly.int_tuple<1>
  %outer = fly.make_layout(%s, %d)
      : (!fly.int_tuple<128>, !fly.int_tuple<1>) -> !fly.layout<128:1>
  // CHECK: fly.make_composed_layout
  // CHECK-SAME: !fly.mod_swizzle<MS<2,3,2>>
  // CHECK-SAME: -> !fly.composed_layout<MS<2,3,2> o 0 o 128:1>
  %cl = fly.make_composed_layout(%ms, %off, %outer)
      : (!fly.mod_swizzle<MS<2,3,2>>, !fly.int_tuple<0>, !fly.layout<128:1>)
      -> !fly.composed_layout<MS<2,3,2> o 0 o 128:1>
  return %cl : !fly.composed_layout<MS<2,3,2> o 0 o 128:1>
}

// CHECK-LABEL: @test_composed_get_inner_mod_swizzle
// CHECK-SAME: %[[CL:.+]]: !fly.composed_layout<MS<2,3,2> o 0 o 128:1>
func.func @test_composed_get_inner_mod_swizzle(
    %cl: !fly.composed_layout<MS<2,3,2> o 0 o 128:1>)
    -> !fly.mod_swizzle<MS<2,3,2>> {
  // CHECK: fly.composed_get_inner(%[[CL]])
  // CHECK-SAME: -> !fly.mod_swizzle<MS<2,3,2>>
  %inner = fly.composed_get_inner(%cl)
      : (!fly.composed_layout<MS<2,3,2> o 0 o 128:1>) -> !fly.mod_swizzle<MS<2,3,2>>
  return %inner : !fly.mod_swizzle<MS<2,3,2>>
}

// CHECK-LABEL: @test_composed_get_offset_mod_swizzle
func.func @test_composed_get_offset_mod_swizzle(
    %cl: !fly.composed_layout<MS<2,3,2> o 0 o 128:1>) -> !fly.int_tuple<0> {
  // CHECK: fly.composed_get_offset
  // CHECK-SAME: -> !fly.int_tuple<0>
  %off = fly.composed_get_offset(%cl)
      : (!fly.composed_layout<MS<2,3,2> o 0 o 128:1>) -> !fly.int_tuple<0>
  return %off : !fly.int_tuple<0>
}

// CHECK-LABEL: @test_composed_get_outer_mod_swizzle
func.func @test_composed_get_outer_mod_swizzle(
    %cl: !fly.composed_layout<MS<2,3,2> o 0 o 128:1>) -> !fly.layout<128:1> {
  // CHECK: fly.composed_get_outer
  // CHECK-SAME: -> !fly.layout<128:1>
  %outer = fly.composed_get_outer(%cl)
      : (!fly.composed_layout<MS<2,3,2> o 0 o 128:1>) -> !fly.layout<128:1>
  return %outer : !fly.layout<128:1>
}

// -----

// Static crd2idx through a ComposedLayout whose inner is MS<2,3,2> and whose
// outer is the identity layout 128:1, so the result is purely applyModSwizzle.
// Hand-computed (mask=3, shift=2, yyy=96, zb=31):
//   120 -> (120 & ~31) | ((120 + ((120&96)>>2)) & 31)
//        = 96 | ((120 + 24) & 31) = 96 | (144 & 31) = 96 | 16 = 112   (carry!)
//    96 -> 96 | ((96 + 24) & 31)  = 96 | 24 = 120
//    64 -> 64 | ((64 + 16) & 31)  = 64 | 16 = 80
// CHECK-LABEL: @test_crd2idx_mod_swizzle_add_wraparound
func.func @test_crd2idx_mod_swizzle_add_wraparound() {
  %off = fly.static : !fly.int_tuple<0>
  %s = fly.static : !fly.int_tuple<128>
  %d = fly.static : !fly.int_tuple<1>
  %outer = fly.make_layout(%s, %d)
      : (!fly.int_tuple<128>, !fly.int_tuple<1>) -> !fly.layout<128:1>
  %ms = fly.static : !fly.mod_swizzle<MS<2,3,2>>
  %cl = fly.make_composed_layout(%ms, %off, %outer)
      : (!fly.mod_swizzle<MS<2,3,2>>, !fly.int_tuple<0>, !fly.layout<128:1>)
      -> !fly.composed_layout<MS<2,3,2> o 0 o 128:1>

  %c120 = fly.static : !fly.int_tuple<120>
  // CHECK: fly.crd2idx
  // CHECK-SAME: -> !fly.int_tuple<112>
  %i120 = fly.crd2idx(%c120, %cl)
      : (!fly.int_tuple<120>, !fly.composed_layout<MS<2,3,2> o 0 o 128:1>) -> !fly.int_tuple<112>

  %c96 = fly.static : !fly.int_tuple<96>
  // CHECK: fly.crd2idx
  // CHECK-SAME: -> !fly.int_tuple<120>
  %i96 = fly.crd2idx(%c96, %cl)
      : (!fly.int_tuple<96>, !fly.composed_layout<MS<2,3,2> o 0 o 128:1>) -> !fly.int_tuple<120>

  %c64 = fly.static : !fly.int_tuple<64>
  // CHECK: fly.crd2idx
  // CHECK-SAME: -> !fly.int_tuple<80>
  %i64 = fly.crd2idx(%c64, %cl)
      : (!fly.int_tuple<64>, !fly.composed_layout<MS<2,3,2> o 0 o 128:1>) -> !fly.int_tuple<80>
  return
}

// Same shape parameters but XOR Swizzle S<2,3,2>: 120 -> 120 ^ 24 = 96, which
// differs from ModSwizzle's 112 above. This pins down that MS adds (with carry)
// while S xors.
// CHECK-LABEL: @test_crd2idx_swizzle_xor_contrast
func.func @test_crd2idx_swizzle_xor_contrast() {
  %off = fly.static : !fly.int_tuple<0>
  %s = fly.static : !fly.int_tuple<128>
  %d = fly.static : !fly.int_tuple<1>
  %outer = fly.make_layout(%s, %d)
      : (!fly.int_tuple<128>, !fly.int_tuple<1>) -> !fly.layout<128:1>
  %sw = fly.static : !fly.swizzle<S<2,3,2>>
  %cl = fly.make_composed_layout(%sw, %off, %outer)
      : (!fly.swizzle<S<2,3,2>>, !fly.int_tuple<0>, !fly.layout<128:1>)
      -> !fly.composed_layout<S<2,3,2> o 0 o 128:1>

  %c120 = fly.static : !fly.int_tuple<120>
  // CHECK: fly.crd2idx
  // CHECK-SAME: -> !fly.int_tuple<96>
  %i120 = fly.crd2idx(%c120, %cl)
      : (!fly.int_tuple<120>, !fly.composed_layout<S<2,3,2> o 0 o 128:1>) -> !fly.int_tuple<96>
  return
}

// Trivial ModSwizzle (mask=0) is the identity mapping under crd2idx.
// CHECK-LABEL: @test_crd2idx_mod_swizzle_trivial_identity
func.func @test_crd2idx_mod_swizzle_trivial_identity() {
  %off = fly.static : !fly.int_tuple<0>
  %s = fly.static : !fly.int_tuple<128>
  %d = fly.static : !fly.int_tuple<1>
  %outer = fly.make_layout(%s, %d)
      : (!fly.int_tuple<128>, !fly.int_tuple<1>) -> !fly.layout<128:1>
  %ms = fly.static : !fly.mod_swizzle<MS<0,3,2>>
  %cl = fly.make_composed_layout(%ms, %off, %outer)
      : (!fly.mod_swizzle<MS<0,3,2>>, !fly.int_tuple<0>, !fly.layout<128:1>)
      -> !fly.composed_layout<MS<0,3,2> o 0 o 128:1>

  %c120 = fly.static : !fly.int_tuple<120>
  // CHECK: fly.crd2idx
  // CHECK-SAME: -> !fly.int_tuple<120>
  %i120 = fly.crd2idx(%c120, %cl)
      : (!fly.int_tuple<120>, !fly.composed_layout<MS<0,3,2> o 0 o 128:1>) -> !fly.int_tuple<120>
  return
}

// crd2idx accepts a standalone ModSwizzle directly (no composed layout) -- the
// Crd2IdxOp operand constraint lists Fly_ModSwizzle alongside Fly_Swizzle /
// Fly_CoordSwizzle. The static result folds via applyModSwizzle (120 -> 112).
// CHECK-LABEL: @test_crd2idx_standalone_mod_swizzle
func.func @test_crd2idx_standalone_mod_swizzle() -> !fly.int_tuple<112> {
  %coord = fly.static : !fly.int_tuple<120>
  %ms = fly.static : !fly.mod_swizzle<MS<2,3,2>>
  // CHECK: fly.crd2idx
  // CHECK-SAME: -> !fly.int_tuple<112>
  %idx = fly.crd2idx(%coord, %ms)
      : (!fly.int_tuple<120>, !fly.mod_swizzle<MS<2,3,2>>) -> !fly.int_tuple<112>
  return %idx : !fly.int_tuple<112>
}

// -----

// The standalone crd2idx folding above exercises applyModSwizzle ->
// intApplyModSwizzle. The MS<2,3,2> cases only cover a positive shift; the
// following pin down the remaining static branches of intApplyModSwizzle.

// Negative shift: shifted = (v & yyy) << -shift (yyy = mask << base since
// max(0,shift)=0). MS<2,3,-1>: mask=3, base=3, shift=-1.
//   yyy = 3<<3 = 24, zb = (1<<5)-1 = 31.
//   24 -> (24 & ~31) | ((24 + ((24&24)<<1)) & 31)
//       = 0 | ((24 + 48) & 31) = 0 | (72 & 31) = 0 | 8 = 8
// CHECK-LABEL: @test_crd2idx_mod_swizzle_negative_shift
func.func @test_crd2idx_mod_swizzle_negative_shift() -> !fly.int_tuple<8> {
  %coord = fly.static : !fly.int_tuple<24>
  %ms = fly.static : !fly.mod_swizzle<MS<2,3,-1>>
  // CHECK: fly.crd2idx
  // CHECK-SAME: -> !fly.int_tuple<8>
  %idx = fly.crd2idx(%coord, %ms)
      : (!fly.int_tuple<24>, !fly.mod_swizzle<MS<2,3,-1>>) -> !fly.int_tuple<8>
  return %idx : !fly.int_tuple<8>
}

// Zero shift: shifted = (v & yyy) >> 0 = v & yyy. MS<2,3,0>: yyy=24, zb=31.
//   24 -> 0 | ((24 + 24) & 31) = (48 & 31) = 16
// CHECK-LABEL: @test_crd2idx_mod_swizzle_zero_shift
func.func @test_crd2idx_mod_swizzle_zero_shift() -> !fly.int_tuple<16> {
  %coord = fly.static : !fly.int_tuple<24>
  %ms = fly.static : !fly.mod_swizzle<MS<2,3,0>>
  // CHECK: fly.crd2idx
  // CHECK-SAME: -> !fly.int_tuple<16>
  %idx = fly.crd2idx(%coord, %ms)
      : (!fly.int_tuple<24>, !fly.mod_swizzle<MS<2,3,0>>) -> !fly.int_tuple<16>
  return %idx : !fly.int_tuple<16>
}

// base = 0 shrinks the low-bit window: zb = (1<<mask)-1. MS<2,0,2>:
//   yyy = 3<<(0+2) = 12, zb = (1<<2)-1 = 3.
//   12 -> (12 & ~3) | ((12 + ((12&12)>>2)) & 3) = 12 | ((12 + 3) & 3) = 12 | 3 = 15
//    4 -> ( 4 & ~3) | (( 4 + (( 4&12)>>2)) & 3) =  4 | (( 4 + 1) & 3) =  4 | 1 =  5
// CHECK-LABEL: @test_crd2idx_mod_swizzle_base_zero
func.func @test_crd2idx_mod_swizzle_base_zero() {
  %ms = fly.static : !fly.mod_swizzle<MS<2,0,2>>

  %c12 = fly.static : !fly.int_tuple<12>
  // CHECK: fly.crd2idx
  // CHECK-SAME: -> !fly.int_tuple<15>
  %i12 = fly.crd2idx(%c12, %ms)
      : (!fly.int_tuple<12>, !fly.mod_swizzle<MS<2,0,2>>) -> !fly.int_tuple<15>

  %c4 = fly.static : !fly.int_tuple<4>
  // CHECK: fly.crd2idx
  // CHECK-SAME: -> !fly.int_tuple<5>
  %i4 = fly.crd2idx(%c4, %ms)
      : (!fly.int_tuple<4>, !fly.mod_swizzle<MS<2,0,2>>) -> !fly.int_tuple<5>
  return
}
