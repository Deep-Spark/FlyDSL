// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --split-input-file | FileCheck %s

// Static crd2idx numeric checks against cute::Swizzle_Mod<2,6,2>:
//   result = (off & ~255) | ((off + ((off & 0x300) >> 2)) & 255)
// For off < 256 the swizzle is the identity. For off >= 256:
//   off=256 -> 320,  off=320 -> 384,  off=448 -> 256.
// The op verifier rejects a wrong declared result type, so a successful
// round-trip already pins the computed value.
//
// NOTE: SM<2,6,2> here is the ELEMENT-space rowxfb8 swizzle (the int8 Row8b
// SmemAtom). The FlyDSL canonical *bit-level* form is SM<2,9,2>; recasting it
// 1-bit -> 8-bit (int8) yields this SM<2,6,2> (base 9 - log2(8) = 6). See
// swizzle_mod_recast.mlir. (Numerics below are only meaningful at base 6: zb_mask
// = 0xFF; at bit-level base 9 these offsets would all fall in the identity range.)

// CHECK-LABEL: @sm_crd_lt256
func.func @sm_crd_lt256() -> !fly.int_tuple<192> {
  %s = fly.static : !fly.swizzle_mod<SM<2,6,2>>
  %off = fly.static : !fly.int_tuple<0>
  %sh = fly.static : !fly.int_tuple<(8, 64)>
  %st = fly.static : !fly.int_tuple<(64, 1)>
  %outer = fly.make_layout(%sh, %st)
      : (!fly.int_tuple<(8, 64)>, !fly.int_tuple<(64, 1)>) -> !fly.layout<(8, 64) : (64, 1)>
  %cl = fly.make_composed_layout(%s, %off, %outer)
      : (!fly.swizzle_mod<SM<2,6,2>>, !fly.int_tuple<0>, !fly.layout<(8, 64) : (64, 1)>)
      -> !fly.composed_layout<SM<2,6,2> o 0 o (8, 64) : (64, 1)>
  %coord = fly.static : !fly.int_tuple<(3, 0)>
  // (3,0) -> linear 192 < 256 -> identity 192
  // CHECK: fly.crd2idx{{.*}} -> !fly.int_tuple<192>
  %idx = fly.crd2idx(%coord, %cl)
      : (!fly.int_tuple<(3, 0)>, !fly.composed_layout<SM<2,6,2> o 0 o (8, 64) : (64, 1)>)
      -> !fly.int_tuple<192>
  return %idx : !fly.int_tuple<192>
}

// -----

// CHECK-LABEL: @sm_crd_256
func.func @sm_crd_256() -> !fly.int_tuple<320> {
  %s = fly.static : !fly.swizzle_mod<SM<2,6,2>>
  %off = fly.static : !fly.int_tuple<0>
  %sh = fly.static : !fly.int_tuple<(8, 64)>
  %st = fly.static : !fly.int_tuple<(64, 1)>
  %outer = fly.make_layout(%sh, %st)
      : (!fly.int_tuple<(8, 64)>, !fly.int_tuple<(64, 1)>) -> !fly.layout<(8, 64) : (64, 1)>
  %cl = fly.make_composed_layout(%s, %off, %outer)
      : (!fly.swizzle_mod<SM<2,6,2>>, !fly.int_tuple<0>, !fly.layout<(8, 64) : (64, 1)>)
      -> !fly.composed_layout<SM<2,6,2> o 0 o (8, 64) : (64, 1)>
  %coord = fly.static : !fly.int_tuple<(4, 0)>
  // (4,0) -> linear 256 -> swizzled 320
  // CHECK: fly.crd2idx{{.*}} -> !fly.int_tuple<320>
  %idx = fly.crd2idx(%coord, %cl)
      : (!fly.int_tuple<(4, 0)>, !fly.composed_layout<SM<2,6,2> o 0 o (8, 64) : (64, 1)>)
      -> !fly.int_tuple<320>
  return %idx : !fly.int_tuple<320>
}

// -----

// CHECK-LABEL: @sm_crd_448
func.func @sm_crd_448() -> !fly.int_tuple<256> {
  %s = fly.static : !fly.swizzle_mod<SM<2,6,2>>
  %off = fly.static : !fly.int_tuple<0>
  %sh = fly.static : !fly.int_tuple<(8, 64)>
  %st = fly.static : !fly.int_tuple<(64, 1)>
  %outer = fly.make_layout(%sh, %st)
      : (!fly.int_tuple<(8, 64)>, !fly.int_tuple<(64, 1)>) -> !fly.layout<(8, 64) : (64, 1)>
  %cl = fly.make_composed_layout(%s, %off, %outer)
      : (!fly.swizzle_mod<SM<2,6,2>>, !fly.int_tuple<0>, !fly.layout<(8, 64) : (64, 1)>)
      -> !fly.composed_layout<SM<2,6,2> o 0 o (8, 64) : (64, 1)>
  %coord = fly.static : !fly.int_tuple<(7, 0)>
  // (7,0) -> linear 448 -> swizzled 256
  // CHECK: fly.crd2idx{{.*}} -> !fly.int_tuple<256>
  %idx = fly.crd2idx(%coord, %cl)
      : (!fly.int_tuple<(7, 0)>, !fly.composed_layout<SM<2,6,2> o 0 o (8, 64) : (64, 1)>)
      -> !fly.int_tuple<256>
  return %idx : !fly.int_tuple<256>
}
