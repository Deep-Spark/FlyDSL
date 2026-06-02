// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s | FileCheck %s

// Tests for the Iluvatar-only SwizzleMod (modular / "add") attribute & type and
// its use as the inner slot of ComposedLayoutAttr. The regular XOR Swizzle is
// unaffected; these only exercise the parallel SwizzleMod path.

// -----

// CHECK-LABEL: @test_swizzle_mod_static
func.func @test_swizzle_mod_static() -> !fly.swizzle_mod<SM<2,6,2>> {
  // CHECK: fly.static : !fly.swizzle_mod<SM<2,6,2>>
  %s = fly.static : !fly.swizzle_mod<SM<2,6,2>>
  return %s : !fly.swizzle_mod<SM<2,6,2>>
}

// CHECK-LABEL: @test_swizzle_mod_trivial
func.func @test_swizzle_mod_trivial() -> !fly.swizzle_mod<SM<0,0,0>> {
  // CHECK: fly.static : !fly.swizzle_mod<SM<0,0,0>>
  %s = fly.static : !fly.swizzle_mod<SM<0,0,0>>
  return %s : !fly.swizzle_mod<SM<0,0,0>>
}

// CHECK-LABEL: @test_composed_layout_with_swizzle_mod_inner
func.func @test_composed_layout_with_swizzle_mod_inner()
    -> !fly.composed_layout<SM<2,6,2> o 0 o (8,64):(64,1)> {
  %s = fly.static : !fly.swizzle_mod<SM<2,6,2>>
  %off = fly.static : !fly.int_tuple<0>
  %sh = fly.static : !fly.int_tuple<(8, 64)>
  %st = fly.static : !fly.int_tuple<(64, 1)>
  %outer = fly.make_layout(%sh, %st)
      : (!fly.int_tuple<(8, 64)>, !fly.int_tuple<(64, 1)>)
      -> !fly.layout<(8, 64) : (64, 1)>
  // CHECK: fly.make_composed_layout
  // CHECK-SAME: !fly.swizzle_mod<SM<2,6,2>>
  // CHECK-SAME: -> !fly.composed_layout<SM<2,6,2> o 0 o (8,64):(64,1)>
  %cl = fly.make_composed_layout(%s, %off, %outer)
      : (!fly.swizzle_mod<SM<2,6,2>>, !fly.int_tuple<0>, !fly.layout<(8, 64) : (64, 1)>)
      -> !fly.composed_layout<SM<2,6,2> o 0 o (8, 64) : (64, 1)>
  return %cl : !fly.composed_layout<SM<2,6,2> o 0 o (8, 64) : (64, 1)>
}

// CHECK-LABEL: @test_composed_get_inner_swizzle_mod
func.func @test_composed_get_inner_swizzle_mod(
    %cl: !fly.composed_layout<SM<2,6,2> o 0 o (8, 64) : (64, 1)>)
    -> !fly.swizzle_mod<SM<2,6,2>> {
  // CHECK: fly.composed_get_inner
  // CHECK-SAME: -> !fly.swizzle_mod<SM<2,6,2>>
  %inner = fly.composed_get_inner(%cl)
      : (!fly.composed_layout<SM<2,6,2> o 0 o (8, 64) : (64, 1)>)
      -> !fly.swizzle_mod<SM<2,6,2>>
  return %inner : !fly.swizzle_mod<SM<2,6,2>>
}
