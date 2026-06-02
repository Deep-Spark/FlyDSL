// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-canonicalize --fly-layout-lowering | FileCheck %s

// Regression: recast_layout on a *swizzled* ComposedLayout must rescale the
// inner swizzle base together with the outer (it has no smem_ptr_flag
// specialization). This pins the lowering to agree with
// RecastLayoutOp::inferReturnTypes.
//
// A dynamic outer (dynamic stride) prevents the recast from constant-folding in
// canonicalize, so it reaches RecastLayoutOpLowering (the path that previously
// used a DRR pattern preserving the inner swizzle). We observe the swizzle base
// via the crd2idx SSA: bit-level SM<2,9,2> recast 1-bit -> 8-bit (int8) must
// lower to the element-space SM<2,6,2> sequence:
//   yyy_mask = 0x300 (768), shift = 2, zb_mask = 255  (base 6 -- CORRECT)
// NOT the preserved base-9 sequence:
//   yyy_mask = 0x1800 (6144), zb_mask = 2047           (base 9 -- BUG)

// CHECK-LABEL: @recast_dyn_composed_swizzle_mod
// CHECK-DAG: %[[C768:.+]] = arith.constant 768 : i32
// CHECK-DAG: %[[C255:.+]] = arith.constant 255 : i32
// CHECK-DAG: %[[CN256:.+]] = arith.constant -256 : i32
// CHECK: %[[OFF:.+]] = arith.muli
// CHECK: arith.andi %[[OFF]], %[[C768]] : i32
// CHECK: arith.andi %{{.*}}, %[[C255]] : i32
// CHECK: arith.andi %[[OFF]], %[[CN256]] : i32
// CHECK-NOT: arith.constant 6144 : i32
// CHECK-NOT: arith.constant 2047 : i32
func.func @recast_dyn_composed_swizzle_mod(%c: i32, %dstride: i32) -> !fly.int_tuple<?> {
  %coord = fly.make_coord(%c) : (i32) -> !fly.int_tuple<?>
  %s = fly.static : !fly.swizzle_mod<SM<2,9,2>>
  %off = fly.static : !fly.int_tuple<0>
  %sh = fly.static : !fly.int_tuple<8192>
  %st = fly.make_coord(%dstride) : (i32) -> !fly.int_tuple<?>
  %outer = fly.make_layout(%sh, %st)
      : (!fly.int_tuple<8192>, !fly.int_tuple<?>) -> !fly.layout<8192:?>
  %cl = fly.make_composed_layout(%s, %off, %outer)
      : (!fly.swizzle_mod<SM<2,9,2>>, !fly.int_tuple<0>, !fly.layout<8192:?>)
      -> !fly.composed_layout<SM<2,9,2> o 0 o 8192:?>
  %rc = fly.recast_layout(%cl) {new_type_bits = 8 : i32, old_type_bits = 1 : i32}
      : (!fly.composed_layout<SM<2,9,2> o 0 o 8192:?>)
      -> !fly.composed_layout<SM<2,6,2> o 0 o 8192:?>
  %idx = fly.crd2idx(%coord, %rc)
      : (!fly.int_tuple<?>, !fly.composed_layout<SM<2,6,2> o 0 o 8192:?>) -> !fly.int_tuple<?>
  return %idx : !fly.int_tuple<?>
}
