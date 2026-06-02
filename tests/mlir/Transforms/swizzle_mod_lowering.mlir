// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-canonicalize --fly-layout-lowering | FileCheck %s

// The dynamic SwizzleMod path lowers crd2idx to the exact modular-add SSA
// sequence for cute::Swizzle_Mod<2,6,2>:
//   yyy_mask = 0x300 (768), shift = 2, zb_mask = 255, ~zb_mask = -256.

// CHECK-LABEL: @sm_crd_dynamic
func.func @sm_crd_dynamic(%c: i32) -> !fly.int_tuple<?> {
  %coord = fly.make_coord(%c) : (i32) -> !fly.int_tuple<?>
  %s = fly.static : !fly.swizzle_mod<SM<2,6,2>>
  %off = fly.static : !fly.int_tuple<0>
  %sh = fly.static : !fly.int_tuple<512>
  %st = fly.static : !fly.int_tuple<1>
  %outer = fly.make_layout(%sh, %st)
      : (!fly.int_tuple<512>, !fly.int_tuple<1>) -> !fly.layout<512:1>
  %cl = fly.make_composed_layout(%s, %off, %outer)
      : (!fly.swizzle_mod<SM<2,6,2>>, !fly.int_tuple<0>, !fly.layout<512:1>)
      -> !fly.composed_layout<SM<2,6,2> o 0 o 512:1>
  // CHECK-DAG: %[[C768:.+]] = arith.constant 768 : i32
  // CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i32
  // CHECK-DAG: %[[C255:.+]] = arith.constant 255 : i32
  // CHECK-DAG: %[[CN256:.+]] = arith.constant -256 : i32
  // CHECK: %[[YYY:.+]] = arith.andi %arg0, %[[C768]] : i32
  // CHECK: %[[SH:.+]] = arith.shrui %[[YYY]], %[[C2]] : i32
  // CHECK: %[[ADD:.+]] = arith.addi %arg0, %[[SH]] : i32
  // CHECK: %[[LOW:.+]] = arith.andi %[[ADD]], %[[C255]] : i32
  // CHECK: %[[HIGH:.+]] = arith.andi %arg0, %[[CN256]] : i32
  // CHECK: arith.ori %[[HIGH]], %[[LOW]] : i32
  %idx = fly.crd2idx(%coord, %cl)
      : (!fly.int_tuple<?>, !fly.composed_layout<SM<2,6,2> o 0 o 512:1>) -> !fly.int_tuple<?>
  return %idx : !fly.int_tuple<?>
}
