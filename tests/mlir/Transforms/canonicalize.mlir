// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-canonicalize | FileCheck %s

// Tests for fly-canonicalize constructor normalization.

// -----

// CHECK-LABEL: @test_make_shape
func.func @test_make_shape(%m: i32, %n: i32) -> !fly.int_tuple<(?, ?)> {
  // CHECK: fly.make_int_tuple(%{{.*}}, %{{.*}}) : (i32, i32) -> !fly.int_tuple<(?,?)>
  // CHECK-NOT: fly.make_shape
  %0 = fly.make_shape(%m, %n) : (i32, i32) -> !fly.int_tuple<(?, ?)>
  return %0 : !fly.int_tuple<(?, ?)>
}

// CHECK-LABEL: @test_make_stride
func.func @test_make_stride(%s0: i32, %s1: i32) -> !fly.int_tuple<(?, ?)> {
  // CHECK: fly.make_int_tuple(%{{.*}}, %{{.*}}) : (i32, i32) -> !fly.int_tuple<(?,?)>
  // CHECK-NOT: fly.make_stride
  %0 = fly.make_stride(%s0, %s1) : (i32, i32) -> !fly.int_tuple<(?, ?)>
  return %0 : !fly.int_tuple<(?, ?)>
}

// CHECK-LABEL: @test_make_coord
func.func @test_make_coord(%c0: i32, %c1: i32) -> !fly.int_tuple<(?, ?)> {
  // CHECK: fly.make_int_tuple(%{{.*}}, %{{.*}}) : (i32, i32) -> !fly.int_tuple<(?,?)>
  // CHECK-NOT: fly.make_coord
  %0 = fly.make_coord(%c0, %c1) : (i32, i32) -> !fly.int_tuple<(?, ?)>
  return %0 : !fly.int_tuple<(?, ?)>
}
