// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-int-swizzle-simplify | FileCheck %s

// -----------------------------------------------------------------------------
// Iluvatar shared-memory copy address after swizzle currently keeps the original
// lane expression. A later IXDL optimization should fold:
//   (lane_id ^ 1) ^ 33 -> lane_id ^ 32.
// -----------------------------------------------------------------------------
// CHECK-LABEL: gpu.func @iluvatar_swizzled_addr
// CHECK:       %[[LANE_IDX:.+]] = gpu.lane_id
// CHECK:       %[[LANE:.+]] = arith.index_cast %[[LANE_IDX]] : index to i32
// CHECK:       %[[C1:.+]] = arith.constant 1 : i32
// CHECK:       %[[C33:.+]] = arith.constant 33 : i32
// CHECK:       %[[SW0:.+]] = arith.xori %[[LANE]], %[[C1]] : i32
// CHECK:       %[[SW1:.+]] = arith.xori %[[SW0]], %[[C33]] : i32
// CHECK:       %[[OFF:.+]] = fly.make_int_tuple(%[[SW1]]) : (i32) -> !fly.int_tuple<?>
// CHECK:       fly.add_offset(%arg0, %[[OFF]])
gpu.module @t [#ixdl.target] {
  gpu.func @iluvatar_swizzled_addr(%src: !fly.ptr<f16, shared>, %dst: !fly.ptr<f16, register>) kernel {
    %lane_idx = gpu.lane_id
    %lane = arith.index_cast %lane_idx : index to i32
    %c1 = arith.constant 1 : i32
    %c33 = arith.constant 33 : i32
    %sw0 = arith.xori %lane, %c1 : i32
    %sw1 = arith.xori %sw0, %c33 : i32
    %off = fly.make_int_tuple(%sw1) : (i32) -> !fly.int_tuple<?>
    %src_offset = fly.add_offset(%src, %off)
        : (!fly.ptr<f16, shared>, !fly.int_tuple<?>) -> !fly.ptr<f16, shared>

    %shape4 = fly.make_int_tuple() : () -> !fly.int_tuple<4>
    %stride1 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %layout = fly.make_layout(%shape4, %stride1)
        : (!fly.int_tuple<4>, !fly.int_tuple<1>) -> !fly.layout<4:1>
    %src_view = fly.make_view(%src_offset, %layout)
        : (!fly.ptr<f16, shared>, !fly.layout<4:1>) -> !fly.memref<f16, shared, 4:1>
    %dst_view = fly.make_view(%dst, %layout)
        : (!fly.ptr<f16, register>, !fly.layout<4:1>) -> !fly.memref<f16, register, 4:1>
    %copy = fly.make_copy_atom {valBits = 16 : i32}
        : !fly.copy_atom<!fly.universal_copy<64>, 16>
    fly.copy_atom_call(%copy, %src_view, %dst_view)
        : (!fly.copy_atom<!fly.universal_copy<64>, 16>, !fly.memref<f16, shared, 4:1>, !fly.memref<f16, register, 4:1>) -> ()
    gpu.return
  }
}
