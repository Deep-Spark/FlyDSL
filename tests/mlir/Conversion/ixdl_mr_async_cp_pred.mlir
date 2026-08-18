// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-ixdl --cse | FileCheck %s --check-prefix=CPASYNC

// Predicated MR G2S, value form: src is sme_gmem, dst is shared, pred is a
// register memref. A false predicate selects sOffset = 0xffffff so the
// hardware drops the transfer (global read included). The atom stays in
// straight-line code.

// CPASYNC-LABEL: @test_mr_async_cp_predicated
// CPASYNC: %[[PRED:.*]] = llvm.load %{{.*}} : !llvm.ptr<5> -> i1
// CPASYNC: %[[SOFF:.*]] = llvm.ptrtoint %{{.*}} : !llvm.ptr<3> to i32
// CPASYNC: %[[BAD:.*]] = arith.constant 16777215 : i32
// CPASYNC: %[[SEL:.*]] = arith.select %[[PRED]], %[[SOFF]], %[[BAD]] : i32
// CPASYNC: ixdl.cp_async.16x16.b32.row %[[SEL]],
func.func @test_mr_async_cp_predicated(
    %src: !fly.memref<f32, #fly_ixdl.sme_gmem, 1:1>,
    %dst: !fly.memref<f32, shared, 1:1>,
    %pred: !fly.memref<i1, register, 1:1>) {
  %atom = fly.make_copy_atom {valBits = 32 : i32}
      : !fly.copy_atom<!fly_ixdl.mr.async_copy<swizzle = 0>, 32>
  fly.copy_atom_call(%atom, %src, %dst, %pred)
      : (!fly.copy_atom<!fly_ixdl.mr.async_copy<swizzle = 0>, 32>,
         !fly.memref<f32, #fly_ixdl.sme_gmem, 1:1>,
         !fly.memref<f32, shared, 1:1>,
         !fly.memref<i1, register, 1:1>) -> ()
  return
}
