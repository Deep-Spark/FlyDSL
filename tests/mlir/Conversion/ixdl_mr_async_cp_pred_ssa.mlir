// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-convert-atom-call-to-ssa-form --fly-promote-regmem-to-vectorssa | FileCheck %s --check-prefix=SSA
// RUN: %fly-opt %s --fly-convert-atom-call-to-ssa-form --fly-promote-regmem-to-vectorssa --convert-fly-to-ixdl --cse | FileCheck %s --check-prefix=LOWERED

// Kernel pipeline for predicated MR G2S: src/dst are not register, only pred
// is. convert-atom-call-to-ssa-form must still select the op so pred becomes
// i1; otherwise promote-regmem leaves a register pointer behind.

// SSA-LABEL: gpu.func @g2s_pred_only
// SSA-NOT: register
// SSA: %[[POISON:.*]] = ub.poison : vector<1xi1>
// SSA: %[[STATE:.*]] = vector.insert %arg2, %[[POISON]] [0] : i1 into vector<1xi1>
// SSA: %[[PRED:.*]] = vector.extract %[[STATE]][0] : i1 from vector<1xi1>
// SSA: fly.copy_atom_call_ssa(%{{.*}}, %{{.*}}, %{{.*}}, %[[PRED]]) {operandSegmentSizes = array<i32: 1, 1, 1, 1>}
// SSA-SAME: i1) -> ()

// LOWERED-LABEL: gpu.func @g2s_pred_only
// LOWERED: %[[BAD:.*]] = arith.constant 16777215 : i32
// LOWERED: %[[SEL:.*]] = arith.select %{{.*}}, %{{.*}}, %[[BAD]] : i32
// LOWERED: ixdl.cp_async.16x16.b32.row %[[SEL]],
gpu.module @promote_pred_only {
  gpu.func @g2s_pred_only(%src: !fly.ptr<f32, #fly_ixdl.sme_gmem>,
                          %dst: !fly.ptr<f32, shared>,
                          %p: i1) kernel {
    %shape1 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %stride1 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %lay1 = fly.make_layout(%shape1, %stride1)
        : (!fly.int_tuple<1>, !fly.int_tuple<1>) -> !fly.layout<1:1>

    %src_view = fly.make_view(%src, %lay1)
        : (!fly.ptr<f32, #fly_ixdl.sme_gmem>, !fly.layout<1:1>)
       -> !fly.memref<f32, #fly_ixdl.sme_gmem, 1:1>
    %dst_view = fly.make_view(%dst, %lay1)
        : (!fly.ptr<f32, shared>, !fly.layout<1:1>) -> !fly.memref<f32, shared, 1:1>

    %pred_ptr = fly.make_ptr() {dictAttrs = {allocSize = 1 : i64}} : () -> !fly.ptr<i1, register>
    fly.ptr.store(%p, %pred_ptr) : (i1, !fly.ptr<i1, register>) -> ()
    %pred_view = fly.make_view(%pred_ptr, %lay1)
        : (!fly.ptr<i1, register>, !fly.layout<1:1>) -> !fly.memref<i1, register, 1:1>

    %atom = fly.make_copy_atom {valBits = 32 : i32}
        : !fly.copy_atom<!fly_ixdl.mr.async_copy<swizzle = 0>, 32>
    fly.copy_atom_call(%atom, %src_view, %dst_view, %pred_view)
        : (!fly.copy_atom<!fly_ixdl.mr.async_copy<swizzle = 0>, 32>,
           !fly.memref<f32, #fly_ixdl.sme_gmem, 1:1>,
           !fly.memref<f32, shared, 1:1>,
           !fly.memref<i1, register, 1:1>) -> ()
    gpu.return
  }
}
