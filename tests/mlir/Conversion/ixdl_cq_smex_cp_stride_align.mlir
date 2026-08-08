// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-ixdl --verify-diagnostics

// CQ SMEX G2S: constant global row stride (stride_byte) must be 16B-aligned.
// i8 leading_stride=24 -> stride_byte=24, not a multiple of 16.
func.func @reject_misaligned_smex_stride(
    %base: !fly.ptr<i8, global>,
    %dst: !fly.memref<i8, shared, 1:1>) {
  %c24 = arith.constant 24 : i32
  %p = fly.make_ptr(%base, %c24)
      : (!fly.ptr<i8, global>, i32) -> !fly.ptr<i8, #fly_ixdl.sme_gmem>
  %layout = fly.static : !fly.layout<1:1>
  %src = fly.make_view(%p, %layout)
      : (!fly.ptr<i8, #fly_ixdl.sme_gmem>, !fly.layout<1:1>)
      -> !fly.memref<i8, #fly_ixdl.sme_gmem, 1:1>
  %atom = fly.make_copy_atom {valBits = 8 : i32}
      : !fly.copy_atom<!fly_ixdl.cq.smex_cp<16, layout = mtx>, 8>
  // expected-error@+2 {{CQ SMEX global row stride must be 16B-aligned}}
  // expected-error@+1 {{failed to legalize operation 'fly.copy_atom_call'}}
  fly.copy_atom_call(%atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.cq.smex_cp<16, layout = mtx>, 8>,
         !fly.memref<i8, #fly_ixdl.sme_gmem, 1:1>,
         !fly.memref<i8, shared, 1:1>) -> ()
  return
}
