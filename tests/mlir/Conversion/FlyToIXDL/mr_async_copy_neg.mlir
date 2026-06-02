// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: not %fly-opt %s --fly-rewrite-func-signature --fly-canonicalize --fly-layout-lowering --convert-fly-to-ixdl 2>&1 | FileCheck %s

// Negative: the async G2S copy requires a Shared destination. A non-shared dst
// must NOT silently fall back — conversion must fail.

// CHECK: failed to legalize operation 'fly.copy_atom_call'
func.func @async_bad_dst(
    %atom: !fly.copy_atom<!fly_ixdl.mr.async_copy.16x64.b8.row<128>, 128>,
    %src: !fly.memref<f16, global, 32:1>,
    %dst: !fly.memref<f16, global, 32:1>) {
  fly.copy_atom_call(%atom, %src, %dst) : (!fly.copy_atom<!fly_ixdl.mr.async_copy.16x64.b8.row<128>, 128>, !fly.memref<f16, global, 32:1>, !fly.memref<f16, global, 32:1>) -> ()
  return
}
