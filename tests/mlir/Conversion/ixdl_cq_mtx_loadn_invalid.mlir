// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --split-input-file --verify-diagnostics

// expected-error@+3 {{unsupported pattern = 2 for CQMtxLoadn (expected 0=loadn16 or 1=loadn64)}}
// expected-error@+2 {{failed to parse Fly_CopyAtom parameter 'copyOp'}}
func.func @reject_bad_pattern(
    %atom: !fly.copy_atom<!fly_ixdl.cq.mtx_loadn<pattern = 2, dir = 0, b = 16, x2 = 1>, 16>) {
  return
}

// -----

// expected-error@+3 {{unsupported b = 32 for CQMtxLoadn (expected 8 or 16)}}
// expected-error@+2 {{failed to parse Fly_CopyAtom parameter 'copyOp'}}
func.func @reject_bad_bitwidth(
    %atom: !fly.copy_atom<!fly_ixdl.cq.mtx_loadn<pattern = 0, dir = 0, b = 32, x2 = 1>, 32>) {
  return
}
