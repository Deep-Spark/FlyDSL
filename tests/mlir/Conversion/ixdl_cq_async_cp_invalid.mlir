// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --split-input-file --verify-diagnostics

// expected-error@+3 {{unsupported enhanced-SME shape (8, 16), transpose = 0 for CQAsyncCp}}
// expected-error@+2 {{failed to parse Fly_CopyAtom parameter 'copyOp'}}
func.func @reject_mr_only_8x16_b32(
    %atom: !fly.copy_atom<!fly_ixdl.cq.async_copy<8, 16, transpose = 0>, 32>) {
  return
}

// -----

// expected-error@+3 {{unsupported enhanced-SME shape (1, 128), transpose = 0 for CQAsyncCp}}
// expected-error@+2 {{failed to parse Fly_CopyAtom parameter 'copyOp'}}
func.func @reject_mr_only_1x8b64(
    %atom: !fly.copy_atom<!fly_ixdl.cq.async_copy<1, 128, transpose = 0>, 32>) {
  return
}
