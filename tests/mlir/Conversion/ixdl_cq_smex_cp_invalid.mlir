// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --split-input-file --verify-diagnostics

// expected-error@+3 {{unsupported CQSmexCp rows = 8 (expected 4, 16, or 64)}}
// expected-error@+2 {{failed to parse Fly_CopyAtom parameter 'copyOp'}}
func.func @reject_bad_rows(
    %atom: !fly.copy_atom<!fly_ixdl.cq.smex_cp<8, layout = mtx>, 8>) {
  return
}

// -----

// expected-error@+3 {{unsupported CQSmexCp rows = 32 (expected 4, 16, or 64)}}
// expected-error@+2 {{failed to parse Fly_CopyAtom parameter 'copyOp'}}
func.func @reject_rows_32(
    %atom: !fly.copy_atom<!fly_ixdl.cq.smex_cp<32, layout = mtx>, 8>) {
  return
}

// -----

// expected-error@+4 {{expected one of [plain, mtx] for CQ SMEX shared-memory layout, got: invalid}}
// expected-error@+3 {{failed to parse FlyIXDL_CopyOpCQSmexCp parameter 'layout' which is to be a `SmexLayout`}}
// expected-error@+2 {{failed to parse Fly_CopyAtom parameter 'copyOp' which is to be a `Type`}}
func.func @reject_bad_layout(
    %atom: !fly.copy_atom<!fly_ixdl.cq.smex_cp<16, layout = invalid>, 8>) {
  return
}
