// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --split-input-file --verify-diagnostics

// expected-error@+3 {{CQMtxLoadn elemBits must be 8 or 16, got 32}}
// expected-error@+2 {{failed to parse Fly_CopyAtom parameter 'copyOp'}}
func.func @reject_32_bit_element(
    %atom: !fly.copy_atom<!fly_ixdl.cq.mtx_loadn<loadn16, row, 32>, 32>) {
  return
}

// -----

// expected-error@+3 {{CQMtxLoadn elemBits must be 8 or 16, got 4}}
// expected-error@+2 {{failed to parse Fly_CopyAtom parameter 'copyOp'}}
func.func @reject_4_bit_element(
    %atom: !fly.copy_atom<!fly_ixdl.cq.mtx_loadn<loadn16, row, 4>, 4>) {
  return
}

// -----

// loadn4 is G2S-only; S2R mtx_loadn accepts only loadn16 / loadn64.
// expected-error@+4 {{expected one of [loadn16, loadn64] for CQ SMEX matrix-load pattern, got: loadn4}}
// expected-error@+3 {{failed to parse FlyIXDL_CopyOpCQMtxLoadn parameter 'pattern' which is to be a `MtxLoadPattern`}}
// expected-error@+2 {{failed to parse Fly_CopyAtom parameter 'copyOp' which is to be a `Type`}}
func.func @reject_loadn4_pattern(
    %atom: !fly.copy_atom<!fly_ixdl.cq.mtx_loadn<loadn4, row, 16>, 16>) {
  return
}

// -----

// expected-error@+4 {{expected one of [loadn16, loadn64] for CQ SMEX matrix-load pattern, got: invalid}}
// expected-error@+3 {{failed to parse FlyIXDL_CopyOpCQMtxLoadn parameter 'pattern' which is to be a `MtxLoadPattern`}}
// expected-error@+2 {{failed to parse Fly_CopyAtom parameter 'copyOp' which is to be a `Type`}}
func.func @reject_bad_pattern(
    %atom: !fly.copy_atom<!fly_ixdl.cq.mtx_loadn<invalid, row, 16>, 16>) {
  return
}

// -----

// expected-error@+4 {{expected one of [row, col] for CQ matrix-load gather direction, got: invalid}}
// expected-error@+3 {{failed to parse FlyIXDL_CopyOpCQMtxLoadn parameter 'direction' which is to be a `MtxGatherDirection`}}
// expected-error@+2 {{failed to parse Fly_CopyAtom parameter 'copyOp' which is to be a `Type`}}
func.func @reject_bad_direction(
    %atom: !fly.copy_atom<!fly_ixdl.cq.mtx_loadn<loadn16, invalid, 16>, 16>) {
  return
}
