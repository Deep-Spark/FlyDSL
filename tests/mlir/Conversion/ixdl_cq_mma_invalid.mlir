// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --split-input-file --verify-diagnostics

// expected-error@+3 {{CQ s8 MMA requires i32/si32 accumulator, got 'ui32'}}
// expected-error@+2 {{failed to parse Fly_MmaAtom parameter 'mmaOp'}}
func.func @reject_s8_u32_accumulator(
    %atom: !fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (i8, i8) -> ui32>>) {
  return
}

// -----

// expected-error@+3 {{CQ u8 MMA requires i32/ui32 accumulator, got 'si32'}}
// expected-error@+2 {{failed to parse Fly_MmaAtom parameter 'mmaOp'}}
func.func @reject_u8_s32_accumulator(
    %atom: !fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (ui8, ui8) -> si32>>) {
  return
}

// -----

// expected-error@+3 {{CQ int8 MMA requires matching A/B signedness, got 'i8' vs 'ui8'}}
// expected-error@+2 {{failed to parse Fly_MmaAtom parameter 'mmaOp'}}
func.func @reject_mixed_s8_u8(
    %atom: !fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (i8, ui8) -> i32>>) {
  return
}
