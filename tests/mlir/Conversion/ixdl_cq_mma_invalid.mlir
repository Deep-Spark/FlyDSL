// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --split-input-file --verify-diagnostics

// expected-error@+3 {{CQ MMA requires (M,N) in {(16,16),(32,32),(16,64),(64,16)}, got 8x8}}
// expected-error@+2 {{failed to parse Fly_MmaAtom parameter 'mmaOp'}}
func.func @reject_illegal_mn(
    %atom: !fly.mma_atom<!fly_ixdl.cq.mma<8, 8, 16, (f16, f16) -> f32>>) {
  return
}

// -----

// expected-error@+3 {{CQ MMA multiplicand type must be f16/bf16/i8/ui8/f8E4M3/f8E5M2, got ('f32', 'f32')}}
// expected-error@+2 {{failed to parse Fly_MmaAtom parameter 'mmaOp'}}
func.func @reject_unsupported_multiplicand(
    %atom: !fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 16, (f32, f32) -> f32>>) {
  return
}

// -----

// expected-error@+3 {{CQ f16/bf16 MMA requires matching A/B element types, got 'f16' vs 'bf16'}}
// expected-error@+2 {{failed to parse Fly_MmaAtom parameter 'mmaOp'}}
func.func @reject_f16_bf16_mismatch(
    %atom: !fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 16, (f16, bf16) -> f32>>) {
  return
}

// -----

// expected-error@+3 {{CQ f16/bf16 MMA requires K = 16, got 32}}
// expected-error@+2 {{failed to parse Fly_MmaAtom parameter 'mmaOp'}}
func.func @reject_f16_wrong_k(
    %atom: !fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (f16, f16) -> f32>>) {
  return
}

// -----

// expected-error@+3 {{CQ f16/bf16 MMA requires f32 accumulator, got 'f16'}}
// expected-error@+2 {{failed to parse Fly_MmaAtom parameter 'mmaOp'}}
func.func @reject_f16_wrong_acc(
    %atom: !fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 16, (f16, f16) -> f16>>) {
  return
}

// -----

// expected-error@+3 {{CQ int8 MMA requires matching A/B signedness, got 'i8' vs 'ui8'}}
// expected-error@+2 {{failed to parse Fly_MmaAtom parameter 'mmaOp'}}
func.func @reject_mixed_s8_u8(
    %atom: !fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (i8, ui8) -> i32>>) {
  return
}

// -----

// expected-error@+3 {{CQ int8 MMA requires K = 32, got 16}}
// expected-error@+2 {{failed to parse Fly_MmaAtom parameter 'mmaOp'}}
func.func @reject_s8_wrong_k(
    %atom: !fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 16, (i8, i8) -> i32>>) {
  return
}

// -----

// expected-error@+3 {{CQ s8 MMA requires i32/si32 accumulator, got 'ui32'}}
// expected-error@+2 {{failed to parse Fly_MmaAtom parameter 'mmaOp'}}
func.func @reject_s8_u32_accumulator(
    %atom: !fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (i8, i8) -> ui32>>) {
  return
}

// -----

// expected-error@+3 {{CQ s8 MMA requires i32/si32 accumulator, got 'f32'}}
// expected-error@+2 {{failed to parse Fly_MmaAtom parameter 'mmaOp'}}
func.func @reject_s8_f32_accumulator(
    %atom: !fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (i8, i8) -> f32>>) {
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

// expected-error@+3 {{CQ FP8 MMA requires K = 32, got 16}}
// expected-error@+2 {{failed to parse Fly_MmaAtom parameter 'mmaOp'}}
func.func @reject_fp8_wrong_k(
    %atom: !fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 16, (f8E4M3, f8E4M3) -> f32>>) {
  return
}

// -----

// expected-error@+3 {{CQ FP8 MMA requires f32 or f16 accumulator, got 'i32'}}
// expected-error@+2 {{failed to parse Fly_MmaAtom parameter 'mmaOp'}}
func.func @reject_fp8_wrong_acc(
    %atom: !fly.mma_atom<!fly_ixdl.cq.mma<16, 16, 32, (f8E4M3, f8E4M3) -> i32>>) {
  return
}
