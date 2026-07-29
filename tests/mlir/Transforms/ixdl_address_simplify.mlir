// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --fly-ixdl-address-simplify | FileCheck %s

// -----------------------------------------------------------------------------
// Iluvatar shared-memory copy address after swizzle is simplified:
//   (lane_id ^ 1) ^ 33  ->  lane_id ^ 32.
// -----------------------------------------------------------------------------
// CHECK-LABEL: gpu.func @iluvatar_swizzled_addr
// CHECK:       %[[LANE_IDX:.+]] = gpu.lane_id
// CHECK:       %[[LANE:.+]] = arith.index_cast %[[LANE_IDX]] : index to i32
// CHECK:       %[[C32:.+]] = arith.constant 32 : i32
// CHECK:       %[[OPT:.+]] = arith.xori %[[LANE]], %[[C32]] : i32
// CHECK:       %[[OFF:.+]] = fly.make_int_tuple(%[[OPT]]) : (i32) -> !fly.int_tuple<?>
// CHECK:       fly.add_offset(%arg0, %[[OFF]])
gpu.module @t [#ixdl.target<chip = "ivcore11">] {
  gpu.func @iluvatar_swizzled_addr(%src: !fly.ptr<f16, shared>, %dst: !fly.ptr<f16, register>) kernel {
    %lane_idx = gpu.lane_id
    %lane = arith.index_cast %lane_idx : index to i32
    %c1 = arith.constant 1 : i32
    %c33 = arith.constant 33 : i32
    %sw0 = arith.xori %lane, %c1 : i32
    %sw1 = arith.xori %sw0, %c33 : i32
    %off = fly.make_int_tuple(%sw1) : (i32) -> !fly.int_tuple<?>
    %src_offset = fly.add_offset(%src, %off)
        : (!fly.ptr<f16, shared>, !fly.int_tuple<?>) -> !fly.ptr<f16, shared>

    %shape4 = fly.make_int_tuple() : () -> !fly.int_tuple<4>
    %stride1 = fly.make_int_tuple() : () -> !fly.int_tuple<1>
    %layout = fly.make_layout(%shape4, %stride1)
        : (!fly.int_tuple<4>, !fly.int_tuple<1>) -> !fly.layout<4:1>
    %src_view = fly.make_view(%src_offset, %layout)
        : (!fly.ptr<f16, shared>, !fly.layout<4:1>) -> !fly.memref<f16, shared, 4:1>
    %dst_view = fly.make_view(%dst, %layout)
        : (!fly.ptr<f16, register>, !fly.layout<4:1>) -> !fly.memref<f16, register, 4:1>
    %copy = fly.make_copy_atom {valBits = 16 : i32}
        : !fly.copy_atom<!fly.universal_copy<64>, 16>
    fly.copy_atom_call(%copy, %src_view, %dst_view)
        : (!fly.copy_atom<!fly.universal_copy<64>, 16>, !fly.memref<f16, shared, 4:1>, !fly.memref<f16, register, 4:1>) -> ()
    gpu.return
  }
}

// -----------------------------------------------------------------------------
// B8 Row8b: Euclidean thr→byte fold
//   4*(lane%16) + 64*(lane/16)  ->  4*lane
//   Also: A=1 and additive/subtractive noise E inside a shallow add/sub tree:
//     (lane%16) + 16*(lane/16)           -> lane
//     4*(lane%16) + 1 + 64*(lane/16)     -> 4*lane + 1
//     4*(lane%16) + 64*(lane/16) - 1     -> 4*lane - 1
//     100 - 4*(lane%16) - 64*(lane/16)   -> 100 - 4*lane
// -----------------------------------------------------------------------------
// CHECK-LABEL: gpu.func @b8_euclidean_lane_addr
// CHECK:       %[[LANE_IDX:.+]] = gpu.lane_id
// CHECK:       %[[LANE:.+]] = arith.index_cast %[[LANE_IDX]] : index to i32
// CHECK:       %[[OUT:.+]] = arith.muli %[[LANE]], %{{.+}} : i32
// CHECK:       gpu.printf "%d", %[[OUT]] : i32
gpu.module @b8_euclid [#ixdl.target] {
  gpu.func @b8_euclidean_lane_addr() kernel {
    %lane_idx = gpu.lane_id
    %lane = arith.index_cast %lane_idx : index to i32
    %c4 = arith.constant 4 : i32
    %c16 = arith.constant 16 : i32
    %c64 = arith.constant 64 : i32
    %rem = arith.remsi %lane, %c16 : i32
    %div = arith.divsi %lane, %c16 : i32
    %lo = arith.muli %rem, %c4 : i32
    %hi = arith.muli %div, %c64 : i32
    %sum = arith.addi %lo, %hi : i32
    gpu.printf "%d", %sum : i32
    gpu.return
  }
}

// CHECK-LABEL: gpu.func @b8_euclidean_identity_a1
// CHECK:       %[[LANE_IDX:.+]] = gpu.lane_id
// CHECK:       %[[LANE:.+]] = arith.index_cast %[[LANE_IDX]] : index to i32
// CHECK:       gpu.printf "%d", %[[LANE]] : i32
// CHECK-NOT:   arith.remsi
gpu.module @b8_euclid_a1 [#ixdl.target] {
  gpu.func @b8_euclidean_identity_a1() kernel {
    %lane_idx = gpu.lane_id
    %lane = arith.index_cast %lane_idx : index to i32
    %c16 = arith.constant 16 : i32
    %rem = arith.remsi %lane, %c16 : i32
    %div = arith.divsi %lane, %c16 : i32
    %hi = arith.muli %div, %c16 : i32
    %sum = arith.addi %rem, %hi : i32
    gpu.printf "%d", %sum : i32
    gpu.return
  }
}

// CHECK-LABEL: gpu.func @b8_euclidean_with_extra
// CHECK:       %[[LANE_IDX:.+]] = gpu.lane_id
// CHECK:       %[[LANE:.+]] = arith.index_cast %[[LANE_IDX]] : index to i32
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : i32
// CHECK:       %[[AX:.+]] = arith.muli %[[LANE]], %{{.+}} : i32
// CHECK:       %[[OUT:.+]] = arith.addi %[[AX]], %[[C1]] : i32
// CHECK:       gpu.printf "%d", %[[OUT]] : i32
gpu.module @b8_euclid_extra [#ixdl.target] {
  gpu.func @b8_euclidean_with_extra() kernel {
    %lane_idx = gpu.lane_id
    %lane = arith.index_cast %lane_idx : index to i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %c16 = arith.constant 16 : i32
    %c64 = arith.constant 64 : i32
    %rem = arith.remsi %lane, %c16 : i32
    %div = arith.divsi %lane, %c16 : i32
    %lo = arith.muli %rem, %c4 : i32
    %mid = arith.addi %lo, %c1 : i32
    %hi = arith.muli %div, %c64 : i32
    %sum = arith.addi %mid, %hi : i32
    gpu.printf "%d", %sum : i32
    gpu.return
  }
}

// CHECK-LABEL: gpu.func @b8_euclidean_with_sub
// CHECK:       %[[LANE_IDX:.+]] = gpu.lane_id
// CHECK:       %[[LANE:.+]] = arith.index_cast %[[LANE_IDX]] : index to i32
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : i32
// CHECK:       %[[AX:.+]] = arith.muli %[[LANE]], %{{.+}} : i32
// CHECK:       %[[OUT:.+]] = arith.subi %[[AX]], %[[C1]] : i32
// CHECK:       gpu.printf "%d", %[[OUT]] : i32
gpu.module @b8_euclid_sub [#ixdl.target] {
  gpu.func @b8_euclidean_with_sub() kernel {
    %lane_idx = gpu.lane_id
    %lane = arith.index_cast %lane_idx : index to i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %c16 = arith.constant 16 : i32
    %c64 = arith.constant 64 : i32
    %rem = arith.remsi %lane, %c16 : i32
    %div = arith.divsi %lane, %c16 : i32
    %lo = arith.muli %rem, %c4 : i32
    %hi = arith.muli %div, %c64 : i32
    %sum = arith.addi %lo, %hi : i32
    %out = arith.subi %sum, %c1 : i32
    gpu.printf "%d", %out : i32
    gpu.return
  }
}

// CHECK-LABEL: gpu.func @b8_euclidean_neg_pair
// CHECK:       %[[LANE_IDX:.+]] = gpu.lane_id
// CHECK:       %[[LANE:.+]] = arith.index_cast %[[LANE_IDX]] : index to i32
// CHECK-DAG:   %[[C100:.+]] = arith.constant 100 : i32
// CHECK:       %[[AX:.+]] = arith.muli %[[LANE]], %{{.+}} : i32
// CHECK:       %[[OUT:.+]] = arith.subi %[[C100]], %[[AX]] : i32
// CHECK:       gpu.printf "%d", %[[OUT]] : i32
gpu.module @b8_euclid_neg [#ixdl.target] {
  gpu.func @b8_euclidean_neg_pair() kernel {
    %lane_idx = gpu.lane_id
    %lane = arith.index_cast %lane_idx : index to i32
    %c100 = arith.constant 100 : i32
    %c4 = arith.constant 4 : i32
    %c16 = arith.constant 16 : i32
    %c64 = arith.constant 64 : i32
    %rem = arith.remsi %lane, %c16 : i32
    %div = arith.divsi %lane, %c16 : i32
    %lo = arith.muli %rem, %c4 : i32
    %hi = arith.muli %div, %c64 : i32
    %t = arith.subi %c100, %lo : i32
    %out = arith.subi %t, %hi : i32
    gpu.printf "%d", %out : i32
    gpu.return
  }
}

// -----------------------------------------------------------------------------
// B8 ModSwizzle identity (MS<2,6,2> / yyyMask=768, zbMask=255, shift=2).
//
// Expanded form used by Row8b S2R addressing:
//   yyy  = x & 768                  // bits [8,9] of x  (768 = 0b11_0000_0000)
//   shr  = yyy >> 2                 // inject those bits into [6,7]
//   low  = (x + shr) & 255          // zb: low 8 bits after wrap add
//   high = x & (~255)               // nzb: bits above zb (mask = -256)
//   MS(x) = high | low
//
// When x < yyyLo (= 256, lowest set bit of 768), yyy==0 so shr==0 and
// high==0, therefore MS(x) == x.  Here x = lane_id ∈ [0,64).
//
// Before: full MS chain above.  After: printf(lane)  (no ori).
// -----------------------------------------------------------------------------
// CHECK-LABEL: gpu.func @b8_modswizzle_identity
// CHECK:       %[[LANE_IDX:.+]] = gpu.lane_id
// CHECK:       %[[LANE:.+]] = arith.index_cast %[[LANE_IDX]] : index to i32
// CHECK:       gpu.printf "%d", %[[LANE]] : i32
// CHECK-NOT:   arith.ori
gpu.module @b8_ms_id [#ixdl.target] {
  gpu.func @b8_modswizzle_identity() kernel {
    %lane_idx = gpu.lane_id
    %lane = arith.index_cast %lane_idx : index to i32
    // MS<2,6,2> with x = lane < 256 → identity.
    %c768 = arith.constant 768 : i32   // yyyMask
    %c255 = arith.constant 255 : i32   // zbMask  (low 8 bits)
    %cnzb = arith.constant -256 : i32  // ~zbMask (high bits)
    %c2 = arith.constant 2 : i32       // shift
    %yyy = arith.andi %lane, %c768 : i32
    %shr = arith.shrui %yyy, %c2 : i32
    %sum = arith.addi %lane, %shr : i32
    %low = arith.andi %sum, %c255 : i32
    %high = arith.andi %lane, %cnzb : i32
    %ms = arith.ori %high, %low : i32
    gpu.printf "%d", %ms : i32
    gpu.return
  }
}

// -----------------------------------------------------------------------------
// B8 Row8b second-word ModSwizzle closed form (same MS<2,6,2>).
//
// Second 256B word of a 512B row tile: x = base + 256 with base < 256
// (here base = lane_id).  Full expansion:
//   x    = base + 256
//   yyy  = x & 768                  // for base∈[0,256): always 256
//   shr  = yyy >> 2                 // = 64
//   low  = (x + shr) & 255          // = (base + 256 + 64) & 255 = (base+320)&255
//   high = x & (~255)               // = 256  (second-word bit)
//   MS(x) = high | low
//
// Closed form (pass folds the chain away):
//   MS(base+256) -> ((base + 320) & 255) | 256
//
// Before: x=base+256 + yyy/shr/sum/low/high/ori.
// After:  sum=base+320; low=sum&255; out=low|256.
// -----------------------------------------------------------------------------
// CHECK-LABEL: gpu.func @b8_modswizzle_second_word
// CHECK:       %[[LANE_IDX:.+]] = gpu.lane_id
// CHECK:       %[[LANE:.+]] = arith.index_cast %[[LANE_IDX]] : index to i32
// CHECK-DAG:   arith.constant 320 : i32
// CHECK-DAG:   arith.constant 256 : i32
// CHECK:       %[[SUM:.+]] = arith.addi %[[LANE]], %{{.+}} : i32
// CHECK:       %[[LOW:.+]] = arith.andi %[[SUM]], %{{.+}} : i32
// CHECK:       %[[OUT:.+]] = arith.ori %[[LOW]], %{{.+}} : i32
// CHECK:       gpu.printf "%d", %[[OUT]] : i32
gpu.module @b8_ms_sw [#ixdl.target] {
  gpu.func @b8_modswizzle_second_word() kernel {
    %lane_idx = gpu.lane_id
    %lane = arith.index_cast %lane_idx : index to i32  // base ∈ [0,64) < 256
    %c256 = arith.constant 256 : i32   // second-word offset
    %c768 = arith.constant 768 : i32   // yyyMask
    %c255 = arith.constant 255 : i32   // zbMask
    %cnzb = arith.constant -256 : i32  // ~zbMask
    %c2 = arith.constant 2 : i32       // shift
    // x = base + 256  (second 256B word)
    %x = arith.addi %lane, %c256 : i32
    // MS(x) = (x & ~255) | (((x + ((x & 768) >> 2)) & 255)
    %yyy = arith.andi %x, %c768 : i32
    %shr = arith.shrui %yyy, %c2 : i32
    %sum = arith.addi %x, %shr : i32
    %low = arith.andi %sum, %c255 : i32
    %high = arith.andi %x, %cnzb : i32
    %ms = arith.ori %high, %low : i32
    gpu.printf "%d", %ms : i32
    gpu.return
  }
}

// -----------------------------------------------------------------------------
// Row S2R: warp_base + 4*lane → readfirstlane(warp_base) + 4*lane
// -----------------------------------------------------------------------------
// CHECK-LABEL: gpu.func @b8_row_tid4_readfirstlane
// CHECK:       %[[LANE_IDX:.+]] = gpu.lane_id
// CHECK:       %[[LANE:.+]] = arith.index_cast %[[LANE_IDX]] : index to i32
// CHECK-DAG:   %[[BASE:.+]] = arith.muli %{{.+}}, %{{.+}} : i32
// CHECK-DAG:   %[[LANE4:.+]] = arith.muli %[[LANE]], %{{.+}} : i32
// CHECK:       %[[UNI:.+]] = llvm.call_intrinsic "llvm.bi.readfirstlane"(%[[BASE]])
// CHECK:       %[[OUT:.+]] = arith.addi %[[UNI]], %[[LANE4]] : i32
// CHECK:       gpu.printf "%d", %[[OUT]] : i32
gpu.module @b8_rfl [#ixdl.target] {
  gpu.func @b8_row_tid4_readfirstlane() kernel {
    %lane_idx = gpu.lane_id
    %lane = arith.index_cast %lane_idx : index to i32
    %wid_idx = gpu.thread_id x
    %wid = arith.index_cast %wid_idx : index to i32
    %c6 = arith.constant 6 : i32
    %c4096 = arith.constant 4096 : i32
    %c4 = arith.constant 4 : i32
    %warp = arith.shrui %wid, %c6 : i32
    %base = arith.muli %warp, %c4096 : i32
    %lane4 = arith.muli %lane, %c4 : i32
    %addr = arith.addi %base, %lane4 : i32
    gpu.printf "%d", %addr : i32
    gpu.return
  }
}
