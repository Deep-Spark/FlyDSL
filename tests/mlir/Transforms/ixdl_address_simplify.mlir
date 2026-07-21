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
gpu.module @t [#ixdl.target] {
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
// B8 Row8b S2R address peeps — pre-commit baseline (current pass leaves these
// unfolded). Follow-up flips CHECKs when Euclidean / ModSwizzle / readfirstlane
// closed forms land.
//
// Intended folds (not yet applied):
//   1) 4*(lane%16)+64*(lane/16) -> 4*lane
//      also A=1: (lane%16)+16*(lane/16) -> lane
//      E in shallow add/sub tree: 4*(lane%16)+1+64*(lane/16) -> 4*lane+1
//                               4*(lane%16)+64*(lane/16)-1 -> 4*lane-1
//                               100-4*(lane%16)-64*(lane/16) -> 100-4*lane
//   2) MS<2,6,2>(x) -> x when x < 256
//   3) MS(base+256) -> ((base+320)&255)|256 when base < 256
//   4) warp_base + 4*lane -> readfirstlane(warp_base) + 4*lane
// -----------------------------------------------------------------------------

// CHECK-LABEL: gpu.func @b8_euclidean_lane_addr
// CHECK:       %[[LANE_IDX:.+]] = gpu.lane_id
// CHECK:       %[[LANE:.+]] = arith.index_cast %[[LANE_IDX]] : index to i32
// CHECK:       %[[REM:.+]] = arith.remsi %[[LANE]], %{{.+}} : i32
// CHECK:       %[[DIV:.+]] = arith.divsi %[[LANE]], %{{.+}} : i32
// CHECK:       %[[LO:.+]] = arith.muli %[[REM]], %{{.+}} : i32
// CHECK:       %[[HI:.+]] = arith.muli %[[DIV]], %{{.+}} : i32
// CHECK:       %[[SUM:.+]] = arith.addi %[[LO]], %[[HI]] : i32
// CHECK:       gpu.printf "%d", %[[SUM]] : i32
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

// A=1 Euclidean identity: (lane%16)+16*(lane/16) -> lane. Baseline keeps rem/div.
//
// CHECK-LABEL: gpu.func @b8_euclidean_identity_a1
// CHECK:       %[[LANE_IDX:.+]] = gpu.lane_id
// CHECK:       %[[LANE:.+]] = arith.index_cast %[[LANE_IDX]] : index to i32
// CHECK:       %[[REM:.+]] = arith.remsi %[[LANE]], %{{.+}} : i32
// CHECK:       %[[DIV:.+]] = arith.divsi %[[LANE]], %{{.+}} : i32
// CHECK:       %[[HI:.+]] = arith.muli %[[DIV]], %{{.+}} : i32
// CHECK:       %[[SUM:.+]] = arith.addi %[[REM]], %[[HI]] : i32
// CHECK:       gpu.printf "%d", %[[SUM]] : i32
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

// Euclidean with additive noise E: 4*(lane%16)+1+64*(lane/16) -> 4*lane+1.
// Baseline keeps rem/div and the mid addi.
//
// CHECK-LABEL: gpu.func @b8_euclidean_with_extra
// CHECK:       %[[LANE_IDX:.+]] = gpu.lane_id
// CHECK:       %[[LANE:.+]] = arith.index_cast %[[LANE_IDX]] : index to i32
// CHECK:       %[[REM:.+]] = arith.remsi %[[LANE]], %{{.+}} : i32
// CHECK:       %[[DIV:.+]] = arith.divsi %[[LANE]], %{{.+}} : i32
// CHECK:       %[[LO:.+]] = arith.muli %[[REM]], %{{.+}} : i32
// CHECK:       %[[MID:.+]] = arith.addi %[[LO]], %{{.+}} : i32
// CHECK:       %[[HI:.+]] = arith.muli %[[DIV]], %{{.+}} : i32
// CHECK:       %[[SUM:.+]] = arith.addi %[[MID]], %[[HI]] : i32
// CHECK:       gpu.printf "%d", %[[SUM]] : i32
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


// Euclidean with subtractive E: 4*(lane%16)+64*(lane/16)-1 -> 4*lane-1.
// Baseline keeps rem/div and the outer subi.
//
// CHECK-LABEL: gpu.func @b8_euclidean_with_sub
// CHECK:       %[[LANE_IDX:.+]] = gpu.lane_id
// CHECK:       %[[LANE:.+]] = arith.index_cast %[[LANE_IDX]] : index to i32
// CHECK:       %[[REM:.+]] = arith.remsi %[[LANE]], %{{.+}} : i32
// CHECK:       %[[DIV:.+]] = arith.divsi %[[LANE]], %{{.+}} : i32
// CHECK:       %[[LO:.+]] = arith.muli %[[REM]], %{{.+}} : i32
// CHECK:       %[[HI:.+]] = arith.muli %[[DIV]], %{{.+}} : i32
// CHECK:       %[[SUM:.+]] = arith.addi %[[LO]], %[[HI]] : i32
// CHECK:       %[[OUT:.+]] = arith.subi %[[SUM]], %{{.+}} : i32
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

// Euclidean neg-pair: 100 - 4*(lane%16) - 64*(lane/16) -> 100 - 4*lane.
// Baseline keeps rem/div and the subi chain.
//
// CHECK-LABEL: gpu.func @b8_euclidean_neg_pair
// CHECK:       %[[LANE_IDX:.+]] = gpu.lane_id
// CHECK:       %[[LANE:.+]] = arith.index_cast %[[LANE_IDX]] : index to i32
// CHECK:       %[[REM:.+]] = arith.remsi %[[LANE]], %{{.+}} : i32
// CHECK:       %[[DIV:.+]] = arith.divsi %[[LANE]], %{{.+}} : i32
// CHECK:       %[[LO:.+]] = arith.muli %[[REM]], %{{.+}} : i32
// CHECK:       %[[HI:.+]] = arith.muli %[[DIV]], %{{.+}} : i32
// CHECK:       %[[T:.+]] = arith.subi %{{.+}}, %[[LO]] : i32
// CHECK:       %[[OUT:.+]] = arith.subi %[[T]], %[[HI]] : i32
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

// B8 ModSwizzle identity (MS<2,6,2> / yyyMask=768, zbMask=255, shift=2).
// Expanded form:
//   yyy  = x & 768; shr = yyy >> 2
//   low  = (x + shr) & 255; high = x & (~255)
//   MS(x) = high | low
// When x < 256, MS(x) == x; baseline still emits the full chain.
//
// CHECK-LABEL: gpu.func @b8_modswizzle_identity
// CHECK:       %[[LANE_IDX:.+]] = gpu.lane_id
// CHECK:       %[[LANE:.+]] = arith.index_cast %[[LANE_IDX]] : index to i32
// CHECK:       %[[YYY:.+]] = arith.andi %[[LANE]], %{{.+}} : i32
// CHECK:       %[[SHR:.+]] = arith.shrui %[[YYY]], %{{.+}} : i32
// CHECK:       %[[SUM:.+]] = arith.addi %[[LANE]], %[[SHR]] : i32
// CHECK:       %[[LOW:.+]] = arith.andi %[[SUM]], %{{.+}} : i32
// CHECK:       %[[HIGH:.+]] = arith.andi %[[LANE]], %{{.+}} : i32
// CHECK:       %[[MS:.+]] = arith.ori %[[HIGH]], %[[LOW]] : i32
// CHECK:       gpu.printf "%d", %[[MS]] : i32
gpu.module @b8_ms_id [#ixdl.target] {
  gpu.func @b8_modswizzle_identity() kernel {
    %lane_idx = gpu.lane_id
    %lane = arith.index_cast %lane_idx : index to i32
    %c768 = arith.constant 768 : i32
    %c255 = arith.constant 255 : i32
    %cnzb = arith.constant -256 : i32
    %c2 = arith.constant 2 : i32
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

// B8 Row8b second-word ModSwizzle: x = base+256, base < 256.
// Intended: MS(base+256) -> ((base+320)&255)|256. Baseline keeps full MS.
//
// CHECK-LABEL: gpu.func @b8_modswizzle_second_word
// CHECK:       %[[LANE_IDX:.+]] = gpu.lane_id
// CHECK:       %[[LANE:.+]] = arith.index_cast %[[LANE_IDX]] : index to i32
// CHECK:       %[[X:.+]] = arith.addi %[[LANE]], %{{.+}} : i32
// CHECK:       %[[YYY:.+]] = arith.andi %[[X]], %{{.+}} : i32
// CHECK:       %[[SHR:.+]] = arith.shrui %[[YYY]], %{{.+}} : i32
// CHECK:       %[[SUM:.+]] = arith.addi %[[X]], %[[SHR]] : i32
// CHECK:       %[[LOW:.+]] = arith.andi %[[SUM]], %{{.+}} : i32
// CHECK:       %[[HIGH:.+]] = arith.andi %[[X]], %{{.+}} : i32
// CHECK:       %[[MS:.+]] = arith.ori %[[HIGH]], %[[LOW]] : i32
// CHECK:       gpu.printf "%d", %[[MS]] : i32
gpu.module @b8_ms_sw [#ixdl.target] {
  gpu.func @b8_modswizzle_second_word() kernel {
    %lane_idx = gpu.lane_id
    %lane = arith.index_cast %lane_idx : index to i32
    %c256 = arith.constant 256 : i32
    %c768 = arith.constant 768 : i32
    %c255 = arith.constant 255 : i32
    %cnzb = arith.constant -256 : i32
    %c2 = arith.constant 2 : i32
    %x = arith.addi %lane, %c256 : i32
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

// Row S2R: warp_base + 4*lane. Intended: readfirstlane(warp_base)+4*lane.
// Baseline leaves a plain addi (no llvm.bi.readfirstlane).
//
// CHECK-LABEL: gpu.func @b8_row_tid4_readfirstlane
// CHECK:       %[[LANE_IDX:.+]] = gpu.lane_id
// CHECK:       %[[LANE:.+]] = arith.index_cast %[[LANE_IDX]] : index to i32
// CHECK-DAG:   %[[BASE:.+]] = arith.muli %{{.+}}, %{{.+}} : i32
// CHECK-DAG:   %[[LANE4:.+]] = arith.muli %[[LANE]], %{{.+}} : i32
// CHECK:       %[[OUT:.+]] = arith.addi %[[BASE]], %[[LANE4]] : i32
// CHECK-NOT:   llvm.call_intrinsic "llvm.bi.readfirstlane"
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
