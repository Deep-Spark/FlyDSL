// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-ixdl | FileCheck %s
// RUN: %fly-opt %s | FileCheck %s --check-prefix=ROUNDTRIP

// CQSmexCp lowers to ixdl.cp_async.smex.{mtx,plain}[.pred].{4,16,64}x1b64.
// State: (row_mask:i64, col_mask:i32, pred:i1, pred_enabled:i8); masks default
// all-1s; pred disabled until set_value(pred=...).

// ROUNDTRIP-LABEL: @test_cq_smex_cp_type
// ROUNDTRIP-SAME: !fly_ixdl.cq.smex_cp<16, layout = mtx>
func.func @test_cq_smex_cp_type(
    %atom: !fly.copy_atom<!fly_ixdl.cq.smex_cp<16, layout = mtx>, 8>) {
  return
}

// CHECK-LABEL: @test_cq_smex_cp_16_default_masks
// CHECK-DAG: %[[UNDEF:.*]] = llvm.mlir.undef : !llvm.struct<(i64, i32, i1, i8)>
// CHECK-DAG: %[[RM_M1:.*]] = arith.constant -1 : i64
// CHECK-DAG: %[[CM_M1:.*]] = arith.constant -1 : i32
// CHECK-DAG: %[[PRED1:.*]] = arith.constant true
// CHECK-DAG: %[[EN0:.*]] = arith.constant 0 : i8
// CHECK: %[[S0:.*]] = llvm.insertvalue %[[RM_M1]], %[[UNDEF]][0]
// CHECK: %[[S1:.*]] = llvm.insertvalue %[[CM_M1]], %[[S0]][1]
// CHECK: %[[S2:.*]] = llvm.insertvalue %[[PRED1]], %[[S1]][2]
// CHECK: %[[ATOM:.*]] = llvm.insertvalue %[[EN0]], %[[S2]][3]
// CHECK: %[[RM:.*]] = llvm.extractvalue %[[ATOM]][0]
// CHECK: %[[CM:.*]] = llvm.extractvalue %[[ATOM]][1]
// CHECK: %[[KOP:.*]] = arith.constant 1 : i32
// CHECK: %[[PF:.*]] = arith.constant 1 : i32
// CHECK: ixdl.cp_async.smex.mtx.16x1b64 %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[RM]], %[[CM]], %[[KOP]], %[[PF]]
func.func @test_cq_smex_cp_16_default_masks(
    %src: !fly.memref<i8, #fly_ixdl.sme_gmem, 1:1>,
    %dst: !fly.memref<i8, shared, 1:1>) {
  %atom = fly.make_copy_atom {valBits = 8 : i32}
      : !fly.copy_atom<!fly_ixdl.cq.smex_cp<16, layout = mtx>, 8>
  fly.copy_atom_call(%atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.cq.smex_cp<16, layout = mtx>, 8>,
         !fly.memref<i8, #fly_ixdl.sme_gmem, 1:1>,
         !fly.memref<i8, shared, 1:1>) -> ()
  return
}

// CHECK-LABEL: @test_cq_smex_cp_16_set_masks
// CHECK-SAME: (%[[ATOM:.*]]: !llvm.struct<(i64, i32, i1, i8)>, %[[RM:.*]]: i64, %[[CM:.*]]: i32,
func.func @test_cq_smex_cp_16_set_masks(
    %atom: !fly.copy_atom<!fly_ixdl.cq.smex_cp<16, layout = mtx>, 16>,
    %rm: i64, %cm: i32,
    %src: !fly.memref<f16, #fly_ixdl.sme_gmem, 1:1>,
    %dst: !fly.memref<f16, shared, 1:1>) {
  // CHECK: %[[A1:.*]] = llvm.insertvalue %[[RM]], %[[ATOM]][0]
  // CHECK: %[[A2:.*]] = llvm.insertvalue %[[CM]], %[[A1]][1]
  // CHECK: %[[RM2:.*]] = llvm.extractvalue %[[A2]][0]
  // CHECK: %[[CM2:.*]] = llvm.extractvalue %[[A2]][1]
  // CHECK: ixdl.cp_async.smex.mtx.16x1b64 %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[RM2]], %[[CM2]]
  %a1 = fly.atom.set_value(%atom, "row_mask", %rm)
      : (!fly.copy_atom<!fly_ixdl.cq.smex_cp<16, layout = mtx>, 16>, i64)
      -> !fly.copy_atom<!fly_ixdl.cq.smex_cp<16, layout = mtx>, 16>
  %a2 = fly.atom.set_value(%a1, "col_mask", %cm)
      : (!fly.copy_atom<!fly_ixdl.cq.smex_cp<16, layout = mtx>, 16>, i32)
      -> !fly.copy_atom<!fly_ixdl.cq.smex_cp<16, layout = mtx>, 16>
  fly.copy_atom_call(%a2, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.cq.smex_cp<16, layout = mtx>, 16>,
         !fly.memref<f16, #fly_ixdl.sme_gmem, 1:1>,
         !fly.memref<f16, shared, 1:1>) -> ()
  return
}

// CHECK-LABEL: @test_cq_smex_cp_64
// CHECK: ixdl.cp_async.smex.mtx.64x1b64
func.func @test_cq_smex_cp_64(
    %src: !fly.memref<i8, #fly_ixdl.sme_gmem, 1:1>,
    %dst: !fly.memref<i8, shared, 1:1>) {
  %atom = fly.make_copy_atom {valBits = 8 : i32}
      : !fly.copy_atom<!fly_ixdl.cq.smex_cp<64, layout = mtx>, 8>
  fly.copy_atom_call(%atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.cq.smex_cp<64, layout = mtx>, 8>,
         !fly.memref<i8, #fly_ixdl.sme_gmem, 1:1>,
         !fly.memref<i8, shared, 1:1>) -> ()
  return
}

// CHECK-LABEL: @test_cq_smex_cp_mtx_4
// CHECK: ixdl.cp_async.smex.mtx.4x1b64
func.func @test_cq_smex_cp_mtx_4(
    %src: !fly.memref<i8, #fly_ixdl.sme_gmem, 1:1>,
    %dst: !fly.memref<i8, shared, 1:1>) {
  %atom = fly.make_copy_atom {valBits = 8 : i32}
      : !fly.copy_atom<!fly_ixdl.cq.smex_cp<4, layout = mtx>, 8>
  fly.copy_atom_call(%atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.cq.smex_cp<4, layout = mtx>, 8>,
         !fly.memref<i8, #fly_ixdl.sme_gmem, 1:1>,
         !fly.memref<i8, shared, 1:1>) -> ()
  return
}

// CHECK-LABEL: @test_cq_smex_cp_plain_16
// CHECK: ixdl.cp_async.smex.plain.16x1b64
func.func @test_cq_smex_cp_plain_16(
    %src: !fly.memref<i8, #fly_ixdl.sme_gmem, 1:1>,
    %dst: !fly.memref<i8, shared, 1:1>) {
  %atom = fly.make_copy_atom {valBits = 8 : i32}
      : !fly.copy_atom<!fly_ixdl.cq.smex_cp<16, layout = plain>, 8>
  fly.copy_atom_call(%atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.cq.smex_cp<16, layout = plain>, 8>,
         !fly.memref<i8, #fly_ixdl.sme_gmem, 1:1>,
         !fly.memref<i8, shared, 1:1>) -> ()
  return
}

// CHECK-LABEL: @test_cq_smex_cp_mtx_16_pred
// CHECK-SAME: (%[[ATOM:.*]]: !llvm.struct<(i64, i32, i1, i8)>, %[[P:.*]]: i1,
// CHECK: %[[A1:.*]] = llvm.insertvalue %[[P]], %[[ATOM]][2]
// CHECK: %[[EN1:.*]] = arith.constant 1 : i8
// CHECK: %[[A2:.*]] = llvm.insertvalue %[[EN1]], %[[A1]][3]
// CHECK: %[[RM:.*]] = llvm.extractvalue %[[A2]][0]
// CHECK: %[[CM:.*]] = llvm.extractvalue %[[A2]][1]
// CHECK: %[[PRED:.*]] = llvm.extractvalue %[[A2]][2]
// CHECK: ixdl.cp_async.smex.mtx.pred.16x1b64 %[[PRED]], %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[RM]], %[[CM]]
func.func @test_cq_smex_cp_mtx_16_pred(
    %atom: !fly.copy_atom<!fly_ixdl.cq.smex_cp<16, layout = mtx>, 8>,
    %p: i1,
    %src: !fly.memref<i8, #fly_ixdl.sme_gmem, 1:1>,
    %dst: !fly.memref<i8, shared, 1:1>) {
  %a1 = fly.atom.set_value(%atom, "pred", %p)
      : (!fly.copy_atom<!fly_ixdl.cq.smex_cp<16, layout = mtx>, 8>, i1)
      -> !fly.copy_atom<!fly_ixdl.cq.smex_cp<16, layout = mtx>, 8>
  fly.copy_atom_call(%a1, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.cq.smex_cp<16, layout = mtx>, 8>,
         !fly.memref<i8, #fly_ixdl.sme_gmem, 1:1>,
         !fly.memref<i8, shared, 1:1>) -> ()
  return
}

// CHECK-LABEL: @test_cq_smex_cp_plain_64_pred
// CHECK: ixdl.cp_async.smex.plain.pred.64x1b64
func.func @test_cq_smex_cp_plain_64_pred(
    %atom: !fly.copy_atom<!fly_ixdl.cq.smex_cp<64, layout = plain>, 8>,
    %p: i1,
    %src: !fly.memref<i8, #fly_ixdl.sme_gmem, 1:1>,
    %dst: !fly.memref<i8, shared, 1:1>) {
  %a1 = fly.atom.set_value(%atom, "pred", %p)
      : (!fly.copy_atom<!fly_ixdl.cq.smex_cp<64, layout = plain>, 8>, i1)
      -> !fly.copy_atom<!fly_ixdl.cq.smex_cp<64, layout = plain>, 8>
  fly.copy_atom_call(%a1, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.cq.smex_cp<64, layout = plain>, 8>,
         !fly.memref<i8, #fly_ixdl.sme_gmem, 1:1>,
         !fly.memref<i8, shared, 1:1>) -> ()
  return
}

// CHECK-LABEL: @test_cq_smex_cp_plain_4
// CHECK: ixdl.cp_async.smex.plain.4x1b64
func.func @test_cq_smex_cp_plain_4(
    %src: !fly.memref<i8, #fly_ixdl.sme_gmem, 1:1>,
    %dst: !fly.memref<i8, shared, 1:1>) {
  %atom = fly.make_copy_atom {valBits = 8 : i32}
      : !fly.copy_atom<!fly_ixdl.cq.smex_cp<4, layout = plain>, 8>
  fly.copy_atom_call(%atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.cq.smex_cp<4, layout = plain>, 8>,
         !fly.memref<i8, #fly_ixdl.sme_gmem, 1:1>,
         !fly.memref<i8, shared, 1:1>) -> ()
  return
}

// CHECK-LABEL: @test_cq_smex_cp_plain_64
// CHECK: ixdl.cp_async.smex.plain.64x1b64
func.func @test_cq_smex_cp_plain_64(
    %src: !fly.memref<i8, #fly_ixdl.sme_gmem, 1:1>,
    %dst: !fly.memref<i8, shared, 1:1>) {
  %atom = fly.make_copy_atom {valBits = 8 : i32}
      : !fly.copy_atom<!fly_ixdl.cq.smex_cp<64, layout = plain>, 8>
  fly.copy_atom_call(%atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.cq.smex_cp<64, layout = plain>, 8>,
         !fly.memref<i8, #fly_ixdl.sme_gmem, 1:1>,
         !fly.memref<i8, shared, 1:1>) -> ()
  return
}

// CHECK-LABEL: @test_cq_smex_cp_mtx_4_pred
// CHECK: ixdl.cp_async.smex.mtx.pred.4x1b64
func.func @test_cq_smex_cp_mtx_4_pred(
    %atom: !fly.copy_atom<!fly_ixdl.cq.smex_cp<4, layout = mtx>, 8>,
    %p: i1,
    %src: !fly.memref<i8, #fly_ixdl.sme_gmem, 1:1>,
    %dst: !fly.memref<i8, shared, 1:1>) {
  %pred_atom = fly.atom.set_value(%atom, "pred", %p)
      : (!fly.copy_atom<!fly_ixdl.cq.smex_cp<4, layout = mtx>, 8>, i1)
      -> !fly.copy_atom<!fly_ixdl.cq.smex_cp<4, layout = mtx>, 8>
  fly.copy_atom_call(%pred_atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.cq.smex_cp<4, layout = mtx>, 8>,
         !fly.memref<i8, #fly_ixdl.sme_gmem, 1:1>,
         !fly.memref<i8, shared, 1:1>) -> ()
  return
}

// CHECK-LABEL: @test_cq_smex_cp_mtx_64_pred
// CHECK: ixdl.cp_async.smex.mtx.pred.64x1b64
func.func @test_cq_smex_cp_mtx_64_pred(
    %atom: !fly.copy_atom<!fly_ixdl.cq.smex_cp<64, layout = mtx>, 8>,
    %p: i1,
    %src: !fly.memref<i8, #fly_ixdl.sme_gmem, 1:1>,
    %dst: !fly.memref<i8, shared, 1:1>) {
  %pred_atom = fly.atom.set_value(%atom, "pred", %p)
      : (!fly.copy_atom<!fly_ixdl.cq.smex_cp<64, layout = mtx>, 8>, i1)
      -> !fly.copy_atom<!fly_ixdl.cq.smex_cp<64, layout = mtx>, 8>
  fly.copy_atom_call(%pred_atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.cq.smex_cp<64, layout = mtx>, 8>,
         !fly.memref<i8, #fly_ixdl.sme_gmem, 1:1>,
         !fly.memref<i8, shared, 1:1>) -> ()
  return
}

// CHECK-LABEL: @test_cq_smex_cp_plain_16_pred
// CHECK: ixdl.cp_async.smex.plain.pred.16x1b64
func.func @test_cq_smex_cp_plain_16_pred(
    %atom: !fly.copy_atom<!fly_ixdl.cq.smex_cp<16, layout = plain>, 8>,
    %p: i1,
    %src: !fly.memref<i8, #fly_ixdl.sme_gmem, 1:1>,
    %dst: !fly.memref<i8, shared, 1:1>) {
  %pred_atom = fly.atom.set_value(%atom, "pred", %p)
      : (!fly.copy_atom<!fly_ixdl.cq.smex_cp<16, layout = plain>, 8>, i1)
      -> !fly.copy_atom<!fly_ixdl.cq.smex_cp<16, layout = plain>, 8>
  fly.copy_atom_call(%pred_atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.cq.smex_cp<16, layout = plain>, 8>,
         !fly.memref<i8, #fly_ixdl.sme_gmem, 1:1>,
         !fly.memref<i8, shared, 1:1>) -> ()
  return
}

// CHECK-LABEL: @test_cq_smex_cp_plain_4_copy_pred
// CHECK: %[[PRED:.*]] = llvm.load %{{.*}} : !llvm.ptr<5>
// CHECK: ixdl.cp_async.smex.plain.pred.4x1b64 %[[PRED]]
func.func @test_cq_smex_cp_plain_4_copy_pred(
    %src: !fly.memref<i8, #fly_ixdl.sme_gmem, 1:1>,
    %dst: !fly.memref<i8, shared, 1:1>,
    %pred: !fly.memref<i1, register, 1:1>) {
  %atom = fly.make_copy_atom {valBits = 8 : i32}
      : !fly.copy_atom<!fly_ixdl.cq.smex_cp<4, layout = plain>, 8>
  fly.copy_atom_call(%atom, %src, %dst, %pred)
      : (!fly.copy_atom<!fly_ixdl.cq.smex_cp<4, layout = plain>, 8>,
         !fly.memref<i8, #fly_ixdl.sme_gmem, 1:1>,
         !fly.memref<i8, shared, 1:1>, !fly.memref<i1, register, 1:1>) -> ()
  return
}
