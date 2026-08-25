// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors
// RUN: %fly-opt %s --convert-fly-to-ixdl --cse | FileCheck %s --check-prefix=CPASYNC
// RUN: %fly-opt %s | FileCheck %s --check-prefix=ROUNDTRIP

// FlyIXDL MRAsyncStore: SME store series (shared -> global), the S2G
// counterpart of ixdl_mr_async_cp.mlir. copy_atom_call (shared -> sme_gmem)
// lowers to ixdl.cp_async.store.b{64,128,256}.

// === !fly_ixdl.mr.async_store<bytes = N> parse/print round-trip (no lowering) ===

// ROUNDTRIP-LABEL: @test_mr_async_store_type
// ROUNDTRIP-SAME: !fly_ixdl.mr.async_store<bytes = 256>
func.func @test_mr_async_store_type(
    %atom: !fly.copy_atom<!fly_ixdl.mr.async_store<bytes = 256>, 32>) {
  return
}

// === end-to-end: copy_atom_call (shared -> sme_gmem) -> ixdl.cp_async.store.* ===
//
// sOffset is the shared SOURCE pointer cast to i32; gBase is the v4i32
// SmeDescriptor packed from the raw (loop-invariant) gmem_ptr of the
// DESTINATION fat pointer; gOffset is the accumulated per-tile byte_offset
// (struct field [2]); kop is 0 (CacheAll).

// CPASYNC-LABEL: @test_mr_async_store_b256_call
// CPASYNC-SAME: (%[[SRC:.*]]: !llvm.ptr<3>, %[[DST:.*]]: !llvm.struct<(ptr<1>, i32, i32)>)
// CPASYNC: %[[SOFF:.*]] = llvm.ptrtoint %[[SRC]] : !llvm.ptr<3> to i32
// SmeGmemFatPtr::smeDescriptorVec -- raw gmem_ptr[0] -> i64 -> vector<2xi32>, packed into vector<4xi32>.
// CPASYNC: %[[BASE:.*]] = llvm.extractvalue %[[DST]][0] : !llvm.struct<(ptr<1>, i32, i32)>
// CPASYNC: %[[PI:.*]] = llvm.ptrtoint %[[BASE]] : !llvm.ptr<1> to i64
// CPASYNC: %[[PAIR:.*]] = llvm.bitcast %[[PI]] : i64 to vector<2xi32>
// CPASYNC-DAG: %[[C0:.*]] = arith.constant 0 : i32
// CPASYNC-DAG: %[[C1:.*]] = arith.constant 1 : i32
// CPASYNC-DAG: %[[C2:.*]] = arith.constant 2 : i32
// CPASYNC-DAG: %[[C3:.*]] = arith.constant 3 : i32
// CPASYNC: %[[LO:.*]] = llvm.extractelement %[[PAIR]][%[[C0]] : i32] : vector<2xi32>
// CPASYNC: %[[HI:.*]] = llvm.extractelement %[[PAIR]][%[[C1]] : i32] : vector<2xi32>
// CPASYNC: %[[U:.*]] = llvm.mlir.undef : vector<4xi32>
// CPASYNC: %[[V0:.*]] = llvm.insertelement %[[LO]], %[[U]][%[[C0]] : i32] : vector<4xi32>
// CPASYNC: %[[V1:.*]] = llvm.insertelement %[[HI]], %[[V0]][%[[C1]] : i32] : vector<4xi32>
// [2] is the 0 placeholder (ivcore11); [3] is stride_byte from struct field [1].
// CPASYNC: %[[V2:.*]] = llvm.insertelement %[[C0]], %[[V1]][%[[C2]] : i32] : vector<4xi32>
// CPASYNC: %[[STRIDE:.*]] = llvm.extractvalue %[[DST]][1] : !llvm.struct<(ptr<1>, i32, i32)>
// CPASYNC: %[[V3:.*]] = llvm.insertelement %[[STRIDE]], %[[V2]][%[[C3]] : i32] : vector<4xi32>
// gOffset = accumulated byte_offset (struct field [2]); kop = 0 (CacheAll).
// CPASYNC: %[[GOFF:.*]] = llvm.extractvalue %[[DST]][2] : !llvm.struct<(ptr<1>, i32, i32)>
// CPASYNC: ixdl.cp_async.store.b256 %[[SOFF]], %[[V3]], %[[GOFF]], %[[C0]] : vector<4xi32> -> i32
func.func @test_mr_async_store_b256_call(
    %src: !fly.memref<f32, shared, 1:1>,
    %dst: !fly.memref<f32, #fly_ixdl.sme_gmem, 1:1>) {
  %atom = fly.make_copy_atom {valBits = 32 : i32}
      : !fly.copy_atom<!fly_ixdl.mr.async_store<bytes = 256>, 32>
  fly.copy_atom_call(%atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.mr.async_store<bytes = 256>, 32>,
         !fly.memref<f32, shared, 1:1>,
         !fly.memref<f32, #fly_ixdl.sme_gmem, 1:1>) -> ()
  return
}

// CPASYNC-LABEL: @test_mr_async_store_b128
// CPASYNC: ixdl.cp_async.store.b128
func.func @test_mr_async_store_b128(
    %src: !fly.memref<f32, shared, 1:1>,
    %dst: !fly.memref<f32, #fly_ixdl.sme_gmem, 1:1>) {
  %atom = fly.make_copy_atom {valBits = 32 : i32}
      : !fly.copy_atom<!fly_ixdl.mr.async_store<bytes = 128>, 32>
  fly.copy_atom_call(%atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.mr.async_store<bytes = 128>, 32>,
         !fly.memref<f32, shared, 1:1>,
         !fly.memref<f32, #fly_ixdl.sme_gmem, 1:1>) -> ()
  return
}

// CPASYNC-LABEL: @test_mr_async_store_b64
// CPASYNC: ixdl.cp_async.store.b64
func.func @test_mr_async_store_b64(
    %src: !fly.memref<f16, shared, 1:1>,
    %dst: !fly.memref<f16, #fly_ixdl.sme_gmem, 1:1>) {
  %atom = fly.make_copy_atom {valBits = 16 : i32}
      : !fly.copy_atom<!fly_ixdl.mr.async_store<bytes = 64>, 16>
  fly.copy_atom_call(%atom, %src, %dst)
      : (!fly.copy_atom<!fly_ixdl.mr.async_store<bytes = 64>, 16>,
         !fly.memref<f16, shared, 1:1>,
         !fly.memref<f16, #fly_ixdl.sme_gmem, 1:1>) -> ()
  return
}
