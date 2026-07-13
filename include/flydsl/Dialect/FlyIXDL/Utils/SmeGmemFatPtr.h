// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYDSL_DIALECT_FLYIXDL_UTILS_SMEGMEMFATPTR_H
#define FLYDSL_DIALECT_FLYIXDL_UTILS_SMEGMEMFATPTR_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/FlyIXDL/IR/Dialect.h"

namespace mlir::fly_ixdl {

// Iluvatar SME global "fat pointer". Engineering template: FlyROCDL BufferFatPtr.
// Semantic fields align with the hardware SME descriptor:
//
//   struct { !llvm.ptr<1> gmem_ptr;   // [0] global base pointer
//            i32          stride_byte; // [1] leading stride in bytes (make_ptr)
//            i32          byte_offset; // [2] accumulated byte offset (add_offset)
//          }
//
// The descriptor is built from the *raw* gmem_ptr (loop-invariant, so it hoists
// out of a tile loop), and byte_offset is emitted as the hardware gOffset operand
// (a 32-bit per-tile offset) rather than folded into the 64-bit base. This lets
// the backend keep one descriptor and advance only the narrow offset -- constant
// offsets fold into the goffimm immediate, variable ones stay 32-bit adds
// (see design doc §10).
class SmeGmemFatPtr {
  static constexpr unsigned kGmemAddrSpace = 1;   // global
  static constexpr unsigned kStrideBitWidth = 32; // leading stride in bytes
  static constexpr unsigned kOffsetBitWidth = 32; // accumulated byte offset

  fly::PointerType ptrTy;
  Value fatPtr;

public:
  SmeGmemFatPtr(fly::PointerType ptrTy, Value v) : ptrTy(ptrTy), fatPtr(v) {
    assert(fly::isTargetAddressSpace<SmeGmemAddressAttr>(ptrTy.getAddressSpace()));
  }

  static LLVM::LLVMStructType getType(MLIRContext *ctx) {
    return LLVM::LLVMStructType::getLiteral(ctx, {LLVM::LLVMPointerType::get(ctx, kGmemAddrSpace),
                                                  IntegerType::get(ctx, kStrideBitWidth),
                                                  IntegerType::get(ctx, kOffsetBitWidth)});
  }

  static Value pack(OpBuilder &b, Location loc, Value gmemPtr, Value strideByte,
                    Value byteOffset = nullptr) {
    auto structTy = getType(b.getContext());
    if (!byteOffset)
      byteOffset = arith::ConstantIntOp::create(b, loc, 0, kOffsetBitWidth);
    Value packed = LLVM::UndefOp::create(b, loc, structTy);
    packed = LLVM::InsertValueOp::create(b, loc, packed, gmemPtr, ArrayRef<int64_t>{0});
    packed = LLVM::InsertValueOp::create(b, loc, packed, strideByte, ArrayRef<int64_t>{1});
    packed = LLVM::InsertValueOp::create(b, loc, packed, byteOffset, ArrayRef<int64_t>{2});
    return packed;
  }

  Value gmemPtr(OpBuilder &b, Location loc) const {
    return LLVM::ExtractValueOp::create(b, loc, fatPtr, ArrayRef<int64_t>{0});
  }

  Value strideByte(OpBuilder &b, Location loc) const {
    return LLVM::ExtractValueOp::create(b, loc, fatPtr, ArrayRef<int64_t>{1});
  }

  Value byteOffset(OpBuilder &b, Location loc) const {
    return LLVM::ExtractValueOp::create(b, loc, fatPtr, ArrayRef<int64_t>{2});
  }

  // Accumulate a byte delta into the byte_offset field and repack.
  Value addByteOffset(OpBuilder &b, Location loc, Value deltaBytes) const {
    Type offTy = IntegerType::get(b.getContext(), kOffsetBitWidth);
    if (deltaBytes.getType() != offTy) {
      if (deltaBytes.getType().isIndex())
        deltaBytes = arith::IndexCastOp::create(b, loc, offTy, deltaBytes);
      else if (deltaBytes.getType().getIntOrFloatBitWidth() < kOffsetBitWidth)
        deltaBytes = arith::ExtSIOp::create(b, loc, offTy, deltaBytes);
      else
        deltaBytes = arith::TruncIOp::create(b, loc, offTy, deltaBytes);
    }
    Value newOff = arith::AddIOp::create(b, loc, byteOffset(b, loc), deltaBytes);
    return pack(b, loc, gmemPtr(b, loc), strideByte(b, loc), newOff);
  }

  // Pack into a vector<4xi32> SmeDescriptor: [0..1] = raw gmem_ptr (loop-invariant,
  // byte_offset is NOT folded in -- it goes to the gOffset operand), [2] =
  // placeholder (0 on ivcore11), [3] = stride_byte.
  Value smeDescriptorVec(OpBuilder &b, Location loc) const {
    auto i32Ty = IntegerType::get(b.getContext(), 32);
    auto i64Ty = IntegerType::get(b.getContext(), 64);
    auto vec2Ty = VectorType::get({2}, i32Ty);
    auto vec4Ty = VectorType::get({4}, i32Ty);

    Value ptrInt = LLVM::PtrToIntOp::create(b, loc, i64Ty, gmemPtr(b, loc));
    Value ptrPair = LLVM::BitcastOp::create(b, loc, vec2Ty, ptrInt);

    Value c0 = arith::ConstantIntOp::create(b, loc, 0, 32);
    Value c1 = arith::ConstantIntOp::create(b, loc, 1, 32);
    Value c2 = arith::ConstantIntOp::create(b, loc, 2, 32);
    Value c3 = arith::ConstantIntOp::create(b, loc, 3, 32);

    Value lo = LLVM::ExtractElementOp::create(b, loc, ptrPair, c0);
    Value hi = LLVM::ExtractElementOp::create(b, loc, ptrPair, c1);
    Value placeholder = arith::ConstantIntOp::create(b, loc, 0, 32);

    Value vec = LLVM::UndefOp::create(b, loc, vec4Ty);
    vec = LLVM::InsertElementOp::create(b, loc, vec, lo, c0);
    vec = LLVM::InsertElementOp::create(b, loc, vec, hi, c1);
    vec = LLVM::InsertElementOp::create(b, loc, vec, placeholder, c2);
    vec = LLVM::InsertElementOp::create(b, loc, vec, strideByte(b, loc), c3);
    return vec;
  }
};

} // namespace mlir::fly_ixdl

#endif // FLYDSL_DIALECT_FLYIXDL_UTILS_SMEGMEMFATPTR_H
