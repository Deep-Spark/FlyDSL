// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/PointerUtils.h"
#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"
#include "flydsl/Dialect/FlyIXDL/IR/Dialect.h"
#include "flydsl/Dialect/FlyIXDL/Utils/SmeGmemFatPtr.h"

using namespace mlir;
using namespace mlir::fly;

namespace mlir::fly_ixdl {

LogicalResult CopyOpCQAsyncCpType::verify(function_ref<InFlightDiagnostic()> emitError, int32_t row,
                                          int32_t col, int32_t transpose) {
  if (transpose != 0 && transpose != 1)
    return emitError() << "unsupported transpose = " << transpose
                       << " for CQAsyncCp (expected 0 or 1)";

  bool isRowShape = transpose == 0 && ((row == 64 && col == 64) || (row == 64 && col == 32) ||
                                       (row == 1 && col == 1024) || (row == 64 && col == 16));
  bool isColShape = transpose == 1 && row == 64 && col == 16;
  if (!isRowShape && !isColShape)
    return emitError() << "unsupported enhanced-SME shape (" << row << ", " << col
                       << "), transpose = " << transpose << " for CQAsyncCp";
  return success();
}

bool CopyOpCQAsyncCpType::isStatic() const { return true; }

Value CopyOpCQAsyncCpType::rebuildStaticValue(OpBuilder &, Location, Value) const {
  return nullptr;
}

Attribute CopyOpCQAsyncCpType::getThrLayout() const {
  // The hardware instruction performs the warp cooperation internally.
  return FxLayout(FxC(1), FxC(1));
}

Attribute CopyOpCQAsyncCpType::getThrBitLayoutSrc() const {
  // Every CQ enhanced-SME shape represented by this atom moves 4096 bytes.
  return FxLayout(FxShape(FxC(1), FxC(32768)), FxStride(FxC(0), FxC(1)));
}

Attribute CopyOpCQAsyncCpType::getThrBitLayoutDst() const { return getThrBitLayoutSrc(); }

Attribute CopyOpCQAsyncCpType::getThrBitLayoutRef() const { return getThrBitLayoutDst(); }

// CQAsyncCp is one-directional: global (#fly_ixdl.sme_gmem) to shared. CQ S2R
// matrix loads (`loadn`) are not modeled by this atom.
LogicalResult CopyOpCQAsyncCpType::emitAtomCall(OpBuilder &builder, Location loc,
                                                Type copyAtomTyArg, Type srcMemTyArg,
                                                Type dstMemTyArg, Value, Value src,
                                                Value dst) const {
  auto copyAtomTy = dyn_cast<fly::CopyAtomType>(copyAtomTyArg);
  auto srcMemTy = dyn_cast<fly::MemRefType>(srcMemTyArg);
  auto dstMemTy = dyn_cast<fly::MemRefType>(dstMemTyArg);
  if (!copyAtomTy || !srcMemTy || !dstMemTy)
    return failure();

  if (!isTargetAddressSpace<SmeGmemAddressAttr>(srcMemTy.getAddressSpace()) ||
      !isGenericAddressSpace<fly::AddressSpace::Shared>(dstMemTy.getAddressSpace()))
    return failure();

  Value sOffset = LLVM::PtrToIntOp::create(builder, loc, builder.getI32Type(), dst);
  SmeGmemFatPtr srcFat(srcMemTy.getPointerType(), src);
  Value gBase = srcFat.smeDescriptorVec(builder, loc, -1);
  Value gOffset = srcFat.byteOffset(builder, loc);
  Value kop = arith::ConstantIntOp::create(builder, loc, 1, 32);

  int32_t row = getRow();
  int32_t col = getCol();
  int32_t valBits = copyAtomTy.getValBits();
  bool transpose = getTranspose() != 0;

  if (valBits == 8 && row == 64 && col == 64 && !transpose) {
    IXDL::CpAsync_64x64_b8_RowOp::create(builder, loc, sOffset, gBase, gOffset, kop);
    return success();
  }
  if (valBits == 16 && row == 64 && col == 32 && !transpose) {
    IXDL::CpAsync_64x32_b16_RowOp::create(builder, loc, sOffset, gBase, gOffset, kop);
    return success();
  }
  if (valBits == 32 && row == 1 && col == 1024 && !transpose) {
    IXDL::CpAsync_1x64b64Op::create(builder, loc, sOffset, gBase, gOffset, kop);
    return success();
  }
  if (valBits == 32 && row == 64 && col == 16) {
    if (transpose)
      IXDL::CpAsync_64x16_b32_ColOp::create(builder, loc, sOffset, gBase, gOffset, kop);
    else
      IXDL::CpAsync_64x16_b32_RowOp::create(builder, loc, sOffset, gBase, gOffset, kop);
    return success();
  }

  return mlir::emitError(loc) << "CQAsyncCp does not support valBits = " << valBits
                              << " for shape (" << row << ", " << col
                              << "), transpose = " << transpose;
}

LogicalResult CopyOpCQAsyncCpType::emitAtomCall(OpBuilder &builder, Location loc, Type, Type, Type,
                                                Type, Value, Value, Value, Value) const {
  return mlir::emitError(loc) << "predicated CQAsyncCp is not implemented";
}

FailureOr<Value> CopyOpCQAsyncCpType::emitAtomCallSSA(OpBuilder &builder, Location loc, Type,
                                                      Type copyAtomTyArg, Type srcTyArg,
                                                      Type dstTyArg, Value atomVal, Value src,
                                                      Value dst) const {
  if (failed(emitAtomCall(builder, loc, copyAtomTyArg, srcTyArg, dstTyArg, atomVal, src, dst)))
    return failure();
  return Value{};
}

FailureOr<Value> CopyOpCQAsyncCpType::emitAtomCallSSA(OpBuilder &builder, Location loc, Type, Type,
                                                      Type, Type, Type, Value, Value, Value,
                                                      Value) const {
  return mlir::emitError(loc) << "predicated CQAsyncCp is not implemented";
}

} // namespace mlir::fly_ixdl
