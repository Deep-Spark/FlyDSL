// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"

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

//===----------------------------------------------------------------------===//
// CQMtxLoadn — SmexMtx shared->register matrix load (ixdl.mtx_loadn_*)
//===----------------------------------------------------------------------===//

namespace {

enum class MtxPattern : int32_t { Loadn16 = 0, Loadn64 = 1 };
enum class MtxDir : int32_t { Row = 0, Col = 1 };

int32_t mtxBitSize(int32_t x2) { return x2 ? 64 : 32; }

FailureOr<Value> emitMtxLoadn(OpBuilder &builder, Location loc, int32_t direction, int32_t bitWidth,
                              int32_t x2, Value srcSharedPtr) {
  Type i32Ty = builder.getI32Type();
  Type resTy = x2 ? VectorType::get({2}, i32Ty) : Type(i32Ty);
  bool row = direction == static_cast<int32_t>(MtxDir::Row);

  Value loaded;
  if (row && bitWidth == 16) {
    loaded = x2 ? IXDL::MtxLoadnB16Rowx2Op::create(builder, loc, resTy, srcSharedPtr).getResult()
                : IXDL::MtxLoadnB16RowOp::create(builder, loc, resTy, srcSharedPtr).getResult();
  } else if (row && bitWidth == 8) {
    loaded = x2 ? IXDL::MtxLoadnB8Rowx2Op::create(builder, loc, resTy, srcSharedPtr).getResult()
                : IXDL::MtxLoadnB8RowOp::create(builder, loc, resTy, srcSharedPtr).getResult();
  } else if (!row) {
    // Col gather is width-agnostic at the IXDL op (packed i32 / v2i32).
    loaded = x2 ? IXDL::MtxLoadnColx2Op::create(builder, loc, resTy, srcSharedPtr).getResult()
                : IXDL::MtxLoadnColOp::create(builder, loc, resTy, srcSharedPtr).getResult();
  } else {
    return mlir::emitError(loc) << "CQMtxLoadn unsupported (dir, bitWidth) = (" << direction << ", "
                                << bitWidth << ")";
  }
  return loaded;
}

} // namespace

LogicalResult CopyOpCQMtxLoadnType::verify(function_ref<InFlightDiagnostic()> emitError,
                                           int32_t pattern, int32_t direction, int32_t bitWidth,
                                           int32_t x2) {
  if (pattern != static_cast<int32_t>(MtxPattern::Loadn16) &&
      pattern != static_cast<int32_t>(MtxPattern::Loadn64))
    return emitError() << "unsupported pattern = " << pattern
                       << " for CQMtxLoadn (expected 0=loadn16 or 1=loadn64)";
  if (direction != static_cast<int32_t>(MtxDir::Row) &&
      direction != static_cast<int32_t>(MtxDir::Col))
    return emitError() << "unsupported dir = " << direction
                       << " for CQMtxLoadn (expected 0=row or 1=col)";
  if (bitWidth != 8 && bitWidth != 16)
    return emitError() << "unsupported b = " << bitWidth << " for CQMtxLoadn (expected 8 or 16)";
  if (x2 != 0 && x2 != 1)
    return emitError() << "unsupported x2 = " << x2 << " for CQMtxLoadn (expected 0 or 1)";
  return success();
}

bool CopyOpCQMtxLoadnType::isStatic() const { return true; }

Value CopyOpCQMtxLoadnType::rebuildStaticValue(OpBuilder &, Location, Value) const {
  return nullptr;
}

Attribute CopyOpCQMtxLoadnType::getThrLayout() const {
  // Warp-collective matrix load: all 64 lanes participate.
  return FxLayout(FxC(kWarpSize), FxC(1));
}

Attribute CopyOpCQMtxLoadnType::getThrBitLayoutSrc() const {
  // Shared-side footprint: same thr hierarchy as the register fragment; each
  // lane owns `bitSize` contiguous bits at its SmexMtx pointer (EmPart is
  // folded into the pointer by layout lowering / teaching kernels — Bypass
  // swizzle).
  int32_t bitSize = mtxBitSize(getX2());
  return FxLayout(FxShape(FxThr(16, 4), FxC(bitSize)), FxStride(FxThr(bitSize, 0), FxC(1)));
}

Attribute CopyOpCQMtxLoadnType::getThrBitLayoutDst() const {
  // Bit-granularity expansion of CQ MMA base-tile A/B ThrVal layouts so
  // make_tiled_copy_A/B can divide a CQMma fragment by this atom.
  int32_t bitSize = mtxBitSize(getX2());
  bool row = getDirection() == static_cast<int32_t>(MtxDir::Row);

  if (getBitWidth() == 16) {
    // Layout_16x16_16b_{A,B}
    if (row) {
      if (bitSize == 64)
        return FxLayout(FxShape(FxThr(16, 4), FxVal(2, 2, 16)),
                        FxStride(FxThr(256, 32), FxVal(16, 128, 1)));
      return FxLayout(FxShape(FxThr(16, 4), FxVal(2, 16)), FxStride(FxThr(256, 32), FxVal(16, 1)));
    }
    if (bitSize == 64)
      return FxLayout(FxShape(FxThr(16, 4), FxVal(2, 2, 16)),
                      FxStride(FxThr(16, 512), FxVal(256, 2048, 1)));
    return FxLayout(FxShape(FxThr(16, 4), FxVal(2, 16)), FxStride(FxThr(16, 512), FxVal(256, 1)));
  }

  // Layout_16x32_8b_{A,B}
  if (row) {
    if (bitSize == 64)
      return FxLayout(FxShape(FxThr(16, 4), FxVal(4, 2, 8)),
                      FxStride(FxThr(128, 32), FxVal(8, 2048, 1)));
    return FxLayout(FxShape(FxThr(16, 4), FxVal(4, 8)), FxStride(FxThr(128, 32), FxVal(8, 1)));
  }
  if (bitSize == 64)
    return FxLayout(FxShape(FxThr(16, 4), FxVal(4, 2, 8)),
                    FxStride(FxThr(8, 512), FxVal(128, 2048, 1)));
  return FxLayout(FxShape(FxThr(16, 4), FxVal(4, 8)), FxStride(FxThr(8, 512), FxVal(128, 1)));
}

Attribute CopyOpCQMtxLoadnType::getThrBitLayoutRef() const { return getThrBitLayoutDst(); }

LogicalResult CopyOpCQMtxLoadnType::emitAtomCall(OpBuilder &builder, Location loc,
                                                 Type copyAtomTyArg, Type srcMemTyArg,
                                                 Type dstMemTyArg, Value atomVal, Value src,
                                                 Value dst) const {
  auto dstSSATy = fly::RegMem2SSAType(cast<fly::MemRefType>(dstMemTyArg), true);
  auto res = emitAtomCallSSA(builder, loc, dstSSATy, copyAtomTyArg, srcMemTyArg, Type{}, atomVal,
                             src, Value{});
  if (failed(res))
    return failure();
  LLVM::StoreOp::create(builder, loc, *res, dst);
  return success();
}

LogicalResult CopyOpCQMtxLoadnType::emitAtomCall(OpBuilder &builder, Location loc, Type, Type, Type,
                                                 Type, Value, Value, Value, Value) const {
  return mlir::emitError(loc) << "predicated CQMtxLoadn is not implemented";
}

FailureOr<Value> CopyOpCQMtxLoadnType::emitAtomCallSSA(OpBuilder &builder, Location loc,
                                                       Type resultTy, Type copyAtomTyArg,
                                                       Type srcTyArg, Type, Value, Value src,
                                                       Value) const {
  auto srcMemTy = dyn_cast<fly::MemRefType>(srcTyArg);
  if (srcMemTy &&
      !isGenericAddressSpace<fly::AddressSpace::Shared>(srcMemTy.getAddressSpace()))
    return mlir::emitError(loc) << "CQMtxLoadn src must be shared address space";

  auto loaded = emitMtxLoadn(builder, loc, getDirection(), getBitWidth(), getX2(), src);
  if (failed(loaded))
    return failure();

  Value out = *loaded;
  if (resultTy && out.getType() != resultTy)
    out = LLVM::BitcastOp::create(builder, loc, resultTy, out);
  return out;
}

FailureOr<Value> CopyOpCQMtxLoadnType::emitAtomCallSSA(OpBuilder &builder, Location loc, Type, Type,
                                                       Type, Type, Type, Value, Value, Value,
                                                       Value) const {
  return mlir::emitError(loc) << "predicated CQMtxLoadn is not implemented";
}

} // namespace mlir::fly_ixdl
