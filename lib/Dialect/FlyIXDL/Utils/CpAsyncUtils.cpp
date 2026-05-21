// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "flydsl/Dialect/FlyIXDL/Utils/CpAsyncUtils.h"

#include <limits>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

using namespace mlir;
using namespace mlir::fly;
using namespace mlir::fly_ixdl;

namespace {

LayoutAttr peelToLayoutAttr(Attribute layoutAttr) {
  if (auto layout = dyn_cast<LayoutAttr>(layoutAttr))
    return layout;
  if (auto composed = dyn_cast<ComposedLayoutAttr>(layoutAttr))
    return peelToLayoutAttr(composed.getOuter());
  return nullptr;
}

std::optional<int64_t> staticStrideElemsAtMode(IntTupleAttr strideTuple, int32_t modeIdx) {
  if (strideTuple.isLeaf()) {
    if (modeIdx != 0)
      return std::nullopt;
    IntAttr leaf = strideTuple.extractIntFromLeaf();
    if (!leaf.isStatic())
      return std::nullopt;
    return leaf.getValue();
  }
  if (modeIdx >= strideTuple.rank())
    return std::nullopt;
  IntTupleAttr modeStride = strideTuple.at(modeIdx);
  if (!modeStride.isLeaf())
    return std::nullopt;
  IntAttr leaf = modeStride.extractIntFromLeaf();
  if (!leaf.isStatic())
    return std::nullopt;
  return leaf.getValue();
}

std::optional<int64_t> staticGlobalPitchElems(LayoutAttr layout, bool colCopy) {
  if (!layout.isStaticStride())
    return std::nullopt;
  IntTupleAttr stride = layout.getStride();
  int32_t modeIdx = 0;
  if (colCopy && stride.rank() > 1)
    modeIdx = 1;
  return staticStrideElemsAtMode(stride, modeIdx);
}

bool isColCopyKind(MRCpAsyncKind kind) {
  switch (kind) {
  case MRCpAsyncKind::k4x64_b8_Col:
  case MRCpAsyncKind::k16x64_b8_Col:
  case MRCpAsyncKind::k4x32_b16_Col:
  case MRCpAsyncKind::k16x32_b16_Col:
  case MRCpAsyncKind::k16x16_b32_Col:
    return true;
  default:
    return false;
  }
}

} // namespace

Value fly_ixdl::scaleSoffsetToBytes(OpBuilder &builder, Location loc, Value soffsetRaw,
                                    int64_t elemBits) {
  if (elemBits == 8)
    return soffsetRaw;
  if (elemBits > 8 && elemBits % 8 == 0) {
    Value scale = arith::ConstantIntOp::create(builder, loc, elemBits / 8, 32);
    return arith::MulIOp::create(builder, loc, soffsetRaw, scale);
  }
  Value scale = arith::ConstantIntOp::create(builder, loc, elemBits, 32);
  Value bits = arith::MulIOp::create(builder, loc, soffsetRaw, scale);
  Value eight = arith::ConstantIntOp::create(builder, loc, 8, 32);
  return arith::DivUIOp::create(builder, loc, bits, eight);
}

Value fly_ixdl::buildSOffset(OpBuilder &builder, Location loc, Value dstPtr, Value immOffset) {
  Value sOffset = LLVM::PtrToIntOp::create(builder, loc, builder.getI32Type(), dstPtr);
  return arith::AddIOp::create(builder, loc, sOffset, immOffset);
}

Value fly_ixdl::buildGBase(OpBuilder &builder, Location loc, Value srcPtr, Value scaledSoffsetBytes,
                           Value strideBytes) {
  Value gAddr = LLVM::PtrToIntOp::create(builder, loc, builder.getI64Type(), srcPtr);
  Value scaledExt =
      arith::ExtUIOp::create(builder, loc, builder.getI64Type(), scaledSoffsetBytes);
  gAddr = arith::AddIOp::create(builder, loc, gAddr, scaledExt);

  Value gAddrLow32 = arith::TruncIOp::create(builder, loc, builder.getI32Type(), gAddr);
  Value shift32 = arith::ConstantIntOp::create(builder, loc, 32, 64);
  Value gAddrHigh32 = arith::TruncIOp::create(
      builder, loc, builder.getI32Type(), arith::ShRUIOp::create(builder, loc, gAddr, shift32));
  Value minusOne = arith::ConstantIntOp::create(builder, loc, 0xFFFFFFFF, 32);

  VectorType vectorType = VectorType::get({4}, builder.getI32Type());
  Value gBase = LLVM::PoisonOp::create(builder, loc, vectorType);
  gBase = vector::InsertOp::create(builder, loc, gAddrLow32, gBase, 0);
  gBase = vector::InsertOp::create(builder, loc, gAddrHigh32, gBase, 1);
  gBase = vector::InsertOp::create(builder, loc, minusOne, gBase, 2);
  gBase = vector::InsertOp::create(builder, loc, strideBytes, gBase, 3);
  return gBase;
}

Value fly_ixdl::buildGOffsetZero(OpBuilder &builder, Location loc) {
  return arith::ConstantIntOp::create(builder, loc, 0, 32);
}

Value fly_ixdl::buildKopOne(OpBuilder &builder, Location loc) {
  return arith::ConstantIntOp::create(builder, loc, 1, 32);
}

Value fly_ixdl::buildStrideBytesFromMemref(OpBuilder &builder, Location loc,
                                           fly::MemRefType srcMemTy, MRCpAsyncKind kind) {
  LayoutAttr layout = peelToLayoutAttr(srcMemTy.getLayout());
  if (!layout)
    return Value();
  std::optional<int64_t> strideElems =
      staticGlobalPitchElems(layout, isColCopyKind(kind));
  if (!strideElems)
    return Value();
  int64_t elemBytes = srcMemTy.getElemTy().getIntOrFloatBitWidth() / 8;
  int64_t strideBytes = *strideElems * elemBytes;
  if (strideBytes < 0 || strideBytes > std::numeric_limits<int32_t>::max())
    return Value();
  // Temporary ISA assumption: GM-side addressing for SME uses 64B granularity
  // ({OFF-Imm, 6'b0}), so require pitch to be 64B aligned.
  if (strideBytes % 64 != 0)
    return Value();
  return arith::ConstantIntOp::create(builder, loc, strideBytes, 32);
}

LogicalResult fly_ixdl::emitCpAsyncByKind(OpBuilder &builder, Location loc, MRCpAsyncKind kind,
                                          Value sOffset, Value gBase, Value gOffset, Value kop) {
  switch (kind) {
  case MRCpAsyncKind::k4x64_b8_Row:
    IXDL::CpAsync_4x64_b8_RowOp::create(builder, loc, sOffset, gBase, gOffset, kop);
    return success();
  case MRCpAsyncKind::k4x64_b8_Col:
    IXDL::CpAsync_4x64_b8_ColOp::create(builder, loc, sOffset, gBase, gOffset, kop);
    return success();
  case MRCpAsyncKind::k16x64_b8_Row:
    IXDL::CpAsync_16x64_b8_RowOp::create(builder, loc, sOffset, gBase, gOffset, kop);
    return success();
  case MRCpAsyncKind::k16x64_b8_Col:
    IXDL::CpAsync_16x64_b8_ColOp::create(builder, loc, sOffset, gBase, gOffset, kop);
    return success();
  case MRCpAsyncKind::k4x32_b16_Row:
    IXDL::CpAsync_4x32_b16_RowOp::create(builder, loc, sOffset, gBase, gOffset, kop);
    return success();
  case MRCpAsyncKind::k4x32_b16_Col:
    IXDL::CpAsync_4x32_b16_ColOp::create(builder, loc, sOffset, gBase, gOffset, kop);
    return success();
  case MRCpAsyncKind::k16x32_b16_Row:
    IXDL::CpAsync_16x32_b16_RowOp::create(builder, loc, sOffset, gBase, gOffset, kop);
    return success();
  case MRCpAsyncKind::k16x32_b16_Col:
    IXDL::CpAsync_16x32_b16_ColOp::create(builder, loc, sOffset, gBase, gOffset, kop);
    return success();
  case MRCpAsyncKind::k1x1b64:
    IXDL::CpAsync_1x1b64Op::create(builder, loc, sOffset, gBase, gOffset, kop);
    return success();
  case MRCpAsyncKind::k1x4b64:
    IXDL::CpAsync_1x4b64Op::create(builder, loc, sOffset, gBase, gOffset, kop);
    return success();
  case MRCpAsyncKind::k1x8b64:
    IXDL::CpAsync_1x8b64Op::create(builder, loc, sOffset, gBase, gOffset, kop);
    return success();
  case MRCpAsyncKind::k1x16b64:
    IXDL::CpAsync_1x16b64Op::create(builder, loc, sOffset, gBase, gOffset, kop);
    return success();
  case MRCpAsyncKind::k4x16_b32_Row:
    IXDL::CpAsync_4x16_b32_RowOp::create(builder, loc, sOffset, gBase, gOffset, kop);
    return success();
  case MRCpAsyncKind::k8x16_b32_Row:
    IXDL::CpAsync_8x16_b32_RowOp::create(builder, loc, sOffset, gBase, gOffset, kop);
    return success();
  case MRCpAsyncKind::k16x16_b32_Row:
    IXDL::CpAsync_16x16_b32_RowOp::create(builder, loc, sOffset, gBase, gOffset, kop);
    return success();
  case MRCpAsyncKind::k16x16_b32_Col:
    IXDL::CpAsync_16x16_b32_ColOp::create(builder, loc, sOffset, gBase, gOffset, kop);
    return success();
  }
  return failure();
}
