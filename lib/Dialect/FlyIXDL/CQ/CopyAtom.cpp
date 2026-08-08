// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/PointerUtils.h"
#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"
#include "flydsl/Dialect/FlyIXDL/IR/Dialect.h"
#include "flydsl/Dialect/FlyIXDL/Utils/SmeGmemFatPtr.h"

using namespace mlir;
using namespace mlir::fly;

namespace mlir::fly_ixdl {

//===----------------------------------------------------------------------===//
// CopyOpCQSmexCpType -- CQ SMEX G2S with runtime RowMask / ColMask / Pred.
// Lowers to ixdl.cp_async.smex.{mtx,plain}[.pred].{4,16,64}x1b64.
//===----------------------------------------------------------------------===//

namespace {

constexpr unsigned kSmexRowMaskSlot = 0;
constexpr unsigned kSmexColMaskSlot = 1;
constexpr unsigned kSmexPredSlot = 2;
constexpr unsigned kSmexPredEnabledSlot = 3;

std::optional<unsigned> smexStateSlot(StringRef name) {
  if (name == "row_mask")
    return kSmexRowMaskSlot;
  if (name == "col_mask")
    return kSmexColMaskSlot;
  if (name == "pred")
    return kSmexPredSlot;
  return std::nullopt;
}

LogicalResult emitSmexIxdl(OpBuilder &builder, Location loc, int32_t rows, SmexLayout layout,
                           bool predEnabled, Value sPtr, Value gPtr, Value stride, Value gOffset,
                           Value rowMask, Value colMask, Value pred) {
  // IXDL SMEX ops require kop / pfHint as compile-time constants (not runtime
  // atom state). Default both to 1 to match the ixcc G2S lowering; expose as
  // CQSmexCp type parameters only if a kernel needs a non-default value.
  Value kop = arith::ConstantIntOp::create(builder, loc, 1, 32);
  Value pfHint = arith::ConstantIntOp::create(builder, loc, 1, 32);
  bool plain = layout == SmexLayout::Plain;

  if (predEnabled) {
    if (plain) {
      if (rows == 4) {
        IXDL::CpAsync_SmexPlain_Pred_4x1b64Op::create(builder, loc, pred, sPtr, gPtr, stride,
                                                      gOffset, rowMask, colMask, kop, pfHint);
        return success();
      }
      if (rows == 16) {
        IXDL::CpAsync_SmexPlain_Pred_16x1b64Op::create(builder, loc, pred, sPtr, gPtr, stride,
                                                       gOffset, rowMask, colMask, kop, pfHint);
        return success();
      }
      if (rows == 64) {
        IXDL::CpAsync_SmexPlain_Pred_64x1b64Op::create(builder, loc, pred, sPtr, gPtr, stride,
                                                       gOffset, rowMask, colMask, kop, pfHint);
        return success();
      }
    } else {
      if (rows == 4) {
        IXDL::CpAsync_SmexMtx_Pred_4x1b64Op::create(builder, loc, pred, sPtr, gPtr, stride, gOffset,
                                                    rowMask, colMask, kop, pfHint);
        return success();
      }
      if (rows == 16) {
        IXDL::CpAsync_SmexMtx_Pred_16x1b64Op::create(builder, loc, pred, sPtr, gPtr, stride,
                                                     gOffset, rowMask, colMask, kop, pfHint);
        return success();
      }
      if (rows == 64) {
        IXDL::CpAsync_SmexMtx_Pred_64x1b64Op::create(builder, loc, pred, sPtr, gPtr, stride,
                                                     gOffset, rowMask, colMask, kop, pfHint);
        return success();
      }
    }
  } else {
    if (plain) {
      if (rows == 4) {
        IXDL::CpAsync_SmexPlain_4x1b64Op::create(builder, loc, sPtr, gPtr, stride, gOffset, rowMask,
                                                 colMask, kop, pfHint);
        return success();
      }
      if (rows == 16) {
        IXDL::CpAsync_SmexPlain_16x1b64Op::create(builder, loc, sPtr, gPtr, stride, gOffset,
                                                  rowMask, colMask, kop, pfHint);
        return success();
      }
      if (rows == 64) {
        IXDL::CpAsync_SmexPlain_64x1b64Op::create(builder, loc, sPtr, gPtr, stride, gOffset,
                                                  rowMask, colMask, kop, pfHint);
        return success();
      }
    } else {
      if (rows == 4) {
        IXDL::CpAsync_SmexMtx_4x1b64Op::create(builder, loc, sPtr, gPtr, stride, gOffset, rowMask,
                                               colMask, kop, pfHint);
        return success();
      }
      if (rows == 16) {
        IXDL::CpAsync_SmexMtx_16x1b64Op::create(builder, loc, sPtr, gPtr, stride, gOffset, rowMask,
                                                colMask, kop, pfHint);
        return success();
      }
      if (rows == 64) {
        IXDL::CpAsync_SmexMtx_64x1b64Op::create(builder, loc, sPtr, gPtr, stride, gOffset, rowMask,
                                                colMask, kop, pfHint);
        return success();
      }
    }
  }

  return mlir::emitError(loc) << "CQSmexCp unsupported rows = " << rows
                              << ", predEnabled = " << predEnabled;
}

// Try to fold SmeGmemFatPtr.stride_byte (struct field [1]) to a constant.
// Handles pack(insertvalue) and stride_byte = muli(const, const) from make_ptr
// lowering. Dynamic / non-foldable strides return nullopt (skip the check).
std::optional<int64_t> matchConstantSmeStrideByte(Value fatPtr) {
  Value v = fatPtr;
  while (auto ins = v.getDefiningOp<LLVM::InsertValueOp>()) {
    ArrayRef<int64_t> pos = ins.getPosition();
    if (pos.size() == 1 && pos[0] == 1) {
      Value strideByte = ins.getValue();
      APInt ap;
      if (matchPattern(strideByte, m_ConstantInt(&ap)))
        return ap.getSExtValue();
      if (auto mul = strideByte.getDefiningOp<arith::MulIOp>()) {
        APInt lhs, rhs;
        if (matchPattern(mul.getLhs(), m_ConstantInt(&lhs)) &&
            matchPattern(mul.getRhs(), m_ConstantInt(&rhs)))
          return (lhs * rhs).getSExtValue();
      }
      return std::nullopt;
    }
    v = ins.getContainer();
  }
  return std::nullopt;
}

// Unpack CQ SMEX G2S operands for IXDL:
//   dst (Fly Shared) -> sPtr as !llvm.ptr<3>
//   src (SmeGmem fat ptr) -> gPtr / stride / gOffset
// Fail if address spaces do not match that contract, or if Shared did not
// lower to ptr<3> (type-converter invariant; do not addrspacecast over it).
// When stride_byte folds to a constant, it must be 16B-aligned: CQ SMEX
// hardware silently drops the low 4 bits of the global row stride.
LogicalResult prepareSmexPtrs(OpBuilder &builder, Location loc, Type srcMemTyArg, Type dstMemTyArg,
                              Value src, Value dst, Value &sPtr, Value &gPtr, Value &stride,
                              Value &gOffset) {
  auto srcMemTy = dyn_cast<fly::MemRefType>(srcMemTyArg);
  auto dstMemTy = dyn_cast<fly::MemRefType>(dstMemTyArg);
  if (!srcMemTy || !dstMemTy)
    return failure();
  if (!isTargetAddressSpace<SmeGmemAddressAttr>(srcMemTy.getAddressSpace()) ||
      !isGenericAddressSpace<fly::AddressSpace::Shared>(dstMemTy.getAddressSpace()))
    return failure();

  sPtr = dst;
  auto sPtrTy = dyn_cast<LLVM::LLVMPointerType>(sPtr.getType());
  if (!sPtrTy || sPtrTy.getAddressSpace() != 3)
    return mlir::emitError(loc) << "CQSmexCp dst must lower to !llvm.ptr<3>, got "
                                << sPtr.getType();

  if (std::optional<int64_t> strideByte = matchConstantSmeStrideByte(src)) {
    if (*strideByte % 16 != 0)
      return mlir::emitError(loc)
             << "CQ SMEX global row stride must be 16B-aligned (hardware silently "
             << "truncates the low 4 bits); got stride_byte=" << *strideByte
             << ", need a multiple of 16";
  }

  SmeGmemFatPtr srcFat(srcMemTy.getPointerType(), src);
  gPtr = srcFat.gmemPtr(builder, loc);
  stride = srcFat.strideByte(builder, loc);
  gOffset = srcFat.byteOffset(builder, loc);
  return success();
}

// Walk insertvalue chain for pred_enabled (slot 3). Default / absent => false.
bool isPredEnabled(Value atomVal) {
  Value v = atomVal;
  while (auto ins = v.getDefiningOp<LLVM::InsertValueOp>()) {
    ArrayRef<int64_t> pos = ins.getPosition();
    if (pos.size() == 1 && pos[0] == static_cast<int64_t>(kSmexPredEnabledSlot)) {
      APInt enabledAP;
      if (matchPattern(ins.getValue(), m_ConstantInt(&enabledAP)))
        return !enabledAP.isZero();
      // Non-constant enable bit: treat as enabled.
      return true;
    }
    v = ins.getContainer();
  }
  return false;
}

} // namespace

LogicalResult CopyOpCQSmexCpType::verify(function_ref<InFlightDiagnostic()> emitError, int32_t rows,
                                         SmexLayout layout) {
  if (rows != 4 && rows != 16 && rows != 64)
    return emitError() << "unsupported CQSmexCp rows = " << rows << " (expected 4, 16, or 64)";
  if (layout != SmexLayout::Plain && layout != SmexLayout::Mtx)
    return emitError() << "unsupported CQSmexCp layout";
  return success();
}

Type CopyOpCQSmexCpType::getConvertedType(MLIRContext *ctx) const {
  // (row_mask:i64, col_mask:i32, pred:i1, pred_enabled:i8)
  return LLVM::LLVMStructType::getLiteral(ctx,
                                          {IntegerType::get(ctx, 64), IntegerType::get(ctx, 32),
                                           IntegerType::get(ctx, 1), IntegerType::get(ctx, 8)});
}

Value CopyOpCQSmexCpType::getDefaultState(OpBuilder &builder, Location loc) const {
  auto structTy = cast<LLVM::LLVMStructType>(getConvertedType(builder.getContext()));
  Value state = LLVM::UndefOp::create(builder, loc, structTy);
  // All-1s masks: full tile. Pred disabled -> non-pred IXDL op.
  Value rowMask = arith::ConstantIntOp::create(builder, loc, -1, 64);
  Value colMask = arith::ConstantIntOp::create(builder, loc, -1, 32);
  Value pred = arith::ConstantIntOp::create(builder, loc, 1, 1);
  Value predEnabled = arith::ConstantIntOp::create(builder, loc, 0, 8);
  state = LLVM::InsertValueOp::create(builder, loc, state, rowMask,
                                      ArrayRef<int64_t>{kSmexRowMaskSlot});
  state = LLVM::InsertValueOp::create(builder, loc, state, colMask,
                                      ArrayRef<int64_t>{kSmexColMaskSlot});
  state = LLVM::InsertValueOp::create(builder, loc, state, pred, ArrayRef<int64_t>{kSmexPredSlot});
  state = LLVM::InsertValueOp::create(builder, loc, state, predEnabled,
                                      ArrayRef<int64_t>{kSmexPredEnabledSlot});
  return state;
}

Value CopyOpCQSmexCpType::setAtomState(OpBuilder &builder, Location loc, Value atomStruct,
                                       Attribute fieldAttr, Value fieldValue) const {
  auto fieldStr = dyn_cast<StringAttr>(fieldAttr);
  if (!fieldStr)
    return nullptr;
  auto idx = smexStateSlot(fieldStr.getValue());
  if (!idx)
    return nullptr;
  Value state =
      LLVM::InsertValueOp::create(builder, loc, atomStruct, fieldValue, ArrayRef<int64_t>{*idx});
  // Setting pred enables the predicated IXDL path.
  if (*idx == kSmexPredSlot) {
    Value enabled = arith::ConstantIntOp::create(builder, loc, 1, 8);
    state = LLVM::InsertValueOp::create(builder, loc, state, enabled,
                                        ArrayRef<int64_t>{kSmexPredEnabledSlot});
  }
  return state;
}

Attribute CopyOpCQSmexCpType::getThrLayout() const {
  // Warp-collective SMEX load: single logical thread owns the whole tile.
  return FxLayout(FxC(1), FxC(1));
}

Attribute CopyOpCQSmexCpType::getThrBitLayoutSrc() const {
  // rows x 512b per row (64B); 4 -> 2048, 16 -> 8192, 64 -> 32768 bits.
  int64_t bits = static_cast<int64_t>(getRows()) * 512;
  return FxLayout(FxShape(FxC(1), FxC(bits)), FxStride(FxC(0), FxC(1)));
}

Attribute CopyOpCQSmexCpType::getThrBitLayoutDst() const { return getThrBitLayoutSrc(); }

Attribute CopyOpCQSmexCpType::getThrBitLayoutRef() const { return getThrBitLayoutDst(); }

LogicalResult CopyOpCQSmexCpType::emitAtomCall(OpBuilder &builder, Location loc, Type copyAtomTyArg,
                                               Type srcMemTyArg, Type dstMemTyArg, Value atomVal,
                                               Value src, Value dst) const {
  auto copyAtomTy = dyn_cast<fly::CopyAtomType>(copyAtomTyArg);
  if (!copyAtomTy)
    return failure();

  Value sPtr, gPtr, stride, gOffset;
  if (failed(prepareSmexPtrs(builder, loc, srcMemTyArg, dstMemTyArg, src, dst, sPtr, gPtr, stride,
                             gOffset)))
    return failure();

  Value rowMask =
      LLVM::ExtractValueOp::create(builder, loc, atomVal, ArrayRef<int64_t>{kSmexRowMaskSlot});
  Value colMask =
      LLVM::ExtractValueOp::create(builder, loc, atomVal, ArrayRef<int64_t>{kSmexColMaskSlot});
  bool predEnabled = isPredEnabled(atomVal);
  Value pred;
  if (predEnabled)
    pred = LLVM::ExtractValueOp::create(builder, loc, atomVal, ArrayRef<int64_t>{kSmexPredSlot});

  return emitSmexIxdl(builder, loc, getRows(), getLayout(), predEnabled, sPtr, gPtr, stride,
                      gOffset, rowMask, colMask, pred);
}

LogicalResult CopyOpCQSmexCpType::emitAtomCall(OpBuilder &builder, Location loc, Type copyAtomTyArg,
                                               Type srcMemTyArg, Type dstMemTyArg, Type predTyArg,
                                               Value atomVal, Value src, Value dst,
                                               Value pred) const {
  auto copyAtomTy = dyn_cast<fly::CopyAtomType>(copyAtomTyArg);
  if (!copyAtomTy)
    return failure();

  auto predMemTy = dyn_cast<fly::MemRefType>(predTyArg);
  if (!predMemTy || !predMemTy.getElemTy().isInteger(1))
    return failure();

  Value sPtr, gPtr, stride, gOffset;
  if (failed(prepareSmexPtrs(builder, loc, srcMemTyArg, dstMemTyArg, src, dst, sPtr, gPtr, stride,
                             gOffset)))
    return failure();

  Value rowMask =
      LLVM::ExtractValueOp::create(builder, loc, atomVal, ArrayRef<int64_t>{kSmexRowMaskSlot});
  Value colMask =
      LLVM::ExtractValueOp::create(builder, loc, atomVal, ArrayRef<int64_t>{kSmexColMaskSlot});
  Value predValue = LLVM::LoadOp::create(builder, loc, predMemTy.getElemTy(), pred);
  return emitSmexIxdl(builder, loc, getRows(), getLayout(), /*predEnabled=*/true, sPtr, gPtr,
                      stride, gOffset, rowMask, colMask, predValue);
}

FailureOr<Value> CopyOpCQSmexCpType::emitAtomCallSSA(OpBuilder &builder, Location loc, Type,
                                                     Type copyAtomTyArg, Type srcTyArg,
                                                     Type dstTyArg, Value atomVal, Value src,
                                                     Value dst) const {
  if (failed(emitAtomCall(builder, loc, copyAtomTyArg, srcTyArg, dstTyArg, atomVal, src, dst)))
    return failure();
  return Value{};
}

FailureOr<Value> CopyOpCQSmexCpType::emitAtomCallSSA(OpBuilder &builder, Location loc, Type,
                                                     Type copyAtomTyArg, Type srcTyArg,
                                                     Type dstTyArg, Type predTyArg, Value atomVal,
                                                     Value src, Value dst, Value pred) const {
  if (failed(emitAtomCall(builder, loc, copyAtomTyArg, srcTyArg, dstTyArg, predTyArg, atomVal, src,
                          dst, pred)))
    return failure();
  return Value{};
}

} // namespace mlir::fly_ixdl
