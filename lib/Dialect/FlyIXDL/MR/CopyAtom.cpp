// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"
#include "flydsl/Dialect/FlyIXDL/IR/Dialect.h"
#include "flydsl/Dialect/FlyIXDL/Utils/CpAsyncUtils.h"

using namespace mlir;
using namespace mlir::fly;
using namespace mlir::fly_ixdl;

namespace {

std::optional<unsigned> sharedFieldIndex(AtomStateField field) {
  switch (field) {
  case AtomStateField::Soffset:
    return 0;
  case AtomStateField::ImmOffset:
    return 1;
  case AtomStateField::StrideByte:
    return 2;
  default:
    return std::nullopt;
  }
}

Type stateStructType(MLIRContext *ctx) {
  return LLVM::LLVMStructType::getLiteral(ctx, {IntegerType::get(ctx, 32),
                                                IntegerType::get(ctx, 32),
                                                IntegerType::get(ctx, 32)});
}

bool isGlobalSrc(LLVM::LLVMPointerType ptrTy) { return ptrTy.getAddressSpace() == 1; }

bool isSharedDst(LLVM::LLVMPointerType ptrTy) { return ptrTy.getAddressSpace() == 3; }

LogicalResult emitMRAsyncCopyG2S(OpBuilder &builder, Location loc, fly::MemRefType srcMemTy,
                                 fly::MemRefType dstMemTy, Value atomVal, Value src, Value dst,
                                 MRCpAsyncKind kind) {
  auto srcPtrTy = dyn_cast<LLVM::LLVMPointerType>(src.getType());
  auto dstPtrTy = dyn_cast<LLVM::LLVMPointerType>(dst.getType());
  if (!srcPtrTy || !dstPtrTy)
    return failure();
  if (!isGlobalSrc(srcPtrTy) || !isSharedDst(dstPtrTy))
    return failure();
  if (!isGenericAddressSpace<AddressSpace::Global>(srcMemTy.getAddressSpace()) ||
      !isGenericAddressSpace<AddressSpace::Shared>(dstMemTy.getAddressSpace()))
    return failure();

  Value soffsetRaw = LLVM::ExtractValueOp::create(
      builder, loc, atomVal, ArrayRef<int64_t>{*sharedFieldIndex(AtomStateField::Soffset)});
  Value immOffset = LLVM::ExtractValueOp::create(
      builder, loc, atomVal, ArrayRef<int64_t>{*sharedFieldIndex(AtomStateField::ImmOffset)});
  Value strideFromAtom = LLVM::ExtractValueOp::create(
      builder, loc, atomVal, ArrayRef<int64_t>{*sharedFieldIndex(AtomStateField::StrideByte)});

  int64_t elemBits = srcMemTy.getElemTy().getIntOrFloatBitWidth();
  Value scaledSoffset = scaleSoffsetToBytes(builder, loc, soffsetRaw, elemBits);
  Value sOffset = buildSOffset(builder, loc, dst, immOffset);

  llvm::APInt strideConst;
  bool hasConstStrideFromAtom = matchPattern(strideFromAtom, m_ConstantInt(&strideConst));
  if (hasConstStrideFromAtom && strideConst.isNegative()) {
    emitError(loc) << "MR async copy requires non-negative stride_byte";
    return failure();
  }
  if (hasConstStrideFromAtom && strideConst != 0 && strideConst.urem(64) != 0) {
    emitError(loc) << "MR async copy requires 64B-aligned stride_byte";
    return failure();
  }

  Value staticStrideBytes = buildStrideBytesFromMemref(builder, loc, srcMemTy, kind);
  Value strideBytes = Value();
  if (staticStrideBytes) {
    // `stride_byte = 0` keeps layout-derived pitch; non-zero uses runtime override.
    Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
    Value hasRuntimeStride =
        arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne, strideFromAtom, zero);
    strideBytes = arith::SelectOp::create(builder, loc, hasRuntimeStride, strideFromAtom,
                                          staticStrideBytes);
  } else {
    if (hasConstStrideFromAtom && strideConst == 0) {
      emitError(loc) << "MR async copy requires either static src layout stride or non-zero "
                     << "stride_byte override";
      return failure();
    }
    strideBytes = strideFromAtom;
  }

  Value gBase = buildGBase(builder, loc, src, scaledSoffset, strideBytes);
  Value gOffset = buildGOffsetZero(builder, loc);
  Value kop = buildKopOne(builder, loc);

  return emitCpAsyncByKind(builder, loc, kind, sOffset, gBase, gOffset, kop);
}

struct StatefulCopyOpMethods {
  static std::optional<unsigned> getFieldIndex(AtomStateField field) {
    return sharedFieldIndex(field);
  }

  static Type getConvertedType(MLIRContext *ctx) { return stateStructType(ctx); }

  static Value getDefaultState(OpBuilder &builder, Location loc) {
    auto structTy = cast<LLVM::LLVMStructType>(stateStructType(builder.getContext()));
    Value state = LLVM::UndefOp::create(builder, loc, structTy);
    Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
    state = LLVM::InsertValueOp::create(builder, loc, state, zero, ArrayRef<int64_t>{0});
    state = LLVM::InsertValueOp::create(builder, loc, state, zero, ArrayRef<int64_t>{1});
    return LLVM::InsertValueOp::create(builder, loc, state, zero, ArrayRef<int64_t>{2});
  }

  static Value setAtomState(OpBuilder &builder, Location loc, Value atomStruct,
                            Attribute fieldAttr, Value fieldValue) {
    auto fieldStr = dyn_cast<StringAttr>(fieldAttr);
    if (!fieldStr)
      return nullptr;
    auto field = symbolizeAtomStateField(fieldStr.getValue());
    if (!field)
      return nullptr;
    auto idx = sharedFieldIndex(*field);
    if (!idx)
      return nullptr;
    return LLVM::InsertValueOp::create(builder, loc, atomStruct, fieldValue,
                                      ArrayRef<int64_t>{*idx});
  }

  static LogicalResult emitAtomCallPredicated(
      function_ref<LogicalResult(OpBuilder &, Location, Type, Type, Type, Value, Value, Value)>
          emitFn,
      OpBuilder &builder, Location loc, Type copyAtomTyArg, Type srcMemTyArg, Type dstMemTyArg,
      Type predMemTyArg, Value atomVal, Value src, Value dst, Value pred) {
    OpBuilder::InsertionGuard guard(builder);
    auto predMemTy = cast<fly::MemRefType>(predMemTyArg);
    Value predVal = LLVM::LoadOp::create(builder, loc, predMemTy.getElemTy(), pred);
    auto ifOp = scf::IfOp::create(builder, loc, TypeRange{}, predVal, /*withElse=*/false);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    return emitFn(builder, loc, copyAtomTyArg, srcMemTyArg, dstMemTyArg, atomVal, src, dst);
  }
};

} // namespace

#define DEFINE_MR_ASYNC_COPY_TYPE(TYPE, KIND)                                              \
  std::optional<unsigned> TYPE::getFieldIndex(AtomStateField field) {                      \
    return StatefulCopyOpMethods::getFieldIndex(field);                                    \
  }                                                                                        \
  Type TYPE::getConvertedType(MLIRContext *ctx) const {                                    \
    return StatefulCopyOpMethods::getConvertedType(ctx);                                   \
  }                                                                                        \
  Value TYPE::getDefaultState(OpBuilder &builder, Location loc) const {                    \
    return StatefulCopyOpMethods::getDefaultState(builder, loc);                           \
  }                                                                                        \
  Value TYPE::setAtomState(OpBuilder &builder, Location loc, Value atomStruct,             \
                           Attribute fieldAttr, Value fieldValue) const {                  \
    return StatefulCopyOpMethods::setAtomState(builder, loc, atomStruct, fieldAttr,        \
                                               fieldValue);                                \
  }                                                                                        \
  Attribute TYPE::getThrLayout() const { return FxLayout(FxC(64), FxC(1)); }               \
  Attribute TYPE::getThrBitLayoutSrc() const {                                             \
    int32_t bitsPerLane = getBitSize() / 64;                                                \
    return FxLayout(FxShape(FxC(64), FxC(bitsPerLane)), FxStride(FxC(bitsPerLane), FxC(1))); \
  }                                                                                        \
  Attribute TYPE::getThrBitLayoutDst() const {                                             \
    int32_t bitsPerLane = getBitSize() / 64;                                                \
    return FxLayout(FxShape(FxC(64), FxC(bitsPerLane)), FxStride(FxC(bitsPerLane), FxC(1))); \
  }                                                                                        \
  Attribute TYPE::getThrBitLayoutRef() const {                                             \
    int32_t bitsPerLane = getBitSize() / 64;                                                \
    return FxLayout(FxShape(FxC(64), FxC(bitsPerLane)), FxStride(FxC(bitsPerLane), FxC(1))); \
  }                                                                                        \
  LogicalResult TYPE::emitAtomCall(OpBuilder &builder, Location loc, Type copyAtomTyArg,   \
                                   Type srcMemTyArg, Type dstMemTyArg, Value atomVal,      \
                                   Value src, Value dst) const {                           \
    return emitMRAsyncCopyG2S(builder, loc, cast<fly::MemRefType>(srcMemTyArg),            \
                            cast<fly::MemRefType>(dstMemTyArg), atomVal, src, dst, KIND);  \
  }                                                                                        \
  LogicalResult TYPE::emitAtomCall(OpBuilder &builder, Location loc, Type copyAtomTyArg,   \
                                   Type srcMemTyArg, Type dstMemTyArg, Type predMemTyArg,  \
                                   Value atomVal, Value src, Value dst, Value pred)        \
      const {                                                                              \
    return StatefulCopyOpMethods::emitAtomCallPredicated(                                  \
        [&](OpBuilder &b, Location l, Type a, Type s, Type d, Value av, Value sr,          \
            Value ds) { return emitAtomCall(b, l, a, s, d, av, sr, ds); },                 \
        builder, loc, copyAtomTyArg, srcMemTyArg, dstMemTyArg, predMemTyArg, atomVal, src, \
        dst, pred);                                                                        \
  }                                                                                        \
  FailureOr<Value> TYPE::emitAtomCallSSA(OpBuilder &builder, Location loc, Type resultTy,  \
                                         Type copyAtomTyArg, Type srcTyArg, Type dstTyArg, \
                                         Value atomVal, Value src, Value dst) const {      \
    (void)resultTy;                                                                        \
    (void)copyAtomTyArg;                                                                   \
    (void)srcTyArg;                                                                        \
    (void)dstTyArg;                                                                        \
    (void)atomVal;                                                                         \
    (void)src;                                                                             \
    (void)dst;                                                                             \
    return failure();                                                                      \
  }                                                                                        \
  FailureOr<Value> TYPE::emitAtomCallSSA(OpBuilder &builder, Location loc, Type resultTy,  \
                                         Type copyAtomTyArg, Type srcTyArg, Type dstTyArg, \
                                         Type predTyArg, Value atomVal, Value src,         \
                                         Value dst, Value pred) const {                    \
    (void)resultTy;                                                                        \
    (void)copyAtomTyArg;                                                                   \
    (void)srcTyArg;                                                                        \
    (void)dstTyArg;                                                                        \
    (void)predTyArg;                                                                       \
    (void)atomVal;                                                                         \
    (void)src;                                                                             \
    (void)dst;                                                                             \
    (void)pred;                                                                            \
    return failure();                                                                      \
  }

DEFINE_MR_ASYNC_COPY_TYPE(CopyOpMRAsyncCopy4x64B8RowType, MRCpAsyncKind::k4x64_b8_Row)
DEFINE_MR_ASYNC_COPY_TYPE(CopyOpMRAsyncCopy4x64B8ColType, MRCpAsyncKind::k4x64_b8_Col)
DEFINE_MR_ASYNC_COPY_TYPE(CopyOpMRAsyncCopy16x64B8RowType, MRCpAsyncKind::k16x64_b8_Row)
DEFINE_MR_ASYNC_COPY_TYPE(CopyOpMRAsyncCopy16x64B8ColType, MRCpAsyncKind::k16x64_b8_Col)
DEFINE_MR_ASYNC_COPY_TYPE(CopyOpMRAsyncCopy4x32B16RowType, MRCpAsyncKind::k4x32_b16_Row)
DEFINE_MR_ASYNC_COPY_TYPE(CopyOpMRAsyncCopy4x32B16ColType, MRCpAsyncKind::k4x32_b16_Col)
DEFINE_MR_ASYNC_COPY_TYPE(CopyOpMRAsyncCopy16x32B16RowType, MRCpAsyncKind::k16x32_b16_Row)
DEFINE_MR_ASYNC_COPY_TYPE(CopyOpMRAsyncCopy16x32B16ColType, MRCpAsyncKind::k16x32_b16_Col)
DEFINE_MR_ASYNC_COPY_TYPE(CopyOpMRAsyncCopy1x1B64Type, MRCpAsyncKind::k1x1b64)
DEFINE_MR_ASYNC_COPY_TYPE(CopyOpMRAsyncCopy1x4B64Type, MRCpAsyncKind::k1x4b64)
DEFINE_MR_ASYNC_COPY_TYPE(CopyOpMRAsyncCopy1x8B64Type, MRCpAsyncKind::k1x8b64)
DEFINE_MR_ASYNC_COPY_TYPE(CopyOpMRAsyncCopy1x16B64Type, MRCpAsyncKind::k1x16b64)
DEFINE_MR_ASYNC_COPY_TYPE(CopyOpMRAsyncCopy4x16B32RowType, MRCpAsyncKind::k4x16_b32_Row)
DEFINE_MR_ASYNC_COPY_TYPE(CopyOpMRAsyncCopy8x16B32RowType, MRCpAsyncKind::k8x16_b32_Row)
DEFINE_MR_ASYNC_COPY_TYPE(CopyOpMRAsyncCopy16x16B32RowType, MRCpAsyncKind::k16x16_b32_Row)
DEFINE_MR_ASYNC_COPY_TYPE(CopyOpMRAsyncCopy16x16B32ColType, MRCpAsyncKind::k16x16_b32_Col)

#undef DEFINE_MR_ASYNC_COPY_TYPE
