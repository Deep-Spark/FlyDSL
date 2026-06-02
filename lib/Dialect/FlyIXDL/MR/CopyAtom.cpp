// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/IXDLDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/PointerUtils.h"
#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"
#include "flydsl/Dialect/FlyIXDL/IR/Dialect.h"
#include "flydsl/Dialect/FlyIXDL/Utils/CpAsyncUtils.h"

using namespace mlir;
using namespace mlir::fly;

namespace mlir::fly_ixdl {

//===----------------------------------------------------------------------===//
// CopyOpMR_AsyncCopy_16x64_b8_Row — stateful global->shared SME cp.async
//===----------------------------------------------------------------------===//

std::optional<unsigned>
CopyOpMRAsyncCopy16x64B8RowType::getFieldIndex(AtomStateField field) {
  switch (field) {
  case AtomStateField::Soffset:
    return 0;
  case AtomStateField::ImmOffset:
    return 1;
  default:
    return std::nullopt;
  }
}

Type CopyOpMRAsyncCopy16x64B8RowType::getConvertedType(MLIRContext *ctx) const {
  auto i32Ty = IntegerType::get(ctx, 32);
  return LLVM::LLVMStructType::getLiteral(ctx, {i32Ty, i32Ty});
}

Value CopyOpMRAsyncCopy16x64B8RowType::getDefaultState(OpBuilder &builder, Location loc) const {
  auto structTy = cast<LLVM::LLVMStructType>(getConvertedType(builder.getContext()));
  Value state = LLVM::UndefOp::create(builder, loc, structTy);
  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  state = LLVM::InsertValueOp::create(builder, loc, state, zero,
                                      ArrayRef<int64_t>{*getFieldIndex(AtomStateField::Soffset)});
  state = LLVM::InsertValueOp::create(builder, loc, state, zero,
                                      ArrayRef<int64_t>{*getFieldIndex(AtomStateField::ImmOffset)});
  return state;
}

Value CopyOpMRAsyncCopy16x64B8RowType::setAtomState(OpBuilder &builder, Location loc,
                                                    Value atomStruct, Attribute fieldAttr,
                                                    Value fieldValue) const {
  auto fieldStr = dyn_cast<StringAttr>(fieldAttr);
  if (!fieldStr)
    return nullptr;
  auto field = symbolizeAtomStateField(fieldStr.getValue());
  if (!field)
    return nullptr;
  auto idx = getFieldIndex(*field);
  if (!idx)
    return nullptr;
  return LLVM::InsertValueOp::create(builder, loc, atomStruct, fieldValue, ArrayRef<int64_t>{*idx});
}

Attribute CopyOpMRAsyncCopy16x64B8RowType::getThrLayout() const { return FxLayout(FxC(1), FxC(1)); }

Attribute CopyOpMRAsyncCopy16x64B8RowType::getThrBitLayoutSrc() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpMRAsyncCopy16x64B8RowType::getThrBitLayoutDst() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpMRAsyncCopy16x64B8RowType::getThrBitLayoutRef() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)));
}

LogicalResult CopyOpMRAsyncCopy16x64B8RowType::emitAtomCall(OpBuilder &builder, Location loc,
                                                            Type copyAtomTyArg, Type srcMemTyArg,
                                                            Type dstMemTyArg, Value atomVal,
                                                            Value src, Value dst) const {
  auto srcMemTy = dyn_cast<fly::MemRefType>(srcMemTyArg);
  auto dstMemTy = dyn_cast<fly::MemRefType>(dstMemTyArg);
  if (!srcMemTy || !dstMemTy)
    return failure();

  // Address-space contract: global src -> shared dst. No silent fallback.
  if (!isGenericAddressSpace<fly::AddressSpace::Global>(srcMemTy.getAddressSpace()) ||
      !isGenericAddressSpace<fly::AddressSpace::Shared>(dstMemTy.getAddressSpace()))
    return failure();

  Value immOffset = LLVM::ExtractValueOp::create(
      builder, loc, atomVal, ArrayRef<int64_t>{*getFieldIndex(AtomStateField::ImmOffset)});

  SmeCpAsyncOperands ops =
      buildSmeCpAsyncOperands(builder, loc, srcMemTy, dstMemTy, src, dst, immOffset);

  // f16 row-major operand (A): 16x32 b16 row transfer -> SME `rowxfb16`
  // (`bi.sme.load.16x1b64.rowxfb16`). The vendor `convert-gpu-to-ixdl` selects
  // the intrinsic from (shape=[row,col], elementSize, transpose):
  //   key(16,32,16,false) -> CpAsync_16x32_b16_RowOp -> rowxfb16.
  // This must match the Row16b SmemAtom that `make_smem_tile` builds (XOR
  // swizzle (1,6,2)-element); the previous [16,64]/es=8/false emitted `rowxfb8`
  // (Swizzle_Mod), which mismatched the layout and produced NaN fragments.
  ArrayAttr shapeAttr = builder.getI64ArrayAttr({16, 32});
  IXDL::CpAsyncOp::create(builder, loc, ops.sOffset, ops.gBase, ops.gOffset, ops.kop, shapeAttr,
                          /*elementSize=*/16u, /*transpose=*/false);
  return success();
}

LogicalResult CopyOpMRAsyncCopy16x64B8RowType::emitAtomCall(OpBuilder &builder, Location loc,
                                                            Type copyAtomTyArg, Type srcMemTyArg,
                                                            Type dstMemTyArg, Type predMemTyArg,
                                                            Value atomVal, Value src, Value dst,
                                                            Value pred) const {
  OpBuilder::InsertionGuard guard(builder);
  auto predMemTy = cast<fly::MemRefType>(predMemTyArg);
  Value predVal = LLVM::LoadOp::create(builder, loc, predMemTy.getElemTy(), pred);
  auto ifOp = scf::IfOp::create(builder, loc, TypeRange{}, predVal, /*withElse=*/false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  return emitAtomCall(builder, loc, copyAtomTyArg, srcMemTyArg, dstMemTyArg, atomVal, src, dst);
}

FailureOr<Value> CopyOpMRAsyncCopy16x64B8RowType::emitAtomCallSSA(
    OpBuilder &builder, Location loc, Type resultTy, Type copyAtomTyArg, Type srcTyArg,
    Type dstTyArg, Value atomVal, Value src, Value dst) const {
  if (failed(emitAtomCall(builder, loc, copyAtomTyArg, srcTyArg, dstTyArg, atomVal, src, dst)))
    return failure();
  return Value{};
}

FailureOr<Value> CopyOpMRAsyncCopy16x64B8RowType::emitAtomCallSSA(
    OpBuilder &builder, Location loc, Type resultTy, Type copyAtomTyArg, Type srcTyArg,
    Type dstTyArg, Type predTyArg, Value atomVal, Value src, Value dst, Value pred) const {
  if (failed(emitAtomCall(builder, loc, copyAtomTyArg, srcTyArg, dstTyArg, predTyArg, atomVal, src,
                          dst, pred)))
    return failure();
  return Value{};
}

//===----------------------------------------------------------------------===//
// CopyOpMR_SLBLoad — shared->register fragment load (lowers to llvm.load)
//===----------------------------------------------------------------------===//

bool CopyOpMRSLBLoadType::isStatic() const { return true; }

Value CopyOpMRSLBLoadType::rebuildStaticValue(OpBuilder &builder, Location loc,
                                              Value currentValue) const {
  if (currentValue && isa<MakeCopyAtomOp>(currentValue.getDefiningOp()))
    return nullptr;
  return MakeCopyAtomOp::create(builder, loc, CopyAtomType::get(*this, getBitSize()), getBitSize());
}

Attribute CopyOpMRSLBLoadType::getThrLayout() const { return FxLayout(FxC(1), FxC(1)); }

Attribute CopyOpMRSLBLoadType::getThrBitLayoutSrc() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpMRSLBLoadType::getThrBitLayoutDst() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpMRSLBLoadType::getThrBitLayoutRef() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)));
}

FailureOr<Value> CopyOpMRSLBLoadType::emitAtomCallSSA(OpBuilder &builder, Location loc,
                                                      Type resultTy, Type copyAtomTyArg,
                                                      Type srcTyArg, Type dstTyArg, Value atomVal,
                                                      Value src, Value dst) const {
  if (!resultTy)
    return failure();
  Value loaded = LLVM::LoadOp::create(builder, loc, resultTy, src);
  return loaded;
}

FailureOr<Value> CopyOpMRSLBLoadType::emitAtomCallSSA(OpBuilder &builder, Location loc,
                                                      Type resultTy, Type copyAtomTyArg,
                                                      Type srcTyArg, Type dstTyArg, Type predTyArg,
                                                      Value atomVal, Value src, Value dst,
                                                      Value pred) const {
  assert(resultTy && "resultTy must be SSA Type");
  OpBuilder::InsertionGuard guard(builder);
  auto ifOp = scf::IfOp::create(builder, loc, resultTy, pred, /*withElseRegion=*/true);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  auto result =
      emitAtomCallSSA(builder, loc, resultTy, copyAtomTyArg, srcTyArg, dstTyArg, atomVal, src, dst);
  if (failed(result))
    return failure();
  scf::YieldOp::create(builder, loc, *result);
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  scf::YieldOp::create(builder, loc, dst);
  return ifOp.getResult(0);
}

LogicalResult CopyOpMRSLBLoadType::emitAtomCall(OpBuilder &builder, Location loc, Type copyAtomTyArg,
                                                Type srcMemTyArg, Type dstMemTyArg, Value atomVal,
                                                Value src, Value dst) const {
  auto dstSSATy = fly::RegMem2SSAType(cast<fly::MemRefType>(dstMemTyArg), /*llvmCompatible=*/true);
  auto res = emitAtomCallSSA(builder, loc, dstSSATy, copyAtomTyArg, srcMemTyArg, Type{}, atomVal,
                             src, Value{});
  if (failed(res))
    return failure();
  LLVM::StoreOp::create(builder, loc, *res, dst);
  return success();
}

LogicalResult CopyOpMRSLBLoadType::emitAtomCall(OpBuilder &builder, Location loc, Type copyAtomTyArg,
                                                Type srcMemTyArg, Type dstMemTyArg,
                                                Type predMemTyArg, Value atomVal, Value src,
                                                Value dst, Value pred) const {
  OpBuilder::InsertionGuard guard(builder);
  auto predMemTy = cast<fly::MemRefType>(predMemTyArg);
  Value predVal = LLVM::LoadOp::create(builder, loc, predMemTy.getElemTy(), pred);
  auto ifOp = scf::IfOp::create(builder, loc, TypeRange{}, predVal, /*withElse=*/false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  return emitAtomCall(builder, loc, copyAtomTyArg, srcMemTyArg, dstMemTyArg, atomVal, src, dst);
}

} // namespace mlir::fly_ixdl
