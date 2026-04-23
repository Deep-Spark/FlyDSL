// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/PointerUtils.h"
#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"
#include "flydsl/Dialect/FlyCQ/IR/Dialect.h"

using namespace mlir;
using namespace mlir::fly;

namespace mlir::fly_cq {

bool CopyOpCQ_ScalarMemType::isStatic() const { return true; }

Value CopyOpCQ_ScalarMemType::rebuildStaticValue(OpBuilder &builder, Location loc,
                                                 Value currentValue) const {
  if (currentValue && isa<MakeCopyAtomOp>(currentValue.getDefiningOp()))
    return nullptr;
  return MakeCopyAtomOp::create(builder, loc, CopyAtomType::get(*this, getBitSize()), getBitSize());
}

Attribute CopyOpCQ_ScalarMemType::getThrLayout() const { return FxLayout(FxC(1), FxC(1)); }

Attribute CopyOpCQ_ScalarMemType::getThrBitLayoutSrc() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpCQ_ScalarMemType::getThrBitLayoutDst() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpCQ_ScalarMemType::getThrBitLayoutRef() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())), FxStride(FxC(1), FxC(1)));
}

LogicalResult CopyOpCQ_ScalarMemType::verify(function_ref<InFlightDiagnostic()> emitError,
                                             int32_t bitSize) {
  if (bitSize != 32)
    return emitError() << "CQ placeholder scalar_mem only supports bitSize=32, got " << bitSize;
  return success();
}

static Type loadTyFromSrcMem(Type srcTyArg) {
  if (!srcTyArg)
    return {};
  auto srcMem = dyn_cast<fly::MemRefType>(srcTyArg);
  if (!srcMem)
    return {};
  return fly::RegMem2SSAType(srcMem, /*llvmCompatibleType=*/true);
}

FailureOr<Value> CopyOpCQ_ScalarMemType::emitAtomCallSSA(OpBuilder &builder, Location loc,
                                                       Type resultTy, Type copyAtomTyArg,
                                                       Type srcTyArg, Type dstTyArg, Value atomVal,
                                                       Value src, Value dst) const {
  (void)copyAtomTyArg;
  (void)atomVal;
  (void)dstTyArg;
  Type loadTy = loadTyFromSrcMem(srcTyArg);
  if (!loadTy)
    return failure();

  Value loaded = LLVM::LoadOp::create(builder, loc, loadTy, src);
  if (resultTy && loaded.getType() != resultTy)
    loaded = LLVM::BitcastOp::create(builder, loc, resultTy, loaded);
  if (dst)
    LLVM::StoreOp::create(builder, loc, loaded, dst);
  return loaded;
}

FailureOr<Value> CopyOpCQ_ScalarMemType::emitAtomCallSSA(OpBuilder &builder, Location loc,
                                                       Type resultTy, Type copyAtomTyArg,
                                                       Type srcTyArg, Type dstTyArg, Type predTyArg,
                                                       Value atomVal, Value src, Value dst,
                                                       Value pred) const {
  assert(resultTy && "resultTy must be SSA Type for predicated copy_atom_call_ssa");
  OpBuilder::InsertionGuard guard(builder);
  auto predMemTy = cast<fly::MemRefType>(predTyArg);
  Value predVal = LLVM::LoadOp::create(builder, loc, predMemTy.getElemTy(), pred);
  auto ifOp = scf::IfOp::create(builder, loc, resultTy, predVal, /*withElseRegion=*/true);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  auto inner = emitAtomCallSSA(builder, loc, resultTy, copyAtomTyArg, srcTyArg, dstTyArg, atomVal,
                               src, dst);
  if (failed(inner))
    return failure();
  scf::YieldOp::create(builder, loc, *inner);

  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  scf::YieldOp::create(builder, loc, dst);
  return ifOp.getResult(0);
}

LogicalResult CopyOpCQ_ScalarMemType::emitAtomCall(OpBuilder &builder, Location loc,
                                                   Type copyAtomTyArg, Type srcMemTyArg,
                                                   Type dstMemTyArg, Value atomVal, Value src,
                                                   Value dst) const {
  (void)srcMemTyArg;
  auto dstMemTy = cast<fly::MemRefType>(dstMemTyArg);
  auto dstSSATy = fly::RegMem2SSAType(dstMemTy, /*llvmCompatibleType=*/true);
  auto res = emitAtomCallSSA(builder, loc, dstSSATy, copyAtomTyArg, srcMemTyArg, Type{}, atomVal,
                             src, Value{});
  if (failed(res))
    return failure();
  LLVM::StoreOp::create(builder, loc, *res, dst);
  return success();
}

LogicalResult CopyOpCQ_ScalarMemType::emitAtomCall(OpBuilder &builder, Location loc,
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

} // namespace mlir::fly_cq
