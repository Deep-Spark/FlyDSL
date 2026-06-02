// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/LLVMIR/IXDLDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"
#include "flydsl/Dialect/FlyIXDL/IR/Dialect.h"

using namespace mlir;
using namespace mlir::fly;

namespace mlir::fly_ixdl {

//===----------------------------------------------------------------------===//
// MmaOpMR_MMAD — 16x16x16 FP16 SME MMAD (warp size 64)
//===----------------------------------------------------------------------===//

bool MmaOpMR_MMADType::isStatic() const { return true; }

Value MmaOpMR_MMADType::rebuildStaticValue(OpBuilder &builder, Location loc,
                                           Value currentValue) const {
  if (currentValue && isa<MakeMmaAtomOp>(currentValue.getDefiningOp()))
    return nullptr;
  return MakeMmaAtomOp::create(builder, loc, MmaAtomType::get(*this));
}

Attribute MmaOpMR_MMADType::getThrLayout() const { return FxLayout(FxC(64), FxC(1)); }

Attribute MmaOpMR_MMADType::getShapeMNK() const {
  return IntTupleAttr::get(ArrayAttr::get(getContext(), {FxC(getM()), FxC(getN()), FxC(getK())}));
}

Type MmaOpMR_MMADType::getValTypeA() const { return getElemTyA(); }
Type MmaOpMR_MMADType::getValTypeB() const { return getElemTyB(); }
Type MmaOpMR_MMADType::getValTypeC() const { return getElemTyAcc(); }
Type MmaOpMR_MMADType::getValTypeD() const { return getElemTyAcc(); }

// ThrValLayouts are strictly per the ivcore11 SME repro (§3). B's stride
// differs from A's (N-major), which is the most common correctness pitfall.
Attribute MmaOpMR_MMADType::getThrValLayoutA() const {
  // Shape Thr(16,4) Val(2,2), Stride Thr(16,2) Val(1,8)
  return FxLayout(FxShape(FxThr(16, 4), FxVal(2, 2)), FxStride(FxThr(16, 2), FxVal(1, 8)));
}
Attribute MmaOpMR_MMADType::getThrValLayoutB() const {
  // Shape Thr(16,4) Val(2,2), Stride Thr(1,32) Val(16,128)
  return FxLayout(FxShape(FxThr(16, 4), FxVal(2, 2)), FxStride(FxThr(1, 32), FxVal(16, 128)));
}
Attribute MmaOpMR_MMADType::getThrValLayoutC() const {
  // Shape Thr(16,4) Val(4), Stride Thr(16,1) Val(4)
  return FxLayout(FxShape(FxThr(16, 4), FxVal(4)), FxStride(FxThr(16, 1), FxVal(4)));
}

LogicalResult MmaOpMR_MMADType::verify(function_ref<InFlightDiagnostic()> emitError, int32_t m,
                                       int32_t n, int32_t k, Type elemTyA, Type elemTyB,
                                       Type elemTyAcc) {
  if (m != 16 || n != 16 || k != 16)
    return emitError() << "MR MMAD only supports 16x16x16, got " << m << "x" << n << "x" << k;
  if (!elemTyA.isF16() || !elemTyB.isF16())
    return emitError() << "MR MMAD operands A/B must be f16";
  if (!elemTyAcc.isF32())
    return emitError() << "MR MMAD accumulator must be f32";
  return success();
}

FailureOr<Value> MmaOpMR_MMADType::emitAtomCallSSA(OpBuilder &builder, Location loc, Type resultTy,
                                                   Type mmaAtomTyArg, Type dTyArg, Type aTyArg,
                                                   Type bTyArg, Type cTyArg, Value atomVal, Value d,
                                                   Value a, Value b, Value c) const {
  MLIRContext *ctx = builder.getContext();
  VectorType abTy = VectorType::get({4}, Float16Type::get(ctx));
  VectorType accTy = VectorType::get({4}, Float32Type::get(ctx));

  if (a.getType() != abTy)
    a = LLVM::BitcastOp::create(builder, loc, abTy, a);
  if (b.getType() != abTy)
    b = LLVM::BitcastOp::create(builder, loc, abTy, b);
  if (c.getType() != accTy)
    c = LLVM::BitcastOp::create(builder, loc, accTy, c);

  std::optional<std::array<IXDL::MMADTypes, 2>> multiplicandTypes(
      {IXDL::MMADTypes::f16, IXDL::MMADTypes::f16});
  std::optional<std::array<IXDL::MMADLayout, 2>> multiplicandLayouts(
      {IXDL::MMADLayout::row, IXDL::MMADLayout::col});

  auto mmad = IXDL::MmadOp::create(builder, loc, accTy, ValueRange{a}, ValueRange{b}, ValueRange{c},
                                   ArrayRef<int64_t>{getM(), getN(), getK()}, multiplicandTypes,
                                   multiplicandLayouts);
  Value res = mmad.getRes();
  if (resultTy && res.getType() != resultTy)
    res = LLVM::BitcastOp::create(builder, loc, resultTy, res);
  return res;
}

LogicalResult MmaOpMR_MMADType::emitAtomCall(OpBuilder &builder, Location loc, Type mmaAtomTy,
                                             Type dMemTy, Type aMemTy, Type bMemTy, Type cMemTy,
                                             Value atomVal, Value dPtr, Value aPtr, Value bPtr,
                                             Value cPtr) const {
  MLIRContext *ctx = builder.getContext();
  VectorType abTy = VectorType::get({4}, Float16Type::get(ctx));
  VectorType accTy = VectorType::get({4}, Float32Type::get(ctx));

  Value a = LLVM::LoadOp::create(builder, loc, abTy, aPtr);
  Value b = LLVM::LoadOp::create(builder, loc, abTy, bPtr);
  Value c = LLVM::LoadOp::create(builder, loc, accTy, cPtr);
  auto res = emitAtomCallSSA(builder, loc, accTy, mmaAtomTy, Type{}, abTy, abTy, accTy, atomVal,
                             Value{}, a, b, c);
  if (failed(res))
    return failure();
  LLVM::StoreOp::create(builder, loc, *res, dPtr);
  return success();
}

} // namespace mlir::fly_ixdl
