// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"
#include "flydsl/Dialect/FlyCQ/IR/Dialect.h"

using namespace mlir;
using namespace mlir::fly;

namespace mlir::fly_cq {

bool MmaOpCQ_MatmulF32Type::isStatic() const { return true; }

Value MmaOpCQ_MatmulF32Type::rebuildStaticValue(OpBuilder &builder, Location loc,
                                                  Value currentValue) const {
  if (currentValue && isa<MakeMmaAtomOp>(currentValue.getDefiningOp()))
    return nullptr;
  return MakeMmaAtomOp::create(builder, loc, MmaAtomType::get(*this));
}

Attribute MmaOpCQ_MatmulF32Type::getThrLayout() const { return FxLayout(FxC(64), FxC(1)); }

Attribute MmaOpCQ_MatmulF32Type::getShapeMNK() const {
  return IntTupleAttr::get(ArrayAttr::get(getContext(), {FxC(getM()), FxC(getN()), FxC(getK())}));
}

Type MmaOpCQ_MatmulF32Type::getValTypeA() const { return getElemTyA(); }
Type MmaOpCQ_MatmulF32Type::getValTypeB() const { return getElemTyB(); }
Type MmaOpCQ_MatmulF32Type::getValTypeC() const { return getElemTyAcc(); }
Type MmaOpCQ_MatmulF32Type::getValTypeD() const { return getElemTyAcc(); }

Attribute MmaOpCQ_MatmulF32Type::getThrValLayoutA() const {
  return FxLayout(FxShape(FxThr(16, 4), FxVal(1)), FxStride(FxThr(1, 16), FxVal(1)));
}
Attribute MmaOpCQ_MatmulF32Type::getThrValLayoutB() const {
  return FxLayout(FxShape(FxThr(16, 4), FxVal(1)), FxStride(FxThr(1, 16), FxVal(1)));
}
Attribute MmaOpCQ_MatmulF32Type::getThrValLayoutC() const {
  return FxLayout(FxShape(FxThr(16, 4), FxVal(1)), FxStride(FxThr(1, 4), FxVal(1)));
}

LogicalResult MmaOpCQ_MatmulF32Type::verify(function_ref<InFlightDiagnostic()> emitError, int32_t m,
                                           int32_t n, int32_t k, Type elemTyA, Type elemTyB,
                                           Type elemTyAcc) {
  if (m != 16 || n != 16 || k != 4)
    return emitError() << "CQ placeholder matmul_f32 only supports 16x16x4, got " << m << "x"
                       << n << "x" << k;
  if (!elemTyA.isF32() || !elemTyB.isF32() || !elemTyAcc.isF32())
    return emitError() << "CQ placeholder matmul_f32 only supports f32 element types";
  return success();
}

static int64_t getAccVecSize(int32_t m, int32_t n, Type elemTyA) {
  if (m == 16 && n == 16 && elemTyA.isF32())
    return 4;
  return 0;
}

FailureOr<Value> MmaOpCQ_MatmulF32Type::emitAtomCallSSA(OpBuilder &builder, Location loc,
                                                        Type resultTy, Type mmaAtomTyArg,
                                                        Type dTyArg, Type aTyArg, Type bTyArg,
                                                        Type cTyArg, Value atomVal, Value d,
                                                        Value a, Value b, Value c) const {
  (void)mmaAtomTyArg;
  (void)dTyArg;
  (void)atomVal;
  (void)d;
  int32_t m = getM();
  int32_t n = getN();
  Type elemTyA = getElemTyA();
  Type elemTyB = getElemTyB();
  MLIRContext *ctx = builder.getContext();

  int64_t accVecSize = getAccVecSize(m, n, elemTyA);
  if (accVecSize == 0)
    return failure();

  Type accElemTy = getElemTyAcc();
  auto accTy = VectorType::get({accVecSize}, accElemTy);

  if (a.getType() != elemTyA)
    a = LLVM::BitcastOp::create(builder, loc, elemTyA, a);
  if (b.getType() != elemTyB)
    b = LLVM::BitcastOp::create(builder, loc, elemTyB, b);
  if (c.getType() != accTy)
    c = LLVM::BitcastOp::create(builder, loc, accTy, c);

  auto va = vector::BroadcastOp::create(builder, loc, accTy, a);
  auto vb = vector::BroadcastOp::create(builder, loc, accTy, b);
  auto prod = arith::MulFOp::create(builder, loc, va, vb);
  auto sum = arith::AddFOp::create(builder, loc, prod, c);
  (void)ctx;
  (void)resultTy;
  return sum.getResult();
}

LogicalResult MmaOpCQ_MatmulF32Type::emitAtomCall(OpBuilder &builder, Location loc, Type mmaAtomTy,
                                                  Type dMemTy, Type aMemTy, Type bMemTy, Type cMemTy,
                                                  Value atomVal, Value dPtr, Value aPtr, Value bPtr,
                                                  Value cPtr) const {
  int32_t m = getM();
  int32_t n = getN();
  Type elemTyA = getElemTyA();
  Type elemTyB = getElemTyB();
  MLIRContext *ctx = builder.getContext();

  int64_t accVecSize = getAccVecSize(m, n, elemTyA);
  if (accVecSize == 0)
    return failure();

  Type accElemTy = getElemTyAcc();
  auto accTy = VectorType::get({accVecSize}, accElemTy);

  Value a = LLVM::LoadOp::create(builder, loc, elemTyA, aPtr);
  Value b = LLVM::LoadOp::create(builder, loc, elemTyB, bPtr);
  Value c = LLVM::LoadOp::create(builder, loc, accTy, cPtr);
  auto res = emitAtomCallSSA(builder, loc, accTy, mmaAtomTy, Type{}, elemTyA, elemTyB, accTy,
                             atomVal, Value{}, a, b, c);
  if (failed(res))
    return failure();
  LLVM::StoreOp::create(builder, loc, *res, dPtr);
  (void)ctx;
  (void)dMemTy;
  (void)aMemTy;
  (void)bMemTy;
  (void)cMemTy;
  return success();
}

} // namespace mlir::fly_cq
