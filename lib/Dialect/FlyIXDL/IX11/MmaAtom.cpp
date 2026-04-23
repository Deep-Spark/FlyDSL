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

//===----------------------------------------------------------------------===//
// IX11 CUTE-style ThrVal layouts for ivcore11 MMAD atoms.
//
// The layouts below are a direct translation of CUTLASS's
// ``cute/atom/mma_traits_ivcorex.hpp`` into FlyDSL's nested
// (Thr, Val) layout algebra:
//
//   Layout_16x16_32b_AC : (T64,V4) -> (M16,N16)  row-major on M
//   Layout_16x16_32b_B  : (T64,V4) -> (M16,N16)  col-major on M (K in rows)
//   Layout_16x16_16b_A  : (T64,V4) -> (M16,K16)  16-bit packed
//   Layout_16x16_16b_B  : (T64,V4) -> (K16,N16)  16-bit packed
//   Layout_16x32_8b_A   : (T64,V8) -> (M16,K32)  8-bit packed
//   Layout_16x32_8b_B   : (T64,V8) -> (K32,N16)  8-bit packed
//===----------------------------------------------------------------------===//

namespace ix11 {

enum class Operand { A, B, C };

// Helper that picks the right layout for an MMA operand given the shape and
// the element bit width. ``getContext()`` is captured by the ``Fx*`` macros,
// so we wrap them in a lambda-bound closure.
static LayoutAttr getThrValLayout(MLIRContext *ctx, int32_t M, int32_t N, int32_t K,
                                  int elemBits, Operand op) {
  auto getContext = [&]() { return ctx; };

  // C/D always live in the 32b AC layout regardless of input bit width.
  if (op == Operand::C) {
    // Layout_16x16_32b_AC : Shape((16,4),(4)) Stride((16,1),(4))
    return FxLayout(FxShape(FxThr(16, 4), FxVal(4)),
                    FxStride(FxThr(16, 1), FxVal(4)));
  }

  if (elemBits == 32) {
    if (op == Operand::A) {
      // Layout_16x16_32b_AC
      return FxLayout(FxShape(FxThr(16, 4), FxVal(4)),
                      FxStride(FxThr(16, 1), FxVal(4)));
    }
    // Layout_16x16_32b_B : Shape((16,4),(4)) Stride((1,16),(64))
    return FxLayout(FxShape(FxThr(16, 4), FxVal(4)),
                    FxStride(FxThr(1, 16), FxVal(64)));
  }

  if (elemBits == 16) {
    if (op == Operand::A) {
      // Layout_16x16_16b_A : Shape((16,4),(2,2)) Stride((16,2),(1,8))
      return FxLayout(FxShape(FxThr(16, 4), FxVal(2, 2)),
                      FxStride(FxThr(16, 2), FxVal(1, 8)));
    }
    // Layout_16x16_16b_B : Shape((16,4),(2,2)) Stride((1,32),(16,128))
    return FxLayout(FxShape(FxThr(16, 4), FxVal(2, 2)),
                    FxStride(FxThr(1, 32), FxVal(16, 128)));
  }

  if (elemBits == 8) {
    if (op == Operand::A) {
      // Layout_16x32_8b_A : Shape((16,4),(4,2)) Stride((16,4),(1,256))
      return FxLayout(FxShape(FxThr(16, 4), FxVal(4, 2)),
                      FxStride(FxThr(16, 4), FxVal(1, 256)));
    }
    // Layout_16x32_8b_B : Shape((16,4),(4,2)) Stride((1,64),(16,256))
    return FxLayout(FxShape(FxThr(16, 4), FxVal(4, 2)),
                    FxStride(FxThr(1, 64), FxVal(16, 256)));
  }

  // Unsupported — verifier should have rejected earlier.
  return nullptr;
}

static int getElemBits(Type t) {
  if (t.isF32() || t.isInteger(32))
    return 32;
  if (t.isF16() || t.isBF16() || t.isInteger(16))
    return 16;
  if (t.isInteger(8))
    return 8;
  return 0;
}

// Per-thread vector type for operand A/B/C according to the IX11 CUTE atom.
static Type getPerThreadOperandType(MLIRContext *ctx, Type elemTy, int32_t m, int32_t n,
                                    int32_t k, Operand op) {
  int64_t total = 0;
  if (op == Operand::C) {
    total = (int64_t)m * n;
  } else {
    total = op == Operand::A ? (int64_t)m * k : (int64_t)n * k;
  }
  int64_t perThread = total / 64;
  if (perThread <= 0)
    return nullptr;
  // For C/D, ivcore11 always accumulates in 32-bit (s32 or f32). Fly lets us
  // keep the caller-supplied acc element type (f32 or i32), which already
  // satisfies this.
  return VectorType::get({perThread}, elemTy);
}

} // namespace ix11

namespace mlir::fly_ixdl {

//===----------------------------------------------------------------------===//
// MmaOpIX11_MMADType
//===----------------------------------------------------------------------===//

bool MmaOpIX11_MMADType::isStatic() const { return true; }

Value MmaOpIX11_MMADType::rebuildStaticValue(OpBuilder &builder, Location loc,
                                             Value currentValue) const {
  if (currentValue && isa<MakeMmaAtomOp>(currentValue.getDefiningOp()))
    return nullptr;
  return MakeMmaAtomOp::create(builder, loc, MmaAtomType::get(*this));
}

Attribute MmaOpIX11_MMADType::getThrLayout() const {
  // A single ivcore11 MMAD engages one warp of 64 threads.
  return FxLayout(FxC(64), FxC(1));
}

Attribute MmaOpIX11_MMADType::getShapeMNK() const {
  return IntTupleAttr::get(
      ArrayAttr::get(getContext(), {FxC(getM()), FxC(getN()), FxC(getK())}));
}

Type MmaOpIX11_MMADType::getValTypeA() const { return getElemTyA(); }
Type MmaOpIX11_MMADType::getValTypeB() const { return getElemTyB(); }
Type MmaOpIX11_MMADType::getValTypeC() const { return getElemTyAcc(); }
Type MmaOpIX11_MMADType::getValTypeD() const { return getElemTyAcc(); }

Attribute MmaOpIX11_MMADType::getThrValLayoutA() const {
  return ix11::getThrValLayout(getContext(), getM(), getN(), getK(),
                               ix11::getElemBits(getElemTyA()), ix11::Operand::A);
}
Attribute MmaOpIX11_MMADType::getThrValLayoutB() const {
  return ix11::getThrValLayout(getContext(), getM(), getN(), getK(),
                               ix11::getElemBits(getElemTyB()), ix11::Operand::B);
}
Attribute MmaOpIX11_MMADType::getThrValLayoutC() const {
  return ix11::getThrValLayout(getContext(), getM(), getN(), getK(),
                               ix11::getElemBits(getElemTyAcc()), ix11::Operand::C);
}

LogicalResult MmaOpIX11_MMADType::verify(function_ref<InFlightDiagnostic()> emitError,
                                         int32_t m, int32_t n, int32_t k, Type elemTyA,
                                         Type elemTyB, Type elemTyAcc) {
  // Only the CUTLASS IX11 TT atoms are supported right now.
  auto sameElem = [](Type a, Type b) { return a == b; };

  bool is161616 = (m == 16 && n == 16 && k == 16);
  bool is161632 = (m == 16 && n == 16 && k == 32);
  if (!is161616 && !is161632)
    return emitError() << "unsupported IX11 MMAD shape " << m << "x" << n << "x" << k;

  if (!sameElem(elemTyA, elemTyB))
    return emitError() << "elemTyA/elemTyB must match for IX11 MMAD";

  if (is161616) {
    if (!(elemTyA.isF32() || elemTyA.isF16() || elemTyA.isBF16()))
      return emitError()
             << "16x16x16 IX11 MMAD needs elemTyA in {f32,f16,bf16}, got " << elemTyA;
    if (!elemTyAcc.isF32())
      return emitError() << "16x16x16 IX11 MMAD accumulates in f32, got " << elemTyAcc;
  } else {
    // 16x16x32: integer 8-bit operands with i32 accumulator.
    if (!elemTyA.isInteger(8))
      return emitError() << "16x16x32 IX11 MMAD needs i8 operands, got " << elemTyA;
    if (!elemTyAcc.isInteger(32))
      return emitError() << "16x16x32 IX11 MMAD accumulates in i32, got " << elemTyAcc;
  }
  return success();
}

FailureOr<Value> MmaOpIX11_MMADType::emitAtomCallSSA(OpBuilder &builder, Location loc,
                                                    Type resultTy, Type /*mmaAtomTyArg*/,
                                                    Type /*dTyArg*/, Type /*aTyArg*/,
                                                    Type /*bTyArg*/, Type /*cTyArg*/,
                                                    Value /*atomVal*/, Value /*d*/,
                                                    Value a, Value b, Value c) const {
  MLIRContext *ctx = builder.getContext();
  int32_t m = getM();
  int32_t n = getN();
  int32_t k = getK();
  Type elemTyA = getElemTyA();
  Type elemTyB = getElemTyB();
  Type elemTyAcc = getElemTyAcc();

  Type abTyA = ix11::getPerThreadOperandType(ctx, elemTyA, m, n, k, ix11::Operand::A);
  Type abTyB = ix11::getPerThreadOperandType(ctx, elemTyB, m, n, k, ix11::Operand::B);
  Type accTy = ix11::getPerThreadOperandType(ctx, elemTyAcc, m, n, k, ix11::Operand::C);
  if (!abTyA || !abTyB || !accTy)
    return failure();

  if (a.getType() != abTyA)
    a = LLVM::BitcastOp::create(builder, loc, abTyA, a);
  if (b.getType() != abTyB)
    b = LLVM::BitcastOp::create(builder, loc, abTyB, b);
  if (c.getType() != accTy)
    c = LLVM::BitcastOp::create(builder, loc, accTy, c);

  // Build ixdl.mmad. The intrinsic table keys are 'row, col' for layoutA/B —
  // which matches ``MmadOp::build`` defaults — so we rely on those.
  // Multiplicand types are inferred from the per-thread element type, except
  // that we need to force u8 (default infers s8) when the user asked for it.
  std::optional<std::array<IXDL::MMADTypes, 2>> multTys;
  if (elemTyA.isInteger(8)) {
    // FlyDSL carries signedness at the dialect level via a dedicated parameter.
    // Since the verifier above only accepts plain i8, we default to s8 here;
    // the unsigned variant is exposed through a distinct Python entry point
    // (``MMA_S32_U8``) below, which passes ``MMADTypes::u8`` explicitly.
    multTys = std::array<IXDL::MMADTypes, 2>{IXDL::MMADTypes::s8, IXDL::MMADTypes::s8};
  }

  Type resTy = resultTy ? resultTy : accTy;
  // Make the produced LLVM result match `accTy` — callers (Fly rewriter)
  // will bitcast back to whatever they need.
  Type ixdlResTy = accTy;

  auto mmad = IXDL::MmadOp::create(builder, loc, ixdlResTy,
                                   /*operandA=*/ValueRange{a},
                                   /*operandB=*/ValueRange{b},
                                   /*operandC=*/ValueRange{c},
                                   /*shape=*/ArrayRef<int64_t>{m, n, k},
                                   /*multiplicandTypes=*/multTys,
                                   /*multiplicandLayouts=*/std::nullopt);
  Value res = mmad.getResult();
  if (res.getType() != resTy)
    res = LLVM::BitcastOp::create(builder, loc, resTy, res);
  return res;
}

LogicalResult MmaOpIX11_MMADType::emitAtomCall(OpBuilder &builder, Location loc,
                                               Type mmaAtomTy, Type /*dMemTy*/,
                                               Type /*aMemTy*/, Type /*bMemTy*/,
                                               Type /*cMemTy*/, Value atomVal, Value dPtr,
                                               Value aPtr, Value bPtr, Value cPtr) const {
  MLIRContext *ctx = builder.getContext();
  int32_t m = getM();
  int32_t n = getN();
  int32_t k = getK();

  Type abTyA = ix11::getPerThreadOperandType(ctx, getElemTyA(), m, n, k, ix11::Operand::A);
  Type abTyB = ix11::getPerThreadOperandType(ctx, getElemTyB(), m, n, k, ix11::Operand::B);
  Type accTy = ix11::getPerThreadOperandType(ctx, getElemTyAcc(), m, n, k, ix11::Operand::C);
  if (!abTyA || !abTyB || !accTy)
    return failure();

  Value a = LLVM::LoadOp::create(builder, loc, abTyA, aPtr);
  Value b = LLVM::LoadOp::create(builder, loc, abTyB, bPtr);
  Value c = LLVM::LoadOp::create(builder, loc, accTy, cPtr);
  auto res = emitAtomCallSSA(builder, loc, accTy, mmaAtomTy, Type{}, abTyA, abTyB, accTy,
                             atomVal, Value{}, a, b, c);
  if (failed(res))
    return failure();
  LLVM::StoreOp::create(builder, loc, *res, dPtr);
  return success();
}

} // namespace mlir::fly_ixdl
