// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include <array>
#include <optional>

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"
#include "flydsl/Dialect/FlyIXDL/IR/Dialect.h"

using namespace mlir;
using namespace mlir::fly;

namespace mlir::fly_ixdl {

namespace {

// MR TCU multiplicand type -> IXDL MMAD element type. Signless/signed i8 -> s8,
// unsigned i8 -> u8 (matches ixcc getMmadIntrinsicId keys).
std::optional<IXDL::MMADTypes> mmadMultiplicandType(Type t) {
  if (t.isF16())
    return IXDL::MMADTypes::f16;
  if (t.isBF16())
    return IXDL::MMADTypes::bf16;
  if (t.isF32())
    return IXDL::MMADTypes::f32;
  if (t.isInteger(8))
    return t.isUnsignedInteger(8) ? IXDL::MMADTypes::u8 : IXDL::MMADTypes::s8;
  return std::nullopt;
}

bool isSupportedMultiplicand(Type t) { return mmadMultiplicandType(t).has_value(); }

bool isFloatMultiplicand(Type t) { return t.isF16() || t.isBF16() || t.isF32(); }

// Coerce a register operand to a vector value: load it if it is a pointer
// (non-SSA call / non-coalescable register), otherwise it is already the
// pre-loaded fragment vector (the convert-atom-call-to-ssa-form path).
Value materializeFragment(OpBuilder &builder, Location loc, Value v, VectorType vecTy) {
  if (isa<LLVM::LLVMPointerType>(v.getType()))
    return LLVM::LoadOp::create(builder, loc, vecTy, v);
  return v;
}

// Build `ixdl.mmad D = A*B + C` from A/B/C fragments (each either a register
// pointer or an already-loaded vector) and return the result vector. Mirrors
// FlyROCDL CDNA3 emitAtomCall.
FailureOr<Value> buildMmad(OpBuilder &builder, Location loc, int32_t m, int32_t n, int32_t k,
                           Type elemTyA, Type elemTyB, Type elemTyAcc, Value aVal, Value bVal,
                           Value cVal) {
  // Per-lane element counts: A/B = M*K/64 = N*K/64; C/D = M*N/64.
  int64_t abCount = static_cast<int64_t>(m) * k / 64;
  int64_t accCount = static_cast<int64_t>(m) * n / 64;
  if (abCount <= 0 || accCount <= 0)
    return failure();

  VectorType aVecTy = VectorType::get({abCount}, elemTyA);
  VectorType bVecTy = VectorType::get({abCount}, elemTyB);
  VectorType accVecTy = VectorType::get({accCount}, elemTyAcc);

  auto mmadTypeA = mmadMultiplicandType(elemTyA);
  auto mmadTypeB = mmadMultiplicandType(elemTyB);
  if (!mmadTypeA || !mmadTypeB)
    return failure();

  Value a = materializeFragment(builder, loc, aVal, aVecTy);
  Value b = materializeFragment(builder, loc, bVal, bVecTy);
  Value c = materializeFragment(builder, loc, cVal, accVecTy);

  std::array<IXDL::MMADTypes, 2> mtypes{*mmadTypeA, *mmadTypeB};
  std::array<IXDL::MMADLayout, 2> mlayouts{IXDL::MMADLayout::row, IXDL::MMADLayout::col};
  std::array<int64_t, 3> shape{m, n, k};

  Value d = IXDL::MmadOp::create(builder, loc, accVecTy, ValueRange{a}, ValueRange{b},
                                 ValueRange{c}, shape, mtypes, mlayouts);
  return d;
}

} // namespace

LogicalResult MmaOpMRMmaType::verify(function_ref<InFlightDiagnostic()> emitError, int32_t m,
                                     int32_t n, int32_t k, Type elemTyA, Type elemTyB,
                                     Type elemTyAcc) {
  if (m != 16 || n != 16)
    return emitError() << "MR MMA requires M = N = 16, got " << m << "x" << n;
  if (!isSupportedMultiplicand(elemTyA) || !isSupportedMultiplicand(elemTyB))
    return emitError() << "MR MMA multiplicand type must be f16/bf16/f32/i8, got (" << elemTyA
                       << ", " << elemTyB << ")";
  if (elemTyA != elemTyB)
    return emitError() << "MR MMA requires matching A/B element types, got " << elemTyA << " vs "
                       << elemTyB;

  if (isFloatMultiplicand(elemTyA)) {
    if (k != 16)
      return emitError() << "MR float MMA requires K = 16, got " << k;
    if (!elemTyAcc.isF32())
      return emitError() << "MR float MMA requires f32 accumulator, got " << elemTyAcc;
  } else {
    // int8 path.
    if (k != 32)
      return emitError() << "MR int8 MMA requires K = 32, got " << k;
    if (!elemTyAcc.isInteger(32))
      return emitError() << "MR int8 MMA requires i32 accumulator, got " << elemTyAcc;
  }
  return success();
}

bool MmaOpMRMmaType::isStatic() const { return true; }

// The inner mma-op type is only ever embedded in a `!fly.mma_atom<...>` wrapper,
// whose own rebuildStaticValue reconstructs the make_mma_atom op. Report
// "already in normal form".
Value MmaOpMRMmaType::rebuildStaticValue(OpBuilder &, Location, Value) const { return nullptr; }

// Warp-collective TCU MMA: all 64 lanes participate and the per-lane fragment
// ownership is exposed to layout algebra (unlike the SME async copy, which hides
// the warp). Mirrors CuTe `ThrID = Layout<_64>`.
Attribute MmaOpMRMmaType::getThrLayout() const { return FxLayout(FxC(64), FxC(1)); }

Attribute MmaOpMRMmaType::getShapeMNK() const {
  return IntTupleAttr::get(ArrayAttr::get(getContext(), {FxC(getM()), FxC(getN()), FxC(getK())}));
}

Type MmaOpMRMmaType::getValTypeA() const { return getElemTyA(); }
Type MmaOpMRMmaType::getValTypeB() const { return getElemTyB(); }
Type MmaOpMRMmaType::getValTypeC() const { return getElemTyAcc(); }
Type MmaOpMRMmaType::getValTypeD() const { return getElemTyAcc(); }

// A/B/C fragment thread-value layouts, transcribed from CuTe ivcore11
// (mma_traits_ivcorex.hpp). Codomain is the (M,N) tile; ThrID part is (16,4) =
// 64 lanes, value part is the per-lane element count.
Attribute MmaOpMRMmaType::getThrValLayoutA() const {
  unsigned bits = getElemTyA().getIntOrFloatBitWidth();
  if (bits == 32) // f32: Layout_16x16_32b_AC
    return FxLayout(FxShape(FxThr(16, 4), FxVal(4)), FxStride(FxThr(16, 1), FxVal(4)));
  if (bits == 16) // f16/bf16: Layout_16x16_16b_A
    return FxLayout(FxShape(FxThr(16, 4), FxVal(2, 2)), FxStride(FxThr(16, 2), FxVal(1, 8)));
  // 8b: Layout_16x32_8b_A
  return FxLayout(FxShape(FxThr(16, 4), FxVal(4, 2)), FxStride(FxThr(16, 4), FxVal(1, 256)));
}

Attribute MmaOpMRMmaType::getThrValLayoutB() const {
  unsigned bits = getElemTyB().getIntOrFloatBitWidth();
  if (bits == 32) // f32: Layout_16x16_32b_B
    return FxLayout(FxShape(FxThr(16, 4), FxVal(4)), FxStride(FxThr(1, 16), FxVal(64)));
  if (bits == 16) // f16/bf16: Layout_16x16_16b_B
    return FxLayout(FxShape(FxThr(16, 4), FxVal(2, 2)), FxStride(FxThr(1, 32), FxVal(16, 128)));
  // 8b: Layout_16x32_8b_B
  return FxLayout(FxShape(FxThr(16, 4), FxVal(4, 2)), FxStride(FxThr(1, 64), FxVal(16, 256)));
}

Attribute MmaOpMRMmaType::getThrValLayoutC() const {
  // Accumulator is always 32-bit, 16x16, 4 per lane: Layout_16x16_32b_AC.
  return FxLayout(FxShape(FxThr(16, 4), FxVal(4)), FxStride(FxThr(16, 1), FxVal(4)));
}

LogicalResult MmaOpMRMmaType::emitAtomCall(OpBuilder &builder, Location loc, Type, Type, Type, Type,
                                           Type, Value, Value dPtr, Value aPtr, Value bPtr,
                                           Value cPtr) const {
  auto res = buildMmad(builder, loc, getM(), getN(), getK(), getElemTyA(), getElemTyB(),
                       getElemTyAcc(), aPtr, bPtr, cPtr);
  if (failed(res))
    return failure();
  LLVM::StoreOp::create(builder, loc, *res, dPtr);
  return success();
}

FailureOr<Value> MmaOpMRMmaType::emitAtomCallSSA(OpBuilder &builder, Location loc, Type resultTy,
                                                 Type, Type, Type, Type, Type, Value, Value dPtr,
                                                 Value aPtr, Value bPtr, Value cPtr) const {
  auto res = buildMmad(builder, loc, getM(), getN(), getK(), getElemTyA(), getElemTyB(),
                       getElemTyAcc(), aPtr, bPtr, cPtr);
  if (failed(res))
    return failure();
  if (resultTy)
    return *res;
  // No SSA result requested: store into the D pointer.
  LLVM::StoreOp::create(builder, loc, *res, dPtr);
  return Value{};
}

} // namespace mlir::fly_ixdl
