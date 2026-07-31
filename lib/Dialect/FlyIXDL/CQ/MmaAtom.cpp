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

// CQ TCU multiplicand type -> IXDL MMAD element type.
// f16/bf16/s8/u8/f8e4m3/f8e5m2 (no f32 multiplicand). Signless/signed i8 -> s8;
// unsigned i8 -> u8.
std::optional<IXDL::MMADTypes> mmadMultiplicandType(Type t) {
  if (t.isF16())
    return IXDL::MMADTypes::f16;
  if (t.isBF16())
    return IXDL::MMADTypes::bf16;
  if (t.isInteger(8))
    return t.isUnsignedInteger(8) ? IXDL::MMADTypes::u8 : IXDL::MMADTypes::s8;
  if (isa<Float8E4M3Type>(t))
    return IXDL::MMADTypes::f8e4m3;
  if (isa<Float8E5M2Type>(t))
    return IXDL::MMADTypes::f8e5m2;
  return std::nullopt;
}

bool isSupportedMultiplicand(Type t) { return mmadMultiplicandType(t).has_value(); }

bool isFloat16Multiplicand(Type t) { return t.isF16() || t.isBF16(); }

bool isInt8Multiplicand(Type t) { return t.isInteger(8); }

bool isFp8Multiplicand(Type t) { return isa<Float8E4M3Type, Float8E5M2Type>(t); }

bool isInt32Accumulator(Type t, IXDL::MMADTypes multiplicandType) {
  auto intTy = dyn_cast<IntegerType>(t);
  if (!intTy || intTy.getWidth() != 32)
    return false;
  if (multiplicandType == IXDL::MMADTypes::u8)
    return !intTy.isSigned();
  return !intTy.isUnsigned();
}

// FeatureLongMtx enlargements of the 16x16 base tile; K stays at the base-tile
// value for the dtype.
bool isLongMtxMN(int32_t m, int32_t n) {
  return (m == 32 && n == 32) || (m == 16 && n == 64) || (m == 64 && n == 16);
}

bool isLegalCQMN(int32_t m, int32_t n) { return (m == 16 && n == 16) || isLongMtxMN(m, n); }

// Coerce a register operand to a vector value: load it if it is a pointer
// (non-SSA call / non-coalescable register), otherwise it is already the
// pre-loaded fragment vector (the convert-atom-call-to-ssa-form path).
Value materializeFragment(OpBuilder &builder, Location loc, Value v, VectorType vecTy) {
  if (isa<LLVM::LLVMPointerType>(v.getType()))
    return LLVM::LoadOp::create(builder, loc, vecTy, v);
  return v;
}

// Build `ixdl.mmad D = A*B + C` from A/B/C fragments (each either a register
// pointer or an already-loaded vector) and return the result vector.
// A/B fragment widths differ for asymmetric long-mtx (16x64 / 64x16).
FailureOr<Value> buildMmad(OpBuilder &builder, Location loc, int32_t m, int32_t n, int32_t k,
                           Type elemTyA, Type elemTyB, Type elemTyAcc, Value aVal, Value bVal,
                           Value cVal) {
  // Per-lane element counts divide each fragment across the warp.
  int64_t aCount = static_cast<int64_t>(m) * k / kWarpSize;
  int64_t bCount = static_cast<int64_t>(n) * k / kWarpSize;
  int64_t accCount = static_cast<int64_t>(m) * n / kWarpSize;
  if (aCount <= 0 || bCount <= 0 || accCount <= 0)
    return failure();

  VectorType aVecTy = VectorType::get({aCount}, elemTyA);
  VectorType bVecTy = VectorType::get({bCount}, elemTyB);
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

LogicalResult MmaOpCQMmaType::verify(function_ref<InFlightDiagnostic()> emitError, int32_t m,
                                     int32_t n, int32_t k, Type elemTyA, Type elemTyB,
                                     Type elemTyAcc) {
  if (!isLegalCQMN(m, n))
    return emitError() << "CQ MMA requires (M,N) in {(16,16),(32,32),(16,64),(64,16)}, got " << m
                       << "x" << n;
  if (!isSupportedMultiplicand(elemTyA) || !isSupportedMultiplicand(elemTyB))
    return emitError() << "CQ MMA multiplicand type must be f16/bf16/i8/ui8/f8E4M3/f8E5M2, got ("
                       << elemTyA << ", " << elemTyB << ")";

  if (isFloat16Multiplicand(elemTyA) || isFloat16Multiplicand(elemTyB)) {
    if (elemTyA != elemTyB)
      return emitError() << "CQ f16/bf16 MMA requires matching A/B element types, got " << elemTyA
                         << " vs " << elemTyB;
    if (k != 16)
      return emitError() << "CQ f16/bf16 MMA requires K = 16, got " << k;
    if (!elemTyAcc.isF32())
      return emitError() << "CQ f16/bf16 MMA requires f32 accumulator, got " << elemTyAcc;
    return success();
  }

  if (isInt8Multiplicand(elemTyA) || isInt8Multiplicand(elemTyB)) {
    auto typeA = *mmadMultiplicandType(elemTyA);
    auto typeB = *mmadMultiplicandType(elemTyB);
    if (typeA != typeB)
      return emitError() << "CQ int8 MMA requires matching A/B signedness, got " << elemTyA
                         << " vs " << elemTyB;
    if (k != 32)
      return emitError() << "CQ int8 MMA requires K = 32, got " << k;
    if (!isInt32Accumulator(elemTyAcc, typeA)) {
      if (typeA == IXDL::MMADTypes::u8)
        return emitError() << "CQ u8 MMA requires i32/ui32 accumulator, got " << elemTyAcc;
      return emitError() << "CQ s8 MMA requires i32/si32 accumulator, got " << elemTyAcc;
    }
    return success();
  }

  // FP8 path: A/B may be any combo of f8E4M3 / f8E5M2.
  if (!isFp8Multiplicand(elemTyA) || !isFp8Multiplicand(elemTyB))
    return emitError() << "CQ MMA requires matching dtype family for A/B, got (" << elemTyA << ", "
                       << elemTyB << ")";
  if (k != 32)
    return emitError() << "CQ FP8 MMA requires K = 32, got " << k;
  if (elemTyAcc.isF32() || elemTyAcc.isF16())
    return success();
  return emitError() << "CQ FP8 MMA requires f32 or f16 accumulator, got " << elemTyAcc;
}

bool MmaOpCQMmaType::isStatic() const { return true; }

// The inner mma-op type is only ever embedded in a `!fly.mma_atom<...>` wrapper,
// whose own rebuildStaticValue reconstructs the make_mma_atom op. Report
// "already in normal form".
Value MmaOpCQMmaType::rebuildStaticValue(OpBuilder &, Location, Value) const { return nullptr; }

// Warp-collective TCU MMA: all 64 lanes participate.
Attribute MmaOpCQMmaType::getThrLayout() const { return FxLayout(FxC(kWarpSize), FxC(1)); }

Attribute MmaOpCQMmaType::getShapeMNK() const {
  return IntTupleAttr::get(ArrayAttr::get(getContext(), {FxC(getM()), FxC(getN()), FxC(getK())}));
}

Type MmaOpCQMmaType::getValTypeA() const { return getElemTyA(); }
Type MmaOpCQMmaType::getValTypeB() const { return getElemTyB(); }
Type MmaOpCQMmaType::getValTypeC() const { return getElemTyAcc(); }
Type MmaOpCQMmaType::getValTypeD() const { return getElemTyAcc(); }

// ThrVal layouts for base (16x16) and long-mtx MN shapes. 64x16 A/C extend the
// 16x64 / 32x16 patterns (64x16 is the MN-swapped long-mtx). FP8 uses the 8-bit
// A/B layouts (same as s8/u8).
Attribute MmaOpCQMmaType::getThrValLayoutA() const {
  unsigned bits = getElemTyA().getIntOrFloatBitWidth();
  int32_t m = getM();
  int32_t n = getN();

  if (bits == 16) {
    // Base 16x16: Layout_16x16_16b_A
    if (m == 16 && n == 16)
      return FxLayout(FxShape(FxThr(16, 4), FxVal(2, 2)), FxStride(FxThr(16, 2), FxVal(1, 8)));
    // 16x64: same A as base (M=16,K=16)
    if (m == 16 && n == 64)
      return FxLayout(FxShape(FxThr(16, 4), FxVal(2, 2)), FxStride(FxThr(16, 2), FxVal(1, 8)));
    // 32x32: Layout_32x16_16b_A
    if (m == 32 && n == 32)
      return FxLayout(FxShape(FxThr(16, 4), FxVal(2, 4)), FxStride(FxThr(32, 2), FxVal(1, 8)));
    // 64x16: Layout_64x16_16b_A (extend 32x16 A)
    return FxLayout(FxShape(FxThr(16, 4), FxVal(2, 8)), FxStride(FxThr(64, 2), FxVal(1, 8)));
  }

  // 8b: s8 / u8 / f8, K=32
  // Base 16x16: Layout_16x32_8b_A
  if (m == 16 && n == 16)
    return FxLayout(FxShape(FxThr(16, 4), FxVal(4, 2)), FxStride(FxThr(16, 4), FxVal(1, 256)));
  // 16x64: same A as base (M=16,K=32)
  if (m == 16 && n == 64)
    return FxLayout(FxShape(FxThr(16, 4), FxVal(4, 2)), FxStride(FxThr(16, 4), FxVal(1, 256)));
  // 32x32: Layout_32x32_8b_A
  if (m == 32 && n == 32)
    return FxLayout(FxShape(FxThr(16, 4), FxVal(4, 2, 2)),
                    FxStride(FxThr(32, 4), FxVal(1, 512, 16)));
  // 64x16: Layout_64x32_8b_A
  return FxLayout(FxShape(FxThr(16, 4), FxVal(4, 2, 4)),
                  FxStride(FxThr(64, 4), FxVal(1, 1024, 16)));
}

Attribute MmaOpCQMmaType::getThrValLayoutB() const {
  unsigned bits = getElemTyB().getIntOrFloatBitWidth();
  int32_t m = getM();
  int32_t n = getN();

  if (bits == 16) {
    // Base 16x16: Layout_16x16_16b_B
    if (m == 16 && n == 16)
      return FxLayout(FxShape(FxThr(16, 4), FxVal(2, 2)), FxStride(FxThr(1, 32), FxVal(16, 128)));
    // 64x16: same B as base (N=16,K=16)
    if (m == 64 && n == 16)
      return FxLayout(FxShape(FxThr(16, 4), FxVal(2, 2)), FxStride(FxThr(1, 32), FxVal(16, 128)));
    // 32x32: Layout_32x16_16b_B
    if (m == 32 && n == 32)
      return FxLayout(FxShape(FxThr(16, 4), FxVal(2, 2, 2)),
                      FxStride(FxThr(1, 64), FxVal(32, 256, 16)));
    // 16x64: Layout_64x16_16b_B
    return FxLayout(FxShape(FxThr(16, 4), FxVal(2, 2, 4)),
                    FxStride(FxThr(1, 128), FxVal(64, 512, 16)));
  }

  // 8b: s8 / u8 / f8
  // Base 16x16: Layout_16x32_8b_B
  if (m == 16 && n == 16)
    return FxLayout(FxShape(FxThr(16, 4), FxVal(4, 2)), FxStride(FxThr(1, 64), FxVal(16, 256)));
  // 64x16: same B as base (N=16,K=32)
  if (m == 64 && n == 16)
    return FxLayout(FxShape(FxThr(16, 4), FxVal(4, 2)), FxStride(FxThr(1, 64), FxVal(16, 256)));
  // 32x32: Layout_32x32_8b_B
  if (m == 32 && n == 32)
    return FxLayout(FxShape(FxThr(16, 4), FxVal(4, 2, 2)),
                    FxStride(FxThr(1, 128), FxVal(32, 512, 16)));
  // 16x64: Layout_64x32_8b_B
  return FxLayout(FxShape(FxThr(16, 4), FxVal(4, 2, 4)),
                  FxStride(FxThr(1, 256), FxVal(64, 1024, 16)));
}

Attribute MmaOpCQMmaType::getThrValLayoutC() const {
  int32_t m = getM();
  int32_t n = getN();

  if (getElemTyAcc().isF16()) {
    // 16x16: Layout_16x16_16b_C
    if (m == 16 && n == 16)
      return FxLayout(FxShape(FxThr(16, 4), FxVal(2, 2)), FxStride(FxThr(16, 2), FxVal(1, 8)));
    // 32x32: expand the base tile by two in both M and N.
    if (m == 32 && n == 32)
      return FxLayout(FxShape(FxThr(16, 4), FxVal(2, 2, 2, 2)),
                      FxStride(FxThr(32, 2), FxVal(1, 8, 512, 16)));
    // 16x64: expand the base tile by four in N.
    if (m == 16 && n == 64)
      return FxLayout(FxShape(FxThr(16, 4), FxVal(2, 2, 4)),
                      FxStride(FxThr(16, 2), FxVal(1, 8, 256)));
    // 64x16: expand the base tile by four in M.
    return FxLayout(FxShape(FxThr(16, 4), FxVal(2, 2, 4)),
                    FxStride(FxThr(64, 2), FxVal(1, 8, 1024)));
  }

  // 32-bit accumulator (f32 / i32 / ui32).
  // Base 16x16: Layout_16x16_32b_AC
  if (m == 16 && n == 16)
    return FxLayout(FxShape(FxThr(16, 4), FxVal(4)), FxStride(FxThr(16, 1), FxVal(4)));
  // 32x32: Layout_32x32_32b_C
  if (m == 32 && n == 32)
    return FxLayout(FxShape(FxThr(16, 4), FxVal(4, 2, 2)),
                    FxStride(FxThr(32, 1), FxVal(4, 512, 16)));
  // 16x64: Layout_16x64_32b_C
  if (m == 16 && n == 64)
    return FxLayout(FxShape(FxThr(16, 4), FxVal(4, 4)), FxStride(FxThr(16, 1), FxVal(4, 256)));
  // 64x16: Layout_64x16_32b_C (MN-swap of 16x64)
  return FxLayout(FxShape(FxThr(16, 4), FxVal(4, 4)), FxStride(FxThr(64, 1), FxVal(4, 1024)));
}

LogicalResult MmaOpCQMmaType::emitAtomCall(OpBuilder &builder, Location loc, Type, Type, Type, Type,
                                           Type, Value, Value dPtr, Value aPtr, Value bPtr,
                                           Value cPtr) const {
  auto res = buildMmad(builder, loc, getM(), getN(), getK(), getElemTyA(), getElemTyB(),
                       getElemTyAcc(), aPtr, bPtr, cPtr);
  if (failed(res))
    return failure();
  LLVM::StoreOp::create(builder, loc, *res, dPtr);
  return success();
}

FailureOr<Value> MmaOpCQMmaType::emitAtomCallSSA(OpBuilder &builder, Location loc, Type resultTy,
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
