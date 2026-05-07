// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/IXDLDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"
#include "flydsl/Dialect/FlyIXDL/IR/Dialect.h"

using namespace mlir;
using namespace mlir::fly;

namespace mlir::fly_ixdl {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

namespace {

// Build the 4xi32 SME descriptor:
//   elt 0: low  32 bits of gmem_ptr
//   elt 1: high 32 bits of gmem_ptr
//   elt 2: placeholder (unused on ix11, set to -1 to match ix30 convention)
//   elt 3: stride_byte
//
// ``src`` is expected to be an LLVM pointer value (addrspace=1 / Global).
// ``strideByte`` is the (static) row-stride (K-major) or column-stride
// (MN-major) of the source tensor, in bytes.
static Value buildSMEDescriptor(OpBuilder &builder, Location loc, Value src,
                                int64_t strideByte) {
  MLIRContext *ctx = builder.getContext();
  auto i32Ty = builder.getIntegerType(32);
  auto i64Ty = builder.getIntegerType(64);
  auto v4i32Ty = VectorType::get({4}, i32Ty);

  // ptr -> i64 -> (lo:i32, hi:i32)
  Value ptrAsI64 = LLVM::PtrToIntOp::create(builder, loc, i64Ty, src);
  Value ptrLo = LLVM::TruncOp::create(builder, loc, i32Ty, ptrAsI64);
  Value shiftAmt = LLVM::ConstantOp::create(builder, loc, i64Ty,
                                            builder.getI64IntegerAttr(32));
  Value ptrHiI64 = LLVM::LShrOp::create(builder, loc, ptrAsI64, shiftAmt);
  Value ptrHi = LLVM::TruncOp::create(builder, loc, i32Ty, ptrHiI64);

  Value negOne = LLVM::ConstantOp::create(builder, loc, i32Ty,
                                          builder.getI32IntegerAttr(-1));
  Value stride = LLVM::ConstantOp::create(
      builder, loc, i32Ty,
      builder.getI32IntegerAttr(static_cast<int32_t>(strideByte)));

  Value desc = LLVM::UndefOp::create(builder, loc, v4i32Ty);
  auto idx = [&](int i) {
    return LLVM::ConstantOp::create(builder, loc, i32Ty,
                                    builder.getI32IntegerAttr(i));
  };
  desc = LLVM::InsertElementOp::create(builder, loc, desc, ptrLo, idx(0));
  desc = LLVM::InsertElementOp::create(builder, loc, desc, ptrHi, idx(1));
  desc = LLVM::InsertElementOp::create(builder, loc, desc, negOne, idx(2));
  desc = LLVM::InsertElementOp::create(builder, loc, desc, stride, idx(3));
  return desc;
  (void)ctx;
}

// Cast a shared-memory pointer (addrspace=3) to a 32-bit integer offset,
// as required by ``ixdl.cp.async``'s ``sOffset`` operand.
static Value smemPtrToI32(OpBuilder &builder, Location loc, Value smemPtr) {
  auto i32Ty = builder.getIntegerType(32);
  return LLVM::PtrToIntOp::create(builder, loc, i32Ty, smemPtr);
}

} // namespace

//===----------------------------------------------------------------------===//
// CopyOpIX11_SMEType
//===----------------------------------------------------------------------===//

LogicalResult
CopyOpIX11_SMEType::verify(function_ref<InFlightDiagnostic()> emitError,
                           Type elemTy, int32_t shape0, int32_t shape1,
                           int64_t strideByte, SMEMajorAttr major,
                           SMECacheOpAttr /*cacheOp*/,
                           SMESwizzleAttr swizzle) {
  if (!elemTy)
    return emitError() << "SME copy atom requires a non-null element type";
  int32_t bits = elemTy.getIntOrFloatBitWidth();
  if (bits != 8 && bits != 16 && bits != 32)
    return emitError() << "SME copy atom only supports 8/16/32-bit elements,"
                          " got "
                       << bits;
  // ``shape`` is an element-space (rows, cols) tile. The ixdl -> LLVM
  // dispatch in ``IXDLToLLVMIRTranslation.cpp`` converts this to a
  // (shape0, segmentsFromMinor) pair via
  //   segmentsFromMinor = shape1 * elemBits / 512
  // and then matches against {1,4,8,16,32,64} intrinsic hardware rows.
  // Require shape1 * elemBits be a positive multiple of 512 (one hw row
  // = 64 bytes) and the resulting segment count to land in the set the
  // dispatch actually knows about.
  int64_t minorBits = int64_t(shape1) * int64_t(bits);
  if (minorBits <= 0 || (minorBits % 512) != 0)
    return emitError()
           << "SME tile minor dim " << shape1 << " of " << bits
           << "-bit elements is not a multiple of 512 bits (one hw row)";
  int64_t segMinor = minorBits / 512;
  bool shapeOk = (segMinor == 1) &&
                 (shape0 == 1 || shape0 == 4 || shape0 == 8 ||
                  shape0 == 16 || shape0 == 32 || shape0 == 64);
  if (!shapeOk)
    return emitError()
           << "unsupported SME tile " << shape0 << "x" << shape1
           << " (need shape0 in {1,4,8,16,32,64} and shape1 * sizeof(elem)"
              " == 64 bytes)";
  if (strideByte < 0)
    return emitError() << "SME strideByte must be non-negative";

  // Swizzle-specific constraints. The dispatch table in
  // ``getSMELoadIntrinsicId`` only knows about a handful of element-type /
  // major / swizzle combinations; reject early so kernel writers get a
  // clear error rather than a link-time ``undefined symbol: not_intrinsic``.
  SMESwizzle sw = swizzle.getValue();
  SMEMajor mj = major.getValue();
  switch (sw) {
  case SMESwizzle::None_:
    // No extra constraints: the fold path below always produces a valid
    // ``sme_load_*x1b64`` dispatch for any 8/16/32-bit elem and major.
    break;
  case SMESwizzle::RowXfb16:
    if (bits != 16)
      return emitError() << "SME swizzle row_xfb16 requires 16-bit elements "
                            "(got "
                         << bits << ")";
    if (mj != SMEMajor::K)
      return emitError() << "SME swizzle row_xfb16 requires major=k";
    if (shape0 != 16 && shape0 != 4)
      return emitError() << "SME swizzle row_xfb16 only supports shape0 in "
                            "{4, 16} (got "
                         << shape0 << ")";
    break;
  case SMESwizzle::ColXfb8:
    if (bits != 16)
      return emitError() << "SME swizzle col_xfb8 (16-bit path) requires "
                            "16-bit elements (got "
                         << bits << ")";
    if (mj != SMEMajor::MN)
      return emitError() << "SME swizzle col_xfb8 requires major=mn";
    if (shape0 != 16)
      return emitError() << "SME swizzle col_xfb8 only supports shape0=16 "
                            "(got "
                         << shape0 << ")";
    break;
  }
  return success();
}

bool CopyOpIX11_SMEType::isStatic() const { return true; }

Value CopyOpIX11_SMEType::rebuildStaticValue(OpBuilder &builder, Location loc,
                                             Value currentValue) const {
  if (currentValue && isa<MakeCopyAtomOp>(currentValue.getDefiningOp()))
    return nullptr;
  int32_t bits = getElemTy().getIntOrFloatBitWidth();
  return MakeCopyAtomOp::create(builder, loc,
                                CopyAtomType::get(*this, bits), bits);
}

//===----------------------------------------------------------------------===//
// Thread/value layouts
//
// SME is a single-issuer primitive: one thread triggers the copy, the
// Shared-Memory-Engine moves the full ``shape0 * shape1 * 64`` bytes into
// smem in the background. We therefore mirror CUTLASS's Copy_Traits:
//   ThrID     = Layout<_1>
//   SrcLayout = Layout<Shape<_1, total_bits>, Stride<_0, _1>>
//   DstLayout = Layout<Shape<_1, total_bits>, Stride<_0, _1>>
//===----------------------------------------------------------------------===//

Attribute CopyOpIX11_SMEType::getThrLayout() const {
  return FxLayout(FxC(1), FxC(1));
}

Attribute CopyOpIX11_SMEType::getThrBitLayoutSrc() const {
  int64_t totalBits = int64_t(getShape0()) * int64_t(getShape1()) * 512;
  return FxLayout(FxShape(FxC(1), FxC(totalBits)), FxStride(FxC(0), FxC(1)));
}
Attribute CopyOpIX11_SMEType::getThrBitLayoutDst() const {
  int64_t totalBits = int64_t(getShape0()) * int64_t(getShape1()) * 512;
  return FxLayout(FxShape(FxC(1), FxC(totalBits)), FxStride(FxC(0), FxC(1)));
}
Attribute CopyOpIX11_SMEType::getThrBitLayoutRef() const {
  return getThrBitLayoutSrc();
}

//===----------------------------------------------------------------------===//
// Lowering: emit ``ixdl.cp.async``
//===----------------------------------------------------------------------===//

LogicalResult CopyOpIX11_SMEType::emitAtomCall(OpBuilder &builder, Location loc,
                                               Type /*copyAtomTyArg*/,
                                               Type srcMemTyArg,
                                               Type dstMemTyArg,
                                               Value /*atomVal*/, Value src,
                                               Value dst) const {
  auto srcMemTy = dyn_cast<fly::MemRefType>(srcMemTyArg);
  auto dstMemTy = dyn_cast<fly::MemRefType>(dstMemTyArg);
  if (!srcMemTy || !dstMemTy)
    return failure();

  // SME is strictly global -> shared.
  if (srcMemTy.getAddressSpace().getValue() != AddressSpace::Global)
    return failure();
  if (dstMemTy.getAddressSpace().getValue() != AddressSpace::Shared)
    return failure();

  Value desc = buildSMEDescriptor(builder, loc, src, getStrideByte());
  Value sOffset = smemPtrToI32(builder, loc, dst);

  auto i32Ty = builder.getIntegerType(32);
  Value gOffset = LLVM::ConstantOp::create(builder, loc, i32Ty,
                                           builder.getI32IntegerAttr(0));
  Value kop = LLVM::ConstantOp::create(
      builder, loc, i32Ty,
      builder.getI32IntegerAttr(static_cast<int32_t>(getCacheOp().getValue())));

  int32_t shape0 = getShape0();
  int32_t shape1 = getShape1();
  uint32_t elementSizeBits = getElemTy().getIntOrFloatBitWidth();
  bool transpose = (getMajor().getValue() == SMEMajor::MN);
  SMESwizzle sw = getSwizzle().getValue();

  // Pick ``(emitShape, emitElemSize)`` such that the SDK dispatch in
  // ``getSMELoadIntrinsicId`` routes to the variant we want:
  //
  //   * ``none``      — fold to an i32 strip (``elemSize=32``). This hits
  //                     ``bi_sme_load_*x1b64`` which writes smem in plain
  //                     linear order, so a natural ``make_layout`` reads
  //                     it back correctly. This is the safe default used
  //                     by examples 07-10.
  //   * ``row_xfb16`` — keep the element size as 16, pass element-space
  //                     shape verbatim. Dispatches to
  //                     ``bi_sme_load_*x1b64_rowxfb16``, which applies an
  //                     MMA-A-friendly swizzle in smem.
  //   * ``col_xfb8``  — same, with ``transpose=true``. Dispatches to
  //                     ``bi_sme_load_16x1b64_colxfb8`` (the SDK folds 16
  //                     down to 8 internally when it picks the 16-bit
  //                     transposed path).
  int32_t emitShape0;
  int32_t emitShape1;
  uint32_t emitElemSize;
  switch (sw) {
  case SMESwizzle::None_:
    emitShape0 = shape0;
    emitShape1 = shape1 * int32_t(elementSizeBits) / 32;
    emitElemSize = 32;
    if (emitShape1 == 0) {
      // Elem narrower than a 32-bit strip (shouldn't happen under the
      // verifier, but keep the fallback for safety).
      emitShape0 = shape0;
      emitShape1 = shape1;
      emitElemSize = elementSizeBits;
    }
    break;
  case SMESwizzle::RowXfb16:
  case SMESwizzle::ColXfb8:
    emitShape0 = shape0;
    emitShape1 = shape1;
    emitElemSize = elementSizeBits;
    break;
  }
  ArrayAttr shape = builder.getI64ArrayAttr({emitShape0, emitShape1});

  IXDL::CpAsyncOp::create(builder, loc, sOffset, desc, gOffset, kop, shape,
                          emitElemSize, transpose);
  return success();
}

LogicalResult CopyOpIX11_SMEType::emitAtomCall(OpBuilder &builder, Location loc,
                                               Type copyAtomTyArg,
                                               Type srcMemTyArg,
                                               Type dstMemTyArg,
                                               Type predMemTyArg, Value atomVal,
                                               Value src, Value dst,
                                               Value pred) const {
  OpBuilder::InsertionGuard guard(builder);
  auto predMemTy = cast<fly::MemRefType>(predMemTyArg);
  Value predVal = LLVM::LoadOp::create(builder, loc, predMemTy.getElemTy(), pred);
  auto ifOp = scf::IfOp::create(builder, loc, TypeRange{}, predVal,
                                /*withElse=*/false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  return emitAtomCall(builder, loc, copyAtomTyArg, srcMemTyArg, dstMemTyArg,
                      atomVal, src, dst);
}

// SME has no register-staging form (global -> shared directly), so the SSA
// entry points just forward to the memref form and return a null value, like
// CDNA3 BufferCopyLDS does.
FailureOr<Value> CopyOpIX11_SMEType::emitAtomCallSSA(OpBuilder &builder,
                                                    Location loc,
                                                    Type /*resultTy*/,
                                                    Type copyAtomTyArg,
                                                    Type srcTyArg,
                                                    Type dstTyArg,
                                                    Value atomVal, Value src,
                                                    Value dst) const {
  if (failed(emitAtomCall(builder, loc, copyAtomTyArg, srcTyArg, dstTyArg,
                          atomVal, src, dst)))
    return failure();
  return Value{};
}

FailureOr<Value> CopyOpIX11_SMEType::emitAtomCallSSA(
    OpBuilder &builder, Location loc, Type /*resultTy*/, Type copyAtomTyArg,
    Type srcTyArg, Type dstTyArg, Type predTyArg, Value atomVal, Value src,
    Value dst, Value pred) const {
  if (failed(emitAtomCall(builder, loc, copyAtomTyArg, srcTyArg, dstTyArg,
                          predTyArg, atomVal, src, dst, pred)))
    return failure();
  return Value{};
}

} // namespace mlir::fly_ixdl
