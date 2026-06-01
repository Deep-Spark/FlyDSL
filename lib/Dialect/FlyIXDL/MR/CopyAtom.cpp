// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/PointerUtils.h"
#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"
#include "flydsl/Dialect/FlyIXDL/IR/Dialect.h"
#include "flydsl/Dialect/FlyIXDL/Utils/SmeGmemFatPtr.h"

using namespace mlir;
using namespace mlir::fly;

namespace mlir::fly_ixdl {

namespace {

Attribute makeSmeAtomBitsLayout(MLIRContext *ctx, Attribute swizzle, Attribute outer) {
  return ComposedLayoutAttr::get(ctx, swizzle, IntTupleAttr::getLeafStatic(ctx, 0), outer);
}

} // namespace

LogicalResult CopyOpMRAsyncCpType::verify(function_ref<InFlightDiagnostic()> emitError,
                                          int32_t smeSwizzle) {
  // CUTLASS IX11::SMESwizzle four-state enum: 0=NoSwizzle, 1=Col, 2=Row8b, 3=Row16b.
  if (smeSwizzle < 0 || smeSwizzle > 3)
    return emitError() << "unsupported smeSwizzle = " << smeSwizzle
                       << " for MRAsyncCp (expected 0..3)";
  return success();
}

bool CopyOpMRAsyncCpType::isStatic() const { return true; }

// The inner copy-op type is only ever embedded in a `!fly.copy_atom<...>`
// wrapper, whose own `rebuildStaticValue` reconstructs the make_copy_atom op.
// This type carries no element-bit width, so there is nothing to rebuild on its
// own; report "already in normal form".
Value CopyOpMRAsyncCpType::rebuildStaticValue(OpBuilder &, Location, Value) const {
  return nullptr;
}

Attribute CopyOpMRAsyncCpType::getThrLayout() const {
  // 1 warp = 64 threads cooperate (SALU uniform issue).
  return FxLayout(FxC(64), FxC(1));
}

Attribute CopyOpMRAsyncCpType::getThrBitLayoutSrc() const {
  // 16 x 64B = 1024 byte tile per SME instruction.
  return FxLayout(FxShape(FxC(64), FxC(128)), FxStride(FxC(1), FxC(1)));
}

Attribute CopyOpMRAsyncCpType::getThrBitLayoutDst() const {
  auto *ctx = getContext();
  switch (getSmeSwizzle()) {
  case 0: // CUTLASS Layout_SME_I_16x512b_MN_INTER_Atom_Bits.
    return makeSmeAtomBitsLayout(
        ctx, SwizzleAttr::get(ctx, 0, 0, 0),
        FxLayout(FxShape(FxC(512), FxC(16)), FxStride(FxC(1), FxC(512))));
  case 1: // CUTLASS Layout_SME_I_16x512b_K_COLXFB8_Atom_Bits.
    return makeSmeAtomBitsLayout(
        ctx, SwizzleAttr::get(ctx, 2, 4, 4),
        FxLayout(FxShape(FxShape(FxC(4), FxC(4)), FxShape(FxC(32), FxC(4), FxC(4))),
                 FxStride(FxStride(FxC(32), FxC(2048)),
                          FxStride(FxC(1), FxC(512), FxC(128)))));
  case 2: // CUTLASS Layout_SME_I_16x512b_MN_ROWXFB8_Atom_Bits.
    return makeSmeAtomBitsLayout(
        ctx, ModSwizzleAttr::get(ctx, 2, 6, 2),
        FxLayout(FxShape(FxShape(FxC(8), FxC(4), FxC(4), FxC(4)),
                         FxShape(FxC(4), FxC(4))),
                 FxStride(FxStride(FxC(1), FxC(32), FxC(128), FxC(2048)),
                          FxStride(FxC(8), FxC(512)))));
  case 3: // CUTLASS Layout_SME_I_16x512b_MN_ROWXFB16_Atom_Bits.
    return makeSmeAtomBitsLayout(
        ctx, SwizzleAttr::get(ctx, 1, 7, 2),
        FxLayout(FxShape(FxShape(FxC(16), FxC(16), FxC(2)), FxShape(FxC(2), FxC(8))),
                 FxStride(FxStride(FxC(1), FxC(32), FxC(4096)),
                          FxStride(FxC(16), FxC(512)))));
  default:
    llvm_unreachable("CopyOpMRAsyncCpType::verify should reject unsupported swizzle values");
  }
}

Attribute CopyOpMRAsyncCpType::getThrBitLayoutRef() const { return getThrBitLayoutDst(); }

// MRAsyncCp lowers a one-directional async copy global(#fly_ixdl.sme_gmem) ->
// shared into the ixcc `ixdl.cp_async.*` op family. The core lives in the
// non-predicated emitAtomCall; SSA / predicated entry points delegate to it
// (mirrors FlyROCDL BufferCopyLDS).

LogicalResult CopyOpMRAsyncCpType::emitAtomCall(OpBuilder &builder, Location loc,
                                                Type copyAtomTyArg, Type srcMemTyArg,
                                                Type dstMemTyArg, Value, Value src,
                                                Value dst) const {
  auto copyAtomTy = dyn_cast<fly::CopyAtomType>(copyAtomTyArg);
  if (!copyAtomTy)
    return failure();

  auto srcMemTy = dyn_cast<fly::MemRefType>(srcMemTyArg);
  auto dstMemTy = dyn_cast<fly::MemRefType>(dstMemTyArg);
  if (!srcMemTy || !dstMemTy)
    return failure();

  if (!isTargetAddressSpace<SmeGmemAddressAttr>(srcMemTy.getAddressSpace()) ||
      !isGenericAddressSpace<fly::AddressSpace::Shared>(dstMemTy.getAddressSpace()))
    return failure();

  // dst shared pointer -> i32 sOffset (CUTLASS casts the smem pointer to uint32).
  Value sOffset = LLVM::PtrToIntOp::create(builder, loc, builder.getI32Type(), dst);

  // src SmeGmemFatPtr -> vector<4xi32> SmeDescriptor. The gmem pointer already
  // includes the accumulated byte_offset, so the hardware gOffset stays 0.
  SmeGmemFatPtr srcFat(srcMemTy.getPointerType(), src);
  Value gBase = srcFat.smeDescriptorVec(builder, loc);

  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  Value gOffset = zero; // tile byte offset already merged into the gBase pointer
  Value kop = zero;     // IX11::CacheOP::CacheAll

  int32_t valBits = copyAtomTy.getValBits();
  switch (getSmeSwizzle()) {
  case 0: // NoSwizzle: b32 row-major -> bi_sme_load_16x1b64
    if (valBits != 32)
      return mlir::emitError(loc) << "MRAsyncCp NoSwizzle requires valBits = 32, got " << valBits;
    IXDL::CpAsync_16x16_b32_RowOp::create(builder, loc, sOffset, gBase, gOffset, kop);
    return success();
  case 1: // Col: b8/b16/b32 col-major swizzle.
    if (valBits == 8) {
      IXDL::CpAsync_16x64_b8_ColOp::create(builder, loc, sOffset, gBase, gOffset, kop);
      return success();
    }
    if (valBits == 16) {
      IXDL::CpAsync_16x32_b16_ColOp::create(builder, loc, sOffset, gBase, gOffset, kop);
      return success();
    }
    if (valBits == 32) {
      IXDL::CpAsync_16x16_b32_ColOp::create(builder, loc, sOffset, gBase, gOffset, kop);
      return success();
    }
    return mlir::emitError(loc) << "MRAsyncCp Col requires valBits in {8, 16, 32}, got " << valBits;
  case 2: // Row8b: 8-bit row-major mod/add swizzle.
    if (valBits != 8)
      return mlir::emitError(loc) << "MRAsyncCp Row8b requires valBits = 8, got " << valBits;
    IXDL::CpAsync_16x64_b8_RowOp::create(builder, loc, sOffset, gBase, gOffset, kop);
    return success();
  case 3: // Row16b: 16-bit row-major xor swizzle.
    if (valBits != 16)
      return mlir::emitError(loc) << "MRAsyncCp Row16b requires valBits = 16, got " << valBits;
    IXDL::CpAsync_16x32_b16_RowOp::create(builder, loc, sOffset, gBase, gOffset, kop);
    return success();
  default:
    llvm_unreachable("CopyOpMRAsyncCpType::verify should reject unsupported swizzle values");
  }
}

LogicalResult CopyOpMRAsyncCpType::emitAtomCall(OpBuilder &builder, Location loc, Type, Type, Type,
                                                Type, Value, Value, Value, Value) const {
  // Predicated MRAsyncCp (smem_ptr = 0xffffff when !pred) is deferred (phase 7.2).
  return mlir::emitError(loc) << "predicated MRAsyncCp is not implemented yet";
}

FailureOr<Value> CopyOpMRAsyncCpType::emitAtomCallSSA(OpBuilder &builder, Location loc, Type,
                                                      Type copyAtomTyArg, Type srcTyArg,
                                                      Type dstTyArg, Value atomVal, Value src,
                                                      Value dst) const {
  if (failed(emitAtomCall(builder, loc, copyAtomTyArg, srcTyArg, dstTyArg, atomVal, src, dst)))
    return failure();
  // Async fire-and-forget: no SSA result.
  return Value{};
}

FailureOr<Value> CopyOpMRAsyncCpType::emitAtomCallSSA(OpBuilder &builder, Location loc, Type, Type,
                                                      Type, Type, Type, Value, Value, Value,
                                                      Value) const {
  // Predicated MRAsyncCp is deferred (phase 7.2).
  return mlir::emitError(loc) << "predicated MRAsyncCp is not implemented yet";
}

} // namespace mlir::fly_ixdl
