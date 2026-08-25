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

LogicalResult CopyOpMRAsyncCpType::verify(function_ref<InFlightDiagnostic()> emitError,
                                          int32_t smeSwizzle) {
  // SME swizzle four-state enum: 0=NoSwizzle, 1=Col, 2=Row8b, 3=Row16b.
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
  // Warp-collective SME load: modeled as a single logical thread that owns the
  // whole tile (thread layout = Layout<1>). The 64-lane cooperation is internal
  // to the hardware instruction and is not exposed to layout algebra /
  // TiledCopy partitioning.
  return FxLayout(FxC(1), FxC(1));
}

Attribute CopyOpMRAsyncCpType::getThrBitLayoutSrc() const {
  // One SME instruction moves a fixed 16 x 512b = 8192-bit footprint, owned by
  // the single logical thread: src layout (1,8192):(0,1) -- thr mode size 1
  // (injective), val mode 8192 contiguous bits.
  return FxLayout(FxShape(FxC(1), FxC(8192)), FxStride(FxC(0), FxC(1)));
}

Attribute CopyOpMRAsyncCpType::getThrBitLayoutDst() const {
  // Keep CopyAtom layout as atom footprint / thread-value mapping only. The SME
  // physical shared-memory swizzle layout is modeled separately by a FlyIXDL
  // shared-layout helper (CopyAtom and shared-layout are kept orthogonal).
  return getThrBitLayoutSrc();
}

Attribute CopyOpMRAsyncCpType::getThrBitLayoutRef() const { return getThrBitLayoutDst(); }

// MRAsyncCp lowers a one-directional async copy global(#fly_ixdl.sme_gmem) ->
// shared into the `ixdl.cp_async.*` op family. All four entry points (plain /
// predicated x value / SSA) share emitMRAsyncCp below.

namespace {

// Predicated MR G2S: a false predicate selects an out-of-range SLB address
// (0xffffff) so the hardware drops the whole transfer, including the global
// read. The atom stays in straight-line code; an scf.if would split a region
// and break SSA dominance in the unrolled G2S loops these atoms sit in.
constexpr int64_t kInvalidSlbOffset = 0xffffff;

// ``predVal`` is a plain i1 (or null for the unpredicated form): the value-form
// entry point loads it out of its register memref, while the SSA form already
// receives it promoted.
LogicalResult emitMRAsyncCp(OpBuilder &builder, Location loc, int32_t smeSwizzle,
                            Type copyAtomTyArg, Type srcMemTyArg, Type dstMemTyArg, Value src,
                            Value dst, Value predVal) {
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

  // dst shared pointer -> i32 sOffset (the smem pointer is cast to uint32).
  Value sOffset = LLVM::PtrToIntOp::create(builder, loc, builder.getI32Type(), dst);
  if (predVal) {
    if (!predVal.getType().isInteger(1))
      return mlir::emitError(loc) << "predicated MRAsyncCp expects an i1 predicate, got "
                                  << predVal.getType();
    Value invalidSlb = arith::ConstantIntOp::create(builder, loc, kInvalidSlbOffset, 32);
    sOffset = arith::SelectOp::create(builder, loc, predVal, sOffset, invalidSlb);
  }

  // src SmeGmemFatPtr -> vector<4xi32> SME descriptor built from the raw,
  // loop-invariant gmem pointer. The accumulated per-tile byte_offset is passed
  // as the hardware gOffset operand (a 32-bit offset added on top of the
  // descriptor base) instead of being folded into the 64-bit base, so the
  // descriptor hoists out of a tile loop and only the narrow offset advances
  // (constant offsets fold into the goffimm immediate).
  SmeGmemFatPtr srcFat(srcMemTy.getPointerType(), src);
  Value gBase = srcFat.smeDescriptorVec(builder, loc);
  Value gOffset = srcFat.byteOffset(builder, loc);

  Value kop = arith::ConstantIntOp::create(builder, loc, 0, 32); // CacheAll cache op

  int32_t valBits = copyAtomTy.getValBits();
  switch (smeSwizzle) {
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

} // namespace

LogicalResult CopyOpMRAsyncCpType::emitAtomCall(OpBuilder &builder, Location loc,
                                                Type copyAtomTyArg, Type srcMemTyArg,
                                                Type dstMemTyArg, Value, Value src,
                                                Value dst) const {
  return emitMRAsyncCp(builder, loc, getSmeSwizzle(), copyAtomTyArg, srcMemTyArg, dstMemTyArg, src,
                       dst, /*predVal=*/Value{});
}

LogicalResult CopyOpMRAsyncCpType::emitAtomCall(OpBuilder &builder, Location loc,
                                                Type copyAtomTyArg, Type srcMemTyArg,
                                                Type dstMemTyArg, Type predMemTyArg, Value,
                                                Value src, Value dst, Value pred) const {
  auto predMemTy = dyn_cast<fly::MemRefType>(predMemTyArg);
  if (!predMemTy)
    return failure();
  Value predVal = LLVM::LoadOp::create(builder, loc, predMemTy.getElemTy(), pred);
  return emitMRAsyncCp(builder, loc, getSmeSwizzle(), copyAtomTyArg, srcMemTyArg, dstMemTyArg, src,
                       dst, predVal);
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

FailureOr<Value> CopyOpMRAsyncCpType::emitAtomCallSSA(OpBuilder &builder, Location loc, Type,
                                                      Type copyAtomTyArg, Type srcTyArg,
                                                      Type dstTyArg, Type, Value, Value src,
                                                      Value dst, Value pred) const {
  // Here ``pred`` has already been promoted out of register memory to an i1.
  if (failed(emitMRAsyncCp(builder, loc, getSmeSwizzle(), copyAtomTyArg, srcTyArg, dstTyArg, src,
                           dst, pred)))
    return failure();
  return Value{};
}

//===----------------------------------------------------------------------===//
// CopyOpMRAsyncStoreType: SME store series (shared -> global), the S2G
// counterpart of MRAsyncCp. One warp-collective instruction moves
// ``storeBytes`` bytes (64/128/256) from shared to global memory.
//===----------------------------------------------------------------------===//

LogicalResult
CopyOpMRAsyncStoreType::verify(function_ref<InFlightDiagnostic()> emitError,
                               int32_t storeBytes) {
  if (storeBytes != 64 && storeBytes != 128 && storeBytes != 256)
    return emitError() << "unsupported storeBytes = " << storeBytes
                       << " for MRAsyncStore (expected 64, 128 or 256)";
  return success();
}

bool CopyOpMRAsyncStoreType::isStatic() const { return true; }

// Same rationale as CopyOpMRAsyncCpType: the enclosing !fly.copy_atom wrapper
// rebuilds the make_copy_atom op; this type has nothing to rebuild on its own.
Value CopyOpMRAsyncStoreType::rebuildStaticValue(OpBuilder &, Location, Value) const {
  return nullptr;
}

Attribute CopyOpMRAsyncStoreType::getThrLayout() const {
  // Warp-collective SME store: modeled as a single logical thread that owns the
  // whole tile; the 64-lane cooperation is internal to the hardware instruction.
  return FxLayout(FxC(1), FxC(1));
}

Attribute CopyOpMRAsyncStoreType::getThrBitLayoutSrc() const {
  // One SME store instruction moves storeBytes bytes, owned by the single
  // logical thread: src layout (1,bits):(0,1) -- thr mode size 1 (injective),
  // val mode contiguous bits.
  int64_t bits = static_cast<int64_t>(getStoreBytes()) * 8;
  return FxLayout(FxShape(FxC(1), FxC(bits)), FxStride(FxC(0), FxC(1)));
}

Attribute CopyOpMRAsyncStoreType::getThrBitLayoutDst() const { return getThrBitLayoutSrc(); }

Attribute CopyOpMRAsyncStoreType::getThrBitLayoutRef() const { return getThrBitLayoutDst(); }

namespace {

// S2G direction of emitMRAsyncCp: the shared pointer becomes the hardware
// sOffset operand and the SmeGmem fat pointer provides the descriptor / gOffset,
// exactly like the G2S path but with src/dst roles swapped.
LogicalResult emitMRAsyncStore(OpBuilder &builder, Location loc, int32_t storeBytes,
                               Type copyAtomTyArg, Type srcMemTyArg, Type dstMemTyArg,
                               Value src, Value dst) {
  auto copyAtomTy = dyn_cast<fly::CopyAtomType>(copyAtomTyArg);
  if (!copyAtomTy)
    return failure();

  auto srcMemTy = dyn_cast<fly::MemRefType>(srcMemTyArg);
  auto dstMemTy = dyn_cast<fly::MemRefType>(dstMemTyArg);
  if (!srcMemTy || !dstMemTy)
    return failure();

  if (!isGenericAddressSpace<fly::AddressSpace::Shared>(srcMemTy.getAddressSpace()) ||
      !isTargetAddressSpace<SmeGmemAddressAttr>(dstMemTy.getAddressSpace()))
    return failure();

  // src shared pointer -> i32 sOffset (the smem pointer is cast to uint32).
  Value sOffset = LLVM::PtrToIntOp::create(builder, loc, builder.getI32Type(), src);

  // dst SmeGmemFatPtr -> vector<4xi32> SME descriptor + 32-bit gOffset (same
  // hoisting contract as the G2S path: descriptor from the raw loop-invariant
  // base, per-tile byte offset as the hardware gOffset operand).
  SmeGmemFatPtr dstFat(dstMemTy.getPointerType(), dst);
  Value gBase = dstFat.smeDescriptorVec(builder, loc);
  Value gOffset = dstFat.byteOffset(builder, loc);

  Value kop = arith::ConstantIntOp::create(builder, loc, 0, 32); // CacheAll cache op

  switch (storeBytes) {
  case 64:
    IXDL::CpAsync_Store_b64Op::create(builder, loc, sOffset, gBase, gOffset, kop);
    return success();
  case 128:
    IXDL::CpAsync_Store_b128Op::create(builder, loc, sOffset, gBase, gOffset, kop);
    return success();
  case 256:
    IXDL::CpAsync_Store_b256Op::create(builder, loc, sOffset, gBase, gOffset, kop);
    return success();
  default:
    llvm_unreachable("CopyOpMRAsyncStoreType::verify should reject unsupported store widths");
  }
}

LogicalResult emitPredicatedStoreUnsupported(Location loc) {
  // The G2S predication trick (false pred -> invalid SLB offset -> hardware
  // drops the transfer) has no S2G analogue: redirecting the shared offset
  // still issues the global write, and redirecting the global offset would
  // fault. There is no safe sink for a suppressed store, so predicated S2G is
  // rejected instead of silently writing.
  return mlir::emitError(loc) << "MRAsyncStore does not support predication: a suppressed "
                                 "shared->global store has no safe sink";
}

} // namespace

LogicalResult CopyOpMRAsyncStoreType::emitAtomCall(OpBuilder &builder, Location loc,
                                                   Type copyAtomTyArg, Type srcMemTyArg,
                                                   Type dstMemTyArg, Value, Value src,
                                                   Value dst) const {
  return emitMRAsyncStore(builder, loc, getStoreBytes(), copyAtomTyArg, srcMemTyArg, dstMemTyArg,
                          src, dst);
}

LogicalResult CopyOpMRAsyncStoreType::emitAtomCall(OpBuilder &builder, Location loc, Type, Type,
                                                   Type, Type, Value, Value, Value, Value) const {
  return emitPredicatedStoreUnsupported(loc);
}

FailureOr<Value> CopyOpMRAsyncStoreType::emitAtomCallSSA(OpBuilder &builder, Location loc, Type,
                                                         Type copyAtomTyArg, Type srcTyArg,
                                                         Type dstTyArg, Value atomVal, Value src,
                                                         Value dst) const {
  if (failed(emitAtomCall(builder, loc, copyAtomTyArg, srcTyArg, dstTyArg, atomVal, src, dst)))
    return failure();
  // Async fire-and-forget: no SSA result.
  return Value{};
}

FailureOr<Value> CopyOpMRAsyncStoreType::emitAtomCallSSA(OpBuilder &builder, Location loc, Type,
                                                         Type, Type, Type, Type, Value, Value,
                                                         Value, Value) const {
  if (failed(emitPredicatedStoreUnsupported(loc)))
    return failure();
  return Value{};
}

} // namespace mlir::fly_ixdl
