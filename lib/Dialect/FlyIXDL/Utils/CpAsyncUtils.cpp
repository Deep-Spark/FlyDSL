// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "flydsl/Dialect/FlyIXDL/Utils/CpAsyncUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::fly;

namespace mlir::fly_ixdl {

/// Number of contiguous elements along the source row, used to compute the SME
/// descriptor stride (word3). Walks to the trailing static leaf of the layout
/// shape; returns 0 when it cannot be determined statically (callers then emit
/// a descriptor whose stride is intended to be overridden via the explicit
/// `sme_g2s_*` arch path).
static int64_t getSrcLineElems(fly::MemRefType srcMemTy) {
  Attribute layoutAttr = srcMemTy.getLayout();
  auto layout = dyn_cast_or_null<LayoutAttr>(layoutAttr);
  if (auto composed = dyn_cast_or_null<ComposedLayoutAttr>(layoutAttr))
    layout = dyn_cast_or_null<LayoutAttr>(composed.getOuter());
  if (!layout)
    return 0;

  IntTupleAttr shape = layout.getShape();
  IntTupleAttr cur = shape;
  while (cur && !cur.isLeaf()) {
    int32_t r = cur.rank();
    if (r == 0)
      return 0;
    cur = dyn_cast_or_null<IntTupleAttr>(cur.at(r - 1));
  }
  if (!cur)
    return 0;
  IntAttr leaf = cur.extractIntFromLeaf();
  if (!leaf.isStatic())
    return 0;
  return leaf.getValue();
}

SmeCpAsyncOperands buildSmeCpAsyncOperands(OpBuilder &builder, Location loc,
                                           fly::MemRefType srcMemTy, fly::MemRefType dstMemTy,
                                           Value srcPtr, Value dstPtr, Value immOffsetBytes) {
  MLIRContext *ctx = builder.getContext();
  Type i32 = IntegerType::get(ctx, 32);
  Type i64 = IntegerType::get(ctx, 64);

  // Shared destination offset: low 32 bits of the shared pointer, plus the
  // per-call immediate byte offset from the atom state.
  Value dstAddr = LLVM::PtrToIntOp::create(builder, loc, i64, dstPtr);
  Value sOffset = arith::TruncIOp::create(builder, loc, i32, dstAddr);
  if (immOffsetBytes)
    sOffset = arith::AddIOp::create(builder, loc, sOffset, immOffsetBytes);

  // Global descriptor: [addr_lo, addr_hi, 0xFFFFFFFF, stride_bytes].
  Value gAddr = LLVM::PtrToIntOp::create(builder, loc, i64, srcPtr);
  Value gAddrLow = arith::TruncIOp::create(builder, loc, i32, gAddr);
  Value c32 = arith::ConstantIntOp::create(builder, loc, 32, 64);
  Value gAddrHigh = arith::ShRUIOp::create(builder, loc, gAddr, c32);
  gAddrHigh = arith::TruncIOp::create(builder, loc, i32, gAddrHigh);
  Value minusOne = arith::ConstantIntOp::create(builder, loc, 0xFFFFFFFF, 32);

  int64_t elemBytes = srcMemTy.getElemTy().getIntOrFloatBitWidth() / 8;
  int64_t lineElems = getSrcLineElems(srcMemTy);
  Value stride = arith::ConstantIntOp::create(builder, loc, lineElems * elemBytes, 32);

  VectorType v4i32 = VectorType::get({4}, i32);
  Value gBase = LLVM::PoisonOp::create(builder, loc, v4i32);
  auto insert = [&](Value vec, Value elem, int64_t pos) -> Value {
    Value idx = arith::ConstantIndexOp::create(builder, loc, pos);
    return vector::InsertOp::create(builder, loc, elem, vec, idx);
  };
  gBase = insert(gBase, gAddrLow, 0);
  gBase = insert(gBase, gAddrHigh, 1);
  gBase = insert(gBase, minusOne, 2);
  gBase = insert(gBase, stride, 3);

  Value gOffset = arith::ConstantIntOp::create(builder, loc, 0, 32);
  Value kop = arith::ConstantIntOp::create(builder, loc, 1, 32);

  return SmeCpAsyncOperands{sOffset, gBase, gOffset, kop};
}

} // namespace mlir::fly_ixdl
