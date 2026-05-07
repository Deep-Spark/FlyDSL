// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYDSL_DIALECT_FLY_UTILS_UNIVERSALCOPYUTILS_H
#define FLYDSL_DIALECT_FLY_UTILS_UNIVERSALCOPYUTILS_H

#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

#include <utility>

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/Fly/Utils/LayoutUtils.h"

namespace mlir::fly {

inline FailureOr<std::pair<int64_t, int64_t>>
getCoalescedLeafCountAndStride(fly::MemRefType memRefTy) {
  auto layoutAttr = dyn_cast<LayoutAttr>(memRefTy.getLayout());
  if (!layoutAttr)
    return failure();
  LayoutBuilder<LayoutAttr> builder(memRefTy.getContext());
  auto coalesced = layoutCoalesce(builder, layoutAttr);
  if (!coalesced.isLeaf())
    return failure();
  auto shape = coalesced.getShape().getLeafAsInt();
  auto stride = coalesced.getStride().getLeafAsInt();
  if (!shape.isStatic() || !stride.isStatic())
    return failure();
  return std::make_pair<int64_t, int64_t>(shape.getValue(), stride.getValue());
}

inline bool isReadyForUniversalCopyAtom(CopyAtomType copyAtomTy, fly::MemRefType memRefTy) {
  auto universalCopy = dyn_cast<CopyOpUniversalCopyType>(copyAtomTy.getCopyOp());
  if (!universalCopy)
    return false;

  auto countAndStride = getCoalescedLeafCountAndStride(memRefTy);
  if (failed(countAndStride))
    return false;

  auto [count, stride] = *countAndStride;
  int64_t elemBits = memRefTy.getElemTy().getIntOrFloatBitWidth();
  int64_t copyBits = universalCopy.getBitSize();
  int64_t totalBits = count * elemBits;
  if (totalBits != copyBits)
    return false;

  int64_t contiguousBits = (count <= 1 || stride == 1) ? totalBits : elemBits;
  return contiguousBits >= copyBits;
}

inline LogicalResult verifyUniversalCopyOperand(Operation *op, StringRef operandName,
                                                CopyAtomType copyAtomTy, fly::MemRefType memRefTy) {
  auto universalCopy = dyn_cast<CopyOpUniversalCopyType>(copyAtomTy.getCopyOp());
  if (!universalCopy)
    return success();

  auto countAndStride = getCoalescedLeafCountAndStride(memRefTy);
  if (failed(countAndStride)) {
    return op->emitOpError() << operandName
                             << " memref layout must coalesce to a single static leaf for "
                             << copyAtomTy;
  }

  auto [count, stride] = *countAndStride;
  int64_t elemBits = memRefTy.getElemTy().getIntOrFloatBitWidth();
  int64_t copyBits = universalCopy.getBitSize();
  int64_t totalBits = count * elemBits;
  if (totalBits != copyBits) {
    return op->emitOpError() << operandName << " memref covers " << totalBits
                             << " bits after coalescing, but " << copyAtomTy << " expects "
                             << copyBits << " bits";
  }

  int64_t contiguousBits = (count <= 1 || stride == 1) ? totalBits : elemBits;
  if (contiguousBits < copyBits) {
    return op->emitOpError() << operandName << " memref contiguous bit count " << contiguousBits
                             << " is smaller than copy granularity " << copyBits;
  }

  return success();
}

} // namespace mlir::fly

#endif // FLYDSL_DIALECT_FLY_UTILS_UNIVERSALCOPYUTILS_H
