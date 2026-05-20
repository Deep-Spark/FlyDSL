// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYDSL_DIALECT_FLYIXDL_UTILS_CPASYNCUTILS_H
#define FLYDSL_DIALECT_FLYIXDL_UTILS_CPASYNCUTILS_H

#include "mlir/Dialect/LLVMIR/IXDLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

namespace mlir::fly_ixdl {

/// MR (ivcore11) async copy shapes supported by ixcc ``createCpAsyncShapedOp``.
enum class MRCpAsyncKind {
  k4x64_b8_Row,
  k4x64_b8_Col,
  k16x64_b8_Row,
  k16x64_b8_Col,
  k4x32_b16_Row,
  k4x32_b16_Col,
  k16x32_b16_Row,
  k16x32_b16_Col,
  k1x1b64,
  k1x4b64,
  k1x8b64,
  k1x16b64,
  k4x16_b32_Row,
  k8x16_b32_Row,
  k16x16_b32_Row,
  k16x16_b32_Col,
};

constexpr int32_t mrAsyncCopyBitSize(int64_t rows, int64_t cols, int64_t elemBits) {
  return static_cast<int32_t>(rows * cols * elemBits);
}

Value scaleSoffsetToBytes(OpBuilder &builder, Location loc, Value soffsetRaw, int64_t elemBits);
Value buildSOffset(OpBuilder &builder, Location loc, Value dstPtr, Value immOffset);
Value buildGBase(OpBuilder &builder, Location loc, Value srcPtr, Value scaledSoffsetBytes,
                 Value strideBytes);
Value buildGOffsetZero(OpBuilder &builder, Location loc);
Value buildKopOne(OpBuilder &builder, Location loc);
/// Derive global pitch (bytes) from static src layout when possible.
/// Returns an empty Value when layout is missing, dynamic, out of i32 range,
/// misaligned to 64B, or otherwise unsupported.
Value buildStrideBytesFromMemref(OpBuilder &builder, Location loc, fly::MemRefType srcMemTy,
                                 MRCpAsyncKind kind);

LogicalResult emitCpAsyncByKind(OpBuilder &builder, Location loc, MRCpAsyncKind kind, Value sOffset,
                                Value gBase, Value gOffset, Value kop);

} // namespace mlir::fly_ixdl

#endif // FLYDSL_DIALECT_FLYIXDL_UTILS_CPASYNCUTILS_H
