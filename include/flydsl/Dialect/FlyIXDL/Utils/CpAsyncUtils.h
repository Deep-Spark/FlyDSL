// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYDSL_DIALECT_FLYIXDL_UTILS_CPASYNCUTILS_H
#define FLYDSL_DIALECT_FLYIXDL_UTILS_CPASYNCUTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

namespace mlir::fly_ixdl {

/// Packed operands for an SME `cp.async` issue, mirroring the descriptor that
/// ixcc's `IXGPUAsyncCopyLowering` assembles. See
/// `.cursor/rules/design/iluvatar-mr-async-copy-op.md` §4.2.
struct SmeCpAsyncOperands {
  Value sOffset; ///< i32 shared-memory offset (low 32 bits of dst pointer).
  Value gBase;   ///< vector<4xi32> global descriptor (addr lo/hi, -1, stride).
  Value gOffset; ///< i32 global byte offset (0 for the base issue).
  Value kop;     ///< i32 kop selector (1).
};

/// Build the SME `cp.async` descriptor operands from the converted (LLVM)
/// pointers and the Fly memref types. `srcPtr` must be a global `!llvm.ptr`
/// (AS 1) and `dstPtr` a shared `!llvm.ptr` (AS 3). `immOffsetBytes` is the
/// per-call shared byte offset taken from the atom state (may be null).
SmeCpAsyncOperands buildSmeCpAsyncOperands(OpBuilder &builder, Location loc,
                                           fly::MemRefType srcMemTy, fly::MemRefType dstMemTy,
                                           Value srcPtr, Value dstPtr, Value immOffsetBytes);

} // namespace mlir::fly_ixdl

#endif // FLYDSL_DIALECT_FLYIXDL_UTILS_CPASYNCUTILS_H
