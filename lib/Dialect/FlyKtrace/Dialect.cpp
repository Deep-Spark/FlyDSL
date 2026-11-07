// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/IR/DialectImplementation.h"
// Generated type printer uses TypeSwitch. LLVM 24 includes it from
// DialectImplementation.h; ixcc's LLVM 22 does not.
#include "llvm/ADT/TypeSwitch.h"

#include "flydsl/Dialect/FlyKtrace/IR/Dialect.h"

using namespace mlir;
using namespace mlir::fly_ktrace;

#include "flydsl/Dialect/FlyKtrace/IR/Dialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "flydsl/Dialect/FlyKtrace/IR/TypeDefs.cpp.inc"

#define GET_OP_CLASSES
#include "flydsl/Dialect/FlyKtrace/IR/Ops.cpp.inc"

void FlyKtraceDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "flydsl/Dialect/FlyKtrace/IR/TypeDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "flydsl/Dialect/FlyKtrace/IR/Ops.cpp.inc"
      >();
}
