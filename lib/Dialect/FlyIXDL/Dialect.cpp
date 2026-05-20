// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "flydsl/Dialect/FlyIXDL/IR/Dialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::fly_ixdl;

#include "flydsl/Dialect/FlyIXDL/IR/AtomStateEnums.cpp.inc"
#include "flydsl/Dialect/FlyIXDL/IR/Dialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "flydsl/Dialect/FlyIXDL/IR/Atom.cpp.inc"

#define GET_OP_CLASSES
#include "flydsl/Dialect/FlyIXDL/IR/Ops.cpp.inc"

void FlyIXDLDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "flydsl/Dialect/FlyIXDL/IR/Atom.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "flydsl/Dialect/FlyIXDL/IR/Ops.cpp.inc"
      >();
}
