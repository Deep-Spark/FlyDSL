// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "flydsl/Dialect/FlyIXDL/IR/Dialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::fly;
using namespace mlir::fly_ixdl;

#include "flydsl/Dialect/FlyIXDL/IR/Dialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "flydsl/Dialect/FlyIXDL/IR/CopyAtom.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "flydsl/Dialect/FlyIXDL/IR/AttrDefs.cpp.inc"

void FlyIXDLDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "flydsl/Dialect/FlyIXDL/IR/CopyAtom.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "flydsl/Dialect/FlyIXDL/IR/AttrDefs.cpp.inc"
      >();
}
