// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "flydsl/Dialect/FlyIXDL/IR/Dialect.h"

using namespace mlir;
using namespace mlir::fly;
using namespace mlir::fly_ixdl;

#include "flydsl/Dialect/FlyIXDL/IR/Dialect.cpp.inc"

#include "flydsl/Dialect/FlyIXDL/IR/Enums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "flydsl/Dialect/FlyIXDL/IR/AttrDefs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "flydsl/Dialect/FlyIXDL/IR/Atom.cpp.inc"

void FlyIXDLDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "flydsl/Dialect/FlyIXDL/IR/AttrDefs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "flydsl/Dialect/FlyIXDL/IR/Atom.cpp.inc"
      >();
}
