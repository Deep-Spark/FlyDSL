// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "flydsl/Dialect/FlyCQ/IR/Dialect.h"

using namespace mlir;
using namespace mlir::fly;
using namespace mlir::fly_cq;

#include "flydsl/Dialect/FlyCQ/IR/Dialect.cpp.inc"
#include "flydsl/Dialect/FlyCQ/IR/AtomEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "flydsl/Dialect/FlyCQ/IR/Atom.cpp.inc"

void FlyCQDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "flydsl/Dialect/FlyCQ/IR/Atom.cpp.inc"
      >();
}
