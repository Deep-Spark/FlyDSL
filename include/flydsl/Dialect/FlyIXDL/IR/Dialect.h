// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYDSL_DIALECT_FLYIXDL_IR_DIALECT_H
#define FLYDSL_DIALECT_FLYIXDL_IR_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/LLVMIR/IXDLDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

#include "flydsl/Dialect/FlyIXDL/IR/Dialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "flydsl/Dialect/FlyIXDL/IR/CopyAtom.h.inc"
#define GET_ATTRDEF_CLASSES
#include "flydsl/Dialect/FlyIXDL/IR/AttrDefs.h.inc"

namespace mlir::fly_ixdl {} // namespace mlir::fly_ixdl

#endif // FLYDSL_DIALECT_FLYIXDL_IR_DIALECT_H
