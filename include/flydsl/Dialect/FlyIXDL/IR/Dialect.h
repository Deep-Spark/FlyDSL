// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYDSL_DIALECT_FLYIXDL_IR_DIALECT_H
#define FLYDSL_DIALECT_FLYIXDL_IR_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/LLVMIR/IXDLDialect.h"
#include "mlir/IR/Dialect.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"

#include "flydsl/Dialect/FlyIXDL/IR/AtomStateEnums.h.inc"
#include "flydsl/Dialect/FlyIXDL/IR/Dialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "flydsl/Dialect/FlyIXDL/IR/Atom.h.inc"

#define GET_OP_CLASSES
#include "flydsl/Dialect/FlyIXDL/IR/Ops.h.inc"

namespace mlir::fly_ixdl {} // namespace mlir::fly_ixdl

#endif // FLYDSL_DIALECT_FLYIXDL_IR_DIALECT_H
