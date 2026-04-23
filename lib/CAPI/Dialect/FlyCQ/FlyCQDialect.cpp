// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "flydsl-c/FlyCQDialect.h"

#include "flydsl/Conversion/Passes.h"
#include "flydsl/Dialect/FlyCQ/IR/Dialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(FlyCQ, fly_cq, mlir::fly_cq::FlyCQDialect)

void mlirRegisterFlyToCQConversionPass(void) { mlir::registerFlyToCQConversionPass(); }
