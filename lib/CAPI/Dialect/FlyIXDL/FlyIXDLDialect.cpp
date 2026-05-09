// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "flydsl-c/FlyIXDLDialect.h"

#include "flydsl/Dialect/FlyIXDL/IR/Dialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(FlyIXDL, fly_ixdl, mlir::fly_ixdl::FlyIXDLDialect)

void flydsl_register_iluvatar_dialects(MlirDialectRegistry registry) {
  unwrap(registry)->insert<mlir::fly_ixdl::FlyIXDLDialect>();
}

void flydsl_register_iluvatar_passes(void) {}
