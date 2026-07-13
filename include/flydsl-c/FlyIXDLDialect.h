// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYDSL_C_FLYIXDLDIALECT_H
#define FLYDSL_C_FLYIXDLDIALECT_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(FlyIXDL, fly_ixdl);

MLIR_CAPI_EXPORTED void mlirRegisterFlyToIXDLConversionPass(void);

/// Backend plugin registration: insert all Iluvatar dialects into \p registry.
MLIR_CAPI_EXPORTED void flydsl_register_iluvatar_dialects(MlirDialectRegistry registry);
/// Backend plugin registration: register all Iluvatar passes.
MLIR_CAPI_EXPORTED void flydsl_register_iluvatar_passes(void);

#ifdef __cplusplus
}
#endif

#endif // FLYDSL_C_FLYIXDLDIALECT_H
