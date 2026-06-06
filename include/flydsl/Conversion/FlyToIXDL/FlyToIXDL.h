// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef CONVERSION_FLYTOIXDL_FLYTOIXDL_H
#define CONVERSION_FLYTOIXDL_FLYTOIXDL_H

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL_FLYTOIXDLCONVERSIONPASS
#define GEN_PASS_REGISTRATION
#include "flydsl/Conversion/FlyToIXDL/Passes.h.inc"
} // namespace mlir

#endif // CONVERSION_FLYTOIXDL_FLYTOIXDL_H
