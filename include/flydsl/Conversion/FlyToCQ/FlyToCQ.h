// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef CONVERSION_FLYTOCQ_FLYTOCQ_H
#define CONVERSION_FLYTOCQ_FLYTOCQ_H

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL_FLYTOCQCONVERSIONPASS
#include "flydsl/Conversion/FlyToCQ/Passes.h.inc"
} // namespace mlir

#endif // CONVERSION_FLYTOCQ_FLYTOCQ_H
