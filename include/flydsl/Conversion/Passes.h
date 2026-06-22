// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYDSL_CONVERSION_PASSES_H
#define FLYDSL_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"

#if __has_include("flydsl/Conversion/FlyToIXDL/Passes.h.inc")
#include "flydsl/Conversion/FlyToIXDL/FlyToIXDL.h"
#endif

#if __has_include("flydsl/Conversion/FlyToROCDL/Passes.h.inc")
#include "flydsl/Conversion/FlyToROCDL/FlyToROCDL.h"
#endif

namespace mlir {

#if __has_include("flydsl/Conversion/FlyToROCDL/Passes.h.inc")
#define GEN_PASS_REGISTRATION
#include "flydsl/Conversion/FlyToROCDL/Passes.h.inc"
#endif

} // namespace mlir

#endif // FLYDSL_CONVERSION_PASSES_H
