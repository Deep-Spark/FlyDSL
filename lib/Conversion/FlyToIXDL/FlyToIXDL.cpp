// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "flydsl/Conversion/FlyToIXDL/FlyToIXDL.h"

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_FLYTOIXDLCONVERSIONPASS
#include "flydsl/Conversion/FlyToIXDL/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

class FlyToIXDLConversionPass
    : public mlir::impl::FlyToIXDLConversionPassBase<FlyToIXDLConversionPass> {
public:
  using mlir::impl::FlyToIXDLConversionPassBase<
      FlyToIXDLConversionPass>::FlyToIXDLConversionPassBase;

  void runOnOperation() override {}
};

} // namespace
