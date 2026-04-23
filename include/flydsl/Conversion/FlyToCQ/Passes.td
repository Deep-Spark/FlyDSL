// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

include "mlir/Pass/PassBase.td"

def FlyToCQConversionPass : Pass<"convert-fly-to-cq"> {
  let summary = "Lower Fly to MLIR upstream dialects for Iluvatar CQ (placeholder pipeline)";
  let dependentDialects = [
    "arith::ArithDialect",
    "scf::SCFDialect",
    "vector::VectorDialect",
    "LLVM::LLVMDialect",
    "ROCDL::ROCDLDialect"
  ];
}
