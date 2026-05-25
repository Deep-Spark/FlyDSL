// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

include "mlir/Pass/PassBase.td"

def FlyToIXDLConversionPass : Pass<"convert-fly-to-ixdl"> {
  let summary = "Lower Fly to MLIR upstream and IXDL dialects";
  let dependentDialects = [
    "arith::ArithDialect",
    "scf::SCFDialect",
    "vector::VectorDialect",
    "LLVM::LLVMDialect",
    "IXDL::IXDLDialect"
  ];
}
