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

def IXDLAddressSimplifyPass : Pass<"fly-ixdl-address-simplify"> {
  let summary = "Simplify IXDL swizzled shared-memory address arithmetic";
  let description = [{
      Recognize lane-id expressions in shared-memory copy addresses on IXDL targets
      and rewrite equivalent affine forms to simpler address arithmetic, such as
      `(lane_id ^ 1) ^ 33` to `lane_id ^ 32`.
    }];

  let dependentDialects = [
    "arith::ArithDialect",
    "gpu::GPUDialect"
  ];
}
