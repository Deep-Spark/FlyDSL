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
  let summary = "Simplify IXDL shared-memory address arithmetic";
  let description = [{
      On IXDL targets:
        * Collapse equivalent lane-affine forms in shared copy offsets
          (e.g. `(lane_id ^ 1) ^ 33` → `lane_id ^ 32`).
        * B8 Row8b S2R: Euclidean thr→byte + ModSwizzle closed forms, then
          `readfirstlane` on warp-uniform bases of `base + 4*lane`.
    }];

  let dependentDialects = [
    "arith::ArithDialect",
    "gpu::GPUDialect",
    "LLVM::LLVMDialect"
  ];
}
