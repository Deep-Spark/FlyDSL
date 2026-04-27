#ifndef FLYDSL_CONVERSION_PASSES
#define FLYDSL_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def FlyToROCDLConversionPass : Pass<"convert-fly-to-rocdl"> {
  let summary = "Lower Fly to MLIR upstream and rocdl dialects ";  
  let dependentDialects = [
    "arith::ArithDialect",
    "scf::SCFDialect",
    "vector::VectorDialect",
    "LLVM::LLVMDialect",
    "ROCDL::ROCDLDialect"
  ];
}

def FlyToIXDLConversionPass : Pass<"convert-fly-to-ixdl"> {
  let summary = "Lower Fly to MLIR upstream and ixdl dialects for Iluvatar";
  let dependentDialects = [
    "arith::ArithDialect",
    "scf::SCFDialect",
    "vector::VectorDialect",
    "LLVM::LLVMDialect",
    "IXDL::IXDLDialect"
  ];
}

#endif // FLYDSL_CONVERSION_PASSES
