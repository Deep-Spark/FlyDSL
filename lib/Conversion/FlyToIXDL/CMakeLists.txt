add_mlir_conversion_library(MLIRFlyToIXDL
  FlyToIXDL.cpp

  DEPENDS
  MLIRFlyIXDLIncGen
  FlyToIXDLPassIncGen

  LINK_LIBS PUBLIC
  MLIRFlyDialect
  MLIRFlyIXDLDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIXDLDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTransforms
  MLIRVectorDialect
)
