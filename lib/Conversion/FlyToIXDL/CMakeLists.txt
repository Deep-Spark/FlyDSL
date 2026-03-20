add_mlir_conversion_library(MLIRFlyToIXDL
  FlyToIXDL.cpp

  DEPENDS
  FlyConversionPassIncGen
  MLIRFlyDialect

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRLLVMDialect
  MLIRSCFDialect
  MLIRVectorDialect
  MLIRFlyDialect
)
