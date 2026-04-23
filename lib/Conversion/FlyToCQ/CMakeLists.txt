add_mlir_conversion_library(MLIRFlyToCQ
  FlyToCQ.cpp

  DEPENDS
  MLIRFlyIncGen
  MLIRFlyCQIncGen
  FlyToCQPassIncGen

  LINK_LIBS PUBLIC
  MLIRFlyDialect
  MLIRFlyCQDialect

  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRROCDLDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTransforms
  MLIRVectorDialect
)
