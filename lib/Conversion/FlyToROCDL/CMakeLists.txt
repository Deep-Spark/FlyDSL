add_mlir_conversion_library(MLIRFlyToROCDL
  FlyToROCDL.cpp

  DEPENDS
  MLIRFlyIncGen
  MLIRFlyROCDLIncGen
  FlyToROCDLPassIncGen

  LINK_LIBS PUBLIC
  MLIRFlyDialect
  MLIRFlyROCDLDialect

  MLIRArithDialect
  MLIRIR
  MLIRIXDLDialect
  MLIRLLVMDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTransforms
  MLIRVectorDialect
)
