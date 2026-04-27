add_mlir_conversion_library(MLIRFlyToIXDL
  FlyToIXDL.cpp

  DEPENDS
  MLIRFlyIncGen
  MLIRFlyIXDLIncGen
  FlyConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRFlyDialect
  MLIRFlyIXDLDialect

  MLIRArithDialect
  MLIRIR
  MLIRIXDLDialect
  MLIRLLVMDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTransforms
  MLIRVectorDialect
)
