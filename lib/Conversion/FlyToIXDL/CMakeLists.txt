add_mlir_conversion_library(MLIRFlyToIXDL
  FlyToIXDL.cpp

  DEPENDS
  MLIRFlyIXDLIncGen
  FlyToIXDLPassIncGen

  LINK_LIBS PUBLIC
  MLIRFlyIXDLDialect
  MLIRIXDLDialect
  MLIRIR
  MLIRPass
)
