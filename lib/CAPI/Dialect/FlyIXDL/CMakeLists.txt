add_mlir_public_c_api_library(MLIRCPIFlyIXDL
  FlyIXDLDialect.cpp
  LINK_LIBS PUBLIC
  MLIRFlyIXDLDialect
  MLIRFlyToIXDL
)
