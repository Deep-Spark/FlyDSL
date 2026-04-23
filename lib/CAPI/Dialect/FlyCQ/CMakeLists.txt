add_mlir_public_c_api_library(MLIRCPIFlyCQ
  FlyCQDialect.cpp

  LINK_LIBS PUBLIC
  MLIRFlyCQDialect
  MLIRFlyToCQ
  MLIRCAPIIR
)
