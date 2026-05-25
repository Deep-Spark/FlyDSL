add_mlir_dialect_library(MLIRFlyIXDLDialect
  Dialect.cpp

  DEPENDS
  MLIRFlyIXDLIncGen

  LINK_LIBS PUBLIC
  MLIRIXDLDialect
)
