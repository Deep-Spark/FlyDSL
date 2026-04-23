add_mlir_dialect_library(MLIRFlyIXDLDialect
  Dialect.cpp
  IX11/MmaAtom.cpp

  DEPENDS
  MLIRFlyIXDLIncGen

  LINK_LIBS PUBLIC
  MLIRFlyDialect
  MLIRIXDLDialect
  MLIRLLVMDialect
)
