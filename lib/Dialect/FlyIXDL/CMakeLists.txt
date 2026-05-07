add_mlir_dialect_library(MLIRFlyIXDLDialect
  Dialect.cpp
  IX11/MmaAtom.cpp
  IX11/CopyAtom.cpp

  DEPENDS
  MLIRFlyIXDLIncGen

  LINK_LIBS PUBLIC
  MLIRFlyDialect
  MLIRIXDLDialect
  MLIRLLVMDialect
)
