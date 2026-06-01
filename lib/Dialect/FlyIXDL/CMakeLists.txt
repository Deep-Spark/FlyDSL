add_mlir_dialect_library(MLIRFlyIXDLDialect
  Dialect.cpp
  MR/CopyAtom.cpp

  DEPENDS
  MLIRFlyIXDLIncGen
  MLIRFlyIXDLCopyAtomIncGen

  LINK_LIBS PUBLIC
  MLIRIXDLDialect
  MLIRFlyDialect
)
