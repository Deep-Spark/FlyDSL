add_mlir_dialect_library(MLIRFlyIXDLDialect
  Dialect.cpp
  MR/CopyAtom.cpp
  MR/MmaAtom.cpp

  DEPENDS
  MLIRFlyIXDLIncGen
  MLIRFlyIXDLCopyAtomIncGen

  LINK_LIBS PUBLIC
  MLIRIXDLDialect
  MLIRFlyDialect
)
