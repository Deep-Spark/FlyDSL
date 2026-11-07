add_mlir_dialect_library(MLIRFlyIXDLDialect
  Dialect.cpp
  Ivcore11/CopyAtom.cpp
  Ivcore11/MmaAtom.cpp

  DEPENDS
  MLIRFlyIXDLIncGen
  MLIRFlyIXDLCopyAtomIncGen

  LINK_LIBS PUBLIC
  MLIRIXDLDialect
  MLIRFlyDialect
)
