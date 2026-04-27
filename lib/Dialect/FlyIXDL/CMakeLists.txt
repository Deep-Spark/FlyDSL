add_mlir_dialect_library(MLIRFlyIXDLDialect
  Dialect.cpp
  Ivcore11/MmaAtom.cpp
  Ivcore11/CopyAtom.cpp

  DEPENDS
  MLIRFlyIXDLIncGen
)
