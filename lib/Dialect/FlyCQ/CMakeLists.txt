add_mlir_dialect_library(MLIRFlyCQDialect
  Dialect.cpp
  CQ/CopyAtom.cpp
  CQ/MmaAtom.cpp

  DEPENDS
  MLIRFlyCQIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFlyDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRSCFDialect
  MLIRVectorDialect
)
