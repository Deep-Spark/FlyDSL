add_mlir_dialect_library(MLIRFlyIXDLDialect
  Dialect.cpp
  MR/CopyAtom.cpp
  Utils/CpAsyncUtils.cpp

  DEPENDS
  MLIRFlyIXDLIncGen

  LINK_LIBS PUBLIC
  MLIRFlyDialect
  MLIRIXDLDialect
  MLIRArithDialect
  MLIRLLVMDialect
  MLIRVectorDialect
)
