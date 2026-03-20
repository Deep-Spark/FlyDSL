add_mlir_public_c_api_library(MLIRCPIFlyIXDL
  FlyIXDLDialect.cpp

  DEPENDS
  FlyConversionPassIncGen
  MLIRFlyDialect
  MLIRFlyToIXDL

  LINK_LIBS PUBLIC
  MLIRFlyDialect
  MLIRFlyToIXDL
)

target_include_directories(MLIRCPIFlyIXDL PRIVATE
  ${PROJECT_SOURCE_DIR}/thirdparty/llvm-project/mlir/include
  ${PROJECT_SOURCE_DIR}/thirdparty/llvm-project/llvm/include
)
