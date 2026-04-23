set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name FlyToCQ)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix FlyToCQ)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix FlyToCQ)

add_mlir_generic_tablegen_target(FlyToCQPassIncGen)
