set(LLVM_TARGET_DEFINITIONS Dialect.td)

mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS Atom.td)
mlir_tablegen(Atom.h.inc -gen-typedef-decls -typedefs-dialect=fly_cq)
mlir_tablegen(Atom.cpp.inc -gen-typedef-defs -typedefs-dialect=fly_cq)
mlir_tablegen(AtomEnums.h.inc -gen-enum-decls -typedefs-dialect=fly_cq)
mlir_tablegen(AtomEnums.cpp.inc -gen-enum-defs -typedefs-dialect=fly_cq)

add_public_tablegen_target(MLIRFlyCQIncGen)
