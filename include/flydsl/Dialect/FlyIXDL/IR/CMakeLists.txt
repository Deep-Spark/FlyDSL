set(LLVM_TARGET_DEFINITIONS Dialect.td)

mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS Enums.td)
mlir_tablegen(Enums.h.inc -gen-enum-decls)
mlir_tablegen(Enums.cpp.inc -gen-enum-defs)
mlir_tablegen(AttrDefs.h.inc -gen-attrdef-decls -attrdefs-dialect=fly_ixdl)
mlir_tablegen(AttrDefs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=fly_ixdl)

set(LLVM_TARGET_DEFINITIONS Atom.td)
mlir_tablegen(Atom.h.inc -gen-typedef-decls -typedefs-dialect=fly_ixdl)
mlir_tablegen(Atom.cpp.inc -gen-typedef-defs -typedefs-dialect=fly_ixdl)

add_public_tablegen_target(MLIRFlyIXDLIncGen)
