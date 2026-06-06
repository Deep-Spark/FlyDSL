set(LLVM_TARGET_DEFINITIONS Dialect.td)

mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)

add_public_tablegen_target(MLIRFlyIXDLIncGen)

set(LLVM_TARGET_DEFINITIONS AttrDefs.td)
mlir_tablegen(AttrDefs.h.inc -gen-attrdef-decls -attrdefs-dialect=fly_ixdl)
mlir_tablegen(AttrDefs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=fly_ixdl)

set(LLVM_TARGET_DEFINITIONS CopyAtom.td)
mlir_tablegen(CopyAtom.h.inc -gen-typedef-decls -typedefs-dialect=fly_ixdl)
mlir_tablegen(CopyAtom.cpp.inc -gen-typedef-defs -typedefs-dialect=fly_ixdl)

add_public_tablegen_target(MLIRFlyIXDLCopyAtomIncGen)
