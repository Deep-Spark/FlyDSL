// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYCQ_DIALECT
#define FLYCQ_DIALECT

include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

include "flydsl/Dialect/Fly/IR/FlyInterfaces.td"

def FlyCQ_Dialect : Dialect {
  let name = "fly_cq";
  let cppNamespace = "::mlir::fly_cq";

  let dependentDialects = [
    "LLVM::LLVMDialect",
    "vector::VectorDialect"
  ];

  let useDefaultTypePrinterParser = 1;
}

class FlyCQCL_MmaOp<string typeName, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<FlyCQ_Dialect, typeName, !listconcat(traits, [
        DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>,
        DeclareTypeInterfaceMethods<Fly_MmaOpTypeInterface>
      ])> {
  let mnemonic = typeMnemonic;
}

class FlyCQCL_CopyOp<string typeName, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<FlyCQ_Dialect, typeName, !listconcat(traits, [
        DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>,
        DeclareTypeInterfaceMethods<Fly_CopyOpTypeInterface>
      ])> {
  let mnemonic = typeMnemonic;
}

#endif // FLYCQ_DIALECT
