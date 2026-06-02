// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYIXDL_DIALECT
#define FLYIXDL_DIALECT

include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

include "flydsl/Dialect/Fly/IR/FlyInterfaces.td"

def FlyIXDL_Dialect : Dialect {
  let name = "fly_ixdl";
  let cppNamespace = "::mlir::fly_ixdl";

  let dependentDialects = [
    "IXDL::IXDLDialect"
  ];

  let useDefaultTypePrinterParser = 1;
}

class FlyIXDL_MmaOp<string typeName, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<FlyIXDL_Dialect, typeName, !listconcat(traits, [
        DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>,
        DeclareTypeInterfaceMethods<Fly_MmaOpTypeInterface>
      ])> {
  let mnemonic = typeMnemonic;
}

class FlyIXDL_CopyOp<string typeName, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<FlyIXDL_Dialect, typeName, !listconcat(traits, [
        DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>,
        DeclareTypeInterfaceMethods<Fly_CopyOpTypeInterface>
      ])> {
  let mnemonic = typeMnemonic;
}

class FlyIXDL_StatefulCopyOp<string typeName, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<FlyIXDL_Dialect, typeName, !listconcat(traits, [
        DeclareTypeInterfaceMethods<Fly_CopyOpTypeInterface>,
        DeclareTypeInterfaceMethods<Fly_StatefulOpTypeInterface>
      ])> {
  let mnemonic = typeMnemonic;
  let extraClassDeclaration = [{
    static std::optional<unsigned> getFieldIndex(AtomStateField field);
  }];
}

#endif // FLYIXDL_DIALECT
