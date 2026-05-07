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

  let description = [{
    Iluvatar-specific MMA/Copy atom types for FlyDSL's layout algebra.

    This dialect mirrors ``fly_rocdl`` in purpose: it defines concrete
    ``Fly_MmaOpTypeInterface`` types (and, later, ``Fly_CopyOpTypeInterface``
    types) for the Iluvatar ``ivcore11`` tensor cores. The lowerings emit
    ops from upstream ixcc's ``ixdl`` dialect (specifically
    ``ixdl.mmad``), so the only hard dependency is on ``IXDL::IXDLDialect``.
  }];

  let dependentDialects = [
    "IXDL::IXDLDialect"
  ];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
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

#endif // FLYIXDL_DIALECT
