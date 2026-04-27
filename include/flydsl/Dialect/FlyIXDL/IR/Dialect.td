#ifndef FLYIXDL_DIALECT
#define FLYIXDL_DIALECT

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
  let usePropertiesForAttributes = 1;
}

class FlyIXDL_MmaAtom<string typeName, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<FlyIXDL_Dialect, typeName, !listconcat(traits, [
        DeclareTypeInterfaceMethods<Fly_MayStaticTypeInterface>,
        DeclareTypeInterfaceMethods<Fly_MmaAtomTypeInterface>
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
