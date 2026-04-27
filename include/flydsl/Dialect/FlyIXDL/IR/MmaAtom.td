#ifndef FLYIXDL_MMAATOM
#define FLYIXDL_MMAATOM

include "flydsl/Dialect/FlyIXDL/IR/Dialect.td"

//===----------------------------------------------------------------------===//
// MmaAtom Ivcore11 MMAD (TCU 16x16x16)
//===----------------------------------------------------------------------===//

def FlyIXDL_MmaAtomIvcore11_MMAD : FlyIXDL_MmaAtom<
    "MmaAtomIvcore11_MMAD", "atom.ivcore11.mmad", []> {
  let parameters = (ins
    "int32_t":$m,
    "int32_t":$n,
    "int32_t":$k,
    "Type":$elemTyA,
    "Type":$elemTyB,
    "Type":$elemTyAcc
  );
  let assemblyFormat = "`<` custom<MNKDimensionList>($m, $n, $k) `,` `(` $elemTyA `,` $elemTyB `)` `->` $elemTyAcc `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "int32_t":$m, "int32_t":$n, "int32_t":$k,
      "Type":$elemTyA, "Type":$elemTyB, "Type":$elemTyAcc
    ), [{
      return $_get(elemTyA.getContext(), m, n, k, elemTyA, elemTyB, elemTyAcc);
    }]>
  ];
  let genVerifyDecl = 1;
}

#endif // FLYIXDL_MMAATOM
