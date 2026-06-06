// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYIXDL_MMAATOM
#define FLYIXDL_MMAATOM

include "flydsl/Dialect/FlyIXDL/IR/Dialect.td"

// Iluvatar MR (ivcore11) TCU matrix-multiply-accumulate atom. Mirrors CuTe's
// MMA_Traits<IX11_...> (5 _TT atoms) and the FlyROCDL MmaOpCDNA3_MFMA factory:
// parameterised by MNK and the (A, B) -> accumulator element types. Lowers to
// the ixcc `ixdl.mmad` op.
def FlyIXDL_MmaOpMRMma : FlyIXDL_MmaOp<"MmaOpMRMma", "mr.mma"> {
  let parameters = (ins
    "int32_t":$m,
    "int32_t":$n,
    "int32_t":$k,
    "Type":$elemTyA,
    "Type":$elemTyB,
    "Type":$elemTyAcc
  );
  let assemblyFormat =
      "`<` $m `,` $n `,` $k `,` `(` $elemTyA `,` $elemTyB `)` `->` $elemTyAcc `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "int32_t":$m, "int32_t":$n, "int32_t":$k,
        "Type":$elemTyA, "Type":$elemTyB, "Type":$elemTyAcc), [{
      return $_get(elemTyA.getContext(), m, n, k, elemTyA, elemTyB, elemTyAcc);
    }]>
  ];
  let genVerifyDecl = 1;
}

#endif // FLYIXDL_MMAATOM
