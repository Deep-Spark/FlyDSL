// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYIXDL_MMAATOM
#define FLYIXDL_MMAATOM

include "flydsl/Dialect/FlyIXDL/IR/Dialect.td"

//===----------------------------------------------------------------------===//
// MmaOp MR (ivcore11) — MMAD
//
// Cooperative warp-level matrix-multiply-accumulate, fixed shape 16x16x16 for
// the FP16 SME GEMM path. Lowers to `ixdl.mmad`.
//===----------------------------------------------------------------------===//

def FlyIXDL_MmaOpMR_MMAD : FlyIXDL_MmaOp<"MmaOpMR_MMAD", "mr.mmad", []> {
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
    TypeBuilderWithInferredContext<(ins "int32_t":$m, "int32_t":$n, "int32_t":$k,
        "Type":$elemTyA, "Type":$elemTyB, "Type":$elemTyAcc), [{
      return $_get(elemTyA.getContext(), m, n, k, elemTyA, elemTyB, elemTyAcc);
    }]>
  ];
  let genVerifyDecl = 1;
}

#endif // FLYIXDL_MMAATOM
