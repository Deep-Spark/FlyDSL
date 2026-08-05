// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYIXDL_MMAATOM
#define FLYIXDL_MMAATOM

include "flydsl/Dialect/FlyIXDL/IR/Dialect.td"

// Iluvatar MR (ivcore11) TCU matrix-multiply-accumulate atom. Parameterised by
// MNK and the (A, B) -> accumulator element types. Lowers to `ixdl.mmad`.
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

// Iluvatar CQ (ivcore30) TCU matrix-multiply-accumulate atom.
// Multiplicands: f16/bf16 (K=16->f32), s8/u8 (K=32->i32), f8E4M3/f8E5M2
// (K=32->f32 or f16; A/B may mix).
// (M,N) in {16x16, 32x32, 16x64, 64x16}; long-mtx keeps K.
def FlyIXDL_MmaOpCQMma : FlyIXDL_MmaOp<"MmaOpCQMma", "cq.mma"> {
  let parameters = (ins "int32_t":$m, "int32_t":$n, "int32_t":$k, "Type":$elemTyA, "Type":$elemTyB,
      "Type":$elemTyAcc);
  let assemblyFormat = "`<` $m `,` $n `,` $k `,` `(` $elemTyA `,` $elemTyB `)` `->` $elemTyAcc `>`";

  let builders = [TypeBuilderWithInferredContext<(ins "int32_t":$m, "int32_t":$n, "int32_t":$k,
                                                     "Type":$elemTyA, "Type":$elemTyB,
                                                     "Type":$elemTyAcc),
                                                 [{
      return $_get(elemTyA.getContext(), m, n, k, elemTyA, elemTyB, elemTyAcc);
    }]>];
  let genVerifyDecl = 1;
}

#endif // FLYIXDL_MMAATOM
