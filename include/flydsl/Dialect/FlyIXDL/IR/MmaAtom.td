// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYIXDL_MMAATOM
#define FLYIXDL_MMAATOM

include "flydsl/Dialect/FlyIXDL/IR/Dialect.td"

//===----------------------------------------------------------------------===//
// MmaOp IX11 — ivcore11 MMAD
//
// Covers all five CUTLASS ``IX11_*_TT`` atoms:
//   * 16x16x16  F32 = F32*F32 + F32
//   * 16x16x16  F32 = F16*F16 + F32
//   * 16x16x16  F32 = BF16*BF16 + F32
//   * 16x16x32  S32 = S8*S8 + S32
//   * 16x16x32  S32 = U8*U8 + S32
//
// The underlying MLIR op is ``ixdl.mmad`` (see ixcc's IXDLOps.td). It is
// configured through a shape + row/col layout attributes + multiplicand
// MMADTypes attribute; we always use row/row (``TT`` in CUTLASS parlance)
// which matches the intrinsics exposed by ``mma_ix11.hpp``.
//===----------------------------------------------------------------------===//

def FlyIXDL_MmaOpIX11_MMAD : FlyIXDL_MmaOp<"MmaOpIX11_MMAD", "ix11.mmad", []> {
  let parameters = (ins
    "int32_t":$m,
    "int32_t":$n,
    "int32_t":$k,
    "Type":$elemTyA,
    "Type":$elemTyB,
    "Type":$elemTyAcc
  );
  let assemblyFormat =
      "`<` custom<MNKDimensionList>($m, $n, $k) `,` `(` $elemTyA `,` $elemTyB `)` `->` $elemTyAcc `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "int32_t":$m, "int32_t":$n, "int32_t":$k,
                                        "Type":$elemTyA, "Type":$elemTyB, "Type":$elemTyAcc), [{
      return $_get(elemTyA.getContext(), m, n, k, elemTyA, elemTyB, elemTyAcc);
    }]>
  ];
  let genVerifyDecl = 1;
}

#endif // FLYIXDL_MMAATOM
