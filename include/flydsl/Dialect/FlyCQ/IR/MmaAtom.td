// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYCQ_MMAATOM
#define FLYCQ_MMAATOM

include "flydsl/Dialect/FlyCQ/IR/Dialect.td"

//===----------------------------------------------------------------------===//
// Placeholder CQ matmul atom (LLVM + vector lowering; not vendor-specific ISA).
//===----------------------------------------------------------------------===//

def FlyCQ_MmaOpCQ_MatmulF32 : FlyCQCL_MmaOp<"MmaOpCQ_MatmulF32", "matmul_f32", []> {
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
    TypeBuilderWithInferredContext<(ins "int32_t":$m, "int32_t":$n, "int32_t":$k, "Type":$elemTyA, "Type":$elemTyB, "Type":$elemTyAcc), [{
      return $_get(elemTyA.getContext(), m, n, k, elemTyA, elemTyB, elemTyAcc);
    }]>
  ];
  let genVerifyDecl = 1;
}

#endif // FLYCQ_MMAATOM
