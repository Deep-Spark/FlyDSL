// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYCQ_COPYATOM
#define FLYCQ_COPYATOM

include "flydsl/Dialect/FlyCQ/IR/Dialect.td"

//===----------------------------------------------------------------------===//
// Stateless CQ placeholder copy: LLVM load/store (register memrefs → llvm.ptr).
//===----------------------------------------------------------------------===//

def FlyCQ_CopyOpCQ_ScalarMem : FlyCQCL_CopyOp<"CopyOpCQ_ScalarMem", "scalar_mem", []> {
  let parameters = (ins "int32_t":$bitSize);
  let assemblyFormat = "`<` $bitSize `>`";

  let builders = [
    TypeBuilder<(ins "MLIRContext *":$ctx, "int32_t":$bitSize), [{
      return $_get(ctx, bitSize);
    }]>
  ];
  let genVerifyDecl = 1;
}

#endif // FLYCQ_COPYATOM
