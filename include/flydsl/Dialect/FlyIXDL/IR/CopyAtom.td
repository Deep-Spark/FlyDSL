// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYIXDL_COPYATOM
#define FLYIXDL_COPYATOM

include "flydsl/Dialect/FlyIXDL/IR/Dialect.td"
include "flydsl/Dialect/FlyIXDL/IR/MmaAtom.td"

def FlyIXDL_CopyOpMRAsyncCp : FlyIXDL_CopyOp<"CopyOpMRAsyncCp", "mr.async_copy"> {
  let parameters = (ins
    // SME swizzle state: 0=NoSwizzle, 1=Col, 2=Row8b, 3=Row16b
    "int32_t":$smeSwizzle
  );
  let assemblyFormat = "`<` `swizzle` `=` $smeSwizzle `>`";
  let genVerifyDecl = 1;
}

// Iluvatar CQ (ivcore30) enhanced-SME asynchronous global-to-shared copy.
// The shape uses ixcc AsyncCopyUtils' normalized (row, col, transpose)
// convention. This atom intentionally covers G2S only; CQ S2R matrix loads
// (`loadn`) require a separate CopyOp.
def FlyIXDL_CopyOpCQAsyncCp : FlyIXDL_CopyOp<"CopyOpCQAsyncCp", "cq.async_copy"> {
  let parameters = (ins "int32_t":$row, "int32_t":$col, "int32_t":$transpose);
  let assemblyFormat = "`<` $row `,` $col `,` `transpose` `=` $transpose `>`";
  let genVerifyDecl = 1;
}

#endif // FLYIXDL_COPYATOM
