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

#endif // FLYIXDL_COPYATOM
