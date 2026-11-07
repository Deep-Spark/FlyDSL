// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYIXDL_COPYATOM
#define FLYIXDL_COPYATOM

include "flydsl/Dialect/FlyIXDL/IR/Dialect.td"
include "flydsl/Dialect/FlyIXDL/IR/MmaAtom.td"

def FlyIXDL_CopyOpIvcore11AsyncCp : FlyIXDL_CopyOp<"CopyOpIvcore11AsyncCp", "ivcore11.async_copy"> {
  let parameters = (ins
      // SME swizzle state: 0=NoSwizzle, 1=Col, 2=Row8b, 3=Row16b
      "int32_t":$smeSwizzle);
  let assemblyFormat = "`<` `swizzle` `=` $smeSwizzle `>`";
  let genVerifyDecl = 1;
}

// Iluvatar ivcore11 async store (SME Store series): warp-collective shared ->
// global copy, the S2G counterpart of ivcore11.async_copy. Lowers to the
// ixdl.cp_async.store.b{64,128,256} ops (bi_sme_store_b64/b128/b256), one
// instruction moving storeBytes bytes (64/128/256) per warp.
def FlyIXDL_CopyOpIvcore11AsyncStore
    : FlyIXDL_CopyOp<"CopyOpIvcore11AsyncStore", "ivcore11.async_store"> {
  let parameters = (ins
    // SME store width in bytes per instruction: 64, 128, or 256
    "int32_t":$storeBytes
  );
  let assemblyFormat = "`<` `bytes` `=` $storeBytes `>`";
  let genVerifyDecl = 1;
}

#endif // FLYIXDL_COPYATOM
