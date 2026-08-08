// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYIXDL_COPYATOM
#define FLYIXDL_COPYATOM

include "flydsl/Dialect/FlyIXDL/IR/Dialect.td"
include "flydsl/Dialect/FlyIXDL/IR/AttrDefs.td"
include "flydsl/Dialect/FlyIXDL/IR/MmaAtom.td"

def FlyIXDL_CopyOpMRAsyncCp : FlyIXDL_CopyOp<"CopyOpMRAsyncCp", "mr.async_copy"> {
  let parameters = (ins
    // SME swizzle state: 0=NoSwizzle, 1=Col, 2=Row8b, 3=Row16b
    "int32_t":$smeSwizzle
  );
  let assemblyFormat = "`<` `swizzle` `=` $smeSwizzle `>`";
  let genVerifyDecl = 1;
}

// Iluvatar CQ SMEX G2S (smex_load{,n}_* / *_pred_*). Stateful atom carries
// row_mask (i64), col_mask (i32), optional pred (i1). Defaults: masks all-1s,
// pred disabled (non-pred IXDL op). rows: 4/16/64.
def FlyIXDL_CopyOpCQSmexCp : FlyIXDL_StatefulCopyOp<"CopyOpCQSmexCp", "cq.smex_cp"> {
  let parameters = (ins "int32_t":$rows, "SmexLayout":$layout);
  let assemblyFormat = "`<` $rows `,` `layout` `=` $layout `>`";
  let genVerifyDecl = 1;
}

#endif // FLYIXDL_COPYATOM
