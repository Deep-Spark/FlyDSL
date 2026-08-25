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

// Iluvatar MR async store (SME Store series): warp-collective shared ->
// global copy, the S2G counterpart of mr.async_copy. Lowers to the
// ixdl.cp_async.store.b{64,128,256} ops (bi_sme_store_b64/b128/b256), one
// instruction moving storeBytes bytes (64/128/256) per warp.
def FlyIXDL_CopyOpMRAsyncStore : FlyIXDL_CopyOp<"CopyOpMRAsyncStore", "mr.async_store"> {
  let parameters = (ins
    // SME store width in bytes per instruction: 64, 128, or 256
    "int32_t":$storeBytes
  );
  let assemblyFormat = "`<` `bytes` `=` $storeBytes `>`";
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

// CQ SMEX matrix shared-to-register load. This atom consumes the SmexMtx
// format written by cq.smex_cp<..., layout = mtx>; it must not consume a
// LegacySme byte-swizzled buffer. The shared pointer is warp-uniform and
// already contains the tile-base and EmPart selection.
//
// $pattern (loadn16 / loadn64) only records which G2S SmexMtx tile height this
// atom is meant to consume. Both patterns lower to the same ixdl.mtx_loadn_*x2
// op (64 bits/lane). A 64-row shared footprint is covered by multiple atom
// calls with different EmPart / slot bases, not a different ISA opcode.
def FlyIXDL_CopyOpCQMtxLoadn : FlyIXDL_CopyOp<"CopyOpCQMtxLoadn", "cq.mtx_loadn"> {
  let parameters = (ins "MtxLoadPattern":$pattern, "MtxGatherDirection":$direction,
      "int32_t":$elemBits);
  let assemblyFormat = "`<` $pattern `,` $direction `,` $elemBits `>`";
  let genVerifyDecl = 1;
}

#endif // FLYIXDL_COPYATOM
