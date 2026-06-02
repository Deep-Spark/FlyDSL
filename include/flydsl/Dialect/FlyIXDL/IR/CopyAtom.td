// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYIXDL_COPYATOM
#define FLYIXDL_COPYATOM

include "flydsl/Dialect/FlyIXDL/IR/Dialect.td"

//===----------------------------------------------------------------------===//
// CopyOp MR (ivcore11) — Stateful async G2S copy (SME `cp.async`)
//
// One `copy_atom_call` issues one fixed-shape global->shared async transfer.
// Mutable state (`soffset`, `imm_offset`) mirrors ROCDL `BufferCopyLDS`.
// Lowers to `ixdl.cp_async.16x64.b8.row`.
//===----------------------------------------------------------------------===//

def FlyIXDL_CopyOpMR_AsyncCopy_16x64_b8_Row
    : FlyIXDL_StatefulCopyOp<"CopyOpMRAsyncCopy16x64B8Row", "mr.async_copy.16x64.b8.row", []> {
  let parameters = (ins
    "int32_t":$bitSize
  );
  let assemblyFormat = "`<` $bitSize `>`";
}

//===----------------------------------------------------------------------===//
// CopyOp MR (ivcore11) — SLB Load (Shared -> Register fragment)
//
// Shared-memory fragment load that feeds MMAD operands. MR has no hardware
// `ldmatrix`; this lowers to `llvm.load` (per the copy-atom capability matrix).
//===----------------------------------------------------------------------===//

def FlyIXDL_CopyOpMR_SLBLoad
    : FlyIXDL_CopyOp<"CopyOpMRSLBLoad", "mr.slb_load", []> {
  let parameters = (ins
    "int32_t":$bitSize
  );
  let assemblyFormat = "`<` $bitSize `>`";
}

#endif // FLYIXDL_COPYATOM
