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
// (`loadn`) require a separate CopyOp (`cq.mtx_loadn` below).
def FlyIXDL_CopyOpCQAsyncCp : FlyIXDL_CopyOp<"CopyOpCQAsyncCp", "cq.async_copy"> {
  let parameters = (ins "int32_t":$row, "int32_t":$col, "int32_t":$transpose);
  let assemblyFormat = "`<` $row `,` $col `,` `transpose` `=` $transpose `>`";
  let genVerifyDecl = 1;
}

// Iluvatar CQ (ivcore30) shared->register matrix load (non-interleave loadn).
//
// SmexMtx vs LegacySme (see IXGPU SharedMemoryLayoutUtils / MtxLoadUtils):
//   * SmexMtx (`pattern` = loadn16|loadn64): pairs with SME mtx G2S
//     (`smex_loadn_{16,64}x1b64_mtx`). Shared layout is dword/SMEX; swizzle
//     policy is Bypass (do not apply MR LegacyByteSwizzle). S2R lowers to
//     `ixdl.mtx_loadn_*`.
//   * LegacySme: MR async-copy-sme / CQ compat `loadi64` / unpatterned SME.
//     Shared layout uses historical byte-bank swizzle; S2R is `ixdl.ldmatrix`
//     (or `mtx_loadi_*`). Mixing SmexMtx and LegacySme evidence on one shared
//     buffer is illegal.
//
// Parameters mirror IXGPU `mtx_pattern` + gather direction:
//   pattern   0 = loadn16, 1 = loadn64 (both select mtx_loadn_*; pattern
//             documents the SmexMtx G2S pairing / EmPart geometry)
//   direction 0 = row (A / transpose gather), 1 = col (B / direct)
//   bitWidth  8 or 16 (multiplicand element width)
//   x2        0 or 1; SoT default is 1 (64b / vector<2xi32> per lane)
def FlyIXDL_CopyOpCQMtxLoadn : FlyIXDL_CopyOp<"CopyOpCQMtxLoadn", "cq.mtx_loadn"> {
  let parameters = (ins "int32_t":$pattern, "int32_t":$direction, "int32_t":$bitWidth,
      "int32_t":$x2);
  let assemblyFormat =
      "`<` `pattern` `=` $pattern `,` `dir` `=` $direction `,` `b` `=` $bitWidth `,` "
      "`x2` `=` $x2 `>`";
  let genVerifyDecl = 1;
}

#endif // FLYIXDL_COPYATOM
