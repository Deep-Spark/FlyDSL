// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYIXDL_ATTR_DEFS
#define FLYIXDL_ATTR_DEFS

include "mlir/IR/AttrTypeBase.td"
include "flydsl/Dialect/FlyIXDL/IR/Dialect.td"

// Iluvatar global address space for SME async copies. Mirrors the engineering
// pattern of FlyROCDL_BufferDescAddressAttr (#fly_rocdl.buffer_desc).
def FlyIXDL_SmeGmemAddressAttr : FlyIXDL_Attr<"SmeGmemAddress", "sme_gmem"> {}

def FlyIXDL_SmexLayoutPlain : I32EnumAttrCase<"Plain", 0, "plain">;
def FlyIXDL_SmexLayoutMtx : I32EnumAttrCase<"Mtx", 1, "mtx">;
def FlyIXDL_SmexLayout
    : FlyIXDL_I32EnumAttr<
          "SmexLayout",
          "CQ SMEX shared-memory layout", [FlyIXDL_SmexLayoutPlain, FlyIXDL_SmexLayoutMtx]>;

// loadn16 / loadn64: G2S consume pairing only (16-row vs 64-row SmexMtx tile).
// Not x1 vs x2 and not a distinct mtx_loadn ISA; both patterns emit the same
// *x2 load. Keep the enum description string stable -- it appears in parse
// diagnostics matched by FileCheck.
def FlyIXDL_MtxLoadPatternLoadn16 : I32EnumAttrCase<"Loadn16", 0, "loadn16">;
def FlyIXDL_MtxLoadPatternLoadn64 : I32EnumAttrCase<"Loadn64", 1, "loadn64">;
def FlyIXDL_MtxLoadPattern
    : FlyIXDL_I32EnumAttr<"MtxLoadPattern", "CQ SMEX matrix-load pattern",
                          [FlyIXDL_MtxLoadPatternLoadn16, FlyIXDL_MtxLoadPatternLoadn64]>;

def FlyIXDL_MtxGatherDirectionRow : I32EnumAttrCase<"Row", 0, "row">;
def FlyIXDL_MtxGatherDirectionCol : I32EnumAttrCase<"Col", 1, "col">;
def FlyIXDL_MtxGatherDirection
    : FlyIXDL_I32EnumAttr<"MtxGatherDirection", "CQ matrix-load gather direction",
                          [FlyIXDL_MtxGatherDirectionRow, FlyIXDL_MtxGatherDirectionCol]>;

#endif // FLYIXDL_ATTR_DEFS
