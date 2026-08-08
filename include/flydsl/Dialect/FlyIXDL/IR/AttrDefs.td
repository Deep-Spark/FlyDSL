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

#endif // FLYIXDL_ATTR_DEFS
