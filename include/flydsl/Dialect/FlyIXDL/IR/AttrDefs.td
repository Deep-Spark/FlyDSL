// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYIXDL_ATTR_DEFS
#define FLYIXDL_ATTR_DEFS

include "mlir/IR/AttrTypeBase.td"
include "flydsl/Dialect/FlyIXDL/IR/Dialect.td"

// Iluvatar global address space for SME async copies. Mirrors the engineering
// pattern of FlyROCDL_BufferDescAddressAttr (#fly_rocdl.buffer_desc).
def FlyIXDL_SmeGmemAddressAttr : FlyIXDL_Attr<"SmeGmemAddress", "sme_gmem"> {}

#endif // FLYIXDL_ATTR_DEFS
