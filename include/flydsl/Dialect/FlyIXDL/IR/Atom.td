// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYIXDL_ATOM
#define FLYIXDL_ATOM

include "flydsl/Dialect/FlyIXDL/IR/Dialect.td"

// State-field mnemonics intentionally mirror FlyROCDL so the shared Python
// `fx.copy(..., soffset=..., imm_offset=...)` / `atom_set_value` plumbing needs
// no backend-specific changes.
def FlyIXDL_AtomStateField : I32EnumAttr<"AtomStateField", "", [
  I32EnumAttrCase<"Soffset",   0, "soffset">,
  I32EnumAttrCase<"ImmOffset", 1, "imm_offset">
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = FlyIXDL_Dialect.cppNamespace;
}

include "flydsl/Dialect/FlyIXDL/IR/MmaAtom.td"
include "flydsl/Dialect/FlyIXDL/IR/CopyAtom.td"

#endif // FLYIXDL_ATOM
