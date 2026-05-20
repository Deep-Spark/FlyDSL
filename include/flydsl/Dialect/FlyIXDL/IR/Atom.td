// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYIXDL_ATOM
#define FLYIXDL_ATOM

include "mlir/IR/EnumAttr.td"

include "flydsl/Dialect/FlyIXDL/IR/Dialect.td"

// MR async-copy atom state mirrors FlyROCDL BufferCopyLDS offsets and adds an
// Iluvatar-specific optional runtime pitch override:
// - soffset: per-call source offset in elements
// - imm_offset: per-call shared-memory byte offset
// - stride_byte: optional runtime global pitch in bytes (0 means "use src layout")
def FlyIXDL_AtomStateField : I32EnumAttr<"AtomStateField", "", [
  I32EnumAttrCase<"Soffset",   0, "soffset">,
  I32EnumAttrCase<"ImmOffset", 1, "imm_offset">,
  I32EnumAttrCase<"StrideByte", 2, "stride_byte">
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = FlyIXDL_Dialect.cppNamespace;
}

include "flydsl/Dialect/FlyIXDL/IR/CopyAtom.td"

#endif // FLYIXDL_ATOM
