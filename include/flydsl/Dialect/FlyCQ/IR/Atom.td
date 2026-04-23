// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYCQ_ATOM
#define FLYCQ_ATOM

include "flydsl/Dialect/FlyCQ/IR/Dialect.td"

def FlyCQ_PlaceholderKind : I32EnumAttr<"PlaceholderKind", "", [
  I32EnumAttrCase<"Reserved", 0, "reserved">
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = FlyCQ_Dialect.cppNamespace;
}

include "flydsl/Dialect/FlyCQ/IR/CopyAtom.td"
include "flydsl/Dialect/FlyCQ/IR/MmaAtom.td"

#endif // FLYCQ_ATOM
