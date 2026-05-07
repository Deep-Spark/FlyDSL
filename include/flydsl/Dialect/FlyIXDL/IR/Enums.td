// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYIXDL_ENUMS
#define FLYIXDL_ENUMS

include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "flydsl/Dialect/FlyIXDL/IR/Dialect.td"

// Whether the source tile is traversed K-major (contiguous in the reduction
// axis) or MN-major (contiguous in the output axis). Maps 1:1 to the
// ``transpose`` arg of ``ixdl.cp.async`` and picks the ``rowxfb*`` vs
// ``colxfb*`` intrinsic family.
def FlyIXDL_SMEMajor : I32EnumAttr<"SMEMajor", "ivcore11 SME major axis", [
  I32EnumAttrCase<"MN", 0, "mn">,
  I32EnumAttrCase<"K",  1, "k">
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::fly_ixdl";
}
def FlyIXDL_SMEMajorAttr : EnumAttr<FlyIXDL_Dialect, FlyIXDL_SMEMajor, "sme_major"> {}

// CacheOP from ``copy_ix11_desc.hpp``. Passed through as the ``kop`` operand
// of ``ixdl.cp.async``.
def FlyIXDL_SMECacheOp : I32EnumAttr<"SMECacheOp", "ivcore11 SME cache policy", [
  I32EnumAttrCase<"CacheAll",   0, "cache_all">,
  I32EnumAttrCase<"BypassL1",   1, "bypass_l1">,
  I32EnumAttrCase<"BypassL2",   2, "bypass_l2">,
  I32EnumAttrCase<"BypassL1L2", 3, "bypass_l1_l2">
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::fly_ixdl";
}
def FlyIXDL_SMECacheOpAttr : EnumAttr<FlyIXDL_Dialect, FlyIXDL_SMECacheOp, "sme_cache_op"> {}

// IX11 SME "swizzle" family. Maps directly to the intrinsic variant picked
// up by ``getSMELoadIntrinsicId`` in ``IXDLToLLVMIRTranslation.cpp``:
//
//   * ``none``      — unswizzled ``sme_load_*x1b64`` (plain linear layout in
//                     smem). The ``emitAtomCall`` fold path treats the tile
//                     as a 32-bit strip so any element type bigger than a
//                     single segment routes here.
//   * ``row_xfb16`` — ``sme_load_*x1b64_rowxfb16`` (16-bit row-major HW
//                     swizzle). Intended for the A operand feeding 16-bit
//                     MMA in K-major form.
//   * ``col_xfb8``  — ``sme_load_*x1b64_colxfb8`` (8-bit column-major HW
//                     swizzle, used by cutlass for the 16-bit transposed
//                     B operand). For 16-bit elements the SDK dispatch
//                     accepts the (elemSize=16, transpose=true,
//                     segmentsFromMinor==1) tuple and folds internally
//                     to the colxfb8 intrinsic.
//
// Other swizzle flavors exposed by the SDK (``rowxfb8``, ``colxfb16``,
// ``colxfb32``, etc.) can be added here as needed; we keep the enum
// intentionally narrow so every case maps to an SME path we have a Python
// user and a lit test for.
def FlyIXDL_SMESwizzle : I32EnumAttr<"SMESwizzle", "ivcore11 SME swizzle family", [
  I32EnumAttrCase<"None_",    0, "none">,
  I32EnumAttrCase<"RowXfb16", 1, "row_xfb16">,
  I32EnumAttrCase<"ColXfb8",  2, "col_xfb8">
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::fly_ixdl";
}
def FlyIXDL_SMESwizzleAttr : EnumAttr<FlyIXDL_Dialect, FlyIXDL_SMESwizzle, "sme_swizzle"> {}

#endif // FLYIXDL_ENUMS
