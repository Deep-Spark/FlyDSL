// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYIXDL_COPYATOM
#define FLYIXDL_COPYATOM

include "flydsl/Dialect/FlyIXDL/IR/Dialect.td"
include "flydsl/Dialect/FlyIXDL/IR/Enums.td"

//===----------------------------------------------------------------------===//
// CopyOpIX11_SME
//
// Single-thread async bulk copy from global to shared memory using ivcore11
// SME intrinsics. Each atom call transfers ``shape0 * 64 bytes`` into smem,
// e.g. 16x1 b64 == 1024 bytes. The underlying swizzle kind (rowxfb8/16 or
// colxfb8/16/32) is inferred from ``(elemTy, major)`` by ``ixdl.cp.async``'s
// own dispatch (see ``getSMELoadIntrinsicId`` in
// ``IXDLToLLVMIRTranslation.cpp``). Thread layout is ``Layout<_1>`` — one
// issuing thread per atom call, the HW copies the full 1024-byte tile.
//
// We intentionally do not expose ``swizzle`` as an independent parameter:
// once ``major`` is fixed, the IX11 SME swizzle mode is fully determined by
// the element byte-width (rowxfb8/16, colxfb8/16/32).
//===----------------------------------------------------------------------===//

def FlyIXDL_CopyOpIX11_SME : FlyIXDL_CopyOp<"CopyOpIX11_SME", "ix11.sme", []> {
  let parameters = (ins
    "Type":$elemTy,
    "int32_t":$shape0,
    "int32_t":$shape1,
    "int64_t":$strideByte,
    "::mlir::fly_ixdl::SMEMajorAttr":$major,
    "::mlir::fly_ixdl::SMECacheOpAttr":$cacheOp,
    "::mlir::fly_ixdl::SMESwizzleAttr":$swizzle
  );
  let assemblyFormat =
      "`<` $elemTy `,` $shape0 `x` $shape1 `,` `stride` `=` $strideByte `,` $major `,` $cacheOp `,` $swizzle `>`";

  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elemTy, "int32_t":$shape0,
                                        "int32_t":$shape1, "int64_t":$strideByte,
                                        "::mlir::fly_ixdl::SMEMajorAttr":$major,
                                        "::mlir::fly_ixdl::SMECacheOpAttr":$cacheOp,
                                        "::mlir::fly_ixdl::SMESwizzleAttr":$swizzle), [{
      return $_get(elemTy.getContext(), elemTy, shape0, shape1, strideByte,
                   major, cacheOp, swizzle);
    }]>
  ];
  let genVerifyDecl = 1;
}

#endif // FLYIXDL_COPYATOM
