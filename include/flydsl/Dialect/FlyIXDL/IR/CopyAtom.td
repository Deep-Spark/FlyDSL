#ifndef FLYIXDL_COPYATOM
#define FLYIXDL_COPYATOM

include "flydsl/Dialect/FlyIXDL/IR/Dialect.td"

//===----------------------------------------------------------------------===//
// CopyOp Ivcore11: SME async load (G2S, block-level descriptor-based)
//===----------------------------------------------------------------------===//

def FlyIXDL_CopyOpIvcore11_SMELoad : FlyIXDL_CopyOp<
    "CopyOpIvcore11_SMELoad", "ivcore11.sme_load", []> {
  let parameters = (ins
    "int32_t":$bitSize
  );
  let assemblyFormat = "`<` $bitSize `>`";
}

//===----------------------------------------------------------------------===//
// CopyOp Ivcore11: SLB load (S2R, per-thread shared memory read)
//===----------------------------------------------------------------------===//

def FlyIXDL_CopyOpIvcore11_SLBLoad : FlyIXDL_CopyOp<
    "CopyOpIvcore11_SLBLoad", "ivcore11.slb_load", []> {
  let parameters = (ins
    "int32_t":$bitSize
  );
  let assemblyFormat = "`<` $bitSize `>`";
}

//===----------------------------------------------------------------------===//
// CopyOp Ivcore11: Descriptor store (R2G, per-thread global write)
//===----------------------------------------------------------------------===//

def FlyIXDL_CopyOpIvcore11_DescStore : FlyIXDL_CopyOp<
    "CopyOpIvcore11_DescStore", "ivcore11.desc_store", []> {
  let parameters = (ins
    "int32_t":$bitSize
  );
  let assemblyFormat = "`<` $bitSize `>`";
}

#endif // FLYIXDL_COPYATOM
