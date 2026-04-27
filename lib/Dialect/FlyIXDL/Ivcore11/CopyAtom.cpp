#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/FlyIXDL/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"

using namespace mlir;
using namespace mlir::fly;

namespace mlir::fly_ixdl {

//===----------------------------------------------------------------------===//
// CopyOpIvcore11_SMELoad: async G2S via SME descriptor
//   Block-level operation (all 64 threads participate collectively).
//   Single-thread view: each thread contributes 1 element of bitSize bits.
//===----------------------------------------------------------------------===//

bool CopyOpIvcore11_SMELoadType::isStatic() const { return true; }

Attribute CopyOpIvcore11_SMELoadType::getThrLayout() const {
  return FxLayout(FxC(1), FxC(1));
}

Attribute CopyOpIvcore11_SMELoadType::getThrBitLayoutSrc() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())),
                  FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpIvcore11_SMELoadType::getThrBitLayoutDst() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())),
                  FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpIvcore11_SMELoadType::getThrBitLayoutRef() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())),
                  FxStride(FxC(1), FxC(1)));
}

//===----------------------------------------------------------------------===//
// CopyOpIvcore11_SLBLoad: per-thread shared memory load (S2R)
//===----------------------------------------------------------------------===//

bool CopyOpIvcore11_SLBLoadType::isStatic() const { return true; }

Attribute CopyOpIvcore11_SLBLoadType::getThrLayout() const {
  return FxLayout(FxC(1), FxC(1));
}

Attribute CopyOpIvcore11_SLBLoadType::getThrBitLayoutSrc() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())),
                  FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpIvcore11_SLBLoadType::getThrBitLayoutDst() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())),
                  FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpIvcore11_SLBLoadType::getThrBitLayoutRef() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())),
                  FxStride(FxC(1), FxC(1)));
}

//===----------------------------------------------------------------------===//
// CopyOpIvcore11_DescStore: per-thread descriptor store (R2G)
//===----------------------------------------------------------------------===//

bool CopyOpIvcore11_DescStoreType::isStatic() const { return true; }

Attribute CopyOpIvcore11_DescStoreType::getThrLayout() const {
  return FxLayout(FxC(1), FxC(1));
}

Attribute CopyOpIvcore11_DescStoreType::getThrBitLayoutSrc() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())),
                  FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpIvcore11_DescStoreType::getThrBitLayoutDst() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())),
                  FxStride(FxC(1), FxC(1)));
}
Attribute CopyOpIvcore11_DescStoreType::getThrBitLayoutRef() const {
  return FxLayout(FxShape(FxC(1), FxC(getBitSize())),
                  FxStride(FxC(1), FxC(1)));
}

} // namespace mlir::fly_ixdl
