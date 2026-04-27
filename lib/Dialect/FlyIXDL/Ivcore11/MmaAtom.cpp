#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/FlyIXDL/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include "flydsl/Dialect/Fly/Utils/ThrValLayoutMacro.h.inc"

using namespace mlir;
using namespace mlir::fly;

// Ivcore11 MMAD 16x16x16 thread-value layout (empirically verified)
//
// Hardware mapping (probed via SME→S2V dump, verified with direct load):
//   laneRow = laneId / 16  (0..3)
//   laneCol = laneId % 16  (0..15)
//
// A operand [M×K = 16×16]:
//   v0 → A[2*laneRow    ][laneCol]
//   v1 → A[2*laneRow + 1][laneCol]
//   v2 → A[2*laneRow + 8][laneCol]
//   v3 → A[2*laneRow + 9][laneCol]
//   m = 2*laneRow + v%2 + 8*(v/2),  k = laneCol
//   Decompose: v0 = v%2, v1 = v/2
//   M-major pos = m + k*16 = (2*laneRow + v0 + 8*v1) + laneCol*16
//   Shape  Thr(16, 4)  Val(2, 2)
//   Stride Thr(16, 2)  Val(1, 8)
//
// B operand [K×N = 16×16, CuTe tile is (N,K)]:
//   Hardware: n = laneCol, k = 2*laneRow + v%2 + 8*(v/2)
//   CuTe (N,K) tile, N-major pos = n + k*N:
//     pos = laneCol + (2*laneRow + v0 + 8*v1)*16
//         = t0*1 + t1*32 + v0*16 + v1*128
//   Shape  Thr(16, 4)  Val(2, 2)
//   Stride Thr(1, 32)  Val(16, 128)
//
// C accumulator [M×N = 16×16]:
//   acc[vi] → C[laneRow + vi*4][laneCol]
//   m = laneRow + vi*4,  n = laneCol
//   M-major pos = m + n*16 = (laneRow + vi*4) + laneCol*16
//   Shape  Thr(16, 4)  Val(4)
//   Stride Thr(16, 1)  Val(4)

namespace mlir::fly_ixdl {

bool MmaAtomIvcore11_MMADType::isStatic() const { return true; }

Attribute MmaAtomIvcore11_MMADType::getThrLayout() const {
  return FxLayout(FxC(64), FxC(1));
}

Attribute MmaAtomIvcore11_MMADType::getShapeMNK() const {
  return IntTupleAttr::get(
      ArrayAttr::get(getContext(), {FxC(getM()), FxC(getN()), FxC(getK())}));
}

Type MmaAtomIvcore11_MMADType::getValTypeA() const { return getElemTyA(); }
Type MmaAtomIvcore11_MMADType::getValTypeB() const { return getElemTyB(); }
Type MmaAtomIvcore11_MMADType::getValTypeC() const { return getElemTyAcc(); }
Type MmaAtomIvcore11_MMADType::getValTypeD() const { return getElemTyAcc(); }

Attribute MmaAtomIvcore11_MMADType::getThrValLayoutA() const {
  // A tile (M×K = 16×16): m = 2*laneRow + v0 + 8*v1, k = laneCol
  // M-major pos = m + k*16  →  Thr(16, 2) Val(1, 8)
  return FxLayout(FxShape(FxThr(16, 4), FxVal(2, 2)),
                  FxStride(FxThr(16, 2), FxVal(1, 8)));
}

Attribute MmaAtomIvcore11_MMADType::getThrValLayoutB() const {
  // B tile (N,K): n = laneCol, k = 2*laneRow + v0 + 8*v1
  // N-major pos = t0*1 + t1*32 + v0*16 + v1*128
  return FxLayout(FxShape(FxThr(16, 4), FxVal(2, 2)),
                  FxStride(FxThr(1, 32), FxVal(16, 128)));
}

Attribute MmaAtomIvcore11_MMADType::getThrValLayoutC() const {
  // C tile (M×N = 16×16): m = laneRow + vi*4, n = laneCol
  // M-major pos = m + n*16  →  Thr(16, 1) Val(4)
  return FxLayout(FxShape(FxThr(16, 4), FxVal(4)),
                  FxStride(FxThr(16, 1), FxVal(4)));
}

LogicalResult MmaAtomIvcore11_MMADType::verify(
    function_ref<InFlightDiagnostic()> emitError, int32_t m, int32_t n,
    int32_t k, Type elemTyA, Type elemTyB, Type elemTyAcc) {
  if (m != 16 || n != 16 || k != 16) {
    return emitError() << "ivcore11 MMAD only supports 16x16x16, got " << m
                       << "x" << n << "x" << k;
  }
  if (!elemTyAcc.isF32() && !elemTyAcc.isInteger(32))
    return emitError() << "elemTyAcc must be f32 or i32, got " << elemTyAcc;

  auto isValidInputType = [](Type ty) {
    return ty.isF16() || ty.isBF16() || ty.isInteger(8);
  };
  if (!isValidInputType(elemTyA))
    return emitError() << "elemTyA must be f16, bf16, or i8, got " << elemTyA;
  if (!isValidInputType(elemTyB))
    return emitError() << "elemTyB must be f16, bf16, or i8, got " << elemTyB;
  return success();
}

} // namespace mlir::fly_ixdl
