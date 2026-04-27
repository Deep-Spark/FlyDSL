#include "flydsl-c/FlyIXDLDialect.h"

#include "flydsl/Conversion/Passes.h"
#include "flydsl/Dialect/FlyIXDL/IR/Dialect.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"

using namespace mlir;
using namespace mlir::fly_ixdl;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(FlyIXDL, fly_ixdl,
                                      mlir::fly_ixdl::FlyIXDLDialect)

//===----------------------------------------------------------------------===//
// MmaAtomIvcore11_MMADType
//===----------------------------------------------------------------------===//

bool mlirTypeIsAFlyIXDLMmaAtomIvcore11_MMADType(MlirType type) {
  return isa<MmaAtomIvcore11_MMADType>(unwrap(type));
}

MlirTypeID mlirFlyIXDLMmaAtomIvcore11_MMADTypeGetTypeID(void) {
  return wrap(MmaAtomIvcore11_MMADType::getTypeID());
}

MlirType mlirFlyIXDLMmaAtomIvcore11_MMADTypeGet(int32_t m, int32_t n,
                                                 int32_t k, MlirType elemTyA,
                                                 MlirType elemTyB,
                                                 MlirType elemTyAcc) {
  return wrap(MmaAtomIvcore11_MMADType::get(m, n, k, unwrap(elemTyA),
                                            unwrap(elemTyB),
                                            unwrap(elemTyAcc)));
}

int32_t mlirFlyIXDLMmaAtomIvcore11_MMADTypeGetM(MlirType type) {
  return cast<MmaAtomIvcore11_MMADType>(unwrap(type)).getM();
}
int32_t mlirFlyIXDLMmaAtomIvcore11_MMADTypeGetN(MlirType type) {
  return cast<MmaAtomIvcore11_MMADType>(unwrap(type)).getN();
}
int32_t mlirFlyIXDLMmaAtomIvcore11_MMADTypeGetK(MlirType type) {
  return cast<MmaAtomIvcore11_MMADType>(unwrap(type)).getK();
}
MlirType mlirFlyIXDLMmaAtomIvcore11_MMADTypeGetElemTyA(MlirType type) {
  return wrap(cast<MmaAtomIvcore11_MMADType>(unwrap(type)).getElemTyA());
}
MlirType mlirFlyIXDLMmaAtomIvcore11_MMADTypeGetElemTyB(MlirType type) {
  return wrap(cast<MmaAtomIvcore11_MMADType>(unwrap(type)).getElemTyB());
}
MlirType mlirFlyIXDLMmaAtomIvcore11_MMADTypeGetElemTyAcc(MlirType type) {
  return wrap(cast<MmaAtomIvcore11_MMADType>(unwrap(type)).getElemTyAcc());
}

//===----------------------------------------------------------------------===//
// CopyOpIvcore11_SMELoadType
//===----------------------------------------------------------------------===//

bool mlirTypeIsAFlyIXDLCopyOpIvcore11_SMELoadType(MlirType type) {
  return isa<CopyOpIvcore11_SMELoadType>(unwrap(type));
}
MlirTypeID mlirFlyIXDLCopyOpIvcore11_SMELoadTypeGetTypeID(void) {
  return wrap(CopyOpIvcore11_SMELoadType::getTypeID());
}
MlirType mlirFlyIXDLCopyOpIvcore11_SMELoadTypeGet(MlirContext ctx,
                                                   int32_t bitSize) {
  return wrap(CopyOpIvcore11_SMELoadType::get(unwrap(ctx), bitSize));
}
int32_t mlirFlyIXDLCopyOpIvcore11_SMELoadTypeGetBitSize(MlirType type) {
  return cast<CopyOpIvcore11_SMELoadType>(unwrap(type)).getBitSize();
}

//===----------------------------------------------------------------------===//
// CopyOpIvcore11_SLBLoadType
//===----------------------------------------------------------------------===//

bool mlirTypeIsAFlyIXDLCopyOpIvcore11_SLBLoadType(MlirType type) {
  return isa<CopyOpIvcore11_SLBLoadType>(unwrap(type));
}
MlirTypeID mlirFlyIXDLCopyOpIvcore11_SLBLoadTypeGetTypeID(void) {
  return wrap(CopyOpIvcore11_SLBLoadType::getTypeID());
}
MlirType mlirFlyIXDLCopyOpIvcore11_SLBLoadTypeGet(MlirContext ctx,
                                                   int32_t bitSize) {
  return wrap(CopyOpIvcore11_SLBLoadType::get(unwrap(ctx), bitSize));
}
int32_t mlirFlyIXDLCopyOpIvcore11_SLBLoadTypeGetBitSize(MlirType type) {
  return cast<CopyOpIvcore11_SLBLoadType>(unwrap(type)).getBitSize();
}

//===----------------------------------------------------------------------===//
// CopyOpIvcore11_DescStoreType
//===----------------------------------------------------------------------===//

bool mlirTypeIsAFlyIXDLCopyOpIvcore11_DescStoreType(MlirType type) {
  return isa<CopyOpIvcore11_DescStoreType>(unwrap(type));
}
MlirTypeID mlirFlyIXDLCopyOpIvcore11_DescStoreTypeGetTypeID(void) {
  return wrap(CopyOpIvcore11_DescStoreType::getTypeID());
}
MlirType mlirFlyIXDLCopyOpIvcore11_DescStoreTypeGet(MlirContext ctx,
                                                     int32_t bitSize) {
  return wrap(CopyOpIvcore11_DescStoreType::get(unwrap(ctx), bitSize));
}
int32_t mlirFlyIXDLCopyOpIvcore11_DescStoreTypeGetBitSize(MlirType type) {
  return cast<CopyOpIvcore11_DescStoreType>(unwrap(type)).getBitSize();
}

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

void mlirRegisterFlyToIXDLConversionPass(void) {
  mlir::registerFlyToIXDLConversionPass();
}
