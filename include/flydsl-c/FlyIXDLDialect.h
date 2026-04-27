#ifndef FLYDSL_C_FLYIXDLDIALECT_H
#define FLYDSL_C_FLYIXDLDIALECT_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(FlyIXDL, fly_ixdl);

//===----------------------------------------------------------------------===//
// MmaAtomIvcore11_MMADType
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyIXDLMmaAtomIvcore11_MMADType(MlirType type);
MLIR_CAPI_EXPORTED MlirTypeID mlirFlyIXDLMmaAtomIvcore11_MMADTypeGetTypeID(void);

MLIR_CAPI_EXPORTED MlirType mlirFlyIXDLMmaAtomIvcore11_MMADTypeGet(
    int32_t m, int32_t n, int32_t k, MlirType elemTyA, MlirType elemTyB,
    MlirType elemTyAcc);

MLIR_CAPI_EXPORTED int32_t mlirFlyIXDLMmaAtomIvcore11_MMADTypeGetM(MlirType type);
MLIR_CAPI_EXPORTED int32_t mlirFlyIXDLMmaAtomIvcore11_MMADTypeGetN(MlirType type);
MLIR_CAPI_EXPORTED int32_t mlirFlyIXDLMmaAtomIvcore11_MMADTypeGetK(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyIXDLMmaAtomIvcore11_MMADTypeGetElemTyA(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyIXDLMmaAtomIvcore11_MMADTypeGetElemTyB(MlirType type);
MLIR_CAPI_EXPORTED MlirType mlirFlyIXDLMmaAtomIvcore11_MMADTypeGetElemTyAcc(MlirType type);

//===----------------------------------------------------------------------===//
// CopyOpIvcore11_SMELoadType
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyIXDLCopyOpIvcore11_SMELoadType(MlirType type);
MLIR_CAPI_EXPORTED MlirTypeID mlirFlyIXDLCopyOpIvcore11_SMELoadTypeGetTypeID(void);
MLIR_CAPI_EXPORTED MlirType mlirFlyIXDLCopyOpIvcore11_SMELoadTypeGet(MlirContext ctx, int32_t bitSize);
MLIR_CAPI_EXPORTED int32_t mlirFlyIXDLCopyOpIvcore11_SMELoadTypeGetBitSize(MlirType type);

//===----------------------------------------------------------------------===//
// CopyOpIvcore11_SLBLoadType
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyIXDLCopyOpIvcore11_SLBLoadType(MlirType type);
MLIR_CAPI_EXPORTED MlirTypeID mlirFlyIXDLCopyOpIvcore11_SLBLoadTypeGetTypeID(void);
MLIR_CAPI_EXPORTED MlirType mlirFlyIXDLCopyOpIvcore11_SLBLoadTypeGet(MlirContext ctx, int32_t bitSize);
MLIR_CAPI_EXPORTED int32_t mlirFlyIXDLCopyOpIvcore11_SLBLoadTypeGetBitSize(MlirType type);

//===----------------------------------------------------------------------===//
// CopyOpIvcore11_DescStoreType
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyIXDLCopyOpIvcore11_DescStoreType(MlirType type);
MLIR_CAPI_EXPORTED MlirTypeID mlirFlyIXDLCopyOpIvcore11_DescStoreTypeGetTypeID(void);
MLIR_CAPI_EXPORTED MlirType mlirFlyIXDLCopyOpIvcore11_DescStoreTypeGet(MlirContext ctx, int32_t bitSize);
MLIR_CAPI_EXPORTED int32_t mlirFlyIXDLCopyOpIvcore11_DescStoreTypeGetBitSize(MlirType type);

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED void mlirRegisterFlyToIXDLConversionPass(void);

#ifdef __cplusplus
}
#endif

#endif // FLYDSL_C_FLYIXDLDIALECT_H
