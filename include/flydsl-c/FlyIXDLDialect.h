#ifndef FLYDSL_C_FLYIXDLDIALECT_H
#define FLYDSL_C_FLYIXDLDIALECT_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

MLIR_CAPI_EXPORTED void mlirRegisterFlyToIXDLConversionPass(void);

#ifdef __cplusplus
}
#endif

#endif // FLYDSL_C_FLYIXDLDIALECT_H
