#include "flydsl-c/FlyIXDLDialect.h"
#include "flydsl/Conversion/Passes.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Pass.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

void mlirRegisterFlyToIXDLConversionPass(void) {
  mlir::registerFlyToIXDLConversionPass();
}
