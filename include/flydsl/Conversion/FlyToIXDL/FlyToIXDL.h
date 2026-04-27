#ifndef CONVERSION_FLYTOIXDL_FLYTOIXDL_H
#define CONVERSION_FLYTOIXDL_FLYTOIXDL_H

#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL_FLYTOIXDLCONVERSIONPASS
#include "flydsl/Conversion/Passes.h.inc"
} // namespace mlir

#endif // CONVERSION_FLYTOIXDL_FLYTOIXDL_H
