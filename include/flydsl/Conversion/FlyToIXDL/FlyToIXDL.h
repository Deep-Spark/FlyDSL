#ifndef FLYDSL_CONVERSION_FLYTOIXDL_FLYTOIXDL_H
#define FLYDSL_CONVERSION_FLYTOIXDL_FLYTOIXDL_H

#include <memory>

namespace mlir {
class Pass;

namespace fly {
std::unique_ptr<Pass> createFlyToIXDLConversionPass();
} // namespace fly
} // namespace mlir

#endif // FLYDSL_CONVERSION_FLYTOIXDL_FLYTOIXDL_H
