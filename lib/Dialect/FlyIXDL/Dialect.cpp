#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "flydsl/Dialect/FlyIXDL/IR/Dialect.h"

using namespace mlir;
using namespace mlir::fly;
using namespace mlir::fly_ixdl;

#include "flydsl/Dialect/FlyIXDL/IR/Dialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "flydsl/Dialect/FlyIXDL/IR/Atom.cpp.inc"

void FlyIXDLDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "flydsl/Dialect/FlyIXDL/IR/Atom.cpp.inc"
      >();
}
