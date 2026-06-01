// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/IR/MLIRContext.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/FlyIXDL/IR/Dialect.h"

#include "BindingUtils.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace ::mlir::fly;
using namespace ::mlir::fly_ixdl;

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {
namespace fly_ixdl {

struct PyCopyOpMRAsyncCpType : PyConcreteType<PyCopyOpMRAsyncCpType> {
  FLYDSL_REGISTER_TYPE_BINDING(CopyOpMRAsyncCpType, "CopyOpMRAsyncCpType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t smeSwizzle, DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          return PyCopyOpMRAsyncCpType(context->getRef(),
                                       wrap(CopyOpMRAsyncCpType::get(ctx, smeSwizzle)));
        },
        "sme_swizzle"_a, nb::kw_only(), "context"_a = nb::none(),
        "Create a CopyOpMRAsyncCpType (sme_swizzle: 0=NoSwizzle, 1=Col, 2=Row8b, 3=Row16b)");
  }
};

} // namespace fly_ixdl
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

NB_MODULE(_mlirDialectsFlyIXDL, m) {
  m.doc() = "MLIR Python FlyIXDL Extension";

  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_ixdl::PyCopyOpMRAsyncCpType::bind(m);
}
