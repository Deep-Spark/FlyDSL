// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"

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

struct PyCopyOpIvcore11AsyncCpType : PyConcreteType<PyCopyOpIvcore11AsyncCpType> {
  FLYDSL_REGISTER_TYPE_BINDING(CopyOpIvcore11AsyncCpType, "CopyOpIvcore11AsyncCpType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t smeSwizzle, DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          return PyCopyOpIvcore11AsyncCpType(context->getRef(),
                                             wrap(CopyOpIvcore11AsyncCpType::get(ctx, smeSwizzle)));
        },
        "sme_swizzle"_a, nb::kw_only(), "context"_a = nb::none(),
        "Create a CopyOpIvcore11AsyncCpType (sme_swizzle: 0=NoSwizzle, 1=Col, 2=Row8b, 3=Row16b)");
  }
};

struct PyCopyOpIvcore11AsyncStoreType : PyConcreteType<PyCopyOpIvcore11AsyncStoreType> {
  FLYDSL_REGISTER_TYPE_BINDING(CopyOpIvcore11AsyncStoreType, "CopyOpIvcore11AsyncStoreType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t storeBytes, DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          return PyCopyOpIvcore11AsyncStoreType(
              context->getRef(), wrap(CopyOpIvcore11AsyncStoreType::get(ctx, storeBytes)));
        },
        "store_bytes"_a, nb::kw_only(), "context"_a = nb::none(),
        "Create a CopyOpIvcore11AsyncStoreType (SME store width in bytes: 64, 128 or 256)");
  }
};

struct PyMmaOpIvcore11MmaType : PyConcreteType<PyMmaOpIvcore11MmaType> {
  FLYDSL_REGISTER_TYPE_BINDING(MmaOpIvcore11MmaType, "MmaOpIvcore11MmaType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t m, int32_t n, int32_t k, PyType &elemTyA, PyType &elemTyB, PyType &elemTyAcc,
           DefaultingPyMlirContext context) {
          return PyMmaOpIvcore11MmaType(
              context->getRef(),
              wrap(MmaOpIvcore11MmaType::get(m, n, k, unwrap(elemTyA), unwrap(elemTyB),
                                             unwrap(elemTyAcc))));
        },
        "m"_a, "n"_a, "k"_a, "elem_ty_a"_a, "elem_ty_b"_a, "elem_ty_acc"_a, nb::kw_only(),
        "context"_a = nb::none(),
        "Create a MmaOpIvcore11MmaType (Iluvatar ivcore11 TCU MMA) with m, n, k dimensions and "
        "(A, B) -> accumulator element types");
  }
};

} // namespace fly_ixdl
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

NB_MODULE(_mlirDialectsFlyIXDL, m) {
  m.doc() = "MLIR Python FlyIXDL Extension";

  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_ixdl::PyCopyOpIvcore11AsyncCpType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_ixdl::PyCopyOpIvcore11AsyncStoreType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_ixdl::PyMmaOpIvcore11MmaType::bind(m);
}
