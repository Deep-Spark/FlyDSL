// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/IR/BuiltinAttributes.h"
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

struct PyMmaOpMR_MMADType : PyConcreteType<PyMmaOpMR_MMADType> {
  FLYDSL_REGISTER_TYPE_BINDING(MmaOpMR_MMADType, "MmaOpMR_MMADType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t m, int32_t n, int32_t k, PyType &elemTyA, PyType &elemTyB, PyType &elemTyAcc,
           DefaultingPyMlirContext context) {
          return PyMmaOpMR_MMADType(context->getRef(),
                                    wrap(MmaOpMR_MMADType::get(m, n, k, unwrap(elemTyA),
                                                               unwrap(elemTyB), unwrap(elemTyAcc))));
        },
        "m"_a, "n"_a, "k"_a, "elem_ty_a"_a, "elem_ty_b"_a, "elem_ty_acc"_a, nb::kw_only(),
        "context"_a = nb::none(),
        "Create a MmaOpMR_MMADType (ivcore11 SME MMAD) with m, n, k and element types");
  }
};

struct PyCopyOpMRAsyncCopy16x64B8RowType : PyConcreteType<PyCopyOpMRAsyncCopy16x64B8RowType> {
  FLYDSL_REGISTER_TYPE_BINDING(CopyOpMRAsyncCopy16x64B8RowType, "CopyOpMRAsyncCopy16x64B8RowType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t bitSize, DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          return PyCopyOpMRAsyncCopy16x64B8RowType(
              context->getRef(), wrap(CopyOpMRAsyncCopy16x64B8RowType::get(ctx, bitSize)));
        },
        "bit_size"_a, nb::kw_only(), "context"_a = nb::none(),
        "Create a CopyOpMRAsyncCopy16x64B8RowType (stateful SME cp.async) with the given bit size");
  }
};

struct PyCopyOpMRSLBLoadType : PyConcreteType<PyCopyOpMRSLBLoadType> {
  FLYDSL_REGISTER_TYPE_BINDING(CopyOpMRSLBLoadType, "CopyOpMRSLBLoadType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t bitSize, DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          return PyCopyOpMRSLBLoadType(context->getRef(),
                                       wrap(CopyOpMRSLBLoadType::get(ctx, bitSize)));
        },
        "bit_size"_a, nb::kw_only(), "context"_a = nb::none(),
        "Create a CopyOpMRSLBLoadType (shared->register SLB load) with the given bit size");
  }
};

} // namespace fly_ixdl
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

NB_MODULE(_mlirDialectsFlyIXDL, m) {
  m.doc() = "MLIR Python FlyIXDL Extension";

  // clang-format off
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_ixdl::PyMmaOpMR_MMADType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_ixdl::PyCopyOpMRAsyncCopy16x64B8RowType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_ixdl::PyCopyOpMRSLBLoadType::bind(m);
  // clang-format on
}
