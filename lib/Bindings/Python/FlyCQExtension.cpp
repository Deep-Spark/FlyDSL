// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"

#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/FlyCQ/IR/Dialect.h"

#include "BindingUtils.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace ::mlir::fly;
using namespace ::mlir::fly_cq;

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {
namespace fly_cq {

struct PyMmaOpCQ_MatmulF32Type : PyConcreteType<PyMmaOpCQ_MatmulF32Type> {
  FLYDSL_REGISTER_TYPE_BINDING(MmaOpCQ_MatmulF32Type, "MmaOpCQ_MatmulF32Type");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t m, int32_t n, int32_t k, PyType &elemTyA, PyType &elemTyB, PyType &elemTyAcc,
           DefaultingPyMlirContext context) {
          return PyMmaOpCQ_MatmulF32Type(
              context->getRef(),
              wrap(MmaOpCQ_MatmulF32Type::get(m, n, k, unwrap(elemTyA), unwrap(elemTyB),
                                             unwrap(elemTyAcc))));
        },
        "m"_a, "n"_a, "k"_a, "elem_ty_a"_a, "elem_ty_b"_a, "elem_ty_acc"_a, nb::kw_only(),
        "context"_a = nb::none(),
        "Create a MmaOpCQ_MatmulF32Type (CQ placeholder matmul atom)");
  }
};

struct PyCopyOpCQ_ScalarMemType : PyConcreteType<PyCopyOpCQ_ScalarMemType> {
  FLYDSL_REGISTER_TYPE_BINDING(CopyOpCQ_ScalarMemType, "CopyOpCQ_ScalarMemType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t bitSize, DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          return PyCopyOpCQ_ScalarMemType(
              context->getRef(), wrap(CopyOpCQ_ScalarMemType::get(ctx, bitSize)));
        },
        "bit_size"_a, nb::kw_only(), "context"_a = nb::none(),
        "Create a CopyOpCQ_ScalarMemType (CQ placeholder scalar load/store copy, bitSize=32)");
  }
};

} // namespace fly_cq
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

NB_MODULE(_mlirDialectsFlyCQ, m) {
  m.doc() = "MLIR Python FlyCQ Extension";
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_cq::PyMmaOpCQ_MatmulF32Type::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_cq::PyCopyOpCQ_ScalarMemType::bind(m);
}
