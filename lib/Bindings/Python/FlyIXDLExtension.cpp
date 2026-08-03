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

struct PyCopyOpCQAsyncCpType : PyConcreteType<PyCopyOpCQAsyncCpType> {
  FLYDSL_REGISTER_TYPE_BINDING(CopyOpCQAsyncCpType, "CopyOpCQAsyncCpType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t row, int32_t col, int32_t transpose, DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          return PyCopyOpCQAsyncCpType(context->getRef(),
                                       wrap(CopyOpCQAsyncCpType::get(ctx, row, col, transpose)));
        },
        "row"_a, "col"_a, "transpose"_a, nb::kw_only(), "context"_a = nb::none(),
        "Create a CopyOpCQAsyncCpType (row, col, transpose: verified enhanced-SME "
        "shapes (64,64,0), (64,32,0), (1,1024,0), (64,16,0), (64,16,1))");
  }
};

struct PyCopyOpCQMtxLoadnType : PyConcreteType<PyCopyOpCQMtxLoadnType> {
  FLYDSL_REGISTER_TYPE_BINDING(CopyOpCQMtxLoadnType, "CopyOpCQMtxLoadnType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t pattern, int32_t direction, int32_t bitWidth, int32_t x2,
           DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context.get()->get());
          return PyCopyOpCQMtxLoadnType(
              context->getRef(),
              wrap(CopyOpCQMtxLoadnType::get(ctx, pattern, direction, bitWidth, x2)));
        },
        "pattern"_a, "direction"_a, "bit_width"_a, "x2"_a, nb::kw_only(), "context"_a = nb::none(),
        "Create CopyOpCQMtxLoadnType (pattern: 0=loadn16/1=loadn64; direction: 0=row/1=col; "
        "bit_width: 8|16; x2: 0|1)");
  }
};

struct PyMmaOpMRMmaType : PyConcreteType<PyMmaOpMRMmaType> {
  FLYDSL_REGISTER_TYPE_BINDING(MmaOpMRMmaType, "MmaOpMRMmaType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t m, int32_t n, int32_t k, PyType &elemTyA, PyType &elemTyB, PyType &elemTyAcc,
           DefaultingPyMlirContext context) {
          return PyMmaOpMRMmaType(context->getRef(),
                                  wrap(MmaOpMRMmaType::get(m, n, k, unwrap(elemTyA),
                                                           unwrap(elemTyB), unwrap(elemTyAcc))));
        },
        "m"_a, "n"_a, "k"_a, "elem_ty_a"_a, "elem_ty_b"_a, "elem_ty_acc"_a, nb::kw_only(),
        "context"_a = nb::none(),
        "Create a MmaOpMRMmaType (Iluvatar MR TCU MMA) with m, n, k dimensions and "
        "(A, B) -> accumulator element types");
  }
};

struct PyMmaOpCQMmaType : PyConcreteType<PyMmaOpCQMmaType> {
  FLYDSL_REGISTER_TYPE_BINDING(MmaOpCQMmaType, "MmaOpCQMmaType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t m, int32_t n, int32_t k, PyType &elemTyA, PyType &elemTyB, PyType &elemTyAcc,
           DefaultingPyMlirContext context) {
          return PyMmaOpCQMmaType(context->getRef(),
                                  wrap(MmaOpCQMmaType::get(m, n, k, unwrap(elemTyA),
                                                           unwrap(elemTyB), unwrap(elemTyAcc))));
        },
        "m"_a, "n"_a, "k"_a, "elem_ty_a"_a, "elem_ty_b"_a, "elem_ty_acc"_a, nb::kw_only(),
        "context"_a = nb::none(),
        "Create a MmaOpCQMmaType (Iluvatar CQ TCU MMA) with m, n, k dimensions and "
        "(A, B) -> accumulator element types");
  }
};

} // namespace fly_ixdl
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

NB_MODULE(_mlirDialectsFlyIXDL, m) {
  m.doc() = "MLIR Python FlyIXDL Extension";

  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_ixdl::PyCopyOpMRAsyncCpType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_ixdl::PyCopyOpCQAsyncCpType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_ixdl::PyCopyOpCQMtxLoadnType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_ixdl::PyMmaOpMRMmaType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_ixdl::PyMmaOpCQMmaType::bind(m);
}
