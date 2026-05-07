// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
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

struct PyMmaOpIX11_MMADType : PyConcreteType<PyMmaOpIX11_MMADType> {
  FLYDSL_REGISTER_TYPE_BINDING(MmaOpIX11_MMADType, "MmaOpIX11_MMADType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t m, int32_t n, int32_t k, PyType &elemTyA, PyType &elemTyB, PyType &elemTyAcc,
           DefaultingPyMlirContext context) {
          return PyMmaOpIX11_MMADType(
              context->getRef(),
              wrap(MmaOpIX11_MMADType::get(m, n, k, unwrap(elemTyA), unwrap(elemTyB),
                                           unwrap(elemTyAcc))));
        },
        "m"_a, "n"_a, "k"_a, "elem_ty_a"_a, "elem_ty_b"_a, "elem_ty_acc"_a, nb::kw_only(),
        "context"_a = nb::none(),
        "Create a MmaOpIX11_MMADType (ivcore11) with m, n, k dimensions and element types");
  }
};

struct PyCopyOpIX11_SMEType : PyConcreteType<PyCopyOpIX11_SMEType> {
  FLYDSL_REGISTER_TYPE_BINDING(CopyOpIX11_SMEType, "CopyOpIX11_SMEType");

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](PyType &elemTy, int32_t shape0, int32_t shape1, int64_t strideByte,
           int32_t major, int32_t cacheOp, int32_t swizzle,
           DefaultingPyMlirContext context) {
          MLIRContext *ctx = unwrap(context->get());
          auto majorAttr = ::mlir::fly_ixdl::SMEMajorAttr::get(
              ctx, ::mlir::fly_ixdl::symbolizeSMEMajor(major).value_or(
                       ::mlir::fly_ixdl::SMEMajor::K));
          auto cacheAttr = ::mlir::fly_ixdl::SMECacheOpAttr::get(
              ctx, ::mlir::fly_ixdl::symbolizeSMECacheOp(cacheOp).value_or(
                       ::mlir::fly_ixdl::SMECacheOp::CacheAll));
          auto swizzleAttr = ::mlir::fly_ixdl::SMESwizzleAttr::get(
              ctx, ::mlir::fly_ixdl::symbolizeSMESwizzle(swizzle).value_or(
                       ::mlir::fly_ixdl::SMESwizzle::None_));
          // Use ``getChecked`` so verifier failures surface as Python
          // exceptions instead of asserting in debug builds.
          std::string err;
          auto emitErr = [&]() -> ::mlir::InFlightDiagnostic {
            auto d = ::mlir::emitError(::mlir::UnknownLoc::get(ctx));
            return d;
          };
          auto ty = CopyOpIX11_SMEType::getChecked(
              emitErr, unwrap(elemTy), shape0, shape1, strideByte,
              majorAttr, cacheAttr, swizzleAttr);
          if (!ty)
            throw nb::value_error("CopyOpIX11_SMEType verification failed "
                                  "(see emitted diagnostic above)");
          return PyCopyOpIX11_SMEType(context->getRef(), wrap(ty));
        },
        "elem_ty"_a, "shape0"_a, "shape1"_a, "stride_byte"_a, "major"_a,
        "cache_op"_a, "swizzle"_a = 0, nb::kw_only(), "context"_a = nb::none(),
        "Create a CopyOpIX11_SMEType (ivcore11 SME async copy) for "
        "global->shared staging. ``major`` picks the row/col xfb family: "
        "0=MN-major (col-xfb*), 1=K-major (row-xfb*). ``cache_op``: "
        "0=CacheAll, 1=BypassL1, 2=BypassL2, 3=BypassL1L2. ``swizzle``: "
        "0=None (plain sme_load_*x1b64), 1=RowXfb16 (16-bit row swizzle, "
        "A-operand), 2=ColXfb8 (16-bit col swizzle, B-operand).");
  }
};

} // namespace fly_ixdl
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

NB_MODULE(_mlirDialectsFlyIXDL, m) {
  m.doc() = "MLIR Python FlyIXDL Extension";

  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_ixdl::PyMmaOpIX11_MMADType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_ixdl::PyCopyOpIX11_SMEType::bind(m);
}
