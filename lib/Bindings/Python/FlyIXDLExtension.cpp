#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Wrap.h"

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>

#include "flydsl-c/FlyIXDLDialect.h"
#include "flydsl/Dialect/Fly/IR/FlyDialect.h"
#include "flydsl/Dialect/FlyIXDL/IR/Dialect.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace mlir {
namespace python {
namespace MLIR_BINDINGS_PYTHON_DOMAIN {
namespace fly_ixdl {

struct PyMmaAtomIvcore11_MMADType
    : PyConcreteType<PyMmaAtomIvcore11_MMADType> {
  static constexpr IsAFunctionTy isaFunction =
      mlirTypeIsAFlyIXDLMmaAtomIvcore11_MMADType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFlyIXDLMmaAtomIvcore11_MMADTypeGetTypeID;
  static constexpr const char *pyClassName = "MmaAtomIvcore11_MMADType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t m, int32_t n, int32_t k, PyType &elemTyA, PyType &elemTyB,
           PyType &elemTyAcc, DefaultingPyMlirContext context) {
          return PyMmaAtomIvcore11_MMADType(
              context->getRef(),
              wrap(::mlir::fly_ixdl::MmaAtomIvcore11_MMADType::get(
                  m, n, k, unwrap(static_cast<MlirType>(elemTyA)),
                  unwrap(static_cast<MlirType>(elemTyB)),
                  unwrap(static_cast<MlirType>(elemTyAcc)))));
        },
        "m"_a, "n"_a, "k"_a, "elem_ty_a"_a, "elem_ty_b"_a,
        "elem_ty_acc"_a, nb::kw_only(), "context"_a = nb::none(),
        "Create a MmaAtomIvcore11_MMADType");

    c.def_prop_ro("m", [](PyMmaAtomIvcore11_MMADType &self) -> int32_t {
      return mlirFlyIXDLMmaAtomIvcore11_MMADTypeGetM(self);
    });
    c.def_prop_ro("n", [](PyMmaAtomIvcore11_MMADType &self) -> int32_t {
      return mlirFlyIXDLMmaAtomIvcore11_MMADTypeGetN(self);
    });
    c.def_prop_ro("k", [](PyMmaAtomIvcore11_MMADType &self) -> int32_t {
      return mlirFlyIXDLMmaAtomIvcore11_MMADTypeGetK(self);
    });
    c.def_prop_ro(
        "elem_ty_a", [](PyMmaAtomIvcore11_MMADType &self) -> MlirType {
          return mlirFlyIXDLMmaAtomIvcore11_MMADTypeGetElemTyA(self);
        });
    c.def_prop_ro(
        "elem_ty_b", [](PyMmaAtomIvcore11_MMADType &self) -> MlirType {
          return mlirFlyIXDLMmaAtomIvcore11_MMADTypeGetElemTyB(self);
        });
    c.def_prop_ro(
        "elem_ty_acc", [](PyMmaAtomIvcore11_MMADType &self) -> MlirType {
          return mlirFlyIXDLMmaAtomIvcore11_MMADTypeGetElemTyAcc(self);
        });

    c.def_prop_ro(
        "thr_layout", [](PyMmaAtomIvcore11_MMADType &self) -> MlirType {
          auto ty = ::mlir::cast<::mlir::fly::MmaAtomTypeInterface>(
              unwrap(static_cast<MlirType>(self)));
          auto attr =
              ::mlir::cast<::mlir::fly::LayoutAttr>(ty.getThrLayout());
          return wrap(::mlir::fly::LayoutType::get(attr));
        });
    c.def_prop_ro(
        "shape_mnk", [](PyMmaAtomIvcore11_MMADType &self) -> MlirType {
          auto ty = ::mlir::cast<::mlir::fly::MmaAtomTypeInterface>(
              unwrap(static_cast<MlirType>(self)));
          auto attr =
              ::mlir::cast<::mlir::fly::IntTupleAttr>(ty.getShapeMNK());
          return wrap(::mlir::fly::IntTupleType::get(attr));
        });
    c.def_prop_ro(
        "tv_layout_a", [](PyMmaAtomIvcore11_MMADType &self) -> MlirType {
          auto ty = ::mlir::cast<::mlir::fly::MmaAtomTypeInterface>(
              unwrap(static_cast<MlirType>(self)));
          auto attr =
              ::mlir::cast<::mlir::fly::LayoutAttr>(ty.getThrValLayoutA());
          return wrap(::mlir::fly::LayoutType::get(attr));
        });
    c.def_prop_ro(
        "tv_layout_b", [](PyMmaAtomIvcore11_MMADType &self) -> MlirType {
          auto ty = ::mlir::cast<::mlir::fly::MmaAtomTypeInterface>(
              unwrap(static_cast<MlirType>(self)));
          auto attr =
              ::mlir::cast<::mlir::fly::LayoutAttr>(ty.getThrValLayoutB());
          return wrap(::mlir::fly::LayoutType::get(attr));
        });
    c.def_prop_ro(
        "tv_layout_c", [](PyMmaAtomIvcore11_MMADType &self) -> MlirType {
          auto ty = ::mlir::cast<::mlir::fly::MmaAtomTypeInterface>(
              unwrap(static_cast<MlirType>(self)));
          auto attr =
              ::mlir::cast<::mlir::fly::LayoutAttr>(ty.getThrValLayoutC());
          return wrap(::mlir::fly::LayoutType::get(attr));
        });
  }
};

struct PyCopyOpIvcore11_SMELoadType
    : PyConcreteType<PyCopyOpIvcore11_SMELoadType> {
  static constexpr IsAFunctionTy isaFunction =
      mlirTypeIsAFlyIXDLCopyOpIvcore11_SMELoadType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFlyIXDLCopyOpIvcore11_SMELoadTypeGetTypeID;
  static constexpr const char *pyClassName = "CopyOpIvcore11_SMELoadType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t bitSize, DefaultingPyMlirContext context) {
          return PyCopyOpIvcore11_SMELoadType(
              context->getRef(),
              mlirFlyIXDLCopyOpIvcore11_SMELoadTypeGet(context->get(),
                                                       bitSize));
        },
        "bit_size"_a, nb::kw_only(), "context"_a = nb::none());
    c.def_prop_ro("bit_size",
                  [](PyCopyOpIvcore11_SMELoadType &self) -> int32_t {
                    return mlirFlyIXDLCopyOpIvcore11_SMELoadTypeGetBitSize(
                        self);
                  });
  }
};

struct PyCopyOpIvcore11_SLBLoadType
    : PyConcreteType<PyCopyOpIvcore11_SLBLoadType> {
  static constexpr IsAFunctionTy isaFunction =
      mlirTypeIsAFlyIXDLCopyOpIvcore11_SLBLoadType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFlyIXDLCopyOpIvcore11_SLBLoadTypeGetTypeID;
  static constexpr const char *pyClassName = "CopyOpIvcore11_SLBLoadType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t bitSize, DefaultingPyMlirContext context) {
          return PyCopyOpIvcore11_SLBLoadType(
              context->getRef(),
              mlirFlyIXDLCopyOpIvcore11_SLBLoadTypeGet(context->get(),
                                                       bitSize));
        },
        "bit_size"_a, nb::kw_only(), "context"_a = nb::none());
    c.def_prop_ro("bit_size",
                  [](PyCopyOpIvcore11_SLBLoadType &self) -> int32_t {
                    return mlirFlyIXDLCopyOpIvcore11_SLBLoadTypeGetBitSize(
                        self);
                  });
  }
};

struct PyCopyOpIvcore11_DescStoreType
    : PyConcreteType<PyCopyOpIvcore11_DescStoreType> {
  static constexpr IsAFunctionTy isaFunction =
      mlirTypeIsAFlyIXDLCopyOpIvcore11_DescStoreType;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      mlirFlyIXDLCopyOpIvcore11_DescStoreTypeGetTypeID;
  static constexpr const char *pyClassName = "CopyOpIvcore11_DescStoreType";
  using Base::Base;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "get",
        [](int32_t bitSize, DefaultingPyMlirContext context) {
          return PyCopyOpIvcore11_DescStoreType(
              context->getRef(),
              mlirFlyIXDLCopyOpIvcore11_DescStoreTypeGet(context->get(),
                                                         bitSize));
        },
        "bit_size"_a, nb::kw_only(), "context"_a = nb::none());
    c.def_prop_ro("bit_size",
                  [](PyCopyOpIvcore11_DescStoreType &self) -> int32_t {
                    return mlirFlyIXDLCopyOpIvcore11_DescStoreTypeGetBitSize(
                        self);
                  });
  }
};

} // namespace fly_ixdl
} // namespace MLIR_BINDINGS_PYTHON_DOMAIN
} // namespace python
} // namespace mlir

NB_MODULE(_fly_ixdl, m) {
  m.doc() = "MLIR Python FlyIXDL Extension";

  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_ixdl::
      PyMmaAtomIvcore11_MMADType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_ixdl::
      PyCopyOpIvcore11_SMELoadType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_ixdl::
      PyCopyOpIvcore11_SLBLoadType::bind(m);
  ::mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::fly_ixdl::
      PyCopyOpIvcore11_DescStoreType::bind(m);
}
