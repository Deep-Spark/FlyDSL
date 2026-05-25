// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 FlyDSL Project Contributors

#ifndef FLYIXDL_DIALECT
#define FLYIXDL_DIALECT

include "mlir/IR/OpBase.td"

def FlyIXDL_Dialect : Dialect {
  let name = "fly_ixdl";
  let cppNamespace = "::mlir::fly_ixdl";

  let dependentDialects = [
    "IXDL::IXDLDialect"
  ];
}

#endif // FLYIXDL_DIALECT
