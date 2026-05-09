# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Iluvatar backend descriptor.
#
# TableGen / header subdirectories under include/flydsl/
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_INCLUDE_DIALECT_SUBDIRS "FlyIXDL")

# C++ library subdirectories under lib/
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_LIB_DIALECT_SUBDIRS "FlyIXDL")

# CAPI wrapper subdirectory under lib/CAPI/Dialect/
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_CAPI_SUBDIRS "FlyIXDL")

# CAPI link targets for _mlirRegisterEverything (EMBED_CAPI_LINK_LIBS)
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_EMBED_CAPI_LIBS "MLIRCPIFlyIXDL")

# Link targets for fly-opt
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_FLYOPT_LINK_LIBS "MLIRCPIFlyIXDL")

# Stubgen modules for this backend
set_property(GLOBAL APPEND PROPERTY FLYDSL_BACKEND_STUBGEN_MODULES
  "flydsl._mlir._mlir_libs._mlirDialectsFlyIXDL")

# Convenience boolean for Python CMakeLists gating of Iluvatar-specific bindings
set(FLYDSL_HAS_ILUVATAR ON)
