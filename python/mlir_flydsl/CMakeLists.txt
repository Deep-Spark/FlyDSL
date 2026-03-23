include(AddMLIRPython)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=flydsl.${MLIR_PYTHON_PACKAGE_PREFIX}.")

declare_mlir_python_sources(FlyPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT FlyPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/"
  TD_FILE dialects/FlyOps.td
  SOURCES
    dialects/fly.py
    _mlir_libs/_mlirRegisterEverything/py.typed
  DIALECT_NAME fly
  GEN_ENUM_BINDINGS
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT FlyPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/"
  TD_FILE dialects/FlyROCDL.td
  SOURCES
    dialects/fly_rocdl.py
  DIALECT_NAME fly_rocdl
  GEN_ENUM_BINDINGS
)

# NOTE: Do NOT link MLIRFlyDialect/MLIRFlyROCDLDialect here via PRIVATE_LINK_LIBS.
# These symbols are already provided by FlyPythonCAPI.so (via MLIRCPIFly's transitive
# dependencies in EMBED_CAPI_LINK_LIBS). Statically linking them here creates DUPLICATE
# TypeID static variables, causing "storage uniquer isn't initialized" errors at runtime.
# The COMMON_CAPI_LINK_LIBS FlyPythonCAPI in add_mlir_python_modules ensures _fly.so
# links to FlyPythonCAPI.so, which provides all needed symbols.

declare_mlir_python_extension(FlyPythonSources.Core.fly
  MODULE_NAME _fly
  ADD_TO_PARENT FlyPythonSources
  ROOT_DIR "${PROJECT_SOURCE_DIR}/lib/Bindings/Python"
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    FlyExtension.cpp
    DLTensorAdaptor.h
  PRIVATE_LINK_LIBS
    LLVMSupport
)

declare_mlir_python_extension(FlyPythonSources.Core.fly_rocdl
  MODULE_NAME _fly_rocdl
  ADD_TO_PARENT FlyPythonSources
  ROOT_DIR "${PROJECT_SOURCE_DIR}/lib/Bindings/Python"
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    FlyROCDLExtension.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
)

# NOTE: Do NOT link MLIRFlyToROCDL or other C++ libs via PRIVATE_LINK_LIBS.
# Doing so statically embeds a copy of MLIRPass (and its pass registry) into
# _mlirRegisterEverything.so. Passes registered via registerFlyPasses() then
# go into that LOCAL registry, while PassManager.parse() queries the GLOBAL
# registry in FlyPythonCAPI.so → "does not refer to a registered pass".
#
# Instead, pass registration is done through CAPI functions
# (mlirRegisterFlyPasses, mlirRegisterFlyToROCDLConversionPass) defined in
# the CAPI libraries (MLIRCPIFly, MLIRCPIFlyROCDL). These are already in
# FlyPythonCAPI.so, so ::mlir::registerPass() resolves to the correct
# global instance.
declare_mlir_python_extension(FlyPythonSources.RegisterEverything
  MODULE_NAME _mlirRegisterEverything
  ADD_TO_PARENT FlyPythonSources
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    FlyRegisterEverything.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCPIFly
    MLIRCPIFlyROCDL
    MLIRCAPIArith
    MLIRCAPIGPU
    MLIRCAPILLVM
    MLIRCAPIMath
    MLIRCAPIVector
    MLIRCAPIConversion
    MLIRCAPITransforms
    MLIRCAPIRegisterEverything
)


set(MLIRFlyDSLSources
  FlyPythonSources
  MLIRPythonSources.Core
  MLIRPythonSources.Dialects.builtin
  MLIRPythonSources.Dialects.arith
  MLIRPythonSources.Dialects.math
  MLIRPythonSources.Dialects.memref
  MLIRPythonSources.Dialects.gpu
  MLIRPythonSources.Dialects.func
  MLIRPythonSources.Dialects.cf
  MLIRPythonSources.Dialects.scf
  MLIRPythonSources.Dialects.rocdl
  MLIRPythonSources.Dialects.vector
  MLIRPythonSources.Dialects.llvm
  MLIRPythonSources.ExecutionEngine
)

add_mlir_python_common_capi_library(FlyPythonCAPI
  INSTALL_COMPONENT FlyPythonModules
  INSTALL_DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES ${MLIRFlyDSLSources}
)

# ---------------------------------------------------------------------------
# Symbol visibility: prevent LLVM/MLIR internals from leaking out of
# FlyPythonCAPI.so, which would conflict with ROCm's own LLVM
# (libamd_comgr.so).
#
# Strategy:
#   1. -fvisibility=hidden  — hide symbols compiled by FlyDSL itself
#   2. Version script       — controls the FINAL exported symbol set:
#                             exports mlir C API + mlir:: C++ symbols
#                             (needed by _fly.so, _mlir.so, etc.),
#                             hides everything else (llvm::, LLVM C API).
#
# NOTE: Triton uses --exclude-libs,ALL because it ships a single
# libtriton.so with no downstream consumers.  FlyDSL cannot use
# --exclude-libs,ALL because GNU ld's --exclude-libs forces symbols
# to LOCAL *before* the version script is evaluated — wildcard patterns
# in the version script cannot re-export them.  The version script
# alone is sufficient: its `local: *;` catch-all hides LLVM internals.
# ---------------------------------------------------------------------------
if(UNIX AND NOT APPLE)
  target_compile_options(FlyPythonCAPI PRIVATE -fvisibility=hidden)

  set(FLYDSL_VERSION_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/flydsl_exports.ld")
  target_link_options(FlyPythonCAPI PRIVATE
    -Wl,--version-script=${FLYDSL_VERSION_SCRIPT})
  set_property(TARGET FlyPythonCAPI APPEND PROPERTY
    LINK_DEPENDS "${FLYDSL_VERSION_SCRIPT}")
endif()


set(FlyPythonModules_ROOT_PREFIX "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}")

################################################################################
# Python Modules
################################################################################

add_mlir_python_modules(FlyPythonModules
  ROOT_PREFIX "${FlyPythonModules_ROOT_PREFIX}"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  # Use "mlir" domain so Fly types are compatible with upstream IR types.
  # The package is self-contained under flydsl._mlir (no upstream mlir loaded).
  MLIR_BINDINGS_PYTHON_NB_DOMAIN "mlir"
  DECLARED_SOURCES "${MLIRFlyDSLSources}"
  COMMON_CAPI_LINK_LIBS
    FlyPythonCAPI
)

################################################################################
# Type Stubs Generation
################################################################################

set(_FLYDSL_PYTHON_PACKAGES_DIR "${MLIR_BINARY_DIR}/python_packages")
set(_MLIR_LIBS_DIR "${FlyPythonModules_ROOT_PREFIX}/_mlir_libs")
set(_STUB_MARKER_FILE "${_MLIR_LIBS_DIR}/.stubs_generated")

add_custom_command(
  OUTPUT "${_STUB_MARKER_FILE}"
  COMMAND /bin/bash -c "\
    PYTHONPATH='${_FLYDSL_PYTHON_PACKAGES_DIR}' \
    '${Python3_EXECUTABLE}' -m nanobind.stubgen \
      -q -r \
      -m flydsl._mlir._mlir_libs._mlir \
      -m flydsl._mlir._mlir_libs._fly \
      -m flydsl._mlir._mlir_libs._fly_rocdl \
      -m flydsl._mlir._mlir_libs._mlirDialectsGPU \
      -m flydsl._mlir._mlir_libs._mlirDialectsLLVM \
      -O '${_MLIR_LIBS_DIR}' \
    || echo 'Warning: nanobind.stubgen not available -- skipping stub generation'"
  COMMAND ${CMAKE_COMMAND} -E touch "${_STUB_MARKER_FILE}"
  DEPENDS CopyFlyPythonSources
  COMMENT "Generating Python stub files for all extension modules"
  VERBATIM
)

add_custom_target(FlyPythonStubs ALL
  DEPENDS "${_STUB_MARKER_FILE}"
)

add_custom_target(CopyFlyPythonSources ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${PROJECT_SOURCE_DIR}/python/flydsl"
    "${MLIR_BINARY_DIR}/python_packages/flydsl"
  # Copy auto-generated Python bindings from mlir-tblgen to the correct location
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${MLIR_BINARY_DIR}/python/mlir_flydsl/dialects/_fly_ops_gen.py"
    "${MLIR_BINARY_DIR}/python_packages/flydsl/_mlir/dialects/_fly_ops_gen.py"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${MLIR_BINARY_DIR}/python/mlir_flydsl/dialects/_fly_enum_gen.py"
    "${MLIR_BINARY_DIR}/python_packages/flydsl/_mlir/dialects/_fly_enum_gen.py"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${MLIR_BINARY_DIR}/python/mlir_flydsl/dialects/_fly_rocdl_ops_gen.py"
    "${MLIR_BINARY_DIR}/python_packages/flydsl/_mlir/dialects/_fly_rocdl_ops_gen.py"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${MLIR_BINARY_DIR}/python/mlir_flydsl/dialects/_fly_rocdl_enum_gen.py"
    "${MLIR_BINARY_DIR}/python_packages/flydsl/_mlir/dialects/_fly_rocdl_enum_gen.py"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE:mlir_c_runner_utils>"
    "${_MLIR_LIBS_DIR}/libmlir_c_runner_utils.so"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE:mlir_float16_utils>"
    "${_MLIR_LIBS_DIR}/$<TARGET_FILE_NAME:mlir_float16_utils>"
  COMMENT "Copying python/flydsl sources to build/python_packages/flydsl"
  DEPENDS FlyPythonModules
)

add_dependencies(CopyFlyPythonSources FlyPythonModules)


################################################################################
# FlyJitRuntime — GPU runtime wrappers
################################################################################

option(FLYDSL_ENABLE_ROCM "Enable ROCm/HIP JIT runtime" ON)
option(FLYDSL_ENABLE_COREX "Enable Iluvatar COREX (CUDA Driver API) JIT runtime" OFF)

set(_FLY_RUNTIME_BUILT OFF)

# --- Iluvatar COREX (CUDA Driver API compatible) ---
if(FLYDSL_ENABLE_COREX AND NOT _FLY_RUNTIME_BUILT)
    set(_COREX_SEARCH_PATHS
        "$ENV{COREX_PATH}"
        "$ENV{CUDA_PATH}"
        "/home/wenhui-liang/sw_home/local/corex"
        "/usr/local/corex"
    )
    set(_COREX_INCLUDE_DIR "")
    set(_COREX_LIB_DIR "")
    foreach(_p ${_COREX_SEARCH_PATHS})
        if(EXISTS "${_p}/include/cuda.h")
            set(_COREX_INCLUDE_DIR "${_p}/include")
            if(EXISTS "${_p}/lib64/libcuda.so")
                set(_COREX_LIB_DIR "${_p}/lib64")
            elseif(EXISTS "${_p}/lib/libcuda.so")
                set(_COREX_LIB_DIR "${_p}/lib")
            endif()
            break()
        endif()
    endforeach()

    if(_COREX_INCLUDE_DIR AND _COREX_LIB_DIR)
        message(STATUS "COREX SDK found: ${_COREX_INCLUDE_DIR} / ${_COREX_LIB_DIR}")
        message(STATUS "Building FlyJitRuntime with CUDA Driver API (COREX)...")

        add_library(FlyJitRuntime SHARED
          ${PROJECT_SOURCE_DIR}/lib/Runtime/FlyCorexRuntimeWrappers.cpp
        )
        target_include_directories(FlyJitRuntime PRIVATE
          ${LLVM_INCLUDE_DIRS}
          ${MLIR_INCLUDE_DIRS}
          ${_COREX_INCLUDE_DIR}
        )
        target_compile_features(FlyJitRuntime PRIVATE cxx_std_17)
        target_link_libraries(FlyJitRuntime PRIVATE
          "${_COREX_LIB_DIR}/libcuda.so"
        )
        set_target_properties(FlyJitRuntime PROPERTIES
          OUTPUT_NAME "fly_jit_runtime"
          LIBRARY_OUTPUT_DIRECTORY "${_MLIR_LIBS_DIR}"
        )
        add_dependencies(FlyPythonCAPI FlyJitRuntime)
        set(_FLY_RUNTIME_BUILT ON)
    else()
        message(WARNING "COREX SDK not found. Skipping FlyJitRuntime (COREX).")
    endif()
endif()

# --- ROCm/HIP (AMD hardware) ---
if(FLYDSL_ENABLE_ROCM AND NOT _FLY_RUNTIME_BUILT)
    file(GLOB _ROCM_SEARCH_PATHS LIST_DIRECTORIES true "/opt/rocm*")
    list(SORT _ROCM_SEARCH_PATHS ORDER DESCENDING)
    find_package(hip QUIET CONFIG PATHS ${_ROCM_SEARCH_PATHS})

    if(hip_FOUND)
        message(STATUS "ROCm/HIP found. Building FlyJitRuntime...")
        add_library(FlyJitRuntime SHARED
          ${PROJECT_SOURCE_DIR}/lib/Runtime/FlyRocmRuntimeWrappers.cpp
        )
        target_include_directories(FlyJitRuntime PRIVATE
          ${LLVM_INCLUDE_DIRS}
          ${MLIR_INCLUDE_DIRS}
        )
        target_compile_features(FlyJitRuntime PRIVATE cxx_std_17)
        target_link_libraries(FlyJitRuntime PRIVATE hip::host hip::amdhip64)
        set_target_properties(FlyJitRuntime PROPERTIES
          OUTPUT_NAME "fly_jit_runtime"
          LIBRARY_OUTPUT_DIRECTORY "${_MLIR_LIBS_DIR}"
        )
        add_dependencies(FlyPythonCAPI FlyJitRuntime)
        target_compile_definitions(FlyJitRuntime PRIVATE FLYDSL_HAS_ROCM=1)
        set(_FLY_RUNTIME_BUILT ON)
    else()
        message(WARNING "ROCm/HIP not found. Skipping FlyJitRuntime.")
        set(FLYDSL_ENABLE_ROCM OFF)
    endif()
endif()

if(NOT _FLY_RUNTIME_BUILT)
    message(STATUS "Building without GPU runtime support.")
    add_definitions(-DFLYDSL_DISABLE_ROCM)
endif()


