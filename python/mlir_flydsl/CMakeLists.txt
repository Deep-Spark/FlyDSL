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
# The COMMON_CAPI_LINK_LIBS FlyPythonCAPI in add_mlir_python_modules ensures _mlirDialectsFly.so
# links to FlyPythonCAPI.so, which provides all needed symbols.

declare_mlir_python_extension(FlyPythonSources.Core.fly
  MODULE_NAME _mlirDialectsFly
  ADD_TO_PARENT FlyPythonSources
  ROOT_DIR "${PROJECT_SOURCE_DIR}/lib/Bindings/Python"
  PYTHON_BINDINGS_LIBRARY nanobind
  SOURCES
    TiledOpTraits.cpp
    FlyExtension.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
)

declare_mlir_python_extension(FlyPythonSources.Core.fly_rocdl
  MODULE_NAME _mlirDialectsFlyROCDL
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
    MLIRCPIFlyIXDL
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
#                             (needed by _mlirDialectsFly.so, _mlir.so, etc.),
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
      -m flydsl._mlir._mlir_libs._mlirDialectsFly \
      -m flydsl._mlir._mlir_libs._mlirDialectsFlyROCDL \
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
  COMMAND bash -c "patchelf --set-rpath '\\$$ORIGIN' '${_MLIR_LIBS_DIR}/libmlir_c_runner_utils.so'"
  COMMAND bash -c "patchelf --set-rpath '\\$$ORIGIN' '${_MLIR_LIBS_DIR}/$<TARGET_FILE_NAME:mlir_float16_utils>'"
  COMMENT "Copying python/flydsl sources to build/python_packages/flydsl"
  DEPENDS FlyPythonModules
)

add_dependencies(CopyFlyPythonSources FlyPythonModules)


################################################################################
# FlyJitRuntime — thin ROCm runtime with GPU module caching
################################################################################

# 1. 定义一个开关，默认如果是 AMD 环境就开启，否则可以手动关闭
option(FLYDSL_ENABLE_ROCM "Enable ROCm/HIP JIT runtime" ON)
 
# 2. 只有在开启且找到相关路径时才进行
if(FLYDSL_ENABLE_ROCM)
    file(GLOB _ROCM_SEARCH_PATHS LIST_DIRECTORIES true "/opt/rocm*")
    list(SORT _ROCM_SEARCH_PATHS ORDER DESCENDING)
     
    # 移除 REQUIRED，改为 QUIET，避免找不到时报错
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
         
        # 传递宏定义给 C++ 源码，告知 ROCM 已启用
        target_compile_definitions(FlyJitRuntime PRIVATE FLYDSL_HAS_ROCM=1)
    else()
        message(WARNING "ROCm/HIP not found. Skipping FlyJitRuntime.")
        set(FLYDSL_ENABLE_ROCM OFF)
    endif()
endif()
 
# 3. 如果没启用 ROCm，确保 FlyPythonCAPI 不会去链接不存在的库
if(NOT FLYDSL_ENABLE_ROCM)
    message(STATUS "Building without ROCm support.")
    # 如果有其他地方引用了 FlyJitRuntime，可能需要在这里定义一些空操作或宏
    add_definitions(-DFLYDSL_DISABLE_ROCM)
endif()

################################################################################
# FlyIXJitRuntime — thin Iluvatar Corex runtime (CUDA-compatible)
################################################################################

option(FLYDSL_ENABLE_ILUVATAR "Enable Iluvatar Corex JIT runtime" ON)

if(FLYDSL_ENABLE_ILUVATAR)
    set(ILUVATAR_ROOT "$ENV{SW_HOME}/local/corex")
    
    find_path(IXDL_INCLUDE_DIR NAMES cuda.h PATHS "${ILUVATAR_ROOT}/include")
    find_library(IXDL_LIBRARY NAMES cuda PATHS "${ILUVATAR_ROOT}/lib64")

    if(IXDL_INCLUDE_DIR AND IXDL_LIBRARY)
        message(STATUS "Iluvatar Corex SDK found at ${ILUVATAR_ROOT}. Building FlyIXJitRuntime...")
        
        add_library(FlyIXJitRuntime SHARED
          ${PROJECT_SOURCE_DIR}/lib/Runtime/FlyIXDLRuntimeWrappers.cpp
        )
        target_include_directories(FlyIXJitRuntime PRIVATE
          ${IXDL_INCLUDE_DIR}
          ${LLVM_INCLUDE_DIRS}
          ${MLIR_INCLUDE_DIRS}
        )
        target_compile_features(FlyIXJitRuntime PRIVATE cxx_std_17)
        target_link_libraries(FlyIXJitRuntime PRIVATE ${IXDL_LIBRARY})
        set_target_properties(FlyIXJitRuntime PROPERTIES
          OUTPUT_NAME "fly_ix_jit_runtime"
          LIBRARY_OUTPUT_DIRECTORY "${_MLIR_LIBS_DIR}"
        )
        add_dependencies(FlyPythonCAPI FlyIXJitRuntime)
    else()
        message(WARNING "Iluvatar Corex SDK not found. Skipping FlyIXJitRuntime. (IXDL_INCLUDE_DIR=${IXDL_INCLUDE_DIR}, IXDL_LIBRARY=${IXDL_LIBRARY})")
        set(FLYDSL_ENABLE_ILUVATAR OFF)
    endif()
endif()
