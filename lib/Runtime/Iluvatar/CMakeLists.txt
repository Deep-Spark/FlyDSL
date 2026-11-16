# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# FlyIluvatarJitRuntime -- CUDA-compatible Iluvatar JIT runtime wrappers.

find_package(CUDAToolkit REQUIRED)

add_library(FlyIluvatarJitRuntime SHARED FlyIluvatarRuntimeWrappers.cpp)
target_include_directories(FlyIluvatarJitRuntime PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)
target_compile_features(FlyIluvatarJitRuntime PRIVATE cxx_std_17)
target_link_libraries(FlyIluvatarJitRuntime PRIVATE CUDA::cuda_driver)
set_target_properties(FlyIluvatarJitRuntime PROPERTIES
  OUTPUT_NAME "fly_iluvatar_jit_runtime")

# Iluvatar AOT exports partially link this PIC archive into the generated host
# object. Its hidden symbols are localized by the relocatable link, leaving the
# CUDA driver and standard system libraries as the only deployment dependencies.
add_library(FlyIluvatarAotRuntime STATIC
  FlyIluvatarAotRuntime.cpp
  FlyIluvatarRuntimeError.cpp
)
target_compile_definitions(FlyIluvatarAotRuntime PRIVATE FLYDSL_AOT_RUNTIME_EMBEDDED=1)
target_compile_features(FlyIluvatarAotRuntime PRIVATE cxx_std_17)
# Without this, the scoped-context destructor gets unwind landing pads, and the
# exported object would need a C++ runtime alongside the CUDA driver and libc.
target_compile_options(FlyIluvatarAotRuntime PRIVATE -fno-exceptions -fno-rtti)
target_link_libraries(FlyIluvatarAotRuntime PRIVATE CUDA::cuda_driver)
set_target_properties(FlyIluvatarAotRuntime PROPERTIES
  OUTPUT_NAME "fly_iluvatar_aot_runtime"
  POSITION_INDEPENDENT_CODE ON
)
