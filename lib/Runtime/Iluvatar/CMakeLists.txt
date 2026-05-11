# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# FlyIluvatarJitRuntime — CUDA-compatible Iluvatar JIT runtime wrappers.

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
