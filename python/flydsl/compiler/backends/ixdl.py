# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from typing import List

from ...utils import env
from .base import BaseBackend, GPUTarget


class IxdlBackend(BaseBackend):
    """Iluvatar IXDL compile backend (CUDA runtime, IXDL lowering)."""

    @staticmethod
    def supports_target(target: GPUTarget) -> bool:
        return target.backend == "ixdl"

    @staticmethod
    def detect_target() -> GPUTarget:
        arch = env.compile.arch or "ivcore11"
        return GPUTarget(backend="ixdl", arch=arch, warp_size=64)

    @classmethod
    def make_target(cls, arch: str) -> GPUTarget:
        return GPUTarget(backend="ixdl", arch=arch or "ivcore11", warp_size=64)

    def pipeline_fragments(self, *, compile_hints: dict) -> List[str]:
        binary_format = "llvm" if env.compile.compile_only else "fatbin"
        return [
            "fly-rewrite-func-signature",
            "fly-canonicalize",
            "fly-layout-lowering",
            "convert-fly-to-rocdl",
            "canonicalize",
            (
                "gpu.module("
                "convert-scf-to-cf,"
                "cse,"
                "convert-gpu-to-ixdl{index-bitwidth=0 use-bare-ptr-memref-call-conv=true},"
                "canonicalize,"
                "cse,"
                "reconcile-unrealized-casts)"
            ),
            "eliminate-extra-extend-trunc",
            "optimize-gep-offset",
            "convert-scf-to-cf",
            "convert-cf-to-llvm",
            "gpu-to-llvm{use-bare-pointers-for-host=true use-bare-pointers-for-kernels=true}",
            "convert-arith-to-llvm",
            "convert-func-to-llvm",
            "reconcile-unrealized-casts",
            f"gpu-module-to-binary{{format={binary_format}}}",
        ]

    def gpu_module_targets(self) -> List[str]:
        chip = self.target.arch
        return [f'#ixdl.target<O = {env.compile.opt_level}, chip = "{chip}">']

    def native_lib_patterns(self) -> List[str]:
        return [
            "_fly*.so",
            "libFly*.so",
            "libmlir_cuda_runtime.so*",
            "_mlirRegisterEverything*.so",
        ]

    def jit_runtime_lib_basenames(self) -> List[str]:
        return [
            "libmlir_cuda_runtime.so",
            "libmlir_c_runner_utils.so",
        ]
