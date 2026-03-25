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
        chip = self.target.arch
        binary_format = "llvm" if env.compile.compile_only else "fatbin"
        return [
            "fly-rewrite-func-signature",
            "fly-canonicalize",
            "fly-layout-lowering",
            "convert-fly-to-rocdl",
            "canonicalize",
            (
                "gpu-lower-to-ixdl-pipeline{"
                f"binary-chip={chip} "
                f"binary-format={binary_format} "
                f"opt-level={env.compile.opt_level} "
                "kernel-bare-ptr-calling-convention=true "
                "host-bare-ptr-calling-convention=true}"
            ),
        ]

    def gpu_module_targets(self) -> List[str]:
        chip = self.target.arch
        return [f'#ixdl.target<chip = "{chip}">']

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
