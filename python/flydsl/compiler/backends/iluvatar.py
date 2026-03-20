# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from typing import List

from ...runtime.device import get_target_arch
from ...utils import env
from .base import BaseBackend, GPUTarget


class IluvatarBackend(BaseBackend):
    """Iluvatar Corex compile backend (IXDL lowering, CUDA-compatible driver API)."""

    @staticmethod
    def supports_target(target: GPUTarget) -> bool:
        return target.backend == "iluvatar"

    @staticmethod
    def detect_target() -> GPUTarget:
        arch = env.compile.arch or get_target_arch()
        return GPUTarget(backend="iluvatar", arch=arch, warp_size=32)

    @classmethod
    def make_target(cls, arch: str) -> GPUTarget:
        return GPUTarget(backend="iluvatar", arch=arch, warp_size=32)

    def pipeline_fragments(self, *, compile_hints: dict) -> List[str]:
        _ = compile_hints
        chip = self.target.arch
        debug_opt = "-g" if env.debug.enable_debug_info else ""
        all_opts = debug_opt.strip()

        return [
            "fly-rewrite-func-signature",
            "fly-canonicalize",
            "fly-layout-lowering",
            "convert-fly-to-ixdl",
            "canonicalize",
            f"gpu.module(convert-scf-to-cf,cse,"
            f"convert-gpu-to-ixdl{{index-bitwidth=0 use-bare-ptr-memref-call-conv=true}})",
            "convert-scf-to-cf",
            "convert-cf-to-llvm",
            "gpu-to-llvm{use-bare-pointers-for-host=true use-bare-pointers-for-kernels=true}",
            "convert-arith-to-llvm",
            "convert-func-to-llvm",
            "reconcile-unrealized-casts",
            *(
                ["ensure-debug-info-scope-on-llvm-func{emission-kind=LineTablesOnly}"]
                if env.debug.enable_debug_info
                else []
            ),
            f'gpu-module-to-binary{{format=fatbin opts="{all_opts}"}}',
        ]

    def gpu_module_targets(self) -> List[str]:
        chip = self.target.arch
        return [f'#ixdl.target<chip = "{chip}">']

    def native_lib_patterns(self) -> List[str]:
        return [
            "_mlirDialectsFly*.so",
            "libFly*.so",
            "libfly_ix_jit_runtime.so",
            "libfly_jit_runtime.so",
            "_mlirRegisterEverything*.so",
        ]

    def jit_runtime_lib_basenames(self) -> List[str]:
        return [
            "libfly_ix_jit_runtime.so",
            "libmlir_c_runner_utils.so",
        ]
