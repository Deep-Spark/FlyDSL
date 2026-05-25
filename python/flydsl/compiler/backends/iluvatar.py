# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

from typing import List

from ...utils import env
from .base import BaseBackend, GPUTarget

_DEFAULT_ARCH = "ivcore11"
_WARP_SIZE = 64


class IluvatarBackend(BaseBackend):
    """Iluvatar compile backend."""

    @staticmethod
    def supports_target(target: GPUTarget) -> bool:
        return target.backend == "iluvatar"

    @staticmethod
    def detect_target() -> GPUTarget:
        return IluvatarBackend.make_target(env.compile.arch or _DEFAULT_ARCH)

    @classmethod
    def make_target(cls, arch: str) -> GPUTarget:
        return GPUTarget(backend="iluvatar", arch=arch or _DEFAULT_ARCH, warp_size=_WARP_SIZE)

    @staticmethod
    def _format_pass_opts(opts: dict) -> str:
        """Format {key: value, ...} as 'key=value key2=value2' for MLIR pass options."""
        return " ".join(f"{k}={v}" for k, v in opts.items())

    def pipeline_fragments(self, *, compile_hints: dict) -> List[str]:
        chip = self.target.arch
        ixdl_opts = {
            "O": 2,
            "chip": chip,
            "triple": "bi-iluvatar-ilurt",
        }

        return [
            "fly-rewrite-func-signature",
            "fly-canonicalize",
            "fly-layout-lowering",
            "fly-int-swizzle-simplify",
            "canonicalize",
            "fly-convert-atom-call-to-ssa-form",
            "fly-promote-regmem-to-vectorssa",
            "convert-fly-to-ixdl",
            "canonicalize",
            "gpu.module(convert-scf-to-cf,cse,"
            "convert-gpu-to-ixdl{index-bitwidth=0 use-bare-ptr-memref-call-conv=true})",
            f"ixdl-attach-target{{{self._format_pass_opts(ixdl_opts)}}}",
            "convert-scf-to-cf",
            "convert-cf-to-llvm",
            "gpu-to-llvm{use-bare-pointers-for-host=true use-bare-pointers-for-kernels=true}",
            "convert-vector-to-llvm",
            "convert-arith-to-llvm",
            "convert-func-to-llvm",
            "reconcile-unrealized-casts",
            "gpu-module-to-binary{format=fatbin}",
        ]

    def gpu_module_targets(self) -> List[str]:
        return [f'#ixdl.target<chip = "{self.target.arch}">']

    def native_lib_patterns(self) -> List[str]:
        return [
            "_mlirDialectsFly*.so",
            "libFly*.so",
            "libfly_iluvatar_jit_runtime.so",
            "_mlirRegisterEverything*.so",
        ]

    def jit_runtime_lib_basenames(self) -> List[str]:
        return [
            "libfly_iluvatar_jit_runtime.so",
            "libmlir_c_runner_utils.so",
        ]
