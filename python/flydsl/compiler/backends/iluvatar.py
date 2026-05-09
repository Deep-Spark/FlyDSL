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

    def pipeline_fragments(self, *, compile_hints: dict) -> List[str]:
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
            "reconcile-unrealized-casts",
        ]

    def gpu_module_targets(self) -> List[str]:
        return [f'#iluvatar.target<arch = "{self.target.arch}">']

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
