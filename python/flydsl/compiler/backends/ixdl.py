# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar IXDL compile backend (``ivcore11``, CUDA-compatible host runtime).

Pipeline (mirrors the ROCm one where possible):

1. ``fly-*`` front passes + ``convert-fly-to-rocdl`` (acts as a generic
   FlyToLLVM lowering for operations that don't depend on the AMDGPU
   dialect — e.g. ``UniversalCopy*``/arith/memref).
2. Per-gpu.module: ``convert-gpu-to-ixdl`` turns ``gpu.thread_id`` /
   ``gpu.block_id`` / ``gpu.barrier`` / etc. into ``ixdl.*`` intrinsics and
   LLVM ops.
3. Host-side finalization (``gpu-to-llvm`` with bare-pointer ABI), then
   ``gpu-module-to-binary`` picks up the ``#ixdl.target`` attribute and
   invokes Iluvatar's ``IluvatarSerializer`` (llc + ``ld.lld``) to produce
   a device binary.

The ``#ixdl.target`` attribute defaults already cover MR-V100 / MR-V50
(``chip = "ivcore11"``, ``triple = "bi-iluvatar-ilurt"``), but we emit it
explicitly so the backend is self-documenting and cache-key stable.
"""

from __future__ import annotations

from typing import List

from ...runtime.ixdl_device import get_ixdl_arch
from ...utils import env
from .base import BaseBackend, GPUTarget


class IxdlBackend(BaseBackend):
    """Iluvatar IXDL compile backend (``ivcore11``)."""

    @staticmethod
    def supports_target(target: GPUTarget) -> bool:
        return target.backend == "ixdl"

    @staticmethod
    def detect_target() -> GPUTarget:
        arch = env.compile.arch or get_ixdl_arch()
        return GPUTarget(backend="ixdl", arch=arch, warp_size=64)

    @classmethod
    def make_target(cls, arch: str) -> GPUTarget:
        return GPUTarget(backend="ixdl", arch=arch, warp_size=64)

    # -- compile pipeline ------------------------------------------------

    def pipeline_fragments(self, *, compile_hints: dict) -> List[str]:
        bin_cli_opts: List[str] = []
        if env.debug.enable_debug_info:
            bin_cli_opts.append("-g")

        return [
            "fly-rewrite-func-signature",
            "fly-canonicalize",
            "fly-layout-lowering",
            "fly-convert-atom-call-to-ssa-form",
            "fly-promote-regmem-to-vectorssa",
            # convert-fly-to-rocdl is the generic FlyToLLVM pass in disguise;
            # when the kernel only uses Universal* atoms it emits plain LLVM
            # (no ROCDL/AMDGPU ops), which is exactly what IXDL wants.
            "convert-fly-to-rocdl",
            "canonicalize",
            (
                "gpu.module(convert-scf-to-cf,cse,"
                "convert-gpu-to-ixdl{index-bitwidth=0 use-bare-ptr-memref-call-conv=true},"
                "convert-math-to-llvm)"
            ),
            "convert-scf-to-cf",
            "convert-cf-to-llvm",
            "gpu-to-llvm{use-bare-pointers-for-host=true use-bare-pointers-for-kernels=true}",
            "convert-vector-to-llvm",
            "convert-arith-to-llvm",
            "convert-func-to-llvm",
            "reconcile-unrealized-casts",
            *(
                ["ensure-debug-info-scope-on-llvm-func{emission-kind=LineTablesOnly}"]
                if env.debug.enable_debug_info
                else []
            ),
            f'gpu-module-to-binary{{format=fatbin opts="{" ".join(bin_cli_opts)}"}}',
        ]

    def gpu_module_targets(self) -> List[str]:
        chip = self.target.arch
        libdevice = "/home/caokefan/sw_home/local/corex/nvvm/libdevice/libdevice.compute_bi.10.bc"
        return [f'#ixdl.target<chip = "{chip}", link = ["{libdevice}"]>']

    # -- cache / fingerprint ---------------------------------------------

    def native_lib_patterns(self) -> List[str]:
        return [
            "_mlirDialectsFly*.so",
            "libFly*.so",
            "libmlir_cuda_runtime.so",
            "_mlirRegisterEverything*.so",
        ]

    def jit_runtime_lib_basenames(self) -> List[str]:
        # ``libmlir_cuda_runtime.so`` ships with ixcc and exposes the standard
        # ``mgpu*`` shims required by the lowered host IR. No FlyDSL-owned
        # runtime glue is needed for Iluvatar; UniversalCopy lowers straight
        # to LLVM, so ``libmlir_c_runner_utils`` covers printf/abort helpers.
        return [
            "libmlir_cuda_runtime.so",
            "libmlir_c_runner_utils.so",
        ]
