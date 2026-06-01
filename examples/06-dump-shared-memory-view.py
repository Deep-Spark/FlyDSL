# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Dump the shared-memory layout produced by MR async-copy Col b32.

The kernel copies one 16x16 f32 tile from global memory to shared memory using
``ixdl.cp_async.16x16.b32.col``. After the async copy completes, only thread 0
reads shared memory back linearly with ordinary memref load/store operations.
This makes the output tensor a direct dump of physical shared-memory order.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "iluvatar")
os.environ.setdefault("FLYDSL_RUNTIME_KIND", "iluvatar")
os.environ.setdefault("ARCH", "ivcore11")
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
os.environ.setdefault("FLYDSL_DEBUG_PRINT_AFTER_ALL", "0")

import torch  # noqa: E402

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
import flydsl.expr.ixdl as ixdl  # noqa: E402
from flydsl.expr import range_constexpr  # noqa: E402


THREADS = 64
M = 16
N = 16
ELEMENTS = M * N
SMEM_BYTES = ELEMENTS * 4


@flyc.kernel
def _dump_shared_memory_view_kernel(src: fx.Tensor, dst: fx.Tensor):
    tid = fx.gpu.thread_id("x")

    smem = fx.make_view(fx.get_dyn_shared(fx.Float32), fx.make_layout(ELEMENTS, 1))

    sme_src = ixdl.make_sme_gmem_tensor(src, leading_stride=N)
    async_atom = fx.make_copy_atom(ixdl.MRAsyncCpCol(), fx.Float32)
    tile_layout = fx.make_layout(ELEMENTS, 1)
    src_tile = fx.make_view(fx.add_offset(fx.get_iter(sme_src), fx.Int32(0)), tile_layout)
    smem_tile = fx.make_view(fx.add_offset(fx.get_iter(smem), 0), tile_layout)
    fx.copy_atom_call(async_atom, src_tile, smem_tile)

    ixdl.cp_async_commit_group()
    ixdl.cp_async_wait_group(0)
    fx.gpu.barrier()

    if tid == fx.Index(0):
        for i in range_constexpr(ELEMENTS):
            idx = fx.Int32(i)
            dst[idx] = smem[idx]


@flyc.jit
def launch_dump_shared_memory_view(src: fx.Tensor, dst: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    _dump_shared_memory_view_kernel(src, dst).launch(
        grid=(1, 1, 1),
        block=(THREADS, 1, 1),
        smem=SMEM_BYTES,
        stream=stream,
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-compatible Iluvatar device is not available")

    src = torch.arange(ELEMENTS, device="cuda", dtype=torch.float32).reshape(M, N)
    dst = torch.empty(ELEMENTS, device="cuda", dtype=torch.float32)

    launch_dump_shared_memory_view(src, dst)
    torch.cuda.synchronize()

    print("Input logical 16x16 row-major tile:")
    print(src.cpu().to(torch.int32))
    print("\nShared memory physical 16x16 view after ixdl.cp_async.16x16.b32.col:")
    print(dst.reshape(M, N).cpu().to(torch.int32))
    print("\nShared memory physical linear order:")
    print(dst.cpu().to(torch.int32).tolist())


if __name__ == "__main__":
    main()
