#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path

os.environ.setdefault("BACKEND", "ixdl")
os.environ.setdefault("COMPILE_ONLY", "1")

_repo_root = Path(__file__).resolve().parents[1]
_build_pkg_dir = _repo_root / "build-fly" / "python_packages"
if _build_pkg_dir.exists():
    sys.path.insert(0, str(_build_pkg_dir))

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

SMEM_ROWS = 16
SMEM_COLS = 32
SMEM_BITS = SMEM_ROWS * 512
SMEM_BYTES = SMEM_ROWS * SMEM_COLS * 2


@flyc.kernel
def ixdl_sme_copy_kernel(src: fx.Tensor, dst: fx.Tensor):
    # Build a shared-memory fly.memref view from the dynamic shared base pointer.
    smem_layout = fx.make_layout((SMEM_ROWS, SMEM_COLS), (SMEM_COLS, 1))
    smem_base = fx.get_dyn_shared()
    smem_ptr_ty = fx.PointerType.get(fx.T.f16(), fx.AddressSpace.Shared)
    smem_ptr = fx.recast_iter(smem_ptr_ty, smem_base)
    smem = fx.make_view(smem_ptr, smem_layout)

    load_atom = fx.make_copy_atom(fx.ixdl.SMELoad16x512b(fx.Float16), fx.Float16)
    store_atom = fx.make_copy_atom(fx.UniversalCopy(SMEM_BITS), fx.Float16)

    fx.copy(load_atom, src, smem)
    fx.ixdl.cp_async_commit_group()
    fx.ixdl.cp_async_wait_group(0)
    fx.copy(store_atom, smem, dst)


@flyc.jit
def ixdl_sme_copy(src: fx.Tensor, dst: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    ixdl_sme_copy_kernel(src, dst).launch(
        grid=(1, 1, 1),
        block=(1, 1, 1),
        smem=SMEM_BYTES,
        stream=stream,
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for this ixdl SME copy demo")

    src = torch.arange(SMEM_ROWS * SMEM_COLS, dtype=torch.int32, device="cuda").to(torch.float16)
    src = src.reshape(SMEM_ROWS, SMEM_COLS).contiguous()
    dst = torch.zeros_like(src)

    stream = torch.cuda.Stream()
    ixdl_sme_copy(src, dst, stream=stream)

    if os.environ.get("COMPILE_ONLY", "1") != "0":
        print("[ixdl_sme_copy] compile-only launch emitted")
        return

    torch.cuda.synchronize()
    ok = bool(torch.equal(src, dst))
    print("[ixdl_sme_copy] allclose =", ok)
    print("[ixdl_sme_copy] src[0, :8] =", src[0, :8].cpu())
    print("[ixdl_sme_copy] dst[0, :8] =", dst[0, :8].cpu())


if __name__ == "__main__":
    main()
