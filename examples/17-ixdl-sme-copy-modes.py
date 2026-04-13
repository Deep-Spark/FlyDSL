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
from flydsl.expr import range_constexpr

ROW_SHAPE = (16, 32)
COL_SHAPE = (16, 32)
SME_BITS = 16 * 512
SME_BYTES = SME_BITS // 8


def _store_logical_view_to_global(src, dst, shape):
    for i in range_constexpr(shape[0]):
        for j in range_constexpr(shape[1]):
            value = fx.memref_load(src, (i, j))
            fx.memref_store(value, dst, (i, j))


@flyc.kernel
def ixdl_sme_copy_row_kernel(src: fx.Tensor, dst: fx.Tensor):
    smem = fx.ixdl.SMEView16x512b(fx.get_dyn_shared(), fx.Float16)

    load_atom = fx.make_copy_atom(fx.ixdl.SMELoad16x512b(fx.Float16), fx.Float16)

    fx.copy_atom_call(load_atom, src, smem)
    fx.ixdl.cp_async_commit_group()
    fx.ixdl.cp_async_wait_group(0)
    fx.gpu.barrier()
    _store_logical_view_to_global(smem, dst, ROW_SHAPE)


@flyc.kernel
def ixdl_sme_copy_col_kernel(src: fx.Tensor, dst: fx.Tensor):
    smem = fx.ixdl.SMEView16x512b(fx.get_dyn_shared(), fx.Float16, transpose=True)

    load_atom = fx.make_copy_atom(fx.ixdl.SMELoad16x512b(fx.Float16, transpose=True), fx.Float16)

    fx.copy_atom_call(load_atom, src, smem)
    fx.ixdl.cp_async_commit_group()
    fx.ixdl.cp_async_wait_group(0)
    fx.gpu.barrier()
    _store_logical_view_to_global(smem, dst, COL_SHAPE)


@flyc.jit
def ixdl_sme_copy_row(src: fx.Tensor, dst: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    ixdl_sme_copy_row_kernel(src, dst).launch(
        grid=(1, 1, 1),
        block=(64, 1, 1),
        smem=SME_BYTES,
        stream=stream,
    )


@flyc.jit
def ixdl_sme_copy_col(src: fx.Tensor, dst: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    ixdl_sme_copy_col_kernel(src, dst).launch(
        grid=(1, 1, 1),
        block=(64, 1, 1),
        smem=SME_BYTES,
        stream=stream,
    )


def _run_case(name, launcher, src):
    dst = torch.zeros_like(src)
    stream = torch.cuda.Stream()
    launcher(src, dst, stream=stream)

    if os.environ.get("COMPILE_ONLY", "1") != "0":
        print(f"[{name}] compile-only launch emitted")
        return

    torch.cuda.synchronize()
    ok = bool(torch.equal(src, dst))
    print(f"[{name}] allclose = {ok}")
    print(f"[{name}] src[0, :8] = {src[0, :8].cpu()}")
    print(f"[{name}] dst[0, :8] = {dst[0, :8].cpu()}")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for this ixdl SME copy demo")

    src_row = torch.arange(ROW_SHAPE[0] * ROW_SHAPE[1], dtype=torch.int32, device="cuda").to(torch.float16)
    src_row = src_row.reshape(*ROW_SHAPE).contiguous()

    src_col = torch.arange(COL_SHAPE[0] * COL_SHAPE[1], dtype=torch.int32, device="cuda").to(torch.float16)
    src_col = src_col.reshape(*COL_SHAPE).contiguous()

    _run_case("row", ixdl_sme_copy_row, src_row)
    _run_case("col", ixdl_sme_copy_col, src_col)


if __name__ == "__main__":
    main()
