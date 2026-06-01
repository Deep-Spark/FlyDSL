# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar MR async-copy examples.

This file is organized as a collection of runnable cases. Each case launches
multiple thread blocks; each block copies one logical matrix from a padded
global-memory tensor to shared memory with MR SME async-copy instructions, waits
for the copies, loads shared-memory data into per-thread registers, and writes
it back to global memory.
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


BLOCKS = 4
THREADS = 64


def _async_copy_case_body(
    src,
    dst,
    *,
    dtype,
    async_copy_op,
    scalar_copy_op,
    matrix_m,
    matrix_n,
    src_stride_n,
    async_tile_m,
    async_tile_n,
):
    bid = fx.gpu.block_id("x")
    tid = fx.gpu.thread_id("x")

    async_tile_elems = async_tile_m * async_tile_n
    tile_rows = matrix_m // async_tile_m
    tile_cols = matrix_n // async_tile_n
    matrix_elems = matrix_m * matrix_n
    elems_per_thread = matrix_elems // THREADS

    src_block_offset = bid * fx.Index(matrix_m * src_stride_n)
    dst_block_offset = bid * fx.Index(matrix_elems)

    smem_layout = fx.make_layout(matrix_elems, 1)
    smem = fx.make_view(fx.get_dyn_shared(dtype), smem_layout)

    sme_src = ixdl.make_sme_gmem_tensor(src, leading_stride=src_stride_n)
    async_atom = fx.make_copy_atom(async_copy_op(), dtype)
    async_tile_layout = fx.make_layout(async_tile_elems, 1)

    for tile_m in range_constexpr(tile_rows):
        for tile_n in range_constexpr(tile_cols):
            tile_id = tile_m * tile_cols + tile_n
            src_offset = tile_m * async_tile_m * src_stride_n + tile_n * async_tile_n
            smem_offset = tile_id * async_tile_elems
            src_tile_offset = fx.Int32(src_block_offset + fx.Index(src_offset))
            src_tile = fx.make_view(fx.add_offset(fx.get_iter(sme_src), src_tile_offset), async_tile_layout)
            smem_tile = fx.make_view(fx.add_offset(fx.get_iter(smem), smem_offset), async_tile_layout)
            fx.copy_atom_call(async_atom, src_tile, smem_tile)
    ixdl.cp_async_commit_group()
    ixdl.cp_async_wait_group(0)
    fx.gpu.barrier()

    scalar_copy = fx.make_copy_atom(scalar_copy_op(), dtype)
    smem_elems = fx.logical_divide(smem, fx.make_layout(1, 1))
    dst_elems = fx.logical_divide(dst, fx.make_layout(1, 1))
    reg = fx.make_rmem_tensor(fx.make_layout(1, 1), dtype)

    base = tid * fx.Index(elems_per_thread)
    for i in range_constexpr(elems_per_thread):
        idx = base + fx.Index(i)
        row = idx // fx.Index(matrix_n)
        col = idx % fx.Index(matrix_n)
        tile_row = row // fx.Index(async_tile_m)
        tile_col = col // fx.Index(async_tile_n)
        inner_row = row % fx.Index(async_tile_m)
        inner_col = col % fx.Index(async_tile_n)
        smem_idx = (
            (tile_row * fx.Index(tile_cols) + tile_col) * fx.Index(async_tile_elems)
            + inner_row * fx.Index(async_tile_n)
            + inner_col
        )
        dst_idx_i32 = fx.Int32(dst_block_offset + idx)
        smem_idx_i32 = fx.Int32(smem_idx)
        fx.copy_atom_call(scalar_copy, fx.slice(smem_elems, (None, smem_idx_i32)), reg)
        fx.copy_atom_call(scalar_copy, reg, fx.slice(dst_elems, (None, dst_idx_i32)))


def _tile_constant_tensors(*, dtype, matrix_m, matrix_n, src_stride_n, async_tile_m, async_tile_n):
    storage = torch.empty((BLOCKS, matrix_m, src_stride_n), device="cuda", dtype=dtype)
    storage.zero_()
    expected = torch.empty((BLOCKS, matrix_m, matrix_n), device="cuda", dtype=dtype)

    tile_rows = matrix_m // async_tile_m
    tile_cols = matrix_n // async_tile_n
    for block in range(BLOCKS):
        for tile_m in range(tile_rows):
            for tile_n in range(tile_cols):
                value = block * tile_rows * tile_cols + tile_m * tile_cols + tile_n + 1
                storage[
                    block,
                    tile_m * async_tile_m : (tile_m + 1) * async_tile_m,
                    tile_n * async_tile_n : (tile_n + 1) * async_tile_n,
                ] = value
                expected[
                    block,
                    tile_m * async_tile_m : (tile_m + 1) * async_tile_m,
                    tile_n * async_tile_n : (tile_n + 1) * async_tile_n,
                ] = value

    src = storage[:, :, :matrix_n]
    dst = torch.empty(BLOCKS * matrix_m * matrix_n, device="cuda", dtype=dtype)
    return src, dst, expected.contiguous().reshape(-1)


def _block_constant_tensors(*, dtype, matrix_m, matrix_n, src_stride_n):
    storage = torch.empty((BLOCKS, matrix_m, src_stride_n), device="cuda", dtype=dtype)
    expected = torch.empty((BLOCKS, matrix_m, matrix_n), device="cuda", dtype=dtype)

    for block in range(BLOCKS):
        value = block + 1
        storage[block].fill_(value)
        expected[block].fill_(value)

    src = storage[:, :, :matrix_n]
    dst = torch.empty(BLOCKS * matrix_m * matrix_n, device="cuda", dtype=dtype)
    return src, dst, expected.contiguous().reshape(-1)


# === f32 row-major / NoSwizzle -> ixdl.cp_async.16x16.b32.row ===

F32_ROW_MAJOR_M = 32
F32_ROW_MAJOR_N = 64
F32_ROW_MAJOR_SRC_STRIDE_N = 80
F32_ROW_MAJOR_ASYNC_TILE_M = 16
F32_ROW_MAJOR_ASYNC_TILE_N = 16
F32_ROW_MAJOR_MATRIX_ELEMS = F32_ROW_MAJOR_M * F32_ROW_MAJOR_N
F32_ROW_MAJOR_SMEM_BYTES = F32_ROW_MAJOR_MATRIX_ELEMS * 4


@flyc.kernel
def _f32_row_major_strided_kernel(src: fx.Tensor, dst: fx.Tensor):
    _async_copy_case_body(
        src,
        dst,
        dtype=fx.Float32,
        async_copy_op=ixdl.MRAsyncCpNoSwizzle,
        scalar_copy_op=fx.UniversalCopy32b,
        matrix_m=F32_ROW_MAJOR_M,
        matrix_n=F32_ROW_MAJOR_N,
        src_stride_n=F32_ROW_MAJOR_SRC_STRIDE_N,
        async_tile_m=F32_ROW_MAJOR_ASYNC_TILE_M,
        async_tile_n=F32_ROW_MAJOR_ASYNC_TILE_N,
    )


@flyc.jit
def launch_f32_row_major_strided(src: fx.Tensor, dst: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    _f32_row_major_strided_kernel(src, dst).launch(
        grid=(BLOCKS, 1, 1),
        block=(THREADS, 1, 1),
        smem=F32_ROW_MAJOR_SMEM_BYTES,
        stream=stream,
    )


def run_f32_row_major_strided():
    src, dst, expected = _tile_constant_tensors(
        dtype=torch.float32,
        matrix_m=F32_ROW_MAJOR_M,
        matrix_n=F32_ROW_MAJOR_N,
        src_stride_n=F32_ROW_MAJOR_SRC_STRIDE_N,
        async_tile_m=F32_ROW_MAJOR_ASYNC_TILE_M,
        async_tile_n=F32_ROW_MAJOR_ASYNC_TILE_N,
    )
    launch_f32_row_major_strided(src, dst)
    torch.cuda.synchronize()
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)
    print("MR async copy f32 row-major strided OK")


# === i8 col-major swizzle -> ixdl.cp_async.16x64.b8.col ===

I8_COL_MAJOR_M = 32
I8_COL_MAJOR_N = 128
I8_COL_MAJOR_SRC_STRIDE_N = 160
I8_COL_MAJOR_ASYNC_TILE_M = 16
I8_COL_MAJOR_ASYNC_TILE_N = 64
I8_COL_MAJOR_MATRIX_ELEMS = I8_COL_MAJOR_M * I8_COL_MAJOR_N
I8_COL_MAJOR_SMEM_BYTES = I8_COL_MAJOR_MATRIX_ELEMS


@flyc.kernel
def _i8_col_major_strided_kernel(src: fx.Tensor, dst: fx.Tensor):
    _async_copy_case_body(
        src,
        dst,
        dtype=fx.Int8,
        async_copy_op=ixdl.MRAsyncCpCol,
        scalar_copy_op=fx.UniversalCopy8b,
        matrix_m=I8_COL_MAJOR_M,
        matrix_n=I8_COL_MAJOR_N,
        src_stride_n=I8_COL_MAJOR_SRC_STRIDE_N,
        async_tile_m=I8_COL_MAJOR_ASYNC_TILE_M,
        async_tile_n=I8_COL_MAJOR_ASYNC_TILE_N,
    )


@flyc.jit
def launch_i8_col_major_strided(src: fx.Tensor, dst: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    _i8_col_major_strided_kernel(src, dst).launch(
        grid=(BLOCKS, 1, 1),
        block=(THREADS, 1, 1),
        smem=I8_COL_MAJOR_SMEM_BYTES,
        stream=stream,
    )


def run_i8_col_major_strided():
    src, dst, expected = _block_constant_tensors(
        dtype=torch.int8,
        matrix_m=I8_COL_MAJOR_M,
        matrix_n=I8_COL_MAJOR_N,
        src_stride_n=I8_COL_MAJOR_SRC_STRIDE_N,
    )
    launch_i8_col_major_strided(src, dst)
    torch.cuda.synchronize()
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)
    print("MR async copy i8 col-major strided OK")


# === f16 col-major swizzle -> ixdl.cp_async.16x32.b16.col ===

F16_COL_MAJOR_M = 32
F16_COL_MAJOR_N = 64
F16_COL_MAJOR_SRC_STRIDE_N = 80
F16_COL_MAJOR_ASYNC_TILE_M = 16
F16_COL_MAJOR_ASYNC_TILE_N = 32
F16_COL_MAJOR_MATRIX_ELEMS = F16_COL_MAJOR_M * F16_COL_MAJOR_N
F16_COL_MAJOR_SMEM_BYTES = F16_COL_MAJOR_MATRIX_ELEMS * 2


@flyc.kernel
def _f16_col_major_strided_kernel(src: fx.Tensor, dst: fx.Tensor):
    _async_copy_case_body(
        src,
        dst,
        dtype=fx.Float16,
        async_copy_op=ixdl.MRAsyncCpCol,
        scalar_copy_op=fx.UniversalCopy16b,
        matrix_m=F16_COL_MAJOR_M,
        matrix_n=F16_COL_MAJOR_N,
        src_stride_n=F16_COL_MAJOR_SRC_STRIDE_N,
        async_tile_m=F16_COL_MAJOR_ASYNC_TILE_M,
        async_tile_n=F16_COL_MAJOR_ASYNC_TILE_N,
    )


@flyc.jit
def launch_f16_col_major_strided(src: fx.Tensor, dst: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    _f16_col_major_strided_kernel(src, dst).launch(
        grid=(BLOCKS, 1, 1),
        block=(THREADS, 1, 1),
        smem=F16_COL_MAJOR_SMEM_BYTES,
        stream=stream,
    )


def run_f16_col_major_strided():
    src, dst, expected = _block_constant_tensors(
        dtype=torch.float16,
        matrix_m=F16_COL_MAJOR_M,
        matrix_n=F16_COL_MAJOR_N,
        src_stride_n=F16_COL_MAJOR_SRC_STRIDE_N,
    )
    launch_f16_col_major_strided(src, dst)
    torch.cuda.synchronize()
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)
    print("MR async copy f16 col-major strided OK")


# === f32 col-major swizzle -> ixdl.cp_async.16x16.b32.col ===

F32_COL_MAJOR_M = 32
F32_COL_MAJOR_N = 32
F32_COL_MAJOR_SRC_STRIDE_N = 48
F32_COL_MAJOR_ASYNC_TILE_M = 16
F32_COL_MAJOR_ASYNC_TILE_N = 16
F32_COL_MAJOR_MATRIX_ELEMS = F32_COL_MAJOR_M * F32_COL_MAJOR_N
F32_COL_MAJOR_SMEM_BYTES = F32_COL_MAJOR_MATRIX_ELEMS * 4


@flyc.kernel
def _f32_col_major_strided_kernel(src: fx.Tensor, dst: fx.Tensor):
    _async_copy_case_body(
        src,
        dst,
        dtype=fx.Float32,
        async_copy_op=ixdl.MRAsyncCpCol,
        scalar_copy_op=fx.UniversalCopy32b,
        matrix_m=F32_COL_MAJOR_M,
        matrix_n=F32_COL_MAJOR_N,
        src_stride_n=F32_COL_MAJOR_SRC_STRIDE_N,
        async_tile_m=F32_COL_MAJOR_ASYNC_TILE_M,
        async_tile_n=F32_COL_MAJOR_ASYNC_TILE_N,
    )


@flyc.jit
def launch_f32_col_major_strided(src: fx.Tensor, dst: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    _f32_col_major_strided_kernel(src, dst).launch(
        grid=(BLOCKS, 1, 1),
        block=(THREADS, 1, 1),
        smem=F32_COL_MAJOR_SMEM_BYTES,
        stream=stream,
    )


def run_f32_col_major_strided():
    src, dst, expected = _block_constant_tensors(
        dtype=torch.float32,
        matrix_m=F32_COL_MAJOR_M,
        matrix_n=F32_COL_MAJOR_N,
        src_stride_n=F32_COL_MAJOR_SRC_STRIDE_N,
    )
    launch_f32_col_major_strided(src, dst)
    torch.cuda.synchronize()
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)
    print("MR async copy f32 col-major strided OK")


# === i8 row-major swizzle -> ixdl.cp_async.16x64.b8.row ===

I8_ROW_MAJOR_M = 32
I8_ROW_MAJOR_N = 128
I8_ROW_MAJOR_SRC_STRIDE_N = 160
I8_ROW_MAJOR_ASYNC_TILE_M = 16
I8_ROW_MAJOR_ASYNC_TILE_N = 64
I8_ROW_MAJOR_MATRIX_ELEMS = I8_ROW_MAJOR_M * I8_ROW_MAJOR_N
I8_ROW_MAJOR_SMEM_BYTES = I8_ROW_MAJOR_MATRIX_ELEMS


@flyc.kernel
def _i8_row_major_strided_kernel(src: fx.Tensor, dst: fx.Tensor):
    _async_copy_case_body(
        src,
        dst,
        dtype=fx.Int8,
        async_copy_op=ixdl.MRAsyncCpRow8b,
        scalar_copy_op=fx.UniversalCopy8b,
        matrix_m=I8_ROW_MAJOR_M,
        matrix_n=I8_ROW_MAJOR_N,
        src_stride_n=I8_ROW_MAJOR_SRC_STRIDE_N,
        async_tile_m=I8_ROW_MAJOR_ASYNC_TILE_M,
        async_tile_n=I8_ROW_MAJOR_ASYNC_TILE_N,
    )


@flyc.jit
def launch_i8_row_major_strided(src: fx.Tensor, dst: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    _i8_row_major_strided_kernel(src, dst).launch(
        grid=(BLOCKS, 1, 1),
        block=(THREADS, 1, 1),
        smem=I8_ROW_MAJOR_SMEM_BYTES,
        stream=stream,
    )


def run_i8_row_major_strided():
    src, dst, expected = _block_constant_tensors(
        dtype=torch.int8,
        matrix_m=I8_ROW_MAJOR_M,
        matrix_n=I8_ROW_MAJOR_N,
        src_stride_n=I8_ROW_MAJOR_SRC_STRIDE_N,
    )
    launch_i8_row_major_strided(src, dst)
    torch.cuda.synchronize()
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)
    print("MR async copy i8 row-major strided OK")


# === f16 row-major swizzle -> ixdl.cp_async.16x32.b16.row ===

F16_ROW_MAJOR_M = 32
F16_ROW_MAJOR_N = 64
F16_ROW_MAJOR_SRC_STRIDE_N = 80
F16_ROW_MAJOR_ASYNC_TILE_M = 16
F16_ROW_MAJOR_ASYNC_TILE_N = 32
F16_ROW_MAJOR_MATRIX_ELEMS = F16_ROW_MAJOR_M * F16_ROW_MAJOR_N
F16_ROW_MAJOR_SMEM_BYTES = F16_ROW_MAJOR_MATRIX_ELEMS * 2


@flyc.kernel
def _f16_row_major_strided_kernel(src: fx.Tensor, dst: fx.Tensor):
    _async_copy_case_body(
        src,
        dst,
        dtype=fx.Float16,
        async_copy_op=ixdl.MRAsyncCpRow16b,
        scalar_copy_op=fx.UniversalCopy16b,
        matrix_m=F16_ROW_MAJOR_M,
        matrix_n=F16_ROW_MAJOR_N,
        src_stride_n=F16_ROW_MAJOR_SRC_STRIDE_N,
        async_tile_m=F16_ROW_MAJOR_ASYNC_TILE_M,
        async_tile_n=F16_ROW_MAJOR_ASYNC_TILE_N,
    )


@flyc.jit
def launch_f16_row_major_strided(src: fx.Tensor, dst: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    _f16_row_major_strided_kernel(src, dst).launch(
        grid=(BLOCKS, 1, 1),
        block=(THREADS, 1, 1),
        smem=F16_ROW_MAJOR_SMEM_BYTES,
        stream=stream,
    )


def run_f16_row_major_strided():
    src, dst, expected = _block_constant_tensors(
        dtype=torch.float16,
        matrix_m=F16_ROW_MAJOR_M,
        matrix_n=F16_ROW_MAJOR_N,
        src_stride_n=F16_ROW_MAJOR_SRC_STRIDE_N,
    )
    launch_f16_row_major_strided(src, dst)
    torch.cuda.synchronize()
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)
    print("MR async copy f16 row-major strided OK")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-compatible Iluvatar device is not available")

    run_f32_row_major_strided()
    run_f32_col_major_strided()
    run_f16_row_major_strided()
    run_f16_col_major_strided()
    run_i8_row_major_strided()
    run_i8_col_major_strided()


if __name__ == "__main__":
    main()