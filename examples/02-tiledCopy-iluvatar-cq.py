#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Compile or check every CQ enhanced-SME asynchronous global-to-shared copy.

This is a static ``ivcore30`` compilation example: it does not require a CQ
device by default. Set ``COMPILE_ONLY=0`` on a CQ system to run G2S, read the
physical SLB footprint back with scalar copies, and compare against the source.
CQ matrix-load ``loadn`` coverage belongs to a separate example.
"""

import os

os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "iluvatar")
os.environ.setdefault("FLYDSL_RUNTIME_KIND", "iluvatar")
os.environ["ARCH"] = "ivcore30"
os.environ.setdefault("COMPILE_ONLY", "1")
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch  # noqa: E402

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
import flydsl.expr.ixdl as ixdl  # noqa: E402
from kernels.gemm.iluvatar.common import WARP_SIZE  # noqa: E402

SMEM_BYTES = 4096


def _compile_case(*, fx_dtype, scalar_atom_factory, row, col, transpose, logical_shape, swizzle):
    logical_m, logical_n = logical_shape
    tile_elems = logical_m * logical_n
    source_stride = col if transpose else logical_n

    @flyc.kernel(known_block_size=[WARP_SIZE, 1, 1])
    def copy_kernel(src: fx.Tensor, dst: fx.Tensor):
        lane_id = fx.Int32(fx.lane_id)
        load_layout = fx.make_layout(logical_shape, (1, logical_m))

        sme_src = ixdl.make_sme_gmem_tensor(src, leading_stride=source_stride)
        smem = fx.make_view(fx.get_dyn_shared(fx_dtype), fx.make_layout(tile_elems, 1))

        async_atom = fx.make_copy_atom(ixdl.CQAsyncCp(row, col, transpose), fx_dtype)
        tiled_ld = fx.make_tiled_copy_tv(
            async_atom,
            fx.make_layout((1, 1), (1, 1)),
            load_layout,
        )

        src_tile = fx.make_view(fx.get_iter(sme_src), load_layout)
        smem_tile = fx.make_view(fx.get_iter(smem), load_layout)
        ld = tiled_ld.get_slice(lane_id)
        fx.copy(async_atom, ld.partition_S(src_tile), ld.partition_D(smem_tile))

        ixdl.cp_async_commit_group()
        ixdl.cp_async_wait_group(0)
        fx.gpu.barrier()

        scalar_atom = fx.make_copy_atom(scalar_atom_factory(), fx_dtype)
        if fx.const_expr(row == 1):
            threads_n = WARP_SIZE
            val_n = logical_n // threads_n
            smem_phys = fx.make_view(
                fx.get_iter(smem),
                fx.make_layout(logical_shape, (logical_n, 1)),
            )
            dst_tile = fx.make_view(
                fx.get_iter(dst),
                fx.make_layout(logical_shape, (logical_n, 1)),
            )
            tiled_st = fx.make_tiled_copy_tv(
                scalar_atom,
                fx.make_layout((1, threads_n), (1, 1)),
                fx.make_layout((1, val_n), (1, 1)),
            )
            st = tiled_st.get_slice(lane_id)
            frag = fx.make_fragment_like(st.partition_S(smem_phys))
            fx.copy(scalar_atom, st.partition_S(smem_phys), frag)
            fx.copy(scalar_atom, frag, st.partition_D(dst_tile))
        else:
            slab_m = 16
            slab_n = col
            slab_elems = slab_m * slab_n
            threads_n = WARP_SIZE // slab_m
            val_n = slab_n // threads_n
            smem_phys_layout = ixdl.make_sme_shared_layout(
                swizzle,
                fx_dtype,
                major=ixdl.SMEMajor.K,
            )
            tiled_st = fx.make_tiled_copy_tv(
                scalar_atom,
                fx.make_layout((slab_m, threads_n), (1, slab_m)),
                fx.make_layout((1, val_n), (1, 1)),
            )
            st = tiled_st.get_slice(lane_id)

            # A 64-row CQ copy writes four consecutive 1024-byte legacy-SME slabs.
            for slab in fx.range_constexpr(4):
                smem_off = fx.Int32(slab * slab_elems)
                smem_tile = fx.make_view(
                    fx.add_offset(fx.get_iter(smem), smem_off),
                    smem_phys_layout,
                )
                if fx.const_expr(transpose):
                    dst_off = fx.Int32(slab * slab_n)
                else:
                    dst_off = fx.Int32(slab * slab_elems)
                dst_tile = fx.make_view(
                    fx.add_offset(fx.get_iter(dst), dst_off),
                    fx.make_layout((slab_m, slab_n), (logical_n, 1)),
                )
                frag = fx.make_fragment_like(st.partition_S(smem_tile))
                fx.copy(scalar_atom, st.partition_S(smem_tile), frag)
                fx.copy(scalar_atom, frag, st.partition_D(dst_tile))

    @flyc.jit
    def launch(src: fx.Tensor, dst: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        copy_kernel(src, dst).launch(
            grid=(1, 1, 1),
            block=(WARP_SIZE, 1, 1),
            smem=SMEM_BYTES,
            stream=stream,
        )

    return launch


def _compile_only():
    return os.environ.get("COMPILE_ONLY", "").lower() in ("1", "true", "yes", "on")


def _position_values(*, shape, torch_dtype, device):
    count = shape[0] * shape[1]
    values = torch.arange(count, device=device, dtype=torch.int32)
    if torch_dtype == torch.int8:
        values = (values * 73 + 19) % 255 - 127
    return values.to(torch_dtype).reshape(shape)


def main():
    cases = [
        (
            "b8_64x64_row",
            "ixdl.cp_async.64x64.b8.row",
            fx.Int8,
            torch.int8,
            fx.UniversalCopy8b,
            64,
            64,
            False,
            (64, 64),
            (64, 64),
            ixdl.SMESwizzle.Row8b,
        ),
        (
            "b16_64x32_row",
            "ixdl.cp_async.64x32.b16.row",
            fx.Float16,
            torch.float16,
            fx.UniversalCopy16b,
            64,
            32,
            False,
            (64, 32),
            (64, 32),
            ixdl.SMESwizzle.Row16b,
        ),
        (
            "b32_1x64b64",
            "ixdl.cp_async.1x64b64",
            fx.Float32,
            torch.float32,
            fx.UniversalCopy32b,
            1,
            1024,
            False,
            (1, 1024),
            (1, 1024),
            None,
        ),
        (
            "b32_64x16_row",
            "ixdl.cp_async.64x16.b32.row",
            fx.Float32,
            torch.float32,
            fx.UniversalCopy32b,
            64,
            16,
            False,
            (64, 16),
            (64, 16),
            ixdl.SMESwizzle.NoSwizzle,
        ),
        (
            "b32_64x16_col",
            "ixdl.cp_async.64x16.b32.col",
            fx.Float32,
            torch.float32,
            fx.UniversalCopy32b,
            64,
            16,
            True,
            (16, 64),
            (64, 16),
            ixdl.SMESwizzle.Col,
        ),
    ]

    compile_only = _compile_only()
    if not compile_only and not torch.cuda.is_available():
        raise RuntimeError("COMPILE_ONLY=0 requires a CUDA-compatible CQ device")
    device = "cpu" if compile_only else "cuda"

    for (
        name,
        expected_op,
        fx_dtype,
        torch_dtype,
        scalar_atom,
        row,
        col,
        transpose,
        logical_shape,
        source_shape,
        swizzle,
    ) in cases:
        src = _position_values(shape=source_shape, torch_dtype=torch_dtype, device=device)
        dst = torch.empty(logical_shape, dtype=torch_dtype, device=device)
        launch = _compile_case(
            fx_dtype=fx_dtype,
            scalar_atom_factory=scalar_atom,
            row=row,
            col=col,
            transpose=transpose,
            logical_shape=logical_shape,
            swizzle=swizzle,
        )
        launch(src, dst)

        if compile_only:
            print(f"COMPILED {name}: {expected_op}")
            continue

        torch.cuda.synchronize()
        expected = src.T.contiguous() if transpose else src
        torch.testing.assert_close(dst, expected, rtol=0, atol=0, msg=f"{name} copy mismatch")
        print(f"PASS {name}: {expected_op}")


if __name__ == "__main__":
    main()
