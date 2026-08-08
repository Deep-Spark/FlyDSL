#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Compile-only smoke for CQ SMEX/mtx G2S with default and partial masks.

Set ``COMPILE_ONLY=0`` on a CQ device to also issue the copy (no SLB readback
correctness check yet -- that needs a matching S2R / scalar path).
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

# One 16x1B64 SMEX tile: 16 rows x 64B = 1024B.
SMEM_BYTES = 1024
ROWS = 16
# b8: 64 elements per 64B row.
COLS = 64


def _compile_case(*, row_mask=None, col_mask=None):
    @flyc.kernel(known_block_size=[WARP_SIZE, 1, 1])
    def copy_kernel(src: fx.Tensor, dst: fx.Tensor):
        lane_id = fx.Int32(fx.lane_id)
        load_layout = fx.make_layout((ROWS, COLS), (1, ROWS))

        sme_src = ixdl.make_sme_gmem_tensor(src, leading_stride=COLS)
        smem = fx.make_view(fx.get_dyn_shared(fx.Int8), fx.make_layout(ROWS * COLS, 1))

        async_atom = fx.make_copy_atom(ixdl.CQSmexCp(rows=ROWS), fx.Int8)
        if row_mask is not None or col_mask is not None:
            kwargs = {}
            if row_mask is not None:
                kwargs["row_mask"] = fx.Int64(row_mask)
            if col_mask is not None:
                kwargs["col_mask"] = fx.Int32(col_mask)
            async_atom = async_atom.set_value(kwargs)

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

        # Scalar dump so the kernel is not DCE'd under COMPILE_ONLY.
        scalar_atom = fx.make_copy_atom(fx.UniversalCopy8b(), fx.Int8)
        threads_n = WARP_SIZE
        val_n = COLS // threads_n
        smem_phys = fx.make_view(
            fx.get_iter(smem),
            fx.make_layout((ROWS, COLS), (COLS, 1)),
        )
        dst_tile = fx.make_view(
            fx.get_iter(dst),
            fx.make_layout((ROWS, COLS), (COLS, 1)),
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


def main():
    compile_only = _compile_only()
    if not compile_only and not torch.cuda.is_available():
        raise RuntimeError("COMPILE_ONLY=0 requires a CUDA-compatible CQ device")
    device = "cpu" if compile_only else "cuda"

    cases = [
        ("full_tile", None, None),
        # First 8 rows, first 4 DWs (16B) of each 64B row.
        ("partial_8x4dw", (1 << 8) - 1, (1 << 4) - 1),
    ]

    for name, row_mask, col_mask in cases:
        src = torch.arange(ROWS * COLS, device=device, dtype=torch.int16).to(torch.int8).reshape(
            ROWS, COLS
        )
        dst = torch.empty((ROWS, COLS), dtype=torch.int8, device=device)
        launch = _compile_case(row_mask=row_mask, col_mask=col_mask)
        launch(src, dst)
        print(f"{'COMPILED' if compile_only else 'RAN'} {name}: cq.smex_cp<{ROWS}>")


if __name__ == "__main__":
    main()
