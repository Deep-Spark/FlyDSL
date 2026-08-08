#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Compile / run smoke for CQ SmexMtx G2S + matrix-load S2R.

Pipeline (one 16-row SMEX tile = 16 x 64B = 1024B):

1. G2S: ``CQSmexCp(rows=16, layout="mtx")`` writes the SmexMtx shared format.
2. S2R: ``CQMtxLoadn(..., pattern="loadn16", direction="row")`` issues one
   warp-collective ``ixdl.mtx_loadn_b8_rowx2`` (64 bits / lane = 8 x i8).
3. R2G: ``UniversalCopy64b`` dumps each lane's fragment so the S2R is not DCE'd.

SmexMtx and LegacySme are incompatible shared-buffer contracts: do not mix
``CQSmexCp(layout="mtx")`` with MR byte-swizzle S2R (or the reverse). This
example uses EmPart=0 (shared base pointer); full tile-slot / EmPart addressing
belongs with a later tiled GEMM / teaching example.

Set ``COMPILE_ONLY=0`` on a CQ device to also launch. Fragment dump is not a
full-tile correctness check.
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
# One mtx_loadn x2 returns 64 bits / lane -> 8 x i8.
FRAG_ELEMS = 8


def _compile_case(*, row_mask=None, col_mask=None):
    @flyc.kernel(known_block_size=[WARP_SIZE, 1, 1])
    def copy_kernel(src: fx.Tensor, dst: fx.Tensor):
        lane_id = fx.Int32(fx.lane_id)
        load_layout = fx.make_layout((ROWS, COLS), (1, ROWS))

        sme_src = ixdl.make_sme_gmem_tensor(src, leading_stride=COLS)
        smem = fx.make_view(fx.get_dyn_shared(fx.Int8), fx.make_layout(ROWS * COLS, 1))

        # Phase 1 (G2S): SmexMtx write. Must pair with CQMtxLoadn / loadn16.
        async_atom = fx.make_copy_atom(ixdl.CQSmexCp(rows=ROWS, layout="mtx"), fx.Int8)
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

        # Phase 2 (S2R): warp-collective matrix load. EmPart=0 on the tile base.
        s2r_atom = fx.make_copy_atom(
            ixdl.CQMtxLoadn(fx.Int8, pattern="loadn16", direction="row"),
            fx.Int8,
        )
        smem_mtx = fx.make_view(fx.get_iter(smem), fx.make_layout(FRAG_ELEMS, 1))
        frag = fx.make_rmem_tensor(fx.make_layout(FRAG_ELEMS, 1), fx.Int8)
        fx.copy(s2r_atom, smem_mtx, frag)

        # Phase 3 (R2G): dump the per-lane fragment (anti-DCE / device smoke).
        dump_atom = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Int8)
        dst_frag = fx.make_view(
            fx.add_offset(fx.get_iter(dst), lane_id * FRAG_ELEMS),
            fx.make_layout(FRAG_ELEMS, 1),
        )
        fx.copy(dump_atom, frag, dst_frag)

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
        print(
            f"{'COMPILED' if compile_only else 'RAN'} {name}: "
            f"cq.smex_cp<{ROWS},mtx> + cq.mtx_loadn<loadn16,row,8>"
        )


if __name__ == "__main__":
    main()
