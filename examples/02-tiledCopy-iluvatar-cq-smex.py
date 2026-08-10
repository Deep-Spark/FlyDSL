#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""CQ SmexMtx tiled-copy teaching example (G2S + loadn S2R).

Minimal CQ analogue of ``02-tiledCopy-iluvatar-mr.py``: one 16-row SMEX tile,
async G2S, matrix-load S2R, and a small-tile host check. Not a GEMM.

Pipeline (one 16-row SMEX tile = 16 x 64B = 1024B):

1. G2S: ``CQSmexCp(rows=16, layout="mtx")`` writes the SmexMtx shared format.
2. Sync: ``cp_async_commit_group`` / ``cp_async_wait_group(0)`` + CTA barrier
   (async-copy group sync shared by MR SME and CQ SMEX).
3. S2R: ``CQMtxLoadn(..., pattern="loadn16", direction="row")`` issues one
   warp-collective ``ixdl.mtx_loadn_b8_rowx2`` (64 bits / lane = 8 x i8), the
   base CQ MMA A fragment for 16x32 i8 (EmPart=0 / tile base pointer).
4. R2G: ``UniversalCopy64b`` dumps each lane's fragment for host comparison.

SmexMtx and LegacySme are incompatible shared-buffer contracts: do not mix
``CQSmexCp(layout="mtx")`` with MR byte-swizzle S2R (or the reverse). Full
tile-slot / EmPart addressing belongs with a later tiled GEMM example.

Usage::

    # CI / no GPU: compile-only smoke (default)
    ARCH=ivcore30 FLYDSL_COMPILE_BACKEND=iluvatar FLYDSL_RUNTIME_KIND=iluvatar \\
      COMPILE_ONLY=1 python examples/02-tiledCopy-iluvatar-cq-smex.py

    # CQ device: run and compare the 16x32 A fragment to the host reference
    ARCH=ivcore30 FLYDSL_COMPILE_BACKEND=iluvatar FLYDSL_RUNTIME_KIND=iluvatar \\
      COMPILE_ONLY=0 python examples/02-tiledCopy-iluvatar-cq-smex.py --check
"""

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# One 16x1B64 SMEX tile: 16 rows x 64B = 1024B.
SMEM_BYTES = 1024
ROWS = 16
# b8: 64 elements per 64B row (SMEX footprint).
COLS = 64
# Base CQ MMA A for i8 is 16x32; one loadn16 x2 covers that fragment.
ATOM_K = 32
# One mtx_loadn x2 returns 64 bits / lane -> 8 x i8.
FRAG_ELEMS = 8
WARP_SIZE = 64


def _configure_env(*, compile_only: bool) -> None:
    os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "iluvatar")
    os.environ.setdefault("FLYDSL_RUNTIME_KIND", "iluvatar")
    os.environ.setdefault("ARCH", "ivcore30")
    os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    if compile_only:
        os.environ.setdefault("COMPILE_ONLY", "1")
    else:
        os.environ.pop("COMPILE_ONLY", None)


def _compile_only() -> bool:
    return os.environ.get("COMPILE_ONLY", "").lower() in ("1", "true", "yes", "on")


def _expected_a_frag_i8_row(src_mk):
    """Host reference for CQ base 8b A / ``CQMtxLoadn`` row-gather fragment.

    TV layout (element granularity)::

        Shape(Thr(16, 4), Val(4, 2)) : Stride(Thr(16, 4), Val(1, 256))

    into column-major ``(M=16, K=32)`` with flat index ``m + k * M``.
    """
    import torch

    if tuple(src_mk.shape) != (ROWS, ATOM_K):
        raise ValueError(f"expected src shape {(ROWS, ATOM_K)}, got {tuple(src_mk.shape)}")
    out = torch.empty((WARP_SIZE, FRAG_ELEMS), dtype=torch.int8, device=src_mk.device)
    for tid in range(WARP_SIZE):
        t0 = tid % 16
        t1 = tid // 16
        for frag in range(FRAG_ELEMS):
            v0 = frag % 4
            v1 = frag // 4
            elem = t0 * 16 + t1 * 4 + v0 + v1 * 256
            m = elem % ROWS
            k = elem // ROWS
            out[tid, frag] = src_mk[m, k]
    return out


def _compile_case(*, row_mask=None, col_mask=None):
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    import flydsl.expr.ixdl as ixdl

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

        # Phase 3 (R2G): dump per-lane fragment as (lane, frag) for host check.
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


def _run_compile_smoke() -> None:
    import torch

    device = "cpu"
    cases = [
        ("full_tile", None, None),
        # First 8 rows, first 4 DWs (16B) of each 64B row.
        ("partial_8x4dw", (1 << 8) - 1, (1 << 4) - 1),
    ]
    for name, row_mask, col_mask in cases:
        src = torch.arange(ROWS * COLS, device=device, dtype=torch.int16).to(torch.int8).reshape(ROWS, COLS)
        dst = torch.empty((WARP_SIZE, FRAG_ELEMS), dtype=torch.int8, device=device)
        launch = _compile_case(row_mask=row_mask, col_mask=col_mask)
        launch(src, dst)
        print(f"COMPILED {name}: cq.smex_cp<{ROWS},mtx> + cq.mtx_loadn<loadn16,row,8>")


def _run_check() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("COMPILE_ONLY=0 / --check requires a CUDA-compatible CQ device")

    src = torch.arange(ROWS * COLS, device="cuda", dtype=torch.int16).to(torch.int8).reshape(ROWS, COLS)
    dst = torch.empty((WARP_SIZE, FRAG_ELEMS), dtype=torch.int8, device="cuda")
    launch = _compile_case(row_mask=None, col_mask=None)
    launch(src, dst)
    torch.cuda.synchronize()

    # EmPart=0 / tile base: base 16x32 i8 A fragment from the left K strip.
    expected = _expected_a_frag_i8_row(src[:, :ATOM_K])
    torch.testing.assert_close(dst, expected, rtol=0, atol=0)
    print(
        f"PASS check: cq.smex_cp<{ROWS},mtx> + cq.mtx_loadn<loadn16,row,8> "
        f"matches host Layout_16x32_8b_A on src[:, :{ATOM_K}]"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--check",
        action="store_true",
        help="run on a CQ device and compare the loadn A fragment to the host reference",
    )
    args = p.parse_args(argv)

    if args.check:
        _configure_env(compile_only=False)
        _run_check()
        return 0

    _configure_env(compile_only=True)
    if _compile_only():
        _run_compile_smoke()
        return 0

    # COMPILE_ONLY=0 without --check: still execute the correctness path.
    _run_check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
