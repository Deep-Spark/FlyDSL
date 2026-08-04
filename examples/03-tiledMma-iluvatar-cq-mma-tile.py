#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""CQ MMA tile / FeatureLongMtx correctness harness.

Single-warp CQMma with constant A/B fragments (no G2S/S2R pipeline). Use this to
validate base vs long-mtx tiles on ivcore30 before full CQ HGEMM/IGEMM. For the
minimal G2S + loadn S2R + MMA + store smoke path, see
``examples/03-tiledMma-iluvatar-cq-pipeline.py``.

Examples::

    ARCH=ivcore30 FLYDSL_COMPILE_BACKEND=iluvatar \\
      python examples/03-tiledMma-iluvatar-cq-mma-tile.py --check

    ARCH=ivcore30 FLYDSL_COMPILE_BACKEND=iluvatar \\
      python examples/03-tiledMma-iluvatar-cq-mma-tile.py --check --mma-tile 32x32

    ARCH=ivcore30 FLYDSL_COMPILE_BACKEND=iluvatar \\
      python examples/03-tiledMma-iluvatar-cq-mma-tile.py --check --dtype s8 --mma-tile 16x64
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _configure_env() -> None:
    os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "iluvatar")
    os.environ.setdefault("FLYDSL_RUNTIME_KIND", "iluvatar")
    os.environ.setdefault("ARCH", "ivcore30")
    os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    os.environ.pop("COMPILE_ONLY", None)


def _run_check(*, dtype: str, mma_tile: str) -> None:
    import torch

    import flydsl.expr as fx
    from kernels.gemm.iluvatar.cq.igemm import compile_iluvatar_cq_igemm
    from kernels.gemm.iluvatar.cq.mma_frag import compile_iluvatar_cq_hgemm_mma_frag

    if not torch.cuda.is_available():
        raise SystemExit("CUDA-compatible Iluvatar device is not available")

    if dtype == "s8":
        launch = compile_iluvatar_cq_igemm(mma_tile=mma_tile, a_value=1, b_value=2)
        tile = launch.cq_mma_tile
        A = torch.empty((tile.atom_m, tile.atom_k), device="cuda", dtype=torch.int8)
        B = torch.empty((tile.atom_n, tile.atom_k), device="cuda", dtype=torch.int8)
        C = torch.empty((tile.atom_m, tile.atom_n), device="cuda", dtype=torch.int32)
        launch(A, B, C)
        torch.cuda.synchronize()
        expected = torch.full(
            (tile.atom_m, tile.atom_n),
            tile.atom_k * 1 * 2,
            device="cuda",
            dtype=torch.int32,
        )
        torch.testing.assert_close(C, expected, rtol=0, atol=0)
    else:
        elem = fx.BFloat16 if dtype == "bf16" else fx.Float16
        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
        launch = compile_iluvatar_cq_hgemm_mma_frag(
            elem_dtype=elem, mma_tile=mma_tile, a_value=1.0, b_value=2.0
        )
        tile = launch.cq_mma_tile
        A = torch.empty((tile.atom_m, tile.atom_k), device="cuda", dtype=torch_dtype)
        B = torch.empty((tile.atom_n, tile.atom_k), device="cuda", dtype=torch_dtype)
        C = torch.empty((tile.atom_m, tile.atom_n), device="cuda", dtype=torch.float32)
        launch(A, B, C)
        torch.cuda.synchronize()
        expected = torch.full(
            (tile.atom_m, tile.atom_n),
            float(tile.atom_k) * 1.0 * 2.0,
            device="cuda",
            dtype=torch.float32,
        )
        torch.testing.assert_close(C, expected, rtol=2e-2, atol=2e-2)

    print(
        f"PASS dtype={dtype} mma_tile={mma_tile} "
        f"atom={tile.atom_m}x{tile.atom_n}x{tile.atom_k}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="run MMA fragment correctness check")
    p.add_argument(
        "--mma-tile",
        default="base",
        help='CQ MMA tile: base|16x16|32x32|16x64|64x16 (default: base)',
    )
    p.add_argument(
        "--dtype",
        default="f16",
        choices=("f16", "bf16", "s8"),
        help="multiplicand dtype (default: f16)",
    )
    args = p.parse_args(argv)

    if not args.check:
        p.print_help()
        return 0

    _configure_env()
    _run_check(dtype=args.dtype, mma_tile=args.mma_tile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
