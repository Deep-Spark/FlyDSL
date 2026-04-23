# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tiled MMA matmul on Iluvatar MR-V100 / MR-V50 via the ``ixdl`` backend.

Structurally mirrors ``03-tiledMma.py`` (CDNA3 MFMA path) but swaps out two
ingredients:

* ``fx.rocdl.MFMA`` -> ``fx.ixdl.MMAD`` (ivcore11 MMAD, 16x16x16 f32).
* ``fx.rocdl.BufferCopy32b`` + ``make_buffer_tensor`` -> ``fx.UniversalCopy32b``.
  Iluvatar does not expose AMDGPU-style buffer descriptors, and the
  generic ``UniversalCopy`` path lowers cleanly through ixcc's
  ``convert-gpu-to-ixdl`` pipeline.

Run constraint: Iluvatar cards hang if two programs share a device. Use
``CUDA_VISIBLE_DEVICES`` to pin, and consult ``ixsmi`` if in doubt.
"""

# NOTE: do NOT add ``from __future__ import annotations`` here (same
# Constexpr-introspection reason as 01-vectorAdd-ixdl.py).

import os
import shutil
import subprocess
import sys

os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "ixdl")
os.environ.setdefault("FLYDSL_RUNTIME_KIND", "ixdl")

import torch  # noqa: E402

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402


block_m = 32
block_n = 32
block_k = 16


def _warn_if_card_busy() -> None:
    if shutil.which("ixsmi") is None:
        return
    try:
        out = subprocess.check_output(
            ["ixsmi"], text=True, timeout=5, stderr=subprocess.DEVNULL
        )
    except Exception:
        return
    for line in out.splitlines():
        ls = line.strip()
        if ls.startswith("|") and ("MiB" in ls) and any(c.isdigit() for c in ls):
            if "python" in ls or ("MiB /" in ls and "0MiB" not in ls.split("/")[0]):
                print(
                    "[WARN] ixsmi suggests this GPU may already be busy; "
                    "running two programs on one Iluvatar card will hang. "
                    "Consider setting CUDA_VISIBLE_DEVICES.",
                    file=sys.stderr,
                )
                return


@flyc.kernel
def gemm_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    bA = fx.zipped_divide(A, (block_m, block_k))
    bB = fx.zipped_divide(B, (block_n, block_k))
    bC = fx.zipped_divide(C, (block_m, block_n))

    bA = fx.slice(bA, (None, bid))
    bB = fx.slice(bB, (None, bid))
    bC = fx.slice(bC, (None, bid))

    mma_atom = fx.make_mma_atom(fx.ixdl.MMAD(16, 16, 16, fx.Float32))
    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))
    thr_mma = tiled_mma.thr_slice(tid)

    copy_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    tiled_copy_A = fx.make_tiled_copy_A(copy_atom, tiled_mma)
    tiled_copy_B = fx.make_tiled_copy_B(copy_atom, tiled_mma)
    tiled_copy_C = fx.make_tiled_copy_C(copy_atom, tiled_mma)

    thr_copy_A = tiled_copy_A.get_slice(tid)
    thr_copy_B = tiled_copy_B.get_slice(tid)
    thr_copy_C = tiled_copy_C.get_slice(tid)

    copy_src_A = thr_copy_A.partition_S(bA)
    copy_src_B = thr_copy_B.partition_S(bB)
    copy_dst_C = thr_copy_C.partition_S(bC)

    frag_A = thr_mma.make_fragment_A(bA)
    frag_B = thr_mma.make_fragment_B(bB)
    frag_C = thr_mma.make_fragment_C(bC)

    copy_frag_A = thr_copy_A.retile(frag_A)
    copy_frag_B = thr_copy_B.retile(frag_B)
    copy_frag_C = thr_copy_C.retile(frag_C)

    fx.copy(copy_atom, copy_src_A, copy_frag_A, pred=None)
    fx.copy(copy_atom, copy_src_B, copy_frag_B, pred=None)

    frag_C.fill(0)
    fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)

    fx.copy(copy_atom, copy_frag_C, copy_dst_C, pred=None)


@flyc.jit
def tiledMma(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    # 2x2x1 atom layout over 16x16x16 atoms => 32x32x16 block, 256 threads.
    gemm_kernel(A, B, C).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)


def run_eager() -> bool:
    M, N, K = block_m, block_n, block_k
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.float32).cuda()
    B = torch.randn(N, K, dtype=torch.float32).cuda()
    C = torch.zeros(M, N, dtype=torch.float32).cuda()

    tiledMma(A, B, C, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    expected = A @ B.T
    ok = torch.allclose(C, expected, atol=1e-4, rtol=1e-4)
    print(f"[ixdl] tiledMma correct: {ok}")
    if not ok:
        diff = (C - expected).abs()
        print(f"  max abs diff: {diff.max().item():.3e}")
        print(f"  mean abs diff: {diff.mean().item():.3e}")
    return ok


if __name__ == "__main__":
    _warn_if_card_busy()
    sys.exit(0 if run_eager() else 1)
