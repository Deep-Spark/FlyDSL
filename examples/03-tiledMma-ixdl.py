# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tiled MMA matmul on Iluvatar MR-V100 / MR-V50 via the ``ixdl`` backend.

Structurally mirrors ``03-tiledMma.py`` (CDNA3 MFMA path) but swaps out two
ingredients:

* ``fx.rocdl.MFMA`` -> ``fx.ixdl.MMAD`` (ivcore11 MMAD).
* ``fx.rocdl.BufferCopy32b`` + ``make_buffer_tensor`` -> ``fx.UniversalCopy32b``.
  Iluvatar does not expose AMDGPU-style buffer descriptors, and the
  generic ``UniversalCopy`` path lowers cleanly through ixcc's
  ``convert-gpu-to-ixdl`` pipeline.

Supports the five CUTLASS-documented IX11 MMAD type triples. Select one
with ``--dtype {f32,f16,bf16,i8}`` or ``FLYDSL_IX11_DTYPE``; default ``f32``.

Run constraint: Iluvatar cards hang if two programs share a device. Use
``CUDA_VISIBLE_DEVICES`` to pin, and consult ``ixsmi`` if in doubt.
"""

# NOTE: do NOT add ``from __future__ import annotations`` here (same
# Constexpr-introspection reason as 01-vectorAdd-ixdl.py).

import argparse
import os
import shutil
import subprocess
import sys

os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "ixdl")
os.environ.setdefault("FLYDSL_RUNTIME_KIND", "ixdl")

import torch  # noqa: E402

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402


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


# ---- Per-dtype MMAD / kernel config -----------------------------------------
# ivcore11 always runs 64 threads per MMAD atom. We arrange 2x2x1 atoms per
# block = 256 threads regardless of element type.

DTYPE_TABLE = {
    "f32":  dict(torch=torch.float32, fly=lambda: fx.Float32,  k=16,
                 atol=1e-4, rtol=1e-4, signed=True),
    "f16":  dict(torch=torch.float16, fly=lambda: fx.Float16,  k=16,
                 atol=2e-2, rtol=2e-2, signed=True),
    "bf16": dict(torch=torch.bfloat16, fly=lambda: fx.BFloat16, k=16,
                 atol=2e-2, rtol=2e-2, signed=True),
    "i8":   dict(torch=torch.int8,    fly=lambda: fx.Int8,     k=32,
                 atol=0, rtol=0, signed=True),
}


def _build_kernel(dtype_name):
    info = DTYPE_TABLE[dtype_name]
    fly_elem = info["fly"]()
    k_mma = info["k"]
    block_m = 32
    block_n = 32
    block_k = k_mma

    # 16-bit element types can still be vector-copied at 32-bit width;
    # 8-bit too. Pick the widest legal UniversalCopy that divides the
    # per-tile value count.
    copy_prim = fx.UniversalCopy32b

    @flyc.kernel
    def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x

        bA = fx.zipped_divide(A, (block_m, block_k))
        bB = fx.zipped_divide(B, (block_n, block_k))
        bC = fx.zipped_divide(C, (block_m, block_n))

        bA = fx.slice(bA, (None, bid))
        bB = fx.slice(bB, (None, bid))
        bC = fx.slice(bC, (None, bid))

        mma_atom = fx.make_mma_atom(fx.ixdl.MMAD(16, 16, k_mma, fly_elem))
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))
        thr_mma = tiled_mma.thr_slice(tid)

        copy_atom_ab = fx.make_copy_atom(copy_prim(), fly_elem)
        # C/D are always f32 (or i32 for i8 input) — use accumulator atom.
        copy_atom_c = fx.make_copy_atom(copy_prim(), frag_dtype_for(dtype_name))

        tiled_copy_A = fx.make_tiled_copy_A(copy_atom_ab, tiled_mma)
        tiled_copy_B = fx.make_tiled_copy_B(copy_atom_ab, tiled_mma)
        tiled_copy_C = fx.make_tiled_copy_C(copy_atom_c, tiled_mma)

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

        fx.copy(copy_atom_ab, copy_src_A, copy_frag_A, pred=None)
        fx.copy(copy_atom_ab, copy_src_B, copy_frag_B, pred=None)

        frag_C.fill(0)
        fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)

        fx.copy(copy_atom_c, copy_frag_C, copy_dst_C, pred=None)

    return gemm_kernel, block_m, block_n, block_k


def frag_dtype_for(dtype_name):
    # ivcore11 accumulates 16-bit/32-bit float in f32, and 8-bit int in i32.
    if dtype_name in ("f32", "f16", "bf16"):
        return fx.Float32
    return fx.Int32


def _launch_jit(dtype_name):
    gemm_kernel, bm, bn, bk = _build_kernel(dtype_name)

    @flyc.jit
    def tiledMma(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
                 stream: fx.Stream = fx.Stream(None)):
        gemm_kernel(A, B, C).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)

    return tiledMma, bm, bn, bk


def run_eager(dtype_name) -> bool:
    info = DTYPE_TABLE[dtype_name]
    tiledMma, M, N, K = _launch_jit(dtype_name)

    torch.manual_seed(0)
    debug_simple = os.environ.get("FLYDSL_IX11_SIMPLE", "0") == "1"
    if dtype_name == "i8":
        if debug_simple:
            A = torch.ones((M, K), dtype=torch.int8).cuda()
            B = torch.ones((N, K), dtype=torch.int8).cuda()
        else:
            A = torch.randint(-8, 8, (M, K), dtype=torch.int8).cuda()
            B = torch.randint(-8, 8, (N, K), dtype=torch.int8).cuda()
        C = torch.zeros(M, N, dtype=torch.int32).cuda()
    else:
        if debug_simple:
            A = torch.ones((M, K), dtype=info["torch"]).cuda()
            B = torch.ones((N, K), dtype=info["torch"]).cuda()
        else:
            A = torch.randn(M, K, dtype=info["torch"]).cuda()
            B = torch.randn(N, K, dtype=info["torch"]).cuda()
        C = torch.zeros(M, N, dtype=torch.float32).cuda()

    tiledMma(A, B, C, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    if dtype_name == "i8":
        # PyTorch CUDA doesn't implement int32 matmul; compute the reference on CPU.
        expected = (A.cpu().to(torch.int32) @ B.cpu().to(torch.int32).T).cuda()
    else:
        # Accumulate reference in f32 regardless of input dtype.
        expected = A.to(torch.float32) @ B.to(torch.float32).T

    ok = torch.allclose(C, expected, atol=info["atol"], rtol=info["rtol"])
    print(f"[ixdl] tiledMma[{dtype_name}] correct: {ok}")
    if not ok:
        diff = (C.to(torch.float32) - expected.to(torch.float32)).abs()
        print(f"  max abs diff: {diff.max().item():.3e}")
        print(f"  mean abs diff: {diff.mean().item():.3e}")
        print(f"  C     any nan: {C.isnan().any().item() if C.is_floating_point() else False}")
        print(f"  C[0,0:4]:      {C.to(torch.float32)[0,0:4].tolist()}")
        print(f"  expect[0,0:4]: {expected.to(torch.float32)[0,0:4].tolist()}")
    return ok


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dtype",
        default=os.environ.get("FLYDSL_IX11_DTYPE", "f32"),
        choices=sorted(DTYPE_TABLE.keys()),
    )
    return parser.parse_args()


if __name__ == "__main__":
    _warn_if_card_busy()
    args = _parse_args()
    sys.exit(0 if run_eager(args.dtype) else 1)
