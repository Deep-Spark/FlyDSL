#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from pathlib import Path

os.environ.setdefault("BACKEND", "ixdl")

import torch

_repo_root = Path(__file__).resolve().parents[1]
_build_pkg_dir = _repo_root / "build-fly" / "python_packages"
if _build_pkg_dir.exists():
    sys.path.insert(0, str(_build_pkg_dir))

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr

M = 256
N = 256
K = 256
ATOM_M = 16
ATOM_N = 16
ATOM_K = 16

# Use a single 16x16x16 MMA atom per CTA.
CTA_M = ATOM_M
CTA_N = ATOM_N
CTA_K = ATOM_K
CTA_TILES_M = M // CTA_M
CTA_TILES_N = N // CTA_N
CTA_TILES_K = K // CTA_K


@flyc.kernel
def gemm256_ixdl_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    D: fx.Tensor,
):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x
    bm = bid // CTA_TILES_N
    bn = bid % CTA_TILES_N

    tile_a = fx.make_tile(CTA_M, CTA_K)
    tile_b = fx.make_tile(CTA_N, CTA_K)
    tile_c = fx.make_tile(CTA_M, CTA_N)

    A_tiles = fx.zipped_divide(A, tile_a)
    B_tiles = fx.zipped_divide(B, tile_b)
    C_tiles = fx.zipped_divide(C, tile_c)
    D_tiles = fx.zipped_divide(D, tile_c)

    A_tiles = fx.slice(A_tiles, (None, (bm, None)))
    B_tiles = fx.slice(B_tiles, (None, (bn, None)))
    C_tile = fx.slice(C_tiles, (None, (bm, bn)))
    D_tile = fx.slice(D_tiles, (None, (bm, bn)))

    mma_atom = fx.make_mma_atom(fx.ixdl.MMAD(ATOM_M, ATOM_N, ATOM_K, fx.Float16, elem_type_acc=fx.Float32))
    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (1, 1, 1)))
    thr_mma = tiled_mma.thr_slice(tid)

    copy_atom_A = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float16)
    copy_atom_B = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float16)
    copy_atom_C = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)

    tiled_copy_A = fx.make_tiled_copy_A(copy_atom_A, tiled_mma)
    tiled_copy_B = fx.make_tiled_copy_B(copy_atom_B, tiled_mma)
    tiled_copy_C = fx.make_tiled_copy_C(copy_atom_C, tiled_mma)

    thr_copy_A = tiled_copy_A.get_slice(tid)
    thr_copy_B = tiled_copy_B.get_slice(tid)
    thr_copy_C = tiled_copy_C.get_slice(tid)

    # part_C = thr_mma.partition_C(C_tile)
    # frag_acc = thr_mma.make_fragment_C(part_C)
    frag_acc = thr_mma.make_fragment_C(C_tile)
    copy_src_C = thr_copy_C.partition_S(C_tile)
    copy_dst_D = thr_copy_C.partition_D(D_tile)
    copy_frag_acc = thr_copy_C.retile(frag_acc)
    fx.copy(copy_atom_C, copy_src_C, copy_frag_acc)

    for bk in range_constexpr(CTA_TILES_K):
        A_tile = fx.slice(A_tiles, (None, bk))
        B_tile = fx.slice(B_tiles, (None, bk))

        # part_A = thr_mma.partition_A(A_tile)
        # part_B = thr_mma.partition_B(B_tile)
        # frag_A = thr_mma.make_fragment_A(part_A)
        # frag_B = thr_mma.make_fragment_B(part_B)
        frag_A = thr_mma.make_fragment_A(A_tile)
        frag_B = thr_mma.make_fragment_B(B_tile)

        copy_src_A = thr_copy_A.partition_S(A_tile)
        copy_src_B = thr_copy_B.partition_S(B_tile)
        copy_frag_A = thr_copy_A.retile(frag_A)
        copy_frag_B = thr_copy_B.retile(frag_B)

        fx.copy(copy_atom_A, copy_src_A, copy_frag_A)
        fx.copy(copy_atom_B, copy_src_B, copy_frag_B)
        fx.gemm(mma_atom, frag_acc, frag_A, frag_B, frag_acc)

    fx.copy(copy_atom_C, copy_frag_acc, copy_dst_D)


@flyc.jit
def gemm256_ixdl(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    D: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    gemm256_ixdl_kernel(A, B, C, D).launch(
        grid=(CTA_TILES_M * CTA_TILES_N, 1, 1),
        block=(64, 1, 1),
        stream=stream,
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for this ixdl GEMM demo")


    torch.manual_seed(0)
    a = torch.randint(-3, 4, (M, K), dtype=torch.int32, device="cuda").to(torch.float16)
    b = torch.randint(-3, 4, (N, K), dtype=torch.int32, device="cuda").to(torch.float16)
    c = torch.randint(-2, 3, (M, N), dtype=torch.int32, device="cuda").to(torch.float32)
    d = torch.zeros((M, N), dtype=torch.float32, device="cuda")

    gemm256_ixdl(a, b, c, d, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    ref = torch.einsum("mk,nk->mn", a.float(), b.float()) + c
    diff = (d - ref).abs()
    ok = bool(torch.allclose(d, ref, atol=1e-3, rtol=1e-3))

    print("[gemm256] CuTeDSL contract: A(M,K), B(N,K), D(M,N)")
    print("[gemm256] allclose =", ok)
    print("[gemm256] max_diff =", float(diff.max()))
    print("[gemm256] D[0, :8] =", d[0, :8].cpu())
    print("[gemm256] ref[0, :8] =", ref[0, :8].cpu())


if __name__ == "__main__":
    main()
