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

M = 256
N = 256
K = 256
ATOM_M = 16
ATOM_N = 16
ATOM_K = 16
SME_K = 32
CTA_M = ATOM_M
CTA_N = ATOM_N
CTA_STAGE_K = SME_K
CTA_TILES_M = M // CTA_M
CTA_TILES_N = N // CTA_N
CTA_STAGE_TILES_K = K // CTA_STAGE_K
STAGE_MMA_STEPS = CTA_STAGE_K // ATOM_K
PIPE_STAGES = 2
SME_TILE_ELEMS = 16 * 32
SME_TILE_BYTES = SME_TILE_ELEMS * 2
STAGE_SMEM_ELEMS = SME_TILE_ELEMS * 2
SMEM_BYTES = SME_TILE_BYTES * 2 * PIPE_STAGES


@flyc.kernel
def gemm256_ixdl_sme_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, D: fx.Tensor):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x
    bm = bid // CTA_TILES_N
    bn = bid % CTA_TILES_N

    tile_a_stage = fx.make_tile(CTA_M, CTA_STAGE_K)
    tile_b_stage = fx.make_tile(CTA_N, CTA_STAGE_K)
    tile_c = fx.make_tile(CTA_M, CTA_N)
    tile_a_atom = fx.make_tile(ATOM_M, ATOM_K)
    tile_b_atom = fx.make_tile(ATOM_N, ATOM_K)

    A_stage_tiles = fx.zipped_divide(A, tile_a_stage)
    B_stage_tiles = fx.zipped_divide(B, tile_b_stage)
    C_tiles = fx.zipped_divide(C, tile_c)
    D_tiles = fx.zipped_divide(D, tile_c)

    A_stage_tiles = fx.slice(A_stage_tiles, (None, (bm, None)))
    B_stage_tiles = fx.slice(B_stage_tiles, (None, (bn, None)))
    C_tile = fx.slice(C_tiles, (None, (bm, bn)))
    D_tile = fx.slice(D_tiles, (None, (bm, bn)))

    mma_atom = fx.make_mma_atom(fx.ixdl.MMAD(ATOM_M, ATOM_N, ATOM_K, fx.Float16, elem_type_acc=fx.Float32))
    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (1, 1, 1)))
    thr_mma = tiled_mma.thr_slice(tid)

    copy_atom_A = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float16)
    copy_atom_B = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float16)
    copy_atom_C = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    sme_atom_A = fx.make_copy_atom(fx.ixdl.SMELoad16x512b(fx.Float16), fx.Float16)
    sme_atom_B = fx.make_copy_atom(fx.ixdl.SMELoad16x512b(fx.Float16, transpose=True), fx.Float16)

    tiled_copy_A = fx.make_tiled_copy_A(copy_atom_A, tiled_mma)
    tiled_copy_B = fx.make_tiled_copy_B(copy_atom_B, tiled_mma)
    tiled_copy_C = fx.make_tiled_copy_C(copy_atom_C, tiled_mma)
    thr_copy_A = tiled_copy_A.get_slice(tid)
    thr_copy_B = tiled_copy_B.get_slice(tid)
    thr_copy_C = tiled_copy_C.get_slice(tid)

    smem_base = fx.get_dyn_shared()
    smem_a0 = fx.ixdl.SMEView16x512b(smem_base, fx.Float16, elem_offset=0)
    smem_b0 = fx.ixdl.SMEView16x512b(smem_base, fx.Float16, transpose=True, elem_offset=SME_TILE_ELEMS)
    smem_a1 = fx.ixdl.SMEView16x512b(smem_base, fx.Float16, elem_offset=STAGE_SMEM_ELEMS)
    smem_b1 = fx.ixdl.SMEView16x512b(
        smem_base,
        fx.Float16,
        transpose=True,
        elem_offset=STAGE_SMEM_ELEMS + SME_TILE_ELEMS,
    )
    smem_a0_tiles = fx.zipped_divide(smem_a0, tile_a_atom)
    smem_b0_tiles = fx.zipped_divide(smem_b0, tile_b_atom)
    smem_a1_tiles = fx.zipped_divide(smem_a1, tile_a_atom)
    smem_b1_tiles = fx.zipped_divide(smem_b1, tile_b_atom)

    frag_acc = thr_mma.make_fragment_C(C_tile)
    copy_src_C = thr_copy_C.partition_S(C_tile)
    copy_dst_D = thr_copy_C.partition_D(D_tile)
    copy_frag_acc = thr_copy_C.retile(frag_acc)
    fx.copy(copy_atom_C, copy_src_C, copy_frag_acc)

    # Prologue: prefetch the first K-stage into buffer 0.
    A_stage0 = fx.slice(A_stage_tiles, (None, 0))
    B_stage0 = fx.slice(B_stage_tiles, (None, 0))
    fx.copy_atom_call(sme_atom_A, A_stage0, smem_a0)
    fx.copy_atom_call(sme_atom_B, B_stage0, smem_b0)
    fx.ixdl.cp_async_commit_group()

    for bk in range_constexpr(CTA_STAGE_TILES_K):
        if bk % 2 == 0:
            cur_smem_a_tiles = smem_a0_tiles
            cur_smem_b_tiles = smem_b0_tiles
            next_smem_a = smem_a1
            next_smem_b = smem_b1
        else:
            cur_smem_a_tiles = smem_a1_tiles
            cur_smem_b_tiles = smem_b1_tiles
            next_smem_a = smem_a0
            next_smem_b = smem_b0

        has_next = bk + 1 < CTA_STAGE_TILES_K
        if has_next:
            A_next = fx.slice(A_stage_tiles, (None, bk + 1))
            B_next = fx.slice(B_stage_tiles, (None, bk + 1))
            fx.copy_atom_call(sme_atom_A, A_next, next_smem_a)
            fx.copy_atom_call(sme_atom_B, B_next, next_smem_b)
            fx.ixdl.cp_async_commit_group()
            fx.ixdl.cp_async_wait_group(1)
        else:
            fx.ixdl.cp_async_wait_group(0)

        fx.gpu.barrier()

        for kk in range_constexpr(STAGE_MMA_STEPS):
            A_tile = fx.slice(cur_smem_a_tiles, (None, kk))
            B_tile = fx.slice(cur_smem_b_tiles, (None, kk))

            frag_A = thr_mma.make_fragment_A(A_tile)
            frag_B = thr_mma.make_fragment_B(B_tile)

            copy_src_A = thr_copy_A.partition_S(A_tile)
            copy_src_B = thr_copy_B.partition_S(B_tile)
            copy_frag_A = thr_copy_A.retile(frag_A)
            copy_frag_B = thr_copy_B.retile(frag_B)

            fx.copy(copy_atom_A, copy_src_A, copy_frag_A)
            fx.copy(copy_atom_B, copy_src_B, copy_frag_B)
            fx.gemm(mma_atom, frag_acc, frag_A, frag_B, frag_acc)

        fx.gpu.barrier()

    fx.copy(copy_atom_C, copy_frag_acc, copy_dst_D)


@flyc.jit
def gemm256_ixdl_sme(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    D: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    gemm256_ixdl_sme_kernel(A, B, C, D).launch(
        grid=(CTA_TILES_M * CTA_TILES_N, 1, 1),
        block=(64, 1, 1),
        smem=SMEM_BYTES,
        stream=stream,
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for this ixdl SME GEMM demo")

    torch.manual_seed(0)
    a = torch.randint(-3, 4, (M, K), dtype=torch.int32, device="cuda").to(torch.float16)
    b = torch.randint(-3, 4, (N, K), dtype=torch.int32, device="cuda").to(torch.float16)
    c = torch.randint(-2, 3, (M, N), dtype=torch.int32, device="cuda").to(torch.float32)
    d = torch.zeros((M, N), dtype=torch.float32, device="cuda")

    gemm256_ixdl_sme(a, b, c, d, stream=torch.cuda.Stream())

    if os.environ.get("COMPILE_ONLY", "1") != "0":
        print("[gemm256_ixdl_sme] compile-only launch emitted")
        return

    torch.cuda.synchronize()
    ref = torch.einsum("mk,nk->mn", a.float(), b.float()) + c
    diff = (d - ref).abs()
    ok = bool(torch.allclose(d, ref, atol=1e-3, rtol=1e-3))
    print("[gemm256_ixdl_sme] CuTeDSL contract: A(M,K), B(N,K), D(M,N)")
    print("[gemm256_ixdl_sme] allclose =", ok)
    print("[gemm256_ixdl_sme] max_diff =", float(diff.max()))
    print("[gemm256_ixdl_sme] D[0, :8] =", d[0, :8].cpu())
    print("[gemm256_ixdl_sme] ref[0, :8] =", ref[0, :8].cpu())


if __name__ == "__main__":
    main()
