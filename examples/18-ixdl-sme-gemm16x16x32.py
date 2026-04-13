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

M = 16
N = 16
K = 32
ATOM_M = 16
ATOM_N = 16
ATOM_K = 16
K_TILES = K // ATOM_K
SME_TILE_ELEMS = 16 * 32
SME_TILE_BYTES = SME_TILE_ELEMS * 2
SMEM_BYTES = SME_TILE_BYTES * 2


@flyc.kernel
def ixdl_sme_gemm16x16x32_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, D: fx.Tensor):
    tid = fx.thread_idx.x

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
    smem_a = fx.ixdl.SMEView16x512b(smem_base, fx.Float16)
    smem_b = fx.ixdl.SMEView16x512b(smem_base, fx.Float16, transpose=True, elem_offset=SME_TILE_ELEMS)

    fx.copy_atom_call(sme_atom_A, A, smem_a)
    fx.copy_atom_call(sme_atom_B, B, smem_b)
    fx.ixdl.cp_async_commit_group()
    fx.ixdl.cp_async_wait_group(0)
    fx.gpu.barrier()

    tile_a = fx.make_tile(ATOM_M, ATOM_K)
    tile_b = fx.make_tile(ATOM_N, ATOM_K)
    smem_a_tiles = fx.zipped_divide(smem_a, tile_a)
    smem_b_tiles = fx.zipped_divide(smem_b, tile_b)

    frag_acc = thr_mma.make_fragment_C(C)
    copy_src_C = thr_copy_C.partition_S(C)
    copy_dst_D = thr_copy_C.partition_D(D)
    copy_frag_acc = thr_copy_C.retile(frag_acc)
    fx.copy(copy_atom_C, copy_src_C, copy_frag_acc)

    for bk in range_constexpr(K_TILES):
        a_tile = fx.slice(smem_a_tiles, (None, bk))
        b_tile = fx.slice(smem_b_tiles, (None, bk))

        frag_A = thr_mma.make_fragment_A(a_tile)
        frag_B = thr_mma.make_fragment_B(b_tile)
        copy_src_A = thr_copy_A.partition_S(a_tile)
        copy_src_B = thr_copy_B.partition_S(b_tile)
        copy_frag_A = thr_copy_A.retile(frag_A)
        copy_frag_B = thr_copy_B.retile(frag_B)

        fx.copy(copy_atom_A, copy_src_A, copy_frag_A)
        fx.copy(copy_atom_B, copy_src_B, copy_frag_B)
        fx.gemm(mma_atom, frag_acc, frag_A, frag_B, frag_acc)

    fx.copy(copy_atom_C, copy_frag_acc, copy_dst_D)


@flyc.jit
def ixdl_sme_gemm16x16x32(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    D: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    ixdl_sme_gemm16x16x32_kernel(A, B, C, D).launch(
        grid=(1, 1, 1),
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

    stream = torch.cuda.Stream()
    ixdl_sme_gemm16x16x32(a, b, c, d, stream=stream)

    if os.environ.get("COMPILE_ONLY", "1") != "0":
        print("[ixdl_sme_gemm16x16x32] compile-only launch emitted")
        return

    torch.cuda.synchronize()
    ref = torch.einsum("mk,nk->mn", a.float(), b.float()) + c
    diff = (d - ref).abs()
    ok = bool(torch.allclose(d, ref, atol=1e-3, rtol=1e-3))
    print("[ixdl_sme_gemm16x16x32] allclose =", ok)
    print("[ixdl_sme_gemm16x16x32] max_diff =", float(diff.max()))
    print("[ixdl_sme_gemm16x16x32] D[0, :8] =", d[0, :8].cpu())
    print("[ixdl_sme_gemm16x16x32] ref[0, :8] =", ref[0, :8].cpu())


if __name__ == "__main__":
    main()
