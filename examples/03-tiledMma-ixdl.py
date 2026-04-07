# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import os
import sys
from pathlib import Path

os.environ.setdefault("BACKEND", "ixdl")
os.environ.setdefault("COMPILE_ONLY", "1")

import torch

_repo_root = Path(__file__).resolve().parents[1]
_build_pkg_dir = _repo_root / "build-fly" / "python_packages"
if _build_pkg_dir.exists():
    sys.path.insert(0, str(_build_pkg_dir))

import flydsl.compiler as flyc
import flydsl.expr as fx


@flyc.kernel
def tiled_mma_ixdl_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    D: fx.Tensor,
):
    tid = fx.thread_idx.x
    bid = fx.block_idx.x

    tile_a = fx.make_tile(16, 16)
    tile_b = fx.make_tile(16, 16)
    tile_c = fx.make_tile(16, 16)
    tile_d = fx.make_tile(16, 16)

    A = fx.zipped_divide(A, tile_a)
    B = fx.zipped_divide(B, tile_b)
    C = fx.zipped_divide(C, tile_c)
    D = fx.zipped_divide(D, tile_d)

    A = fx.slice(A, (None, bid))
    B = fx.slice(B, (None, bid))
    C = fx.slice(C, (None, bid))
    D = fx.slice(D, (None, bid))

    mma_atom = fx.make_mma_atom(fx.ixdl.MMAD(16, 16, 16, fx.Float16, elem_type_acc=fx.Float32))
    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (1, 1, 1)))
    thr_mma = tiled_mma.thr_slice(tid)

    part_A = thr_mma.partition_A(A)
    part_B = thr_mma.partition_B(B)
    part_C = thr_mma.partition_C(C)

    frag_A = thr_mma.make_fragment_A(part_A)
    frag_B = thr_mma.make_fragment_B(part_B)
    frag_C = thr_mma.make_fragment_C(part_C)
    frag_D = thr_mma.make_fragment_C(part_C)

    copy_atom_A = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float16)
    copy_atom_B = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float16)
    copy_atom_C = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)

    tiled_copy_A = fx.make_tiled_copy_A(copy_atom_A, tiled_mma)
    tiled_copy_B = fx.make_tiled_copy_B(copy_atom_B, tiled_mma)
    tiled_copy_C = fx.make_tiled_copy_C(copy_atom_C, tiled_mma)

    thr_copy_A = tiled_copy_A.get_slice(tid)
    thr_copy_B = tiled_copy_B.get_slice(tid)
    thr_copy_C = tiled_copy_C.get_slice(tid)

    copy_src_A = thr_copy_A.partition_S(A)
    copy_src_B = thr_copy_B.partition_S(B)
    copy_src_C = thr_copy_C.partition_S(C)
    copy_dst_D = thr_copy_C.partition_D(D)

    copy_frag_A = thr_copy_A.retile(frag_A)
    copy_frag_B = thr_copy_B.retile(frag_B)
    copy_frag_C = thr_copy_C.retile(frag_C)
    copy_frag_D = thr_copy_C.retile(frag_D)

    fx.copy(copy_atom_A, copy_src_A, copy_frag_A)
    fx.copy(copy_atom_B, copy_src_B, copy_frag_B)
    fx.copy(copy_atom_C, copy_src_C, copy_frag_C)
    fx.gemm(mma_atom, frag_D, frag_A, frag_B, frag_C)
    fx.copy(copy_atom_C, copy_frag_D, copy_dst_D)


@flyc.jit
def tiled_mma_ixdl(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    D: fx.Tensor,
    stream: fx.Stream = fx.Stream(None),
):
    tiled_mma_ixdl_kernel(A, B, C, D).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for this compile-only ixdl example")

    if os.environ.get("COMPILE_ONLY", "1") != "0":
        a = torch.zeros((16, 16), dtype=torch.float16, device="cuda")
        b = torch.zeros((16, 16), dtype=torch.float16, device="cuda")
        c = torch.zeros((16, 16), dtype=torch.float32, device="cuda")
        d = torch.zeros((16, 16), dtype=torch.float32, device="cuda")

        tiled_mma_ixdl(a, b, c, d, stream=torch.cuda.Stream())
        print("[flydsl] ixdl tiledMma compile-only example completed")
        return

    def run_case(a, b, c):
        d = torch.zeros((16, 16), dtype=torch.float32, device="cuda")
        tiled_mma_ixdl(a, b, c, d, stream=torch.cuda.Stream())
        torch.cuda.synchronize()
        return d

    a_base = torch.arange(16 * 16, dtype=torch.float16, device="cuda").reshape(16, 16)
    b_base = torch.eye(16, dtype=torch.float16, device="cuda")
    c_zero = torch.zeros((16, 16), dtype=torch.float32, device="cuda")

    d_base = run_case(a_base, b_base, c_zero)
    print("[baseline] D[0, :16] =", d_base[0, :16].cpu())

    b_ones = torch.ones((16, 16), dtype=torch.float16, device="cuda")
    d_ones = run_case(a_base, b_ones, c_zero).cpu()
    # Follow the CuTeDSL operand contract: A is (M, K), B is (N, K).
    ref_ones = torch.einsum("mk,nk->mn", a_base.float(), b_ones.float()).cpu()
    diff_ones = (d_ones - ref_ones).abs()
    print("\n[B=ones] result row0[:16] =", d_ones[0, :16])
    print("[B=ones] ref row0[:16]    =", ref_ones[0, :16])
    print("[B=ones] result col0[:16] =", d_ones[:16, 0])
    print("[B=ones] ref col0[:16]    =", ref_ones[:16, 0])
    print("[B=ones] max_diff =", float(diff_ones.max()))
    print("[B=ones] allclose =", bool(torch.allclose(d_ones, ref_ones, atol=1e-3, rtol=1e-3)))

    probes = [
        ("A[0,0]+=100", "A", 0, 0, 100.0),
        ("A[0,2]+=100", "A", 0, 2, 100.0),
        ("A[0,4]+=100", "A", 0, 4, 100.0),
        # ("B[1,1]=2", "B", 1, 1, 2.0),
        # ("B[4,4]=2", "B", 4, 4, 2.0),
    ]

    for name, which, i, j, value in probes:
        a = a_base.clone()
        b = b_base.clone()
        if which == "A":
            a[i, j] += value
        else:
            b[i, j] = value

        d = run_case(a, b, c_zero)
        delta = (d - d_base).cpu()
        nz = (delta.abs() > 1e-3).nonzero(as_tuple=False)

        print(f"\n[{name}] nonzero count = {len(nz)}")
        print(f"[{name}] first nonzero coords = {nz[:16].tolist()}")
        if len(nz) <= 16:
            for r, c in nz.tolist():
                print(f"[{name}] delta[{r}, {c}] = {float(delta[r, c])}")
        print(f"[{name}] delta row0[:16] = {delta[0, :16]}")


if __name__ == "__main__":
    main()
