# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import os

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

BLOCK_DIM = 256


@flyc.kernel
def vector_add_kernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    n: fx.Int32,
):
    tid = fx.block_idx.x * BLOCK_DIM + fx.thread_idx.x

    if tid < n:
        elem_layout = fx.make_layout(1, 1)
        reg_ty = fx.MemRefType.get(fx.T.i32(), fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
        copy_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)

        src_a = fx.make_view(fx.add_offset(fx.get_iter(A), fx.make_int_tuple(tid)), elem_layout)
        src_b = fx.make_view(fx.add_offset(fx.get_iter(B), fx.make_int_tuple(tid)), elem_layout)
        dst_c = fx.make_view(fx.add_offset(fx.get_iter(C), fx.make_int_tuple(tid)), elem_layout)

        rA = fx.memref_alloca(reg_ty, elem_layout)
        rB = fx.memref_alloca(reg_ty, elem_layout)
        rC = fx.memref_alloca(reg_ty, elem_layout)

        fx.copy_atom_call(copy_atom, src_a, rA)
        fx.copy_atom_call(copy_atom, src_b, rB)

        vC = fx.arith.addi(fx.memref_load_vec(rA), fx.memref_load_vec(rB))
        fx.memref_store_vec(vC, rC)
        fx.copy_atom_call(copy_atom, rC, dst_c)


@flyc.jit
def vector_add(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    n: fx.Int32,
    stream: fx.Stream = fx.Stream(None),
):
    grid_x = (n + BLOCK_DIM - 1) // BLOCK_DIM
    vector_add_kernel(A, B, C, n).launch(grid=(grid_x, 1, 1), block=(BLOCK_DIM, 1, 1), stream=stream)


def _configure_iluvatar_env():
    os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "iluvatar")
    os.environ.setdefault("FLYDSL_RUNTIME_KIND", "iluvatar")
    os.environ.setdefault("ARCH", "ivcore11")


if __name__ == "__main__":
    _configure_iluvatar_env()

    n = 4096
    A = torch.randint(0, 100, (n,), dtype=torch.int32, device="cuda")
    B = torch.randint(0, 100, (n,), dtype=torch.int32, device="cuda")
    C = torch.zeros(n, dtype=torch.int32, device="cuda")

    vector_add(A, B, C, n, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    ok = torch.equal(C, A + B)
    print("Result correct:", ok)
    if not ok:
        print("A[:16]:", A[:16])
        print("B[:16]:", B[:16])
        print("C[:16]:", C[:16])
