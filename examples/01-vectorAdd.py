# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Portable vectorAdd example for Iluvatar backend."""

import os
import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "iluvatar")
os.environ.setdefault("FLYDSL_RUNTIME_KIND", "iluvatar")
os.environ.setdefault("ARCH", "ivcore11")
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
os.environ.setdefault("FLYDSL_DEBUG_PRINT_AFTER_ALL", "1")


@flyc.kernel
def vector_add_kernel(
    a: fx.Tensor,
    c: fx.Tensor,
    block_dim: fx.Constexpr[int],
):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x

    ta = fx.logical_divide(a, fx.make_layout(block_dim, 1))
    tc = fx.logical_divide(c, fx.make_layout(block_dim, 1))

    ta = fx.slice(ta, (None, bid))
    tc = fx.slice(tc, (None, bid))
    ta = fx.logical_divide(ta, fx.make_layout(1, 1))
    tc = fx.logical_divide(tc, fx.make_layout(1, 1))

    reg_ty = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
    copy_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)

    ra = fx.memref_alloca(reg_ty, fx.make_layout(1, 1))
    rc = fx.memref_alloca(reg_ty, fx.make_layout(1, 1))

    fx.copy_atom_call(copy_atom, fx.slice(ta, (None, tid)), ra)

    vc = fx.arith.addf(fx.memref_load_vec(ra), fx.memref_load_vec(ra))
    fx.memref_store_vec(vc, rc)

    fx.copy_atom_call(copy_atom, rc, fx.slice(tc, (None, tid)))


@flyc.jit
def vector_add(
    a: fx.Tensor,
    c: fx.Tensor,
    n: fx.Int32,
    const_n: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    block_dim = 64
    grid_x = (n + block_dim - 1) // block_dim
    vector_add_kernel(a, c, block_dim).launch(grid=(grid_x, 1, 1), block=[block_dim, 1, 1], stream=stream)


def run_eager() -> bool:
    n = 256 
    a = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
    c = torch.zeros(n, dtype=torch.float32).cuda()
    ta = flyc.from_dlpack(a).mark_layout_dynamic(leading_dim=0, divisibility=4)

    vector_add(ta, c, n, n + 1, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    ok = torch.allclose(c, a + a)
    print(f"[Eager] Result correct: {ok}")
    if not ok:
        print("a:", a[:16])
        print("c:", c[:16])
    return ok


if __name__ == "__main__":
    print("=" * 50)
    print("Test: vectorAdd on Iluvatar backend")
    print("=" * 50)
    print(f"Passed: {run_eager()}")