#!/usr/bin/env python3
"""Minimal Python-DSL reproducer for SME pointers carried by ``scf.for``.

Run from the FlyDSL checkout:

    PYTHONPATH=.:build-fly/python_packages python \
      tests/kernels/repro_scf_sme_fat_pointer.py

Before FlyToIXDL registers SCF structural type conversion, ``flyc.compile``
fails because the Python ``for`` below lowers to an ``scf.for`` whose carried
value is ``!fly.ptr<..., #fly_ixdl.sme_gmem>``.  The loop body lowers
``fx.add_offset`` to the SME LLVM fat-pointer struct instead.
"""

import os

import torch

os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "iluvatar")
os.environ.setdefault("FLYDSL_RUNTIME_KIND", "iluvatar")
os.environ.setdefault("ARCH", "ivcore11")

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.ixdl as ixdl


@flyc.kernel(known_block_size=[64, 1, 1])
def carry_sme_pointer(A: fx.Tensor, steps: fx.Int32):
    # A normal global-memory tensor becomes an SME fat pointer:
    # !fly.ptr<f32, #fly_ixdl.sme_gmem>.
    cursor = fx.get_iter(ixdl.make_sme_gmem_tensor(A, leading_stride=16))

    # A dynamic Python range becomes scf.for.  `cursor` is an inferred
    # loop-carried value because the next iteration consumes its updated value.
    for _ in range(fx.Int32(0), steps, fx.Int32(1)):
        cursor = fx.add_offset(cursor, fx.make_int_tuple(fx.Int32(16 * 16)))

    # Keep the loop result live by consuming it in an async SME copy.
    tid = fx.thread_idx.x
    load_layout = fx.make_layout((16, 16), (1, 16))
    smem = fx.make_view(
        fx.get_dyn_shared(fx.Float32),
        fx.make_layout(16 * 16, 1),
    )
    async_atom = fx.make_copy_atom(
        ixdl.MRAsyncCp(ixdl.SMESwizzle.NoSwizzle),
        fx.Float32,
    )
    tiled_copy = fx.make_tiled_copy_tv(
        async_atom,
        fx.make_layout((1, 1), (1, 1)),
        load_layout,
    )
    src_tile = fx.make_view(cursor, load_layout)
    dst_tile = fx.make_view(fx.get_iter(smem), load_layout)
    thread_copy = tiled_copy.get_slice(tid)
    fx.copy(
        async_atom,
        thread_copy.partition_S(src_tile),
        thread_copy.partition_D(dst_tile),
    )


@flyc.jit
def launch(A: fx.Tensor, steps: fx.Int32, stream=fx.Stream(None)):
    carry_sme_pointer(A, steps).launch(
        grid=(1, 1, 1),
        block=(64, 1, 1),
        smem=16 * 16 * 4,
        stream=stream,
    )


def main():
    torch.cuda.set_device(0)
    a = torch.empty(16 * 16, device="cuda", dtype=torch.float32)
    stream = fx.Stream(torch.cuda.current_stream())
    # This is intentionally compile-only; the kernel does no data access.
    flyc.compile(launch, a, fx.Int32(2), stream)


if __name__ == "__main__":
    main()
