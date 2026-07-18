# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Global Split-K helpers for the Iluvatar MR HGEMM.

Modes (see ``compile_iluvatar_mr_hgemm_splitk(..., split_k_mode=...)``):

* **serial** — CUTLASS SplitKSerial: single ``grid.z=split_k`` launch; per-tile
  GMEM locks + ``ixdl.atomic_cas`` turnstile order the load-add-store into C
  (compute overlaps; only the epilogue is serialized per output tile).
* **parallel** — each K-slice writes an fp32 partial to workspace; a separate
  reduce kernel sums along ``split_k`` into C.
* **atomic** — host/stream-ordered C-zero then scalar ``UniversalAtomicAdd``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import flydsl.expr.ixdl as ixdl
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.ir import IntegerType
from flydsl.expr import arith as _arith

SPLIT_K_MODE_SERIAL = "serial"
SPLIT_K_MODE_PARALLEL = "parallel"
SPLIT_K_MODE_ATOMIC = "atomic"
SPLIT_K_MODE_CHOICES = (SPLIT_K_MODE_SERIAL, SPLIT_K_MODE_PARALLEL, SPLIT_K_MODE_ATOMIC)
DEFAULT_SPLIT_K_MODE = SPLIT_K_MODE_SERIAL


def serial_turnstile_wait(lock_ptr, expected):
    """Spin until ``*lock_ptr == expected`` via no-op CAS (acquire).

    Emits a raw ``scf.while`` so it works when called from a nested helper
    (AST ``while`` rewrite only applies to the ``@flyc.kernel`` source body).
    """
    from flydsl._mlir.dialects import arith, scf

    i32 = IntegerType.get_signless(32)
    expected_v = _arith.unwrap(expected)
    init_old = _arith.unwrap(
        ixdl.atomic_cas(
            lock_ptr,
            expected,
            expected,
            success_ordering=llvm.AtomicOrdering.acquire,
            failure_ordering=llvm.AtomicOrdering.acquire,
        )
    )
    w = scf.WhileOp([i32], [init_old])
    before = ir.Block.create_at_start(w.before, [i32])
    after = ir.Block.create_at_start(w.after, [i32])
    with ir.InsertionPoint(before):
        cur = before.arguments[0]
        need_wait = arith.CmpIOp(arith.CmpIPredicate.ne, cur, expected_v).result
        scf.ConditionOp(need_wait, [cur])
    with ir.InsertionPoint(after):
        new_old = _arith.unwrap(
            ixdl.atomic_cas(
                lock_ptr,
                expected,
                expected,
                success_ordering=llvm.AtomicOrdering.acquire,
                failure_ordering=llvm.AtomicOrdering.acquire,
            )
        )
        scf.YieldOp([new_old])


def serial_turnstile_arrive(lock_ptr, expected):
    """Release the turnstile: CAS ``expected → expected+1`` (release)."""
    nextv = expected + fx.Int32(1)
    ixdl.atomic_cas(
        lock_ptr,
        expected,
        nextv,
        success_ordering=llvm.AtomicOrdering.release,
        failure_ordering=llvm.AtomicOrdering.acquire,
    )


def build_c_zero_kernel(*, bm: int, bn: int, n: int, threads: int, elem_dtype):
    """Return a kernel that zeros C in (bm, bn) tiles matching the GEMM grid.

    Used by the serial and atomic split-K paths. Grid is ``(M // bm, N // bn, 1)``.
    """
    tile_elems = bm * bn
    assert tile_elems % threads == 0, "C tile must divide evenly across threads"
    elems_per_thread = tile_elems // threads

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def zero_c_kernel(C: fx.Tensor):
        tid = fx.thread_idx.x
        bid_x, bid_y, _ = fx.block_idx

        gC = fx.slice(fx.flat_divide(C, (bm, bn)), (None, None, bid_x, bid_y))
        c_ptr = fx.get_iter(gC)
        zero = fx.arith.constant(0.0, type=elem_dtype.ir_type)

        for e in fx.range_constexpr(elems_per_thread):
            linear = tid + fx.Int32(e * threads)
            row = linear // fx.Int32(bn)
            col = linear - row * fx.Int32(bn)
            off = row * fx.Int32(n) + col
            fx.ptr_store(zero, fx.add_offset(c_ptr, fx.make_int_tuple(off)))

    return zero_c_kernel


def build_splitk_reduce_kernel(*, bm: int, bn: int, m: int, n: int, split_k: int, threads: int, elem_dtype):
    """Sum ``Workspace[split_k, M, N]`` (fp32) along z into ``C`` (``elem_dtype``).

    Grid is ``(M // bm, N // bn, 1)``, matching the GEMM MN tiling.
    """
    tile_elems = bm * bn
    assert tile_elems % threads == 0, "C tile must divide evenly across threads"
    elems_per_thread = tile_elems // threads
    mn_elems = m * n

    @flyc.kernel(known_block_size=[threads, 1, 1])
    def reduce_kernel(Workspace: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        bid_x, bid_y, _ = fx.block_idx

        gC = fx.slice(fx.flat_divide(C, (bm, bn)), (None, None, bid_x, bid_y))
        c_ptr = fx.get_iter(gC)
        ws_ptr = fx.recast_iter(
            fx.PointerType.get(fx.Float32.ir_type, fx.AddressSpace.Global),
            fx.get_iter(Workspace),
        )

        for e in fx.range_constexpr(elems_per_thread):
            linear = tid + fx.Int32(e * threads)
            row = linear // fx.Int32(bn)
            col = linear - row * fx.Int32(bn)
            g_row = bid_x * fx.Int32(bm) + row
            g_col = bid_y * fx.Int32(bn) + col
            mn_off = g_row * fx.Int32(n) + g_col

            acc = fx.arith.constant(0.0, type=fx.Float32.ir_type)
            for z in fx.range_constexpr(split_k):
                ws_off = fx.Int32(z * mn_elems) + mn_off
                acc = acc + fx.ptr_load(
                    fx.add_offset(ws_ptr, fx.make_int_tuple(ws_off)),
                    result_type=fx.Float32.ir_type,
                )
            out = fx.arith.truncf(elem_dtype.ir_type, acc)
            c_off = row * fx.Int32(n) + col
            fx.ptr_store(out, fx.add_offset(c_ptr, fx.make_int_tuple(c_off)))

    return reduce_kernel


def make_splitk_workspace(split_k: int, m: int, n: int, device):
    """Host fp32 workspace ``[split_k, M, N]`` for CUTLASS SplitKParallel."""
    import torch

    return torch.empty((split_k, m, n), dtype=torch.float32, device=device)


def make_splitk_locks(tiles_m: int, tiles_n: int, device):
    """Per-output-tile i32 turnstile locks for SplitKSerial (``tiles_m * tiles_n``)."""
    import torch

    return torch.empty((tiles_m * tiles_n,), dtype=torch.int32, device=device)


def resolve_device_from_tensor(t):
    """Best-effort CUDA device from a torch / DLPack-backed tensor."""
    import torch

    if hasattr(t, "device"):
        return t.device
    return torch.device("cuda")
