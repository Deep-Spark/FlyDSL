# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Opt-in Iluvatar device tests for ``flydsl.expr.ixdl.atomic_cas``.

Covers:

1. Success / failure return value and memory update (single CTA, i32).
2. Multi-CTA exclusive claim: only one CTA wins ``CAS(0 → 1)``.
3. Ordered turnstile: CTA ``z`` waits for ``lock == z`` then advances to ``z+1``.
4. i16 success / failure smoke (call site uses ``Int16``).
5. f16 / f32 success / failure smoke (frontend bitcasts to i16 / i32 for cmpxchg).

Set ``FLYDSL_ILUVATAR_RUN_ATOMIC_CAS=1`` to run (needs an Iluvatar ivcore11 device).
"""

import os
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.l2_device]

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from kernels.gemm.iluvatar.common import WARP_SIZE  # noqa: E402


def _require_enabled() -> None:
    if os.environ.get("FLYDSL_ILUVATAR_RUN_ATOMIC_CAS", "").lower() not in {"1", "true", "yes", "on"}:
        pytest.skip("set FLYDSL_ILUVATAR_RUN_ATOMIC_CAS=1 to run the Iluvatar atomic CAS device tests")


def _require_imports():
    try:
        import flydsl

        generated_pkg = Path(os.environ.get("FLYDSL_PYTHON_PACKAGES", _REPO_ROOT / "build-fly" / "python_packages"))
        generated_flydsl = generated_pkg / "flydsl"
        if generated_flydsl.is_dir() and str(generated_flydsl) not in flydsl.__path__:
            flydsl.__path__.append(str(generated_flydsl))

        import flydsl.compiler as flyc
        import flydsl.expr as fx
        import flydsl.expr.ixdl as ixdl
        from flydsl._mlir.dialects import llvm
    except ModuleNotFoundError as exc:
        pytest.fail(f"FlyDSL Python package is not importable: {exc}")
    return flyc, fx, ixdl, llvm


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        pytest.skip(f"torch is required for the Iluvatar atomic CAS device tests: {exc}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA-compatible device is not available")
    return torch


def _set_iluvatar_env(monkeypatch) -> None:
    monkeypatch.setenv("FLYDSL_COMPILE_BACKEND", "iluvatar")
    monkeypatch.setenv("FLYDSL_RUNTIME_KIND", "iluvatar")
    monkeypatch.setenv("ARCH", os.environ.get("ARCH", "ivcore11"))
    monkeypatch.setenv("FLYDSL_RUNTIME_ENABLE_CACHE", "0")
    monkeypatch.delenv("COMPILE_ONLY", raising=False)


def test_atomic_cas_success_and_failure(monkeypatch):
    """CAS returns the old value; success updates memory, failure leaves it."""
    _require_enabled()
    flyc, fx, ixdl, llvm = _require_imports()
    torch = _require_torch()
    _set_iluvatar_env(monkeypatch)

    @flyc.kernel(known_block_size=[WARP_SIZE, 1, 1])
    def cas_probe_kernel(Lock: fx.Tensor, Out: fx.Tensor):
        tid = fx.thread_idx.x
        if tid == fx.Int32(0):
            lock_ptr = fx.recast_iter(
                fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global),
                fx.get_iter(Lock),
            )
            out_ptr = fx.recast_iter(
                fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global),
                fx.get_iter(Out),
            )
            # Lock starts at 0. Success: 0 -> 7, returns 0.
            old0 = ixdl.atomic_cas(
                lock_ptr,
                fx.Int32(0),
                fx.Int32(7),
                success_ordering=llvm.AtomicOrdering.acq_rel,
                failure_ordering=llvm.AtomicOrdering.acquire,
            )
            fx.ptr_store(old0, out_ptr)
            # Failure: expect 0 but lock is 7, returns 7 and leaves 7.
            old1 = ixdl.atomic_cas(
                lock_ptr,
                fx.Int32(0),
                fx.Int32(9),
                success_ordering=llvm.AtomicOrdering.acq_rel,
                failure_ordering=llvm.AtomicOrdering.acquire,
            )
            fx.ptr_store(old1, fx.add_offset(out_ptr, fx.make_int_tuple(1)))

    @flyc.jit
    def launch(Lock, Out, stream: fx.Stream = fx.Stream(None)):
        cas_probe_kernel(Lock, Out).launch(grid=(1, 1, 1), block=(WARP_SIZE, 1, 1), stream=stream)

    lock = torch.zeros((1,), dtype=torch.int32, device="cuda")
    out = torch.full((2,), -1, dtype=torch.int32, device="cuda")
    launch(lock, out)
    torch.cuda.synchronize()

    assert out[0].item() == 0, f"successful CAS should return old=0, got {out[0].item()}"
    assert lock[0].item() == 7, f"successful CAS should store 7, got {lock[0].item()}"
    assert out[1].item() == 7, f"failed CAS should return current=7, got {out[1].item()}"
    assert lock[0].item() == 7, f"failed CAS must not update lock, got {lock[0].item()}"


def test_atomic_cas_i16_success_and_failure(monkeypatch):
    """i16 CAS smoke: success updates memory, failure leaves it (CMPSWAP_B16)."""
    _require_enabled()
    flyc, fx, ixdl, llvm = _require_imports()
    torch = _require_torch()
    _set_iluvatar_env(monkeypatch)

    @flyc.kernel(known_block_size=[WARP_SIZE, 1, 1])
    def cas_i16_probe_kernel(Lock: fx.Tensor, Out: fx.Tensor):
        tid = fx.thread_idx.x
        if tid == fx.Int32(0):
            lock_ptr = fx.recast_iter(
                fx.PointerType.get(fx.Int16.ir_type, fx.AddressSpace.Global),
                fx.get_iter(Lock),
            )
            out_ptr = fx.recast_iter(
                fx.PointerType.get(fx.Int16.ir_type, fx.AddressSpace.Global),
                fx.get_iter(Out),
            )
            old0 = ixdl.atomic_cas(
                lock_ptr,
                fx.Int16(0),
                fx.Int16(7),
                success_ordering=llvm.AtomicOrdering.acq_rel,
                failure_ordering=llvm.AtomicOrdering.acquire,
            )
            fx.ptr_store(old0, out_ptr)
            old1 = ixdl.atomic_cas(
                lock_ptr,
                fx.Int16(0),
                fx.Int16(9),
                success_ordering=llvm.AtomicOrdering.acq_rel,
                failure_ordering=llvm.AtomicOrdering.acquire,
            )
            fx.ptr_store(old1, fx.add_offset(out_ptr, fx.make_int_tuple(1)))

    @flyc.jit
    def launch(Lock, Out, stream: fx.Stream = fx.Stream(None)):
        cas_i16_probe_kernel(Lock, Out).launch(grid=(1, 1, 1), block=(WARP_SIZE, 1, 1), stream=stream)

    lock = torch.zeros((1,), dtype=torch.int16, device="cuda")
    out = torch.full((2,), -1, dtype=torch.int16, device="cuda")
    launch(lock, out)
    torch.cuda.synchronize()

    assert out[0].item() == 0, f"successful i16 CAS should return old=0, got {out[0].item()}"
    assert lock[0].item() == 7, f"successful i16 CAS should store 7, got {lock[0].item()}"
    assert out[1].item() == 7, f"failed i16 CAS should return current=7, got {out[1].item()}"
    assert lock[0].item() == 7, f"failed i16 CAS must not update lock, got {lock[0].item()}"


@pytest.mark.parametrize(
    "fx_dtype_name,torch_dtype,desired,fail_desired",
    [
        ("Float16", "float16", 7.0, 9.0),
        ("Float32", "float32", 7.0, 9.0),
    ],
)
def test_atomic_cas_float_success_and_failure(monkeypatch, fx_dtype_name, torch_dtype, desired, fail_desired):
    """Float CAS smoke: same-width bitcast to integer for llvm.cmpxchg."""
    _require_enabled()
    flyc, fx, ixdl, llvm = _require_imports()
    torch = _require_torch()
    _set_iluvatar_env(monkeypatch)

    fx_dtype = getattr(fx, fx_dtype_name)
    tt = getattr(torch, torch_dtype)

    @flyc.kernel(known_block_size=[WARP_SIZE, 1, 1])
    def cas_float_probe_kernel(Lock: fx.Tensor, Out: fx.Tensor):
        tid = fx.thread_idx.x
        if tid == fx.Int32(0):
            lock_ptr = fx.recast_iter(
                fx.PointerType.get(fx_dtype.ir_type, fx.AddressSpace.Global),
                fx.get_iter(Lock),
            )
            out_ptr = fx.recast_iter(
                fx.PointerType.get(fx_dtype.ir_type, fx.AddressSpace.Global),
                fx.get_iter(Out),
            )
            old0 = ixdl.atomic_cas(
                lock_ptr,
                fx_dtype(0.0),
                fx_dtype(desired),
                success_ordering=llvm.AtomicOrdering.acq_rel,
                failure_ordering=llvm.AtomicOrdering.acquire,
            )
            fx.ptr_store(old0, out_ptr)
            old1 = ixdl.atomic_cas(
                lock_ptr,
                fx_dtype(0.0),
                fx_dtype(fail_desired),
                success_ordering=llvm.AtomicOrdering.acq_rel,
                failure_ordering=llvm.AtomicOrdering.acquire,
            )
            fx.ptr_store(old1, fx.add_offset(out_ptr, fx.make_int_tuple(1)))

    @flyc.jit
    def launch(Lock, Out, stream: fx.Stream = fx.Stream(None)):
        cas_float_probe_kernel(Lock, Out).launch(grid=(1, 1, 1), block=(WARP_SIZE, 1, 1), stream=stream)

    lock = torch.zeros((1,), dtype=tt, device="cuda")
    out = torch.full((2,), -1.0, dtype=tt, device="cuda")
    launch(lock, out)
    torch.cuda.synchronize()

    assert out[0].item() == 0.0, f"successful {fx_dtype_name} CAS should return old=0, got {out[0].item()}"
    assert lock[0].item() == desired, f"successful {fx_dtype_name} CAS should store {desired}, got {lock[0].item()}"
    assert out[1].item() == desired, f"failed {fx_dtype_name} CAS should return current={desired}, got {out[1].item()}"
    assert lock[0].item() == desired, f"failed {fx_dtype_name} CAS must not update lock, got {lock[0].item()}"


def test_atomic_cas_multi_cta_exclusive(monkeypatch):
    """Exactly one of many CTAs wins ``CAS(0 → 1)`` on a shared lock."""
    _require_enabled()
    flyc, fx, ixdl, llvm = _require_imports()
    torch = _require_torch()
    _set_iluvatar_env(monkeypatch)

    grid = 32

    @flyc.kernel(known_block_size=[WARP_SIZE, 1, 1])
    def exclusive_kernel(Lock: fx.Tensor, Wins: fx.Tensor):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        if tid == fx.Int32(0):
            lock_ptr = fx.recast_iter(
                fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global),
                fx.get_iter(Lock),
            )
            wins_ptr = fx.recast_iter(
                fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global),
                fx.get_iter(Wins),
            )
            old = ixdl.atomic_cas(
                lock_ptr,
                fx.Int32(0),
                fx.Int32(1),
                success_ordering=llvm.AtomicOrdering.acq_rel,
                failure_ordering=llvm.AtomicOrdering.acquire,
            )
            # Record 1 if this CTA observed the pre-claim value.
            won = fx.arith.select(old == fx.Int32(0), fx.Int32(1), fx.Int32(0))
            fx.ptr_store(won, fx.add_offset(wins_ptr, fx.make_int_tuple(bid)))

    @flyc.jit
    def launch(Lock, Wins, stream: fx.Stream = fx.Stream(None)):
        exclusive_kernel(Lock, Wins).launch(grid=(grid, 1, 1), block=(WARP_SIZE, 1, 1), stream=stream)

    lock = torch.zeros((1,), dtype=torch.int32, device="cuda")
    wins = torch.zeros((grid,), dtype=torch.int32, device="cuda")
    launch(lock, wins)
    torch.cuda.synchronize()

    assert lock[0].item() == 1
    assert int(wins.sum().item()) == 1, f"exactly one CTA should win, got wins={wins.cpu().tolist()}"


def test_atomic_cas_ordered_turnstile(monkeypatch):
    """CTAs advance a per-grid lock in ``bid`` order (wait then arrive)."""
    _require_enabled()
    flyc, fx, ixdl, llvm = _require_imports()
    torch = _require_torch()
    _set_iluvatar_env(monkeypatch)

    from flydsl._mlir import ir
    from flydsl._mlir.dialects import arith, scf
    from flydsl._mlir.ir import IntegerType
    from flydsl.expr import arith as _arith

    grid = 8

    @flyc.kernel(known_block_size=[WARP_SIZE, 1, 1])
    def turnstile_kernel(Lock: fx.Tensor):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        if tid == fx.Int32(0):
            lock_ptr = fx.recast_iter(
                fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global),
                fx.get_iter(Lock),
            )
            expected = fx.Int32(bid)
            # Spin-wait with no-op CAS until lock == bid (raw scf.while: helper
            # while loops are not AST-rewritten).
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
            # Arrive: bid -> bid+1
            ixdl.atomic_cas(
                lock_ptr,
                expected,
                expected + fx.Int32(1),
                success_ordering=llvm.AtomicOrdering.release,
                failure_ordering=llvm.AtomicOrdering.acquire,
            )

    @flyc.jit
    def launch(Lock, stream: fx.Stream = fx.Stream(None)):
        turnstile_kernel(Lock).launch(grid=(grid, 1, 1), block=(WARP_SIZE, 1, 1), stream=stream)

    lock = torch.zeros((1,), dtype=torch.int32, device="cuda")
    launch(lock)
    torch.cuda.synchronize()

    assert lock[0].item() == grid, f"turnstile should finish at {grid}, got {lock[0].item()}"
