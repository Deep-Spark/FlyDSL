# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar async-copy synchronization primitives.

Two coexisting schemes (pick one per kernel):

* Scheme A: :func:`cp_async_commit_group` / :func:`cp_async_wait_group`.
* Scheme B: :func:`sl_waitmem` / :func:`sl_pipebar_arrive` / :func:`sl_pipebar_wait`.

These wrappers emit Iluvatar LLVM intrinsics via ``llvm.call_intrinsic``.
"""

from ..._mlir.dialects import llvm as _llvm
from .. import arith as _arith
from ..typing import T


def _const_i32(value):
    return _arith.unwrap(_arith.constant(int(value), type=T.i32))


def _const_i64(value):
    return _arith.unwrap(_arith.constant(int(value), type=T.i64))


# --- Scheme A: CUDA-style commit / wait group ---


def cp_async_commit_group():
    """Commit all prior async copies into a new group (``ixdl.cp.async.commit.group``)."""
    return _llvm.call_intrinsic(None, "llvm.bi.cp.async.commit.group", [], [], [])


def cp_async_wait_group(n=0):
    """Wait until at most ``n`` async-copy groups are pending (``ixdl.cp.async.wait.group``)."""
    return _llvm.call_intrinsic(None, "llvm.bi.cp.async.wait.group", [_const_i32(n)], [], [])


# --- Scheme B: multi-stage pipeline (sl_waitmem + pipebar) ---


def sl_waitmem(n):
    """Wait for outstanding memory operations."""
    return _llvm.call_intrinsic(None, "llvm.bi.sl.waitcnt", [_const_i64(n)], [], [])


def sl_pipebar_arrive(value=0):
    """Pipeline-barrier arrive / report."""
    return _llvm.call_intrinsic(None, "llvm.bi.pipebar.req", [_const_i32(value)], [], [])


def sl_pipebar_wait(value=0):
    """Pipeline-barrier wait."""
    return _llvm.call_intrinsic(None, "llvm.bi.pipebar.wait", [_const_i32(value)], [], [])
