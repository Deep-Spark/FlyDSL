# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar async-copy / pipeline synchronization primitives.

Pick the scheme that matches the chip generation. Do not mix MR pipebar with
CQ named barriers in the same kernel (and do not emit pipebar on CQ paths).

* Scheme A (async-copy groups): :func:`cp_async_commit_group` /
  :func:`cp_async_wait_group`. Shared by MR SME and CQ SMEX G2S.

* Scheme B (MR pipeline barrier): :func:`sl_waitmem` /
  :func:`sl_barrier_alu` / :func:`sl_pipebar_arrive` / :func:`sl_pipebar_wait`.
  Requires ``pipe-bar`` (ivcore11 / MR).

* Scheme C (CQ named barrier): :func:`nbarrier_reach` / :func:`nbarrier_wait` /
  :func:`nbarrier_sync`. Requires ``named-bar`` (CQ);
  :func:`nbarrier_sync` additionally requires ``named-bar-sync`` (CQ).
  Use these for CQ multi-stage / double-buffer pipelines that would use
  pipebar on MR.

These wrappers emit Iluvatar LLVM intrinsics via ``llvm.call_intrinsic``.
"""

from ..._mlir.dialects import llvm as _llvm
from .. import arith as _arith
from ..typing import T


def _const_i32(value):
    return _arith.unwrap(_arith.constant(int(value), type=T.i32))


def _const_i64(value):
    return _arith.unwrap(_arith.constant(int(value), type=T.i64))


def _i32_arg(value):
    """Materialize an ``i32`` operand from a Python int or DSL scalar."""
    if isinstance(value, int):
        return _const_i32(value)
    return _arith.unwrap(value)


# --- Scheme A: CUDA-style commit / wait group ---


def cp_async_commit_group():
    """Commit all prior async copies into a new group (``ixdl.cp.async.commit.group``)."""
    return _llvm.call_intrinsic(None, "llvm.bi.cp.async.commit.group", [], [], [])


def cp_async_wait_group(n=0):
    """Wait until at most ``n`` async-copy groups are pending (``ixdl.cp.async.wait.group``)."""
    return _llvm.call_intrinsic(None, "llvm.bi.cp.async.wait.group", [_const_i32(n)], [], [])


# --- Scheme B: multi-stage pipeline (sl_waitmem + sl_barrier_alu + pipebar; MR) ---

# Layout of the ivcore11 ``sl.waitcnt`` bitfield. It is a packed structure holding,
# in declaration order, a 1-bit enable flag per counter followed by each counter's
# count field (same order). Only the declaration -- ``(counter name, count-field
# width)`` -- is transcribed here; bit offsets are derived from the field widths.
_WAITCNT_LAYOUT = (
    ("vm", 6),  # global memory
    ("sm", 6),  # constant memory
    ("lm", 4),  # shared memory
    ("g2s", 6),  # async copy global -> shared
    ("s2g", 5),  # async copy shared -> global
    ("mba", 4),  # memory-barrier arrive
    ("mbt", 1),  # memory-barrier test
)


def _waitcnt_value(**counters) -> int:
    """Encode an ivcore11 ``sl.waitcnt`` bitfield from named counters.
    Each keyword's value is the threshold to wait down to (``0`` fully drains that
    counter); only the named counters are enabled. Offsets are derived from
    `_WAITCNT_LAYOUT` so this stays a plain transcription of ``union WaitCount``
    args in C++ intrinsic.
    """
    order = [name for name, _ in _WAITCNT_LAYOUT]
    # enable flags occupy one bit each at the front (bit == field index); the count
    # fields follow, so the first one starts just past every enable flag and each
    # subsequent offset accumulates the previous field's width.
    count_shift = len(order)
    value = 0
    for name, width in _WAITCNT_LAYOUT:
        threshold = counters.pop(name, None)
        if threshold is not None:
            threshold = int(threshold)
            if not 0 <= threshold < (1 << width):
                raise ValueError(f"{name} wait threshold {threshold} out of range [0, {1 << width})")
            value |= (1 << order.index(name)) | (threshold << count_shift)
        count_shift += width
    if counters:
        raise ValueError(f"unknown wait counter(s): {', '.join(sorted(counters))}")
    return value


def sl_waitmem(**counters):
    """Wait for outstanding memory operations (``ixdl.sl.waitcnt``).
    Name the hardware counters to wait on, matching the ``union WaitCount`` fields
    (``vm``, ``sm``, ``lm``, ``g2s``, ``s2g``, ``mba``, ``mbt``); each value is the
    threshold to wait down to. The GEMM pipeline uses ``sl_waitmem(g2s=stages - 1, lm=0)``.
    """
    return _llvm.call_intrinsic(None, "llvm.bi.sl.waitcnt", [_const_i64(_waitcnt_value(**counters))], [], [])


def sl_barrier_alu():
    """ALU-only CTA sync (``llvm.bi.sl.barrier.alu`` / ``__syncthreads_alu``).

    ``fx.gpu.barrier`` lowers to ``sl_barrier``, which embeds a memory wait and
    drains outstanding G2S. This intrinsic does not: on MR it becomes a pipebar
    pair so a 2-stage SME pipeline can keep the next G2S in flight.
    """
    return _llvm.call_intrinsic(None, "llvm.bi.sl.barrier.alu", [], [], [])


def sl_pipebar_arrive(value=0):
    """MR pipeline-barrier arrive / report (``llvm.bi.pipebar.req``).

    Requires ``pipe-bar`` (ivcore11 / MR). Do not use on CQ kernels -- use
    :func:`nbarrier_reach` / :func:`nbarrier_sync` instead.
    """
    return _llvm.call_intrinsic(None, "llvm.bi.pipebar.req", [_const_i32(value)], [], [])


def sl_pipebar_wait(value=0):
    """MR pipeline-barrier wait (``llvm.bi.pipebar.wait``).

    Requires ``pipe-bar`` (ivcore11 / MR). Do not use on CQ kernels -- use
    :func:`nbarrier_wait` / :func:`nbarrier_sync` instead.
    """
    return _llvm.call_intrinsic(None, "llvm.bi.pipebar.wait", [_const_i32(value)], [], [])


# --- Scheme C: named barrier (CQ / named-bar; sync needs named-bar-sync) ---


def nbarrier_reach(tag_id, warp_num):
    """Arrive at a named barrier without stalling (``llvm.bi.nbarrier.reach``).

    ``tag_id`` is the barrier tag; ``warp_num`` is the expected participant warp
    count. Requires ``named-bar`` (CQ). On CQ double-buffer pipelines this
    is the arrive half of the MR ``sl_pipebar_arrive`` / ``sl_pipebar_wait`` pair.
    """
    return _llvm.call_intrinsic(
        None,
        "llvm.bi.nbarrier.reach",
        [_i32_arg(tag_id), _i32_arg(warp_num)],
        [],
        [],
    )


def nbarrier_wait(tag_id, warp_num):
    """Wait on a named barrier (``llvm.bi.nbarrier.wait``).

    Blocks until ``warp_num`` participants have reached barrier ``tag_id``.
    Requires ``named-bar`` (CQ). Pair with :func:`nbarrier_reach`, or use
    :func:`nbarrier_sync` for a combined arrive+wait on CQ.
    """
    return _llvm.call_intrinsic(
        None,
        "llvm.bi.nbarrier.wait",
        [_i32_arg(tag_id), _i32_arg(warp_num)],
        [],
        [],
    )


def nbarrier_sync(tag_id, warp_num):
    """Combined arrive+wait on a named barrier (``llvm.bi.nbarrier.sync``).

    Requires ``named-bar-sync`` (CQ). Prefer this when a single sync point is
    enough; use :func:`nbarrier_reach` + :func:`nbarrier_wait` when arrive and
    wait must be separated around other work (CQ analogue of pipebar split).
    """
    return _llvm.call_intrinsic(
        None,
        "llvm.bi.nbarrier.sync",
        [_i32_arg(tag_id), _i32_arg(warp_num)],
        [],
        [],
    )
