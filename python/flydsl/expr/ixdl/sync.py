# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar async-copy / pipeline synchronization primitives.

Pick the barrier scheme that matches the chip generation — do **not** mix them
in one kernel (ixcc chipset gates reject the wrong family when ``ixdl.chip`` /
``#ixdl.target`` is set):

====================== ========================================================
Generation             Sync for multi-stage / producer-consumer buffering
====================== ========================================================
**MR (ivcore11)**      Scheme B — ``sl_waitmem`` + ``sl_pipebar_arrive`` /
                       ``sl_pipebar_wait`` (FeaturePipeBar). No named-bar.
**CQ (ivcore30)**      Scheme C — ``nbarrier_reach`` / ``nbarrier_wait`` /
                       ``nbarrier_sync`` (FeatureNamedBar / NamedBarSync).
                       **No pipebar** on CQ.
**QS / BZ**            Named-bar (QS: wait/reach; BZ: + sync). No pipebar.
====================== ========================================================

Scheme A (``cp_async_commit_group`` / ``cp_async_wait_group``) fences async-copy
groups and is orthogonal to the barrier choice above. CTA-wide
``fx.gpu.barrier`` remains available on all chipsets.

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


# --- Scheme B: MR multi-stage pipeline (sl_waitmem + pipebar) ---
# MR only (ivcore11). CQ kernels must use Scheme C instead.

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


def sl_pipebar_arrive(value=0):
    """MR pipeline-barrier arrive / report (``llvm.bi.pipebar.req``).

    **MR only.** Do not call from CQ kernels — ixcc rejects pipebar on CQ/QS/BZ
    when the chip target is set. CQ double-buffering must use :func:`nbarrier_reach`
    / :func:`nbarrier_wait` / :func:`nbarrier_sync`.
    """
    return _llvm.call_intrinsic(None, "llvm.bi.pipebar.req", [_const_i32(value)], [], [])


def sl_pipebar_wait(value=0):
    """MR pipeline-barrier wait (``llvm.bi.pipebar.wait``).

    **MR only.** See :func:`sl_pipebar_arrive`.
    """
    return _llvm.call_intrinsic(None, "llvm.bi.pipebar.wait", [_const_i32(value)], [], [])


# --- Scheme C: CQ / QS / BZ named barrier (nbarrier) ---
# CQ double-buffer / producer-consumer sync. Never emit pipebar on this path.


def nbarrier_wait(bar_id=0, cnt=0):
    """Wait on a named barrier (``llvm.bi.nbarrier.wait`` / ``ixdl.nbarrier.wait``).

    Blocks until named barrier ``bar_id`` has been reached by ``cnt`` participants
    (participant warp count; ``0`` matches CQ reference kernels' CTA-default
    encoding). Requires FeatureNamedBar (QS/CQ/BZ). **Not available on MR.**
    """
    return _llvm.call_intrinsic(
        None,
        "llvm.bi.nbarrier.wait",
        [_const_i32(bar_id), _const_i32(cnt)],
        [],
        [],
    )


def nbarrier_reach(bar_id=0, cnt=0):
    """Arrive at a named barrier (``llvm.bi.nbarrier.reach`` / ``ixdl.nbarrier.reach``).

    Signals arrival at named barrier ``bar_id`` for ``cnt`` participants.
    Requires FeatureNamedBar (QS/CQ/BZ). **Not available on MR.** Pair with
    :func:`nbarrier_wait`, or use :func:`nbarrier_sync` for a combined arrive+wait
    on CQ/BZ.
    """
    return _llvm.call_intrinsic(
        None,
        "llvm.bi.nbarrier.reach",
        [_const_i32(bar_id), _const_i32(cnt)],
        [],
        [],
    )


def nbarrier_sync(bar_id=0, cnt=0):
    """Combined arrive+wait on a named barrier (``llvm.bi.nbarrier.sync``).

    Requires FeatureNamedBarSync (CQ/BZ). **Not available on MR or QS.** Prefer
    this for CQ double-buffer phase sync when split reach/wait is unnecessary.
    """
    return _llvm.call_intrinsic(
        None,
        "llvm.bi.nbarrier.sync",
        [_const_i32(bar_id), _const_i32(cnt)],
        [],
        [],
    )
