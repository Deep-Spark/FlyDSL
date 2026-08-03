#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Tiled copy teaching example: CQ SmexMtx G2S + loadn S2R (ivcore30).

Minimal single-warp path for debugging layout / Bypass swizzle / ``loadn`` before
CQ MMA/GEMM. Companion to ``examples/02-tiledCopy-iluvatar-mr.py``. Exhaustive
LegacySme enhanced-SME shape compile/check lives in
``examples/02-tiledCopy-iluvatar-cq-async-shapes.py``.

Phases (CuTe-style TiledCopy on the S2R side):

1. Global -> shared (G2S): warp-collective SmexMtx async copy
   ``llvm.bi.smex.loadn.16x1b64.mtx`` moves one ``16x32`` f16 tile (64 B/row).
   This is the SME mtx G2S that pairs with ``CQMtxLoadn`` (SmexMtx / Bypass);
   do **not** mix with LegacySme ``CQAsyncCp`` / byte swizzle on the same buffer.
2. Sync: ``cp_async_commit/wait_group`` (async-copy fence) + ``nbarrier_sync``
   (CQ Scheme C) + CTA ``barrier``.
3. Shared -> register -> global (S2R): ``CQMtxLoadn`` (loadn16, Row, b16, x2)
   fills the base ``CQMma`` A fragment for the first K-slice (EmPart=0);
   ``UniversalCopy`` dumps the fragment via ``make_tiled_copy_A`` for host check.

Usage::

    export FLYDSL_COMPILE_BACKEND=iluvatar
    export FLYDSL_RUNTIME_KIND=iluvatar
    export ARCH=ivcore30
    # Reserved CQ GPU for this workspace:
    export CUDA_VISIBLE_DEVICES=15

    # Device correctness (small 16x16 A-fragment vs host)
    python examples/02-tiledCopy-iluvatar-cq.py --check

    # Compile without launching (CI / no GPU)
    COMPILE_ONLY=1 python examples/02-tiledCopy-iluvatar-cq.py --check
"""

import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _configure_env() -> None:
    os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "iluvatar")
    os.environ.setdefault("FLYDSL_RUNTIME_KIND", "iluvatar")
    os.environ.setdefault("ARCH", "ivcore30")
    os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")


def _compile_only() -> bool:
    return os.environ.get("COMPILE_ONLY", "").lower() in ("1", "true", "yes", "on")


_configure_env()

import torch  # noqa: E402

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
import flydsl.expr.ixdl as ixdl  # noqa: E402
from flydsl._mlir import ir  # noqa: E402
from flydsl._mlir.dialects import llvm as _llvm  # noqa: E402
from flydsl.expr import arith as _arith  # noqa: E402
from flydsl.expr.primitive import ptrtoint  # noqa: E402
from flydsl.expr.typing import T  # noqa: E402
from kernels.gemm.iluvatar.common import ATOM_K_B16, ATOM_M, ATOM_N, WARP_SIZE  # noqa: E402

# loadn16 G2S footprint: 16 rows x 64 B = 16x32 f16. S2R checks the first
# CQMma A K-slice (16x16) at EmPart=0.
TILE_M = ATOM_M
TILE_K = 32
ATOM_K = ATOM_K_B16
SMEM_BYTES = TILE_M * TILE_K * 2


def _llvm_ptr(ptr, addrspace: int):
    """Materialize ``!fly.ptr`` as ``!llvm.ptr<AS>`` for SmexMtx G2S (interim)."""
    addr = _arith.unwrap(ptrtoint(ptr))
    return _llvm.inttoptr(ir.Type.parse(f"!llvm.ptr<{int(addrspace)}>"), addr)


def _const_i32(v):
    return _arith.unwrap(_arith.constant(int(v), type=T.i32))


def _const_i64(v):
    return _arith.unwrap(_arith.constant(int(v), type=T.i64))


@flyc.kernel(known_block_size=[WARP_SIZE, 1, 1])
def copy_kernel_iluvatar_cq(A: fx.Tensor, Out: fx.Tensor):
    lane_id = fx.Int32(fx.lane_id)

    smem = fx.make_view(
        fx.get_dyn_shared(fx.Float16),
        fx.make_layout(TILE_M * TILE_K, 1),
    )

    # Phase 1 (G2S): SmexMtx async copy (pairs with CQMtxLoadn / Bypass).
    # Interim: emit the IXDL intrinsic until a FlyIXDL CQAsyncCpMtx CopyAtom lands.
    s_ptr = _llvm_ptr(fx.get_iter(smem), 3)
    g_ptr = _llvm_ptr(fx.get_iter(A), 1)
    _llvm.call_intrinsic(
        None,
        "llvm.bi.smex.loadn.16x1b64.mtx",
        [
            s_ptr,
            g_ptr,
            _const_i32(TILE_K * 2),
            _const_i32(0),
            _const_i64(-1),
            _const_i32(-1),
            _const_i32(1),
            _const_i32(1),
        ],
        [],
        [],
    )
    ixdl.cp_async_commit_group()
    ixdl.cp_async_wait_group(0)
    ixdl.nbarrier_sync(0, 0)
    fx.gpu.barrier()

    # Phase 2 (S2R): CQMtxLoadn into base CQMma A fragment, then dump to Out.
    mma_atom = fx.make_mma_atom(
        ixdl.CQMma(ATOM_M, ATOM_N, ATOM_K, fx.Float16, fx.Float16, fx.Float32)
    )
    tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (1, 1, 1)))
    thr_mma = tiled_mma.thr_slice(lane_id)

    loadn_atom = fx.make_copy_atom(
        ixdl.CQMtxLoadn(ixdl.CQMtxPattern.Loadn16, ixdl.CQMtxDir.Row, 16, x2=True),
        fx.Float16,
    )
    tiled_copy_a = fx.make_tiled_copy_A(loadn_atom, tiled_mma)
    thr_copy_a = tiled_copy_a.get_slice(lane_id)

    smem_a = fx.make_view(
        fx.get_iter(smem),
        fx.make_layout((ATOM_M, ATOM_K), (1, ATOM_M)),
    )
    frag_a = thr_mma.make_fragment_A(smem_a)
    fx.copy(loadn_atom, thr_copy_a.partition_S(smem_a), thr_copy_a.retile(frag_a))

    scalar_atom = fx.make_copy_atom(fx.UniversalCopy16b(), fx.Float16)
    tiled_dump = fx.make_tiled_copy_A(scalar_atom, tiled_mma)
    thr_dump = tiled_dump.get_slice(lane_id)
    dst = fx.make_view(
        fx.get_iter(Out),
        fx.make_layout((ATOM_K, ATOM_M), (1, ATOM_K)),
    )
    fx.copy(scalar_atom, thr_dump.retile(frag_a), thr_dump.partition_D(dst))


@flyc.jit
def tiledCopyIluvatarCQ(A: fx.Tensor, Out: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
    copy_kernel_iluvatar_cq(A, Out).launch(
        grid=(1, 1, 1),
        block=(WARP_SIZE, 1, 1),
        smem=SMEM_BYTES,
        stream=stream,
    )


def _run_check() -> None:
    if _compile_only():
        A = torch.zeros(TILE_M, TILE_K, dtype=torch.float16)
        Out = torch.zeros(ATOM_M, ATOM_K, dtype=torch.float16)
        tiledCopyIluvatarCQ(A, Out)
        print("COMPILED cq SmexMtx G2S + loadn S2R (16x16 A fragment)")
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA-compatible CQ device is not available "
            "(set CUDA_VISIBLE_DEVICES to the reserved CQ GPU, e.g. 15)"
        )

    A = (
        torch.arange(TILE_M * TILE_K, dtype=torch.int32)
        .reshape(TILE_M, TILE_K)
        .cuda()
        .to(torch.float16)
    )
    Out = torch.zeros(ATOM_M, ATOM_K, dtype=torch.float16, device="cuda")
    tiledCopyIluvatarCQ(A, Out, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    # make_tiled_copy_A partition_D is K-major; host compares after .T
    got = Out.T.contiguous()
    expected = A[:, :ATOM_K].contiguous()
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    print(
        f"PASS dtype=f16 tile={TILE_M}x{TILE_K} "
        f"checked_A_fragment={ATOM_M}x{ATOM_K}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--check",
        action="store_true",
        help="run small-tile SmexMtx G2S + loadn S2R correctness check",
    )
    args = p.parse_args(argv)
    if not args.check:
        p.print_help()
        return 0
    _run_check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
