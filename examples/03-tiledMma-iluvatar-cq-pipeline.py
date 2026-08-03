#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""CQ tiled MMA teaching / smoke example (ivcore30): G2S + loadn S2R + MMA + store.

Minimal single-warp path that wires the atoms a future ``cq/hgemm.py`` pipeline
will reuse — **no** double-buffer / multi-stage scheduling (that belongs in the
HGEMM PR). Companion pieces:

* ``examples/02-tiledCopy-iluvatar-cq.py`` — SmexMtx G2S + loadn S2R only (A frag dump)
* ``examples/03-tiledMma-iluvatar-cq-mma-tile.py`` — CQMma fragment fill only (no G2S/S2R)
* ``examples/03-tiledMma-iluvatar-mr-pipeline-hgemm.py`` — production MR HGEMM harness

Phases (single K-tile, base CQMma):

1. Global -> shared (G2S): warp-collective SmexMtx async copy
   ``llvm.bi.smex.loadn.16x1b64.mtx`` for A and B (64 B/row). Pairs with
   ``CQMtxLoadn`` (SmexMtx / Bypass); do **not** mix LegacySme ``CQAsyncCp`` on
   the same buffer.
2. Sync: ``cp_async_commit/wait_group`` + ``nbarrier_sync`` (CQ Scheme C) + CTA
   ``barrier``.
3. Shared -> register (S2R): ``CQMtxLoadn`` Row (A) / Col (B) via
   ``make_tiled_copy_A/B`` into base ``CQMma`` fragments (EmPart=0 / first K-slice).
4. MMA: ``fx.gemm`` on ``ixdl.CQMma``.
5. Epilogue: ``UniversalCopy`` + ``make_tiled_copy_C`` stores the C fragment.

Usage::

    export FLYDSL_COMPILE_BACKEND=iluvatar
    export FLYDSL_RUNTIME_KIND=iluvatar
    export ARCH=ivcore30
    # Reserved CQ GPU for this workspace:
    export CUDA_VISIBLE_DEVICES=15

    # Device correctness (fp16 base tile 16x16x16)
    python examples/03-tiledMma-iluvatar-cq-pipeline.py --check

    # Optional s8 base tile 16x16x32
    python examples/03-tiledMma-iluvatar-cq-pipeline.py --check --dtype s8

    # Compile without launching (CI / no GPU)
    COMPILE_ONLY=1 python examples/03-tiledMma-iluvatar-cq-pipeline.py --check
"""

from __future__ import annotations

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
from kernels.gemm.iluvatar.common import (  # noqa: E402
    ATOM_K_B8,
    ATOM_K_B16,
    ATOM_M,
    ATOM_N,
    WARP_SIZE,
)

# loadn16 G2S footprint: 16 rows x 64 B/row. f16 -> 32 elems/row; s8 -> 64.
# S2R/MMA consume the first K-slice (EmPart=0): f16 K=16, s8 K=32.
TILE_M = ATOM_M
TILE_N = ATOM_N
G2S_BYTES_PER_ROW = 64


def _llvm_ptr(ptr, addrspace: int):
    """Materialize ``!fly.ptr`` as ``!llvm.ptr<AS>`` for SmexMtx G2S (interim)."""
    addr = _arith.unwrap(ptrtoint(ptr))
    return _llvm.inttoptr(ir.Type.parse(f"!llvm.ptr<{int(addrspace)}>"), addr)


def _const_i32(v):
    return _arith.unwrap(_arith.constant(int(v), type=T.i32))


def _const_i64(v):
    return _arith.unwrap(_arith.constant(int(v), type=T.i64))


def _smex_g2s_mtx(dst_shared, src_global, *, row_stride_elems: int, elem_bytes: int):
    """Issue one warp-collective SmexMtx G2S (``smex.loadn.16x1b64.mtx``)."""
    _llvm.call_intrinsic(
        None,
        "llvm.bi.smex.loadn.16x1b64.mtx",
        [
            _llvm_ptr(dst_shared, 3),
            _llvm_ptr(src_global, 1),
            _const_i32(row_stride_elems * elem_bytes),
            _const_i32(0),
            _const_i64(-1),
            _const_i32(-1),
            _const_i32(1),
            _const_i32(1),
        ],
        [],
        [],
    )


def _compile_cq_tiled_mma(*, dtype: str):
    """Build the single-warp CQ G2S+S2R+MMA+store launcher for ``f16`` or ``s8``."""
    if dtype == "s8":
        elem_dtype = fx.Int8
        acc_dtype = fx.Int32
        torch_ab = torch.int8
        torch_c = torch.int32
        atom_k = ATOM_K_B8
        bit_width = 8
        elem_bytes = 1
        copy_c_factory = fx.UniversalCopy32b
    elif dtype == "f16":
        elem_dtype = fx.Float16
        acc_dtype = fx.Float32
        torch_ab = torch.float16
        torch_c = torch.float32
        atom_k = ATOM_K_B16
        bit_width = 16
        elem_bytes = 2
        copy_c_factory = fx.UniversalCopy32b
    else:
        raise ValueError(f"unsupported dtype={dtype!r}; expected 'f16' or 's8'")

    g2s_k = G2S_BYTES_PER_ROW // elem_bytes
    smem_a_elems = TILE_M * g2s_k
    smem_b_elems = TILE_N * g2s_k
    smem_bytes = (smem_a_elems + smem_b_elems) * elem_bytes

    @flyc.kernel(known_block_size=[WARP_SIZE, 1, 1])
    def mma_kernel_iluvatar_cq(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        lane_id = fx.Int32(fx.lane_id)

        smem = fx.make_view(
            fx.get_dyn_shared(elem_dtype),
            fx.make_layout(smem_a_elems + smem_b_elems, 1),
        )
        smem_a_base = fx.get_iter(smem)
        smem_b_base = fx.add_offset(smem_a_base, fx.Int32(smem_a_elems))

        # Phase 1 (G2S): SmexMtx async copy for A and B (pairs with CQMtxLoadn).
        _smex_g2s_mtx(
            smem_a_base,
            fx.get_iter(A),
            row_stride_elems=g2s_k,
            elem_bytes=elem_bytes,
        )
        _smex_g2s_mtx(
            smem_b_base,
            fx.get_iter(B),
            row_stride_elems=g2s_k,
            elem_bytes=elem_bytes,
        )
        ixdl.cp_async_commit_group()
        ixdl.cp_async_wait_group(0)
        ixdl.nbarrier_sync(0, 0)
        fx.gpu.barrier()

        # Phase 2 (S2R): loadn into base CQMma A/B fragments (first K-slice).
        mma_atom = fx.make_mma_atom(
            ixdl.CQMma(ATOM_M, ATOM_N, atom_k, elem_dtype, elem_dtype, acc_dtype)
        )
        tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (1, 1, 1)))
        thr_mma = tiled_mma.thr_slice(lane_id)

        loadn_a = fx.make_copy_atom(
            ixdl.CQMtxLoadn(ixdl.CQMtxPattern.Loadn16, ixdl.CQMtxDir.Row, bit_width, x2=True),
            elem_dtype,
        )
        loadn_b = fx.make_copy_atom(
            ixdl.CQMtxLoadn(ixdl.CQMtxPattern.Loadn16, ixdl.CQMtxDir.Col, bit_width, x2=True),
            elem_dtype,
        )
        thr_copy_a = fx.make_tiled_copy_A(loadn_a, tiled_mma).get_slice(lane_id)
        thr_copy_b = fx.make_tiled_copy_B(loadn_b, tiled_mma).get_slice(lane_id)

        smem_a = fx.make_view(smem_a_base, fx.make_layout((ATOM_M, atom_k), (1, ATOM_M)))
        smem_b = fx.make_view(smem_b_base, fx.make_layout((ATOM_N, atom_k), (1, ATOM_N)))

        frag_a = thr_mma.make_fragment_A(smem_a)
        frag_b = thr_mma.make_fragment_B(smem_b)
        fx.copy(loadn_a, thr_copy_a.partition_S(smem_a), thr_copy_a.retile(frag_a))
        fx.copy(loadn_b, thr_copy_b.partition_S(smem_b), thr_copy_b.retile(frag_b))

        # Phase 3 (MMA) + Phase 4 (epilogue store).
        gC = fx.make_view(fx.get_iter(C), fx.make_layout((ATOM_M, ATOM_N), (ATOM_N, 1)))
        frag_c = thr_mma.make_fragment_C(gC)
        frag_c.fill(0)
        fx.gemm(mma_atom, frag_c, frag_a, frag_b, frag_c)

        copy_c = fx.make_copy_atom(copy_c_factory(), acc_dtype)
        thr_copy_c = fx.make_tiled_copy_C(copy_c, tiled_mma).get_slice(lane_id)
        fx.copy(copy_c, thr_copy_c.retile(frag_c), thr_copy_c.partition_D(gC), pred=None)

    @flyc.jit
    def launch(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor, stream: fx.Stream = fx.Stream(None)):
        mma_kernel_iluvatar_cq(A, B, C).launch(
            grid=(1, 1, 1),
            block=(WARP_SIZE, 1, 1),
            smem=smem_bytes,
            stream=stream,
        )

    launch.meta = {  # type: ignore[attr-defined]
        "dtype": dtype,
        "atom_m": ATOM_M,
        "atom_n": ATOM_N,
        "atom_k": atom_k,
        "g2s_k": g2s_k,
        "torch_ab": torch_ab,
        "torch_c": torch_c,
        "smem_bytes": smem_bytes,
    }
    return launch


def _pad_ab(src: torch.Tensor, *, g2s_k: int) -> torch.Tensor:
    """Pad K to the SmexMtx G2S row width (EmPart=0 uses ``[:, :atom_k]``)."""
    m, k = src.shape
    if k == g2s_k:
        return src.contiguous()
    out = torch.zeros(m, g2s_k, dtype=src.dtype, device=src.device)
    out[:, :k] = src
    return out.contiguous()


def _run_check(*, dtype: str) -> None:
    launch = _compile_cq_tiled_mma(dtype=dtype)
    meta = launch.meta
    atom_m, atom_n, atom_k = meta["atom_m"], meta["atom_n"], meta["atom_k"]
    g2s_k = meta["g2s_k"]
    torch_ab, torch_c = meta["torch_ab"], meta["torch_c"]

    if _compile_only():
        A = torch.zeros(atom_m, g2s_k, dtype=torch_ab)
        B = torch.zeros(atom_n, g2s_k, dtype=torch_ab)
        C = torch.zeros(atom_m, atom_n, dtype=torch_c)
        launch(A, B, C)
        print(
            f"COMPILED cq G2S+loadn S2R+MMA+store "
            f"dtype={dtype} atom={atom_m}x{atom_n}x{atom_k} g2s_k={g2s_k}"
        )
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA-compatible CQ device is not available "
            "(set CUDA_VISIBLE_DEVICES to the reserved CQ GPU, e.g. 15)"
        )

    torch.manual_seed(0)
    if dtype == "s8":
        A_tile = torch.randint(-3, 4, (atom_m, atom_k), dtype=torch.int32).to(torch.int8)
        B_tile = torch.randint(-3, 4, (atom_n, atom_k), dtype=torch.int32).to(torch.int8)
        expected = (A_tile.to(torch.int32) @ B_tile.to(torch.int32).T).to(torch.int32)
        rtol, atol = 0, 0
    else:
        A_tile = torch.randn(atom_m, atom_k, dtype=torch.float16)
        B_tile = torch.randn(atom_n, atom_k, dtype=torch.float16)
        expected = A_tile.to(torch.float32) @ B_tile.to(torch.float32).T
        rtol, atol = 2e-2, 2e-2

    A = _pad_ab(A_tile.cuda(), g2s_k=g2s_k)
    B = _pad_ab(B_tile.cuda(), g2s_k=g2s_k)
    C = torch.empty(atom_m, atom_n, dtype=torch_c, device="cuda")
    C.fill_(7777 if dtype == "s8" else 7777.0)

    launch(A, B, C, stream=torch.cuda.Stream())
    torch.cuda.synchronize()

    torch.testing.assert_close(C.cpu(), expected, rtol=rtol, atol=atol)
    print(
        f"PASS dtype={dtype} atom={atom_m}x{atom_n}x{atom_k} "
        f"g2s_k={g2s_k} shape=({atom_m},{atom_n})<-({atom_m},{atom_k})@({atom_n},{atom_k}).T"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--check",
        action="store_true",
        help="run single-tile G2S+S2R+MMA+store correctness check",
    )
    p.add_argument(
        "--dtype",
        default="f16",
        choices=("f16", "s8"),
        help="multiplicand dtype (default: f16)",
    )
    args = p.parse_args(argv)
    if not args.check:
        p.print_help()
        return 0
    _run_check(dtype=args.dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
