# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Vector add on Iluvatar MR-V100 / MR-V50 via the ``ixdl`` backend.

Key differences vs ``01-vectorAdd.py``:

* Pure ``UniversalCopy32b`` path — no AMDGPU buffer descriptors / buffer ops.
* Forces ``FLYDSL_COMPILE_BACKEND=ixdl`` and ``FLYDSL_RUNTIME_KIND=ixdl``
  before importing ``flydsl`` so the JIT picks the right pipeline.
* No CUDA Graph capture path: MR-V100 hangs when multiple programs share a
  physical card; one card / one program at a time is all we run here.

Env knobs (optional):
* ``CUDA_VISIBLE_DEVICES`` — selects a single card (Iluvatar honors it).
* ``FLYDSL_GPU_ARCH`` — overrides the default ``ivcore11`` chip name.
"""

# NOTE: do NOT add ``from __future__ import annotations`` here — @flyc.kernel
# inspects live annotation objects to detect ``fx.Constexpr[int]`` params,
# and stringified annotations (PEP 563) would route them into the runtime
# arg path instead, producing "Cannot derive IR types from <int>".

import os
import shutil
import subprocess
import sys

os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "ixdl")
os.environ.setdefault("FLYDSL_RUNTIME_KIND", "ixdl")

import torch  # noqa: E402

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402


def _warn_if_card_busy() -> None:
    """Best-effort: print a warning if the target card already has a process.

    Iluvatar MR-V100 hangs when two programs share one card. We check via
    ``ixsmi`` and print a note; we do not abort because ``ixsmi`` output
    formatting has drifted across releases and we don't want false negatives
    to block runs.
    """
    if shutil.which("ixsmi") is None:
        return
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    target = cvd.split(",")[0].strip() if cvd else "0"
    try:
        out = subprocess.check_output(
            ["ixsmi"], text=True, timeout=5, stderr=subprocess.DEVNULL
        )
    except Exception:
        return
    # Heuristic: any "python" / "MiB" row under a GPU header means busy.
    busy = False
    for line in out.splitlines():
        ls = line.strip()
        if ls.startswith("|") and ("MiB" in ls) and any(c.isdigit() for c in ls):
            if "python" in ls or "MiB /" in ls and "0MiB" not in ls.split("/")[0]:
                busy = True
                break
    if busy:
        print(
            f"[WARN] ixsmi suggests GPU {target} may already be in use; "
            f"running two programs on one Iluvatar card will hang. "
            f"Set CUDA_VISIBLE_DEVICES to a free card if possible.",
            file=sys.stderr,
        )


@flyc.kernel
def vectorAddKernel(
    A: fx.Tensor,
    B: fx.Tensor,
    C: fx.Tensor,
    block_dim: fx.Constexpr[int],
):
    bid = fx.block_idx.x
    tid = fx.thread_idx.x

    tA = fx.logical_divide(A, fx.make_layout(block_dim, 1))
    tB = fx.logical_divide(B, fx.make_layout(block_dim, 1))
    tC = fx.logical_divide(C, fx.make_layout(block_dim, 1))

    tA = fx.slice(tA, (None, bid))
    tB = fx.slice(tB, (None, bid))
    tC = fx.slice(tC, (None, bid))
    tA = fx.logical_divide(tA, fx.make_layout(1, 1))
    tB = fx.logical_divide(tB, fx.make_layout(1, 1))
    tC = fx.logical_divide(tC, fx.make_layout(1, 1))

    RABMemRefTy = fx.MemRefType.get(fx.T.f32(), fx.LayoutType.get(1, 1), fx.AddressSpace.Register)

    copyAtom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)

    rA = fx.memref_alloca(RABMemRefTy, fx.make_layout(1, 1))
    rB = fx.memref_alloca(RABMemRefTy, fx.make_layout(1, 1))
    rC = fx.memref_alloca(RABMemRefTy, fx.make_layout(1, 1))

    fx.copy_atom_call(copyAtom, fx.slice(tA, (None, tid)), rA)
    fx.copy_atom_call(copyAtom, fx.slice(tB, (None, tid)), rB)

    vC = fx.arith.addf(fx.memref_load_vec(rA), fx.memref_load_vec(rB))
    fx.memref_store_vec(vC, rC)

    fx.copy_atom_call(copyAtom, rC, fx.slice(tC, (None, tid)))


@flyc.jit
def vectorAdd(
    A: fx.Tensor,
    B: fx.Tensor,
    C,
    n: fx.Int32,
    const_n: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    block_dim = 64  # == warp_size on MR-V100 / MR-V50 (ivcore11)
    grid_x = (n + block_dim - 1) // block_dim
    vectorAddKernel(A, B, C, block_dim).launch(
        grid=(grid_x, 1, 1), block=[block_dim, 1, 1], stream=stream
    )


def run_eager() -> bool:
    n = 128
    A = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
    B = torch.randint(0, 10, (n,), dtype=torch.float32).cuda()
    C = torch.zeros(n, dtype=torch.float32).cuda()
    tA = flyc.from_dlpack(A).mark_layout_dynamic(leading_dim=0, divisibility=4)
    vectorAdd(tA, B, C, n, n + 1, stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    ok = torch.allclose(C, A + B)
    print(f"[ixdl] eager vectorAdd correct: {ok}")
    if not ok:
        print("  A[:8]:", A[:8])
        print("  B[:8]:", B[:8])
        print("  C[:8]:", C[:8])
    return ok


if __name__ == "__main__":
    _warn_if_card_busy()
    sys.exit(0 if run_eager() else 1)
