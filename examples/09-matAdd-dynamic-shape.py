"""2D matrix addition with **dynamic shapes** — Triton-ish parameter style.

Compared with 09-matAdd-iluvatar.py:
    - M, N are *runtime* values passed as fx.Int32 kernel args
    - Tensors are marked layout-dynamic so their shape/stride are stamped
      into the memref descriptor at launch time
    - The compiled module is reused for any (M, N) multiple of (BM, BN)

Why DynI32 wrapper?
    @flyc.jit's Python-level cache key is computed from arg *objects*
    (see jit_function.py::_get_type_signature). A bare Python ``int`` is
    signed as ``"int:<value>"`` — so calling with M=128 vs M=64 produces
    different keys and retriggers compilation.  Subclassing ``int`` and
    providing ``__cache_signature__`` collapses all values to a single
    key, giving true "compile-once / reuse-for-many-shapes" behavior.

Usage:
    ARCH=ivcore11 python3 examples/09-matAdd-dynamic-shape.py
    ARCH=gfx942   python3 examples/09-matAdd-dynamic-shape.py
"""

import time

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx

BM, BN = 8, 8


class DynI32(int):
    """Python ``int`` whose flydsl JIT cache-signature is shape-invariant.

    Without this, every unique (M, N) value pair creates a new cache key
    at the Python layer and forces a full recompile, even though the
    kernel signature already declares the arg as ``fx.Int32`` (dynamic).
    """

    def __cache_signature__(self) -> str:
        return "i32"


@flyc.kernel
def matAddKernel(
    A: fx.Tensor,           # 2D tensor, shape=(M_dyn, N_dyn)
    B: fx.Tensor,
    C: fx.Tensor,
    M: fx.Int32,            # dynamic runtime size
    N: fx.Int32,
):
    tid = fx.thread_idx.x
    bid_m = fx.block_idx.x
    bid_n = fx.block_idx.y

    tile = fx.make_tile(BM, BN)
    gA = fx.zipped_divide(A, tile)
    gB = fx.zipped_divide(B, tile)
    gC = fx.zipped_divide(C, tile)

    bA = fx.slice(gA, (None, (bid_m, bid_n)))
    bB = fx.slice(gB, (None, (bid_m, bid_n)))
    bC = fx.slice(gC, (None, (bid_m, bid_n)))

    thr_layout = fx.make_layout((BM, BN), (BN, 1))
    val_layout = fx.make_layout((1, 1), (1, 1))
    layout_thr_val = fx.raked_product(thr_layout, val_layout)
    tile_mn = fx.make_tile(BM, BN)

    copy_atom = fx.make_copy_atom(fx.UniversalCopy(32), fx.Float32)
    tiled_copy = fx.make_tiled_copy(copy_atom, layout_thr_val, tile_mn)
    thr_copy = tiled_copy.get_slice(tid)

    part_A = thr_copy.partition_S(bA)
    part_B = thr_copy.partition_S(bB)
    part_C = thr_copy.partition_D(bC)

    rA = fx.make_fragment_like(part_A)
    rB = fx.make_fragment_like(part_B)
    rC = fx.make_fragment_like(part_C)

    fx.copy(copy_atom, part_A, rA)
    fx.copy(copy_atom, part_B, rB)

    vA = fx.memref_load_vec(rA)
    vB = fx.memref_load_vec(rB)
    vC = fx.arith.addf(vA, vB)
    fx.memref_store_vec(vC, rC)

    fx.copy(copy_atom, rC, part_C)


@flyc.jit
def matAdd(
    A: fx.Tensor,
    B: fx.Tensor,
    C,
    M: fx.Int32,
    N: fx.Int32,
    stream: fx.Stream = fx.Stream(None),
):
    grid_m = (M + BM - 1) // BM
    grid_n = (N + BN - 1) // BN
    matAddKernel(A, B, C, M, N).launch(
        grid=(grid_m, grid_n, 1),
        block=(BM * BN, 1, 1),
        stream=stream,
    )


def run_once(M, N, label=""):
    A = torch.randn(M, N, dtype=torch.float32).cuda()
    B = torch.randn(M, N, dtype=torch.float32).cuda()
    C = torch.zeros(M, N, dtype=torch.float32).cuda()

    tA = flyc.from_dlpack(A).mark_layout_dynamic(leading_dim=1, divisibility=4)
    tB = flyc.from_dlpack(B).mark_layout_dynamic(leading_dim=1, divisibility=4)
    tC = flyc.from_dlpack(C).mark_layout_dynamic(leading_dim=1, divisibility=4)

    t0 = time.perf_counter()
    matAdd(tA, tB, tC, DynI32(M), DynI32(N), stream=torch.cuda.Stream())
    torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) * 1000.0

    ok = torch.allclose(C, A + B, atol=1e-5)
    tag = f"[{label}] " if label else ""
    print(f"  {tag}M={M:5d}  N={N:5d}  ok={ok}  elapsed={dt_ms:8.2f} ms")
    return ok


if __name__ == "__main__":
    torch.manual_seed(0)
    print("=" * 60)
    print("2D matAdd — compile-once / reuse-for-many-shapes")
    print("=" * 60)
    r1 = run_once(128, 128, "call-1 (cold compile)")
    r2 = run_once(128, 128, "call-2 (same shape  )")
    r3 = run_once(64,  256, "call-3 (new  shape 1)")
    r4 = run_once(256,  64, "call-4 (new  shape 2)")
    r5 = run_once(128, 128, "call-5 (back to #1  )")
    print()
    print(f"All passed: {all([r1, r2, r3, r4, r5])}")
