# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""bf16 GEMM benchmark for Iluvatar MR-V100 / MR-V50 via the ``ixdl`` backend.

Extends ``03-tiledMma-ixdl.py`` (single-tile correctness sample) into a
full multi-block GEMM:

* 2-D grid over (M, N) tiles, one block per output tile.
* Python-unrolled K-loop inside the kernel, accumulating into a single
  32x32 fragment of ``frag_C``.
* Path-2 timing: compile once with ``flyc.compile``, then run a tight
  ``for`` loop with CUDA events. Reference matmul is computed in fp32
  because PyTorch's CUDA bf16 matmul is not a reliable baseline on IX11.

Run constraint (same as 03-tiledMma-ixdl.py):
    Iluvatar cards hang if two programs share a device. Use
    ``CUDA_VISIBLE_DEVICES`` to pin.

Typical invocations::

    # Correctness only (small shape, ~sub-second):
    python examples/05-gemm-ixdl-bf16.py --check --shape 128 128 128

    # Perf run (default 512x512x512, 20 warmup + 100 iters):
    python examples/05-gemm-ixdl-bf16.py --shape 512 512 512

    # Sweep:
    for M in 256 512 1024; do
        python examples/05-gemm-ixdl-bf16.py --shape $M $M $M
    done
"""

# NOTE: do NOT add ``from __future__ import annotations`` here (same
# Constexpr-introspection reason as 01-vectorAdd-ixdl.py / 03-tiledMma-ixdl.py).

import argparse
import os
import shutil
import subprocess
import sys
import time

os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "ixdl")
os.environ.setdefault("FLYDSL_RUNTIME_KIND", "ixdl")

import torch  # noqa: E402

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402


# ---- Kernel tile configuration ---------------------------------------------
# Must stay aligned with the ivcore11 bf16 MMAD atom (16x16x16) and the
# 2x2x1 tiled_mma we pick below -> each block writes a 32x32 output tile.
BM = 32
BN = 32
BK = 16


def _warn_if_card_busy() -> None:
    if shutil.which("ixsmi") is None:
        return
    try:
        out = subprocess.check_output(
            ["ixsmi"], text=True, timeout=5, stderr=subprocess.DEVNULL
        )
    except Exception:
        return
    for line in out.splitlines():
        ls = line.strip()
        if ls.startswith("|") and ("MiB" in ls) and any(c.isdigit() for c in ls):
            if "python" in ls or ("MiB /" in ls and "0MiB" not in ls.split("/")[0]):
                print(
                    "[WARN] ixsmi suggests this GPU may already be busy; "
                    "running two programs on one Iluvatar card will hang. "
                    "Consider setting CUDA_VISIBLE_DEVICES.",
                    file=sys.stderr,
                )
                return


def _build_kernel(M, N, K):
    """Build a closed-over @flyc.kernel for a fixed (M,N,K) shape.

    ``K`` is baked in because the Python ``for`` loop below is trace-time
    unrolled (same pattern as examples/04-preshuffle_gemm.py). Changing
    shape means re-tracing, which JitFunction's cache keys handle
    automatically if we re-enter via @flyc.jit.
    """
    k_tiles = K // BK
    assert K % BK == 0, f"K={K} must be divisible by BK={BK}"
    assert M % BM == 0, f"M={M} must be divisible by BM={BM}"
    assert N % BN == 0, f"N={N} must be divisible by BN={BN}"

    @flyc.kernel
    def gemm_kernel(A: fx.Tensor, B: fx.Tensor, C: fx.Tensor):
        tid = fx.thread_idx.x
        bid_x, bid_y, _ = fx.block_idx

        # flat_divide(T, (tile_m, tile_k)) -> view of shape
        # (tile_m, tile_k, num_m_tiles, num_k_tiles). We slice the outer
        # tile axes with bid_{x,y} to produce per-block views.
        #
        # We must use ``fx.slice`` (which lowers via ``fly.make_coord``
        # after unwrapping ``Integer`` wrappers) rather than the raw
        # ``memref[...]`` subscript: the latter dispatches to
        # ``fly.make_int_tuple`` which rejects index-typed SSA values
        # such as the GPU block-id. ``fx.slice`` is the pattern used in
        # the single-tile ``03-tiledMma-ixdl.py`` and handles this
        # correctly.
        gA = fx.slice(fx.flat_divide(A, (BM, BK)),
                      (None, None, bid_x, None))                        # (BM, BK, k)
        gB = fx.slice(fx.flat_divide(B, (BN, BK)),
                      (None, None, bid_y, None))                        # (BN, BK, k)
        gC = fx.slice(fx.flat_divide(C, (BM, BN)),
                      (None, None, bid_x, bid_y))                       # (BM, BN)

        # 16x16x16 bf16 MMAD, 2x2x1 tiling -> 32x32 output per block.
        mma_atom = fx.make_mma_atom(fx.ixdl.MMAD(16, 16, BK, fx.BFloat16))
        tiled_mma = fx.make_tiled_mma(
            mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0))
        )
        thr_mma = tiled_mma.thr_slice(tid)

        copy_atom_a = fx.make_copy_atom(fx.UniversalCopy16b(), fx.BFloat16)
        copy_atom_b = fx.make_copy_atom(fx.UniversalCopy32b(), fx.BFloat16)
        # bf16 inputs accumulate in f32.
        copy_atom_c = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)

        tiled_copy_A = fx.make_tiled_copy_A(copy_atom_a, tiled_mma)
        tiled_copy_B = fx.make_tiled_copy_B(copy_atom_b, tiled_mma)
        tiled_copy_C = fx.make_tiled_copy_C(copy_atom_c, tiled_mma)

        thr_copy_A = tiled_copy_A.get_slice(tid)
        thr_copy_B = tiled_copy_B.get_slice(tid)
        thr_copy_C = tiled_copy_C.get_slice(tid)

        # Accumulator lives across the K-reduction.
        frag_C = thr_mma.make_fragment_C(gC)
        frag_C.fill(0)

        # ---- K-loop (trace-time unrolled via range_constexpr) -------------
        # Plain Python ``range()`` is rewritten by @flyc.kernel's AST
        # pass into an ``scf.for`` with an ``index``-typed iv, and that
        # iv then feeds ``gA[None, None, iv]`` -> ``fly.make_int_tuple``
        # which only accepts i32/i64 operands. ``fx.range_constexpr``
        # keeps the loop at Python level, so ``k`` is a literal Python
        # int and every coord element here is static.
        for k in fx.range_constexpr(k_tiles):
            k_A = gA[None, None, k]                                     # (BM, BK)
            k_B = gB[None, None, k]                                     # (BN, BK)

            frag_A = thr_mma.make_fragment_A(k_A)
            frag_B = thr_mma.make_fragment_B(k_B)

            fx.copy(
                copy_atom_a,
                thr_copy_A.partition_S(k_A),
                thr_copy_A.retile(frag_A),
                pred=None,
            )
            fx.copy(
                copy_atom_b,
                thr_copy_B.partition_S(k_B),
                thr_copy_B.retile(frag_B),
                pred=None,
            )
            fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)

        # Epilogue: write accumulator back to gC (f32 tile).
        fx.copy(
            copy_atom_c,
            thr_copy_C.retile(frag_C),
            thr_copy_C.partition_S(gC),
            pred=None,
        )

    return gemm_kernel


def _build_launcher(M, N, K):
    """Return a @flyc.jit launcher closed over (M,N,K)."""
    gemm_kernel = _build_kernel(M, N, K)
    grid = (M // BM, N // BN, 1)

    @flyc.jit
    def gemm_bf16(
        A: fx.Tensor, B: fx.Tensor, C: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        gemm_kernel(A, B, C).launch(grid=grid, block=(256, 1, 1), stream=stream)

    return gemm_bf16


# ---- Reference / correctness -----------------------------------------------

def _reference(A, B):
    """Compute A @ B.T at f32 precision, regardless of input dtype.

    We never trust bf16 CUDA matmul as a baseline on IX11 — the reference
    is computed in f32 on the same device to match the kernel's f32
    accumulator.
    """
    return A.to(torch.float32) @ B.to(torch.float32).T


def _correctness_once(M, N, K, seed=0):
    torch.manual_seed(seed)
    A = torch.randn(M, K, dtype=torch.bfloat16).cuda()
    B = torch.randn(N, K, dtype=torch.bfloat16).cuda()
    C = torch.zeros(M, N, dtype=torch.float32).cuda()

    launcher = _build_launcher(M, N, K)
    stream = torch.cuda.Stream()
    launcher(A, B, C, stream=stream)
    torch.cuda.synchronize()

    expected = _reference(A, B)
    # bf16 inputs + f32 accumulator: K-reduction error grows ~sqrt(K)*eps_bf16.
    atol = 2e-2 * max(1.0, (K / 16) ** 0.5)
    rtol = 2e-2
    ok = torch.allclose(C, expected, atol=atol, rtol=rtol)
    diff = (C - expected).abs()
    print(
        f"[check] M={M} N={N} K={K}  ok={ok}  "
        f"max_abs={diff.max().item():.3e}  mean_abs={diff.mean().item():.3e}  "
        f"atol={atol:.2e}"
    )
    if not ok:
        print(f"  C[0,0:4]     = {C[0,0:4].tolist()}")
        print(f"  expect[0,0:4]= {expected[0,0:4].tolist()}")
    return ok


# ---- Benchmark -------------------------------------------------------------

def _bench(M, N, K, iters, warmup, seed=0):
    torch.manual_seed(seed)
    A = torch.randn(M, K, dtype=torch.bfloat16).cuda()
    B = torch.randn(N, K, dtype=torch.bfloat16).cuda()
    C = torch.zeros(M, N, dtype=torch.float32).cuda()

    launcher = _build_launcher(M, N, K)
    stream = torch.cuda.Stream()

    # -------- Path 2: flyc.compile once, reuse the CompiledFunction --------
    # flyc.compile() internally calls the JIT once to seed the cache, so
    # this doubles as the first warmup. Subsequent calls go through the
    # fast CallState path (~µs Python dispatch).
    t0 = time.perf_counter()
    compiled = flyc.compile(launcher, A, B, C, fx.Stream(stream))
    torch.cuda.synchronize()
    compile_s = time.perf_counter() - t0
    print(f"[compile] flyc.compile() took {compile_s*1e3:.1f} ms (including 1 warm launch)")

    # -------- Warmup --------
    for _ in range(warmup):
        compiled(A, B, C, fx.Stream(stream))
    torch.cuda.synchronize()

    # -------- Timed loop (single-stream, CUDA events) --------
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record()
        for _ in range(iters):
            compiled(A, B, C, fx.Stream(stream))
        end.record()
    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)
    per_iter_us = total_ms * 1e3 / iters

    # FLOPs: bf16 GEMM = 2 * M * N * K MACs.
    flops = 2.0 * M * N * K
    tflops = flops / (per_iter_us * 1e-6) / 1e12

    # Bandwidth lower bound (no smem reuse in this kernel): each block
    # loads BM*K bf16 + BN*K bf16, writes BM*BN f32. Sum over grid.
    grid_m = M // BM
    grid_n = N // BN
    bytes_read = grid_m * grid_n * (BM * K * 2 + BN * K * 2)
    bytes_write = M * N * 4
    gbps = (bytes_read + bytes_write) / (per_iter_us * 1e-6) / 1e9

    print(
        f"[bench]  M={M} N={N} K={K}  iters={iters}  "
        f"{per_iter_us:.1f} us/iter  "
        f"{tflops:.2f} TFLOPS  "
        f"BW~{gbps:.1f} GB/s"
    )

    # Sanity: validate the result we just computed is still correct after
    # the timing loop (detects aliased-output / data-race bugs).
    expected = _reference(A, B)
    atol = 2e-2 * max(1.0, (K / 16) ** 0.5)
    ok = torch.allclose(C, expected, atol=atol, rtol=2e-2)
    if not ok:
        diff = (C - expected).abs()
        print(f"  [WARN] post-bench correctness FAILED "
              f"(max_abs={diff.max().item():.3e})")
    return per_iter_us, tflops


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", nargs=3, type=int, metavar=("M", "N", "K"),
                   default=[4096, 4096, 4096])
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--check", action="store_true",
                   help="run correctness check only (no perf loop)")
    p.add_argument("--check-shape", nargs=3, type=int, metavar=("M", "N", "K"),
                   default=None,
                   help="use a different shape for the correctness check "
                        "(default: same as --shape)")
    return p.parse_args()


if __name__ == "__main__":
    _warn_if_card_busy()
    args = _parse_args()
    M, N, K = args.shape

    # Always run correctness first (cheap, and catches regressions early).
    cm, cn, ck = args.check_shape if args.check_shape else (M, N, K)
    ok = _correctness_once(cm, cn, ck)
    if not ok:
        sys.exit(1)
    if args.check:
        sys.exit(0)

    _bench(M, N, K, iters=args.iters, warmup=args.warmup)
