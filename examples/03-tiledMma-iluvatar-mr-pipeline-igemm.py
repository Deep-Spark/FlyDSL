# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Iluvatar MR (ivcore11) tiledMma pipeline IGEMM: int8 inputs, int32/int8 output.

Companion to the f16 pipeline HGEMM ({hgemm_example}). Same double-buffered SME
G2S + Ki-deferred S2R/MMA mainloop, but with the int8 path:

* 16x16x32 ``MRMma`` atom (int8 A/B, int32/int8 accumulator) -- vs f16's 16x16x16.
* 64-value SME rows (``MRAsyncCpRow8b`` / ``MRAsyncCpCol``, ``SMESwizzle.Row8b``
  = rowxfb8 mod-swizzle for the row operand) -- vs f16's 32-value Row16b.
* int8 MN-major bricks span two physical K=16 SME bricks per K=32 MMA atom.

Epilogue (``--epilogue``), both ``D = A @ B.T``:

* ``i32`` (default) -- **direct int32 store** (``UniversalCopy32b`` via
  ``make_tiled_copy_C`` + ``partition_S``). Unlike the f16 ``no_c_read`` path, the
  int32 output is already 4 bytes/element, so there is **no warp-shuffle pack**
* ``i8`` -- **packed CShuffle int8 store**: i32 accumulator -> saturate ``[-127,127]``
  -> int8 -> row-major SMEM staging -> coalesced 32-bit (4x int8) global store.
  This is the int8 analog of the f16 shuffle pack).
   Quant scale/bias/relu fusion is omitted for now.

``major_pattern`` -- CUTLASS BLAS layout tag on logical ``A(m, k)`` / ``B(n, k)``.
``tn`` (default; both operands k-major) is the fast path; the mn-major patterns
(``nn`` / ``nt`` / ``tt``) use i8 k-spanning S2R.

Run::

    python examples/03-tiledMma-iluvatar-mr-pipeline-igemm.py --check
    python examples/03-tiledMma-iluvatar-mr-pipeline-igemm.py --bench
"""

import argparse  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "iluvatar")
os.environ.setdefault("FLYDSL_RUNTIME_KIND", "iluvatar")
os.environ.setdefault("ARCH", "ivcore11")

import torch  # noqa: E402

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from kernels.gemm.iluvatar.common import (  # noqa: E402
    DEFAULT_MAJOR_PATTERN,
    MAJOR_PATTERN_CHOICES,
    WARP_SIZE,
    remap_gemm_tensors,
)
from kernels.gemm.iluvatar.mr.igemm import (  # noqa: E402
    ATOM_K_I8,
    DEFAULT_EPILOGUE,
    DEFAULT_K_REP,
    EPILOGUE_CHOICES,
    EPILOGUE_I8,
    compile_iluvatar_mr_igemm,
    resolve_igemm_stages,
)

_OUT_DTYPE = {"i32": torch.int32, "i8": torch.int8}

# (warps_m, warps_n, warp_atoms_m, warp_atoms_n, default_k_rep) -> 256x256 CTA tile.
CTA_PRESETS = {
    "1024": (4, 4, 4, 4, 2),  # 16 warps x 64 lanes, 64x64/warp
    "2048": (4, 8, 4, 2, 4),  # 32 warps x 64 lanes, 64x32/warp
}
DEFAULT_CTA = "1024"

def _reference(A, B, epilogue):
    acc = A.cpu().to(torch.int32) @ B.cpu().to(torch.int32).T
    if epilogue == EPILOGUE_I8:
        # Truncating cast (wrap on overflow).
        return acc.to(torch.int8)
    return acc


def _ops(m: int, n: int, k: int) -> float:
    return 2.0 * float(m) * float(n) * float(k)


def _build_launcher(
    m, n, k, *, warps_m, warps_n, k_rep, warp_atoms_m, warp_atoms_n, major_pattern, epilogue=DEFAULT_EPILOGUE, stages=2
):
    launcher = compile_iluvatar_mr_igemm(
        M=m,
        N=n,
        K=k,
        warps_m=warps_m,
        warps_n=warps_n,
        k_rep=k_rep,
        warp_atoms_m=warp_atoms_m,
        warp_atoms_n=warp_atoms_n,
        major_pattern=major_pattern,
        epilogue=epilogue,
        stages=stages,
    )
    warp_m = 16 * warp_atoms_m
    warp_n = 16 * warp_atoms_n
    bm = warp_m * warps_m
    bn = warp_n * warps_n
    bk = ATOM_K_I8 * k_rep
    threads = warps_m * warps_n * WARP_SIZE
    grid = (m // bm, n // bn, 1)
    block = (threads, 1, 1)
    resolved_stages = resolve_igemm_stages(stages, m, n, k, bm, bn, bk, major_pattern=major_pattern)
    pipeline_smem = (bm + bn) * bk * resolved_stages  # int8 = 1 byte, ``stages``-buffered
    smem = max(pipeline_smem, bm * bn) if epilogue == EPILOGUE_I8 else pipeline_smem
    return launcher, grid, block, smem


def _check(
    m, n, k, *, warps_m, warps_n, k_rep, warp_atoms_m, warp_atoms_n, major_pattern, epilogue=DEFAULT_EPILOGUE, seed=0, stages=2
):
    torch.manual_seed(seed)
    A = torch.randint(-8, 8, (m, k), dtype=torch.int8, device="cuda")
    B = torch.randint(-8, 8, (n, k), dtype=torch.int8, device="cuda")
    C = torch.zeros(m, n, dtype=_OUT_DTYPE[epilogue], device="cuda")
    launcher, grid, block, smem = _build_launcher(
        m,
        n,
        k,
        warps_m=warps_m,
        warps_n=warps_n,
        k_rep=k_rep,
        warp_atoms_m=warp_atoms_m,
        warp_atoms_n=warp_atoms_n,
        major_pattern=major_pattern,
        epilogue=epilogue,
        stages=stages,
    )
    a_dev, b_dev = remap_gemm_tensors(A, B, major_pattern)
    stream = torch.cuda.Stream()
    launcher(a_dev, b_dev, C, stream=stream)
    torch.cuda.synchronize()

    expected = _reference(A, B, epilogue).to("cuda")
    diff = (C.to(torch.int32) - expected.to(torch.int32)).abs()
    ok = torch.equal(C, expected)
    cta_note = f" cta={warps_m}x{warps_n}warps atoms={warp_atoms_m}x{warp_atoms_n} threads={block[0]}"
    print(
        f"[check] epilogue={epilogue} pattern={major_pattern} M={m} N={n} K={k}{cta_note} "
        f"grid={grid} block={block} smem={smem} ok={ok} max_abs={int(diff.max())}"
    )
    if not ok:
        print(f"  C[0,0:4]      = {C[0, 0:4].tolist()}")
        print(f"  expect[0,0:4] = {expected[0, 0:4].tolist()}")
        print(f"  n_mismatch    = {int((diff != 0).sum())}/{m * n}")
    return bool(ok)


def _bench(
    m, n, k, *, warps_m, warps_n, k_rep, warp_atoms_m, warp_atoms_n, major_pattern, epilogue, iters, warmup, stages=2
):
    torch.manual_seed(0)
    A = torch.randint(-8, 8, (m, k), dtype=torch.int8, device="cuda")
    B = torch.randint(-8, 8, (n, k), dtype=torch.int8, device="cuda")
    C = torch.zeros(m, n, dtype=_OUT_DTYPE[epilogue], device="cuda")
    launcher, grid, block, smem = _build_launcher(
        m,
        n,
        k,
        warps_m=warps_m,
        warps_n=warps_n,
        k_rep=k_rep,
        warp_atoms_m=warp_atoms_m,
        warp_atoms_n=warp_atoms_n,
        major_pattern=major_pattern,
        epilogue=epilogue,
        stages=stages,
    )
    a_dev, b_dev = remap_gemm_tensors(A, B, major_pattern)
    stream = torch.cuda.Stream()
    # Wrap as static memrefs (shape/stride baked in) + hoist one stream wrapper so
    # the hot launch path skips per-call layout-buffer packing / stream extraction.
    # This removes ~3us/launch of Python marshaling overhead that otherwise inflates
    # small-shape numbers (host-launch bound when device exec is only a few us).
    a_arg, b_arg, c_arg = flyc.from_dlpack(a_dev), flyc.from_dlpack(b_dev), flyc.from_dlpack(C)
    fxs = fx.Stream(stream)

    t0 = time.perf_counter()
    compiled = flyc.compile(launcher, a_arg, b_arg, c_arg, fxs)
    torch.cuda.synchronize()
    print(f"[compile] flyc.compile() took {1e3 * (time.perf_counter() - t0):.1f} ms")

    for _ in range(warmup):
        compiled(a_arg, b_arg, c_arg, fxs)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record()
        for _ in range(iters):
            compiled(a_arg, b_arg, c_arg, fxs)
        end.record()
    torch.cuda.synchronize()

    us = start.elapsed_time(end) * 1e3 / iters
    tops = _ops(m, n, k) / (us * 1e-6) / 1e12

    # torch int8 reference (torch._int_mm: (m,k) int8 @ (k,n) int8 -> (m,n) int32).
    torch_us = None
    torch_tops = None
    try:
        Bt = B.t().contiguous()

        def torch_ref():
            torch._int_mm(A, Bt)

        for _ in range(warmup):
            torch_ref()
        torch.cuda.synchronize()
        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            t_start.record()
            for _ in range(iters):
                torch_ref()
            t_end.record()
        torch.cuda.synchronize()
        torch_us = t_start.elapsed_time(t_end) * 1e3 / iters
        torch_tops = _ops(m, n, k) / (torch_us * 1e-6) / 1e12
    except Exception as exc:  # noqa: BLE001
        print(f"  [info] torch._int_mm reference unavailable: {repr(exc)[:120]}")

    torch_note = (
        f"  (torch {torch_us:.1f} us, {torch_tops:.2f} TOPS, {torch_us / us:.2f}x)"
        if torch_us is not None
        else ""
    )
    print(
        f"[bench] epilogue={epilogue} pattern={major_pattern} M={m} N={n} K={k} grid={grid} block={block} "
        f"threads={block[0]} smem={smem} {us:.1f} us/iter  {tops:.2f} TOPS(int8){torch_note}"
    )

    expected = _reference(A, B, epilogue).to("cuda")
    if not torch.equal(C, expected):
        diff = (C.to(torch.int32) - expected.to(torch.int32)).abs()
        print(f"  [WARN] post-bench correctness FAILED (max_abs={int(diff.max())})")


def _parse_args(argv):
    p = argparse.ArgumentParser(description="Iluvatar ivcore11 tiledMma pipeline IGEMM (int8)")
    p.add_argument("--m", type=int, default=1024)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--k", type=int, default=512)
    p.add_argument(
        "--major-pattern",
        choices=MAJOR_PATTERN_CHOICES,
        default=DEFAULT_MAJOR_PATTERN,
        help="G2S global layout tag for A/B (see kernels.gemm.iluvatar.mr.igemm)",
    )
    p.add_argument(
        "--epilogue",
        choices=EPILOGUE_CHOICES,
        default=DEFAULT_EPILOGUE,
        help="output store: i32 (direct int32) or i8 (packed CShuffle, saturating cast, int8 C)",
    )
    p.add_argument(
        "--cta",
        choices=sorted(CTA_PRESETS),
        default=DEFAULT_CTA,
        help="thread-block preset: 1024 (4x4 warps, 64x64/warp) or 2048 (4x8 warps, 64x32/warp)",
    )
    p.add_argument("--warps-m", type=int, default=None, help="override preset warps_m")
    p.add_argument("--warps-n", type=int, default=None, help="override preset warps_n")
    p.add_argument("--warp-atoms-m", type=int, default=None, help="MMA atoms per warp in M")
    p.add_argument("--warp-atoms-n", type=int, default=None, help="MMA atoms per warp in N")
    p.add_argument("--k-rep", type=int, default=None, help="BK = 32 * k_rep (even); default per preset")
    p.add_argument(
        "--stages",
        default="auto",
        help="SMEM pipeline stages: 'auto' (small grids->3, large->2), or 2/3 to force",
    )
    p.add_argument(
        "--check-shape",
        nargs=3,
        type=int,
        metavar=("M", "N", "K"),
        default=[256, 256, 64],
        help="correctness shape (default 256 256 64)",
    )
    p.add_argument("--check", action="store_true", help="correctness only")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    return p.parse_args(argv)


def _finalize_cta(args):
    warps_m, warps_n, warp_atoms_m, warp_atoms_n, default_k_rep = CTA_PRESETS[args.cta]
    args.warps_m = warps_m if args.warps_m is None else args.warps_m
    args.warps_n = warps_n if args.warps_n is None else args.warps_n
    args.warp_atoms_m = warp_atoms_m if args.warp_atoms_m is None else args.warp_atoms_m
    args.warp_atoms_n = warp_atoms_n if args.warp_atoms_n is None else args.warp_atoms_n
    args.k_rep = default_k_rep if args.k_rep is None else args.k_rep
    args.stages = None if str(args.stages).lower() == "auto" else int(args.stages)


def main(argv=None):
    args = _parse_args(argv or sys.argv[1:])
    _finalize_cta(args)

    m, n, k = args.m, args.n, args.k
    cm, cn, ck = args.check_shape

    compile_only = os.environ.get("COMPILE_ONLY", "").lower() in {"1", "true", "yes", "on"}
    if compile_only or not torch.cuda.is_available():
        os.environ["COMPILE_ONLY"] = "1"
        a = torch.randint(-8, 8, (m, k), dtype=torch.int8)
        b = torch.randint(-8, 8, (n, k), dtype=torch.int8)
        c = torch.zeros(m, n, dtype=_OUT_DTYPE[args.epilogue])
        launcher, grid, block, smem = _build_launcher(
            m,
            n,
            k,
            warps_m=args.warps_m,
            warps_n=args.warps_n,
            k_rep=args.k_rep,
            warp_atoms_m=args.warp_atoms_m,
            warp_atoms_n=args.warp_atoms_n,
            major_pattern=args.major_pattern,
            epilogue=args.epilogue,
            stages=args.stages,
        )
        a_dev, b_dev = remap_gemm_tensors(a, b, args.major_pattern)
        launcher(a_dev, b_dev, c)
        print(
            f"Compiled tiledMma pipeline IGEMM (COMPILE_ONLY; epilogue={args.epilogue}, "
            f"pattern={args.major_pattern}, {m}x{n}x{k}, cta={args.cta}, grid={grid}, "
            f"block={block}, smem={smem})."
        )
        return 0

    ok = _check(
        cm,
        cn,
        ck,
        warps_m=args.warps_m,
        warps_n=args.warps_n,
        k_rep=args.k_rep,
        warp_atoms_m=args.warp_atoms_m,
        warp_atoms_n=args.warp_atoms_n,
        major_pattern=args.major_pattern,
        epilogue=args.epilogue,
        stages=args.stages,
    )
    if not ok:
        return 1
    if args.check:
        return 0

    if args.bench:
        _bench(
            m,
            n,
            k,
            warps_m=args.warps_m,
            warps_n=args.warps_n,
            k_rep=args.k_rep,
            warp_atoms_m=args.warp_atoms_m,
            warp_atoms_n=args.warp_atoms_n,
            major_pattern=args.major_pattern,
            epilogue=args.epilogue,
            iters=args.iters,
            warmup=args.warmup,
            stages=args.stages,
        )
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
