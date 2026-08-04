#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Iluvatar CQ (ivcore30) tiledMma pipeline HGEMM harness: f16 / bf16.

Wraps ``kernels.gemm.iluvatar.cq.hgemm.compile_iluvatar_cq_hgemm``. Companion
teaching example (single-warp G2S+S2R+MMA): ``examples/03-tiledMma-iluvatar-cq-pipeline.py``.
MR production harness: ``examples/03-tiledMma-iluvatar-mr-pipeline-hgemm.py``.

Bring-up scope: ``major_pattern=tn`` only; no s8 / long-mtx / split-K.

Usage::

    export FLYDSL_COMPILE_BACKEND=iluvatar
    export FLYDSL_RUNTIME_KIND=iluvatar
    export ARCH=ivcore30
    export CUDA_VISIBLE_DEVICES=15

    # Correctness (small shape)
    python examples/03-tiledMma-iluvatar-cq-pipeline-hgemm.py --check

    # Correctness + bench (bench uses --m/--n/--k; check uses --check-shape)
    python examples/03-tiledMma-iluvatar-cq-pipeline-hgemm.py --bench \\
      --dtype fp16 --m 1024 --n 1024 --k 1024 --cta 1024 --k-atoms 2 \\
      --epilogue no_c_read --epilogue-store shfl --warmup 5 --iters 50

    # Quick k_atoms sweep
    for ka in 2 4; do
      python examples/03-tiledMma-iluvatar-cq-pipeline-hgemm.py --bench \\
        --m 1024 --n 1024 --k 1024 --cta 1024 --k-atoms "$ka" --warmup 1 --iters 30
    done
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "iluvatar")
os.environ.setdefault("FLYDSL_RUNTIME_KIND", "iluvatar")
os.environ.setdefault("ARCH", "ivcore30")
os.environ.setdefault("FLYDSL_RUNTIME_ENABLE_CACHE", "0")

import torch  # noqa: E402

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from kernels.gemm.iluvatar.common import remap_gemm_tensors  # noqa: E402
from kernels.gemm.iluvatar.cq.common import DEFAULT_SMEM_CAP_BYTES  # noqa: E402
from kernels.gemm.iluvatar.cq.hgemm import (  # noqa: E402
    ATOM_K_B16,
    DEFAULT_EPILOGUE,
    DEFAULT_EPILOGUE_STORE,
    DEFAULT_K_ATOMS,
    DEFAULT_MAJOR_PATTERN,
    DEFAULT_SWIZZLE_CTA,
    EPILOGUE_NO_C_READ,
    EPILOGUE_READ_C_ACCUM,
    EPILOGUE_STORE_SHFL,
    EPILOGUE_STORE_TILED,
    MAJOR_PATTERN_CHOICES,
    SUPPORTED_MAJOR_PATTERNS,
    SWIZZLE_CTA_PRESETS,
    SwizzleCtaPreset,
    _swizzle_atom_work_ok,
    _swizzle_cta_shape,
    compile_iluvatar_cq_hgemm,
)

EPILOGUE_BOTH = "both"
_DTYPE_CHOICES = ("fp16", "bf16", "float16", "bfloat16")
_DTYPE_TO_FX = {
    "fp16": fx.Float16,
    "float16": fx.Float16,
    "bf16": fx.BFloat16,
    "bfloat16": fx.BFloat16,
}
_DTYPE_TO_TORCH = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def _resolve_dtype(dtype_name: str):
    key = dtype_name.lower()
    return _DTYPE_TO_FX[key], _DTYPE_TO_TORCH[key]


def _reference(A, B, C_in=None):
    ab = A.to(torch.float32) @ B.to(torch.float32).T
    if C_in is None:
        return ab
    return ab + C_in.to(torch.float32)


def _epilogue_modes(epilogue: str) -> list[str]:
    if epilogue == EPILOGUE_BOTH:
        return [EPILOGUE_NO_C_READ, EPILOGUE_READ_C_ACCUM]
    return [epilogue]


def _epilogue_label(epilogue: str, *, epilogue_store: str, dtype_name: str) -> str:
    if epilogue == EPILOGUE_NO_C_READ:
        return f"no_c_read (D=A@B.T, {dtype_name}, store={epilogue_store})"
    return "read_c_accum (C=A@B.T+C, fp32)"


def _make_c_tensor(m: int, n: int, epilogue: str, *, seed: int, torch_dtype=torch.float16):
    if epilogue == EPILOGUE_NO_C_READ:
        C = torch.empty(m, n, dtype=torch_dtype, device="cuda")
        C.fill_(7777.0)
        return C
    torch.manual_seed(seed)
    return torch.randn(m, n, dtype=torch.float32, device="cuda")


def _compare_atol(k: int, k_atoms: int, *, dtype_name: str = "fp16") -> float:
    bk = ATOM_K_B16 * k_atoms
    scale = 2.0 if dtype_name in {"bf16", "bfloat16"} else 1.0
    return 2e-2 * scale * max(1.0, (k / bk) ** 0.5)


def _gemm_flops(m: int, n: int, k: int) -> float:
    return 2.0 * float(m) * float(n) * float(k)


def _finalize_cta(args: argparse.Namespace) -> SwizzleCtaPreset:
    preset = SWIZZLE_CTA_PRESETS[args.cta]
    args.warps_m = preset.warps_m if args.warps_m is None else args.warps_m
    args.warps_n = preset.warps_n if args.warps_n is None else args.warps_n
    args.warp_atoms_m = preset.warp_atoms_m if args.warp_atoms_m is None else args.warp_atoms_m
    args.warp_atoms_n = preset.warp_atoms_n if args.warp_atoms_n is None else args.warp_atoms_n
    return preset


def _validate_shape(m: int, n: int, k: int, args: argparse.Namespace) -> str | None:
    if args.major_pattern not in SUPPORTED_MAJOR_PATTERNS:
        return (
            f"CQ HGEMM bring-up supports major_pattern in {SUPPORTED_MAJOR_PATTERNS}, "
            f"got {args.major_pattern!r}"
        )
    bm, bn, bk, threads, smem_bytes = _swizzle_cta_shape(
        args.warps_m,
        args.warps_n,
        args.k_atoms,
        warp_atoms_m=args.warp_atoms_m,
        warp_atoms_n=args.warp_atoms_n,
    )
    if args.k_atoms < 2 or args.k_atoms % 2:
        return f"k_atoms must be a positive even integer, got {args.k_atoms}"
    if k % bk:
        return f"K must be a multiple of {bk} (16 * k_atoms)"
    if m % bm or n % bn:
        return f"M,N must be multiples of {bm}/{bn} for CQ CTA"
    if not _swizzle_atom_work_ok(bm, bn, bk, args.warps_m, args.warps_n):
        return (
            f"SmexMtx brick count must divide evenly across {args.warps_m}x{args.warps_n} warps; "
            f"try different k_atoms (current BK={bk})"
        )
    if smem_bytes > DEFAULT_SMEM_CAP_BYTES:
        return (
            f"CTA smem {smem_bytes} B exceeds device cap {DEFAULT_SMEM_CAP_BYTES} B "
            f"({bm}x{bn}x{bk}, {threads} threads); use smaller tile or k_atoms"
        )
    return None


def _build_launcher(
    m: int,
    n: int,
    k: int,
    *,
    warps_m: int,
    warps_n: int,
    k_atoms: int,
    warp_atoms_m: int,
    warp_atoms_n: int,
    epilogue: str,
    epilogue_store: str,
    major_pattern: str,
    elem_dtype,
):
    launcher = compile_iluvatar_cq_hgemm(
        M=m,
        N=n,
        K=k,
        warps_m=warps_m,
        warps_n=warps_n,
        k_atoms=k_atoms,
        warp_atoms_m=warp_atoms_m,
        warp_atoms_n=warp_atoms_n,
        epilogue=epilogue,
        epilogue_store=epilogue_store,
        major_pattern=major_pattern,
        elem_dtype=elem_dtype,
    )
    bm, bn, bk, threads, smem = _swizzle_cta_shape(
        warps_m,
        warps_n,
        k_atoms,
        warp_atoms_m=warp_atoms_m,
        warp_atoms_n=warp_atoms_n,
    )
    grid = (m // bm, n // bn, 1)
    block = (threads, 1, 1)
    return launcher, grid, block, smem


def _check(
    m: int,
    n: int,
    k: int,
    *,
    warps_m: int,
    warps_n: int,
    k_atoms: int,
    warp_atoms_m: int,
    warp_atoms_n: int,
    epilogue: str,
    epilogue_store: str,
    major_pattern: str,
    dtype_name: str,
    seed: int = 0,
) -> bool:
    elem_dtype, torch_dtype = _resolve_dtype(dtype_name)
    torch.manual_seed(seed)
    A = torch.randn(m, k, dtype=torch_dtype, device="cuda")
    B = torch.randn(n, k, dtype=torch_dtype, device="cuda")
    C = _make_c_tensor(m, n, epilogue, seed=seed + 1, torch_dtype=torch_dtype)
    C_in = C.clone()

    launcher, grid, block, smem = _build_launcher(
        m,
        n,
        k,
        warps_m=warps_m,
        warps_n=warps_n,
        k_atoms=k_atoms,
        warp_atoms_m=warp_atoms_m,
        warp_atoms_n=warp_atoms_n,
        epilogue=epilogue,
        epilogue_store=epilogue_store,
        major_pattern=major_pattern,
        elem_dtype=elem_dtype,
    )
    a_dev, b_dev = remap_gemm_tensors(A, B, major_pattern)
    stream = torch.cuda.Stream()
    launcher(a_dev, b_dev, C, stream=stream)
    torch.cuda.synchronize()

    expected = _reference(A, B) if epilogue == EPILOGUE_NO_C_READ else _reference(A, B, C_in)
    got = C.to(torch.float32) if epilogue == EPILOGUE_NO_C_READ else C
    atol = _compare_atol(k, k_atoms, dtype_name=dtype_name)
    ok = bool(torch.allclose(got, expected, atol=atol, rtol=2e-2)) and bool(
        torch.isfinite(got).all().item()
    )
    bm, bn, bk, threads, _ = _swizzle_cta_shape(
        warps_m,
        warps_n,
        k_atoms,
        warp_atoms_m=warp_atoms_m,
        warp_atoms_n=warp_atoms_n,
    )
    store_note = f" store={epilogue_store}" if epilogue == EPILOGUE_NO_C_READ else ""
    status = "PASS" if ok else "FAIL"
    if not ok:
        diff = (got - expected).abs()
        print(
            f"[{status}] dtype={dtype_name} epilogue={epilogue}{store_note} "
            f"pattern={major_pattern} M={m} N={n} K={k} "
            f"cta={bm}x{bn}x{bk} grid={grid} block={block} smem={smem} "
            f"max_abs={diff.max().item():.3e} atol={atol:.3e}"
        )
    else:
        print(
            f"[{status}] dtype={dtype_name} epilogue={epilogue}{store_note} "
            f"pattern={major_pattern} M={m} N={n} K={k} "
            f"cta={bm}x{bn}x{bk} grid={grid} block={block} smem={smem}"
        )
    return ok


def _bench(
    m: int,
    n: int,
    k: int,
    *,
    warps_m: int,
    warps_n: int,
    k_atoms: int,
    warp_atoms_m: int,
    warp_atoms_n: int,
    epilogue: str,
    epilogue_store: str,
    major_pattern: str,
    dtype_name: str,
    iters: int,
    warmup: int,
) -> None:
    elem_dtype, torch_dtype = _resolve_dtype(dtype_name)
    print(
        f"[bench] === {_epilogue_label(epilogue, epilogue_store=epilogue_store, dtype_name=dtype_name)} ==="
    )
    torch.manual_seed(0)
    A = torch.randn(m, k, dtype=torch_dtype, device="cuda")
    B = torch.randn(n, k, dtype=torch_dtype, device="cuda")
    C = _make_c_tensor(m, n, epilogue, seed=1, torch_dtype=torch_dtype)
    C_in = C.clone()

    launcher, grid, block, smem = _build_launcher(
        m,
        n,
        k,
        warps_m=warps_m,
        warps_n=warps_n,
        k_atoms=k_atoms,
        warp_atoms_m=warp_atoms_m,
        warp_atoms_n=warp_atoms_n,
        epilogue=epilogue,
        epilogue_store=epilogue_store,
        major_pattern=major_pattern,
        elem_dtype=elem_dtype,
    )
    a_dev, b_dev = remap_gemm_tensors(A, B, major_pattern)
    stream = torch.cuda.Stream()
    fxs = fx.Stream(stream)

    t0 = time.perf_counter()
    compiled = flyc.compile(launcher, a_dev, b_dev, C, fxs)
    torch.cuda.synchronize()
    print(f"[compile] flyc.compile() took {1e3 * (time.perf_counter() - t0):.1f} ms")

    for _ in range(warmup):
        if epilogue == EPILOGUE_READ_C_ACCUM:
            C.copy_(C_in)
        compiled(a_dev, b_dev, C, fxs)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record()
        for _ in range(iters):
            if epilogue == EPILOGUE_READ_C_ACCUM:
                C.copy_(C_in)
            compiled(a_dev, b_dev, C, fxs)
        end.record()
    torch.cuda.synchronize()

    us = start.elapsed_time(end) * 1e3 / iters
    tflops = _gemm_flops(m, n, k) / (us * 1e-6) / 1e12

    c_ref = torch.empty(m, n, dtype=torch_dtype, device="cuda")
    ref_f32 = torch.empty(m, n, dtype=torch.float32, device="cuda")

    def torch_ref():
        if epilogue == EPILOGUE_NO_C_READ:
            c_ref.copy_(A @ B.T)
        else:
            ref_f32.copy_(A.float() @ B.float().T + C_in)

    t_start = torch.cuda.Event(enable_timing=True)
    t_end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        torch_ref()
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        t_start.record()
        for _ in range(iters):
            torch_ref()
        t_end.record()
    torch.cuda.synchronize()
    torch_us = t_start.elapsed_time(t_end) * 1e3 / iters
    torch_tflops = _gemm_flops(m, n, k) / (torch_us * 1e-6) / 1e12

    store_note = f" store={epilogue_store}" if epilogue == EPILOGUE_NO_C_READ else ""
    print(
        f"[bench] dtype={dtype_name} epilogue={epilogue}{store_note} "
        f"pattern={major_pattern} k_atoms={k_atoms} M={m} N={n} K={k} "
        f"grid={grid} block={block} threads={block[0]} smem={smem} "
        f"{us:.1f} us/iter  {tflops:.2f} TFLOPS  "
        f"(torch {torch_us:.1f} us, {torch_tflops:.2f} TFLOPS)"
    )

    expected = _reference(A, B) if epilogue == EPILOGUE_NO_C_READ else _reference(A, B, C_in)
    got = C.to(torch.float32) if epilogue == EPILOGUE_NO_C_READ else C
    atol = _compare_atol(k, k_atoms, dtype_name=dtype_name)
    if not torch.allclose(got, expected, atol=atol, rtol=2e-2):
        diff = (got - expected).abs()
        print(f"  [WARN] post-bench correctness FAILED (max_abs={diff.max().item():.3e})")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Iluvatar ivcore30 tiledMma pipeline HGEMM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "quick profiling examples:\n"
            "  python examples/03-tiledMma-iluvatar-cq-pipeline-hgemm.py --bench \\\n"
            "    --m 1024 --n 1024 --k 1024 --cta 1024 --k-atoms 2 --warmup 5 --iters 50\n"
            "  for ka in 2 4; do\n"
            "    python examples/03-tiledMma-iluvatar-cq-pipeline-hgemm.py --bench \\\n"
            '      --m 1024 --n 1024 --k 1024 --cta 1024 --k-atoms "$ka" --warmup 1 --iters 30\n'
            "  done"
        ),
    )
    p.add_argument("--m", type=int, default=1024, help="bench M (default 1024)")
    p.add_argument("--n", type=int, default=1024, help="bench N (default 1024)")
    p.add_argument("--k", type=int, default=1024, help="bench K (default 1024)")
    p.add_argument(
        "--dtype",
        choices=_DTYPE_CHOICES,
        default="fp16",
        help="A/B (and no_c_read C) element type: fp16 (default) or bf16",
    )
    p.add_argument(
        "--epilogue",
        choices=(EPILOGUE_NO_C_READ, EPILOGUE_READ_C_ACCUM, EPILOGUE_BOTH),
        default=DEFAULT_EPILOGUE,
        help="no_c_read / read_c_accum / both",
    )
    p.add_argument(
        "--epilogue-store",
        choices=(EPILOGUE_STORE_TILED, EPILOGUE_STORE_SHFL),
        default=DEFAULT_EPILOGUE_STORE,
        help="no_c_read store path: tiled or shfl",
    )
    p.add_argument(
        "--major-pattern",
        choices=MAJOR_PATTERN_CHOICES,
        default=DEFAULT_MAJOR_PATTERN,
        help="BLAS layout tag (CQ bring-up: tn only)",
    )
    p.add_argument(
        "--cta",
        choices=sorted(SWIZZLE_CTA_PRESETS),
        default=DEFAULT_SWIZZLE_CTA,
        help="CTA preset: 256 (64x64) or 1024 (256x256)",
    )
    p.add_argument("--warps-m", type=int, default=None)
    p.add_argument("--warps-n", type=int, default=None)
    p.add_argument("--warp-atoms-m", type=int, default=None)
    p.add_argument("--warp-atoms-n", type=int, default=None)
    p.add_argument("--k-atoms", type=int, default=DEFAULT_K_ATOMS, help="BK = 16 * k_atoms (even)")
    p.add_argument(
        "--check-shape",
        nargs=3,
        type=int,
        metavar=("M", "N", "K"),
        default=[64, 64, 64],
        help="correctness shape (default 64 64 64)",
    )
    p.add_argument("--check", action="store_true", help="correctness only (skip bench)")
    p.add_argument("--bench", action="store_true", help="run CUDA-event profiling after check")
    p.add_argument("--warmup", type=int, default=10, help="bench warmup iters (default 10)")
    p.add_argument("--iters", type=int, default=30, help="bench timed iters (default 30)")
    args = p.parse_args(argv)

    if not args.check and not args.bench:
        p.print_help()
        return 0

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA-compatible CQ device is not available "
            "(set CUDA_VISIBLE_DEVICES to the reserved CQ GPU, e.g. 15)"
        )

    _finalize_cta(args)
    epilogues = _epilogue_modes(args.epilogue)
    if args.epilogue_store == EPILOGUE_STORE_SHFL and EPILOGUE_READ_C_ACCUM in epilogues:
        print(
            "[WARN] --epilogue-store shfl applies to no_c_read only; "
            "read_c_accum still uses f32 tiled_copy",
            file=sys.stderr,
        )

    # Always run a correctness check first (MR harness behavior).
    cm, cn, ck = args.check_shape
    err = _validate_shape(cm, cn, ck, args)
    if err:
        if args.bench:
            bm, bn, bk, _, _ = _swizzle_cta_shape(
                args.warps_m,
                args.warps_n,
                args.k_atoms,
                warp_atoms_m=args.warp_atoms_m,
                warp_atoms_n=args.warp_atoms_n,
            )
            print(
                f"[WARN] --check-shape {cm} {cn} {ck} incompatible with CTA ({err}); "
                f"using one CTA tile {bm} {bn} {bk} for pre-bench check",
                file=sys.stderr,
            )
            cm, cn, ck = bm, bn, bk
        else:
            print(f"Error (--check-shape): {err}", file=sys.stderr)
            return 2

    all_ok = True
    for epilogue in epilogues:
        ok = _check(
            cm,
            cn,
            ck,
            warps_m=args.warps_m,
            warps_n=args.warps_n,
            k_atoms=args.k_atoms,
            warp_atoms_m=args.warp_atoms_m,
            warp_atoms_n=args.warp_atoms_n,
            epilogue=epilogue,
            epilogue_store=args.epilogue_store,
            major_pattern=args.major_pattern,
            dtype_name=args.dtype,
        )
        all_ok = all_ok and ok
    if not all_ok:
        return 1
    if args.check and not args.bench:
        return 0

    err = _validate_shape(args.m, args.n, args.k, args)
    if err:
        print(f"Error (--m/--n/--k): {err}", file=sys.stderr)
        return 2

    for epilogue in epilogues:
        _bench(
            args.m,
            args.n,
            args.k,
            warps_m=args.warps_m,
            warps_n=args.warps_n,
            k_atoms=args.k_atoms,
            warp_atoms_m=args.warp_atoms_m,
            warp_atoms_n=args.warp_atoms_n,
            epilogue=epilogue,
            epilogue_store=args.epilogue_store,
            major_pattern=args.major_pattern,
            dtype_name=args.dtype,
            iters=args.iters,
            warmup=args.warmup,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
