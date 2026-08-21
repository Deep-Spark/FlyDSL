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

* ``i32`` (default) -- **direct int32 store**
* ``i8`` -- packed int8 store (truncating cast). For k-major B (``tn`` / ``nn``)
  this uses PackOnly; mn-major B keeps PackSlb. No quant scale.
* ``scaled_bf16`` / ``scaled_fp16`` -- dequant
  ``D = acc * scale_a[m] * scale_b[n] [+ bias[n]]`` (optional ``--bias``).

``major_pattern`` -- CUTLASS BLAS layout tag on logical ``A(m, k)`` / ``B(n, k)``.
``tn`` (default; both operands k-major) is the fast path; the mn-major patterns
(``nn`` / ``nt`` / ``tt``) use i8 k-spanning S2R.

Run::

    python examples/03-tiledMma-iluvatar-mr-pipeline-igemm.py --check
    python examples/03-tiledMma-iluvatar-mr-pipeline-igemm.py --check --epilogue scaled_bf16
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
    remap_gemm_tensors,
)
from kernels.gemm.iluvatar.mr.igemm import (  # noqa: E402
    DEFAULT_EPILOGUE,
    EPILOGUE_CHOICES,
    EPILOGUE_I8,
    _is_scaled_epilogue,
    compile_iluvatar_mr_igemm,
)

_OUT_DTYPE = {
    "i32": torch.int32,
    "i8": torch.int8,
    "scaled_bf16": torch.bfloat16,
    "scaled_fp16": torch.float16,
}

# (warps_m, warps_n, warp_atoms_m, warp_atoms_n, default_k_rep) -> 256x256 CTA tile.
CTA_PRESETS = {
    "1024": (4, 4, 4, 4, 2),  # 16 warps x 64 lanes, 64x64/warp
    "2048": (4, 8, 4, 2, 4),  # 32 warps x 64 lanes, 64x32/warp
}
DEFAULT_CTA = "1024"


def _reference(A, B, epilogue, scale_a=None, scale_b=None, bias=None):
    acc = A.cpu().to(torch.int32) @ B.cpu().to(torch.int32).T
    if epilogue == EPILOGUE_I8:
        # Truncating cast (wrap on overflow).
        return acc.to(torch.int8)
    if _is_scaled_epilogue(epilogue):
        # acc * scale_a[row] * scale_b[col] [+ bias]; no alpha/beta/relu
        out = acc.to(torch.float32)
        out = out * scale_a.cpu().view(-1, 1).to(torch.float32)
        out = out * scale_b.cpu().view(1, -1).to(torch.float32)
        if bias is not None:
            out = out + bias.cpu().view(1, -1).to(torch.float32)
        return out.to(_OUT_DTYPE[epilogue])
    return acc


def _ops(m: int, n: int, k: int) -> float:
    return 2.0 * float(m) * float(n) * float(k)


def _cta_bm(warps_m, warp_atoms_m) -> int:
    return 16 * warp_atoms_m * warps_m


def _build_launcher(
    m,
    n,
    k,
    *,
    warps_m,
    warps_n,
    k_rep,
    warp_atoms_m,
    warp_atoms_n,
    major_pattern,
    epilogue=DEFAULT_EPILOGUE,
    stages=2,
    apply_bias=False,
    allow_dynamic_m=False,
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
        apply_bias=apply_bias,
        allow_dynamic_m=allow_dynamic_m,
    )
    return launcher, launcher.grid, launcher.block, launcher.smem_bytes


def _check(
    m,
    n,
    k,
    *,
    warps_m,
    warps_n,
    k_rep,
    warp_atoms_m,
    warp_atoms_n,
    major_pattern,
    epilogue=DEFAULT_EPILOGUE,
    seed=0,
    stages=2,
    apply_bias=False,
):
    torch.manual_seed(seed)
    short_m = _is_scaled_epilogue(epilogue) and (m % _cta_bm(warps_m, warp_atoms_m) != 0)
    A = torch.randint(-8, 8, (m, k), dtype=torch.int8, device="cuda")
    B = torch.randint(-8, 8, (n, k), dtype=torch.int8, device="cuda")
    scale_a = scale_b = bias = None
    if _is_scaled_epilogue(epilogue):
        scale_a = torch.empty(m, device="cuda", dtype=torch.float32).uniform_(0.01, 0.5)
        scale_b = torch.empty(n, device="cuda", dtype=torch.float32).uniform_(0.01, 0.5)
        if apply_bias:
            bias = torch.empty(n, device="cuda", dtype=torch.float32).uniform_(-1.0, 1.0)
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
        apply_bias=apply_bias,
        allow_dynamic_m=short_m,
    )
    m_ceil = launcher.m_ceil
    C = torch.zeros(m_ceil, n, dtype=_OUT_DTYPE[epilogue], device="cuda")
    a_dev, b_dev = remap_gemm_tensors(A, B, major_pattern)
    stream = torch.cuda.Stream()
    if _is_scaled_epilogue(epilogue):
        if apply_bias:
            launcher(a_dev, b_dev, scale_a, scale_b, C, bias, stream=stream)
        else:
            launcher(a_dev, b_dev, scale_a, scale_b, C, stream=stream)
    else:
        launcher(a_dev, b_dev, C, stream=stream)
    torch.cuda.synchronize()

    expected = _reference(A, B, epilogue, scale_a=scale_a, scale_b=scale_b, bias=bias).to("cuda")
    C_live = C[:m]
    if _is_scaled_epilogue(epilogue):
        # bf16/fp16 rounding vs fp32 reference
        diff = (C_live.float() - expected.float()).abs()
        tol = 2e-2 * expected.float().abs().max().clamp(min=1.0) + 1e-3
        ok = bool((diff <= tol).all().item())
        max_abs = float(diff.max())
    else:
        diff = (C_live.to(torch.int32) - expected.to(torch.int32)).abs()
        ok = torch.equal(C_live, expected)
        max_abs = int(diff.max())
    cta_note = f" cta={warps_m}x{warps_n}warps atoms={warp_atoms_m}x{warp_atoms_n} threads={block[0]}"
    bias_note = f" bias={apply_bias}" if _is_scaled_epilogue(epilogue) else ""
    ceil_note = f" m_ceil={m_ceil}" if m_ceil != m else ""
    print(
        f"[check] epilogue={epilogue}{bias_note} pattern={major_pattern} M={m}{ceil_note} N={n} K={k}{cta_note} "
        f"grid={grid} block={block} smem={smem} ok={ok} max_abs={max_abs}"
    )
    if not ok:
        print(f"  C[0,0:4]      = {C_live[0, 0:4].tolist()}")
        print(f"  expect[0,0:4] = {expected[0, 0:4].tolist()}")
        if _is_scaled_epilogue(epilogue):
            print(f"  n_mismatch    = {int((diff > tol).sum())}/{m * n}")
        else:
            print(f"  n_mismatch    = {int((diff != 0).sum())}/{m * n}")
    return bool(ok)


def _bench(
    m,
    n,
    k,
    *,
    warps_m,
    warps_n,
    k_rep,
    warp_atoms_m,
    warp_atoms_n,
    major_pattern,
    epilogue,
    iters,
    warmup,
    stages=2,
    apply_bias=False,
):
    torch.manual_seed(0)
    A = torch.randint(-8, 8, (m, k), dtype=torch.int8, device="cuda")
    B = torch.randint(-8, 8, (n, k), dtype=torch.int8, device="cuda")
    C = torch.zeros(m, n, dtype=_OUT_DTYPE[epilogue], device="cuda")
    scaled = _is_scaled_epilogue(epilogue)
    scale_a = scale_b = bias = None
    if scaled:
        scale_a = (torch.rand(m, dtype=torch.float32, device="cuda") * 0.01 + 0.001).contiguous()
        scale_b = (torch.rand(n, dtype=torch.float32, device="cuda") * 0.01 + 0.001).contiguous()
        if apply_bias:
            bias = (torch.randn(n, dtype=torch.float32, device="cuda") * 0.01).contiguous()
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
        apply_bias=apply_bias,
    )
    a_dev, b_dev = remap_gemm_tensors(A, B, major_pattern)
    stream = torch.cuda.Stream()
    # Wrap as static memrefs (shape/stride baked in) + hoist one stream wrapper so
    # the hot launch path skips per-call layout-buffer packing / stream extraction.
    # This removes ~3us/launch of Python marshaling overhead that otherwise inflates
    # small-shape numbers (host-launch bound when device exec is only a few us).
    a_arg, b_arg, c_arg = flyc.from_dlpack(a_dev), flyc.from_dlpack(b_dev), flyc.from_dlpack(C)
    fxs = fx.Stream(stream)
    if scaled:
        sa_arg, sb_arg = flyc.from_dlpack(scale_a), flyc.from_dlpack(scale_b)
        if apply_bias:
            compile_args = (a_arg, b_arg, sa_arg, sb_arg, c_arg, flyc.from_dlpack(bias), fxs)
        else:
            compile_args = (a_arg, b_arg, sa_arg, sb_arg, c_arg, fxs)
    else:
        compile_args = (a_arg, b_arg, c_arg, fxs)

    t0 = time.perf_counter()
    compiled = flyc.compile(launcher, *compile_args)
    torch.cuda.synchronize()
    print(f"[compile] flyc.compile() took {1e3 * (time.perf_counter() - t0):.1f} ms")

    for _ in range(warmup):
        compiled(*compile_args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record()
        for _ in range(iters):
            compiled(*compile_args)
        end.record()
    torch.cuda.synchronize()

    us = start.elapsed_time(end) * 1e3 / iters
    tops = _ops(m, n, k) / (us * 1e-6) / 1e12

    # torch int8 reference (torch._int_mm) -- skip for scaled (different epilogue).
    torch_us = None
    torch_tops = None
    if not scaled:
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
        f"  (torch {torch_us:.1f} us, {torch_tops:.2f} TOPS, {torch_us / us:.2f}x)" if torch_us is not None else ""
    )
    bias_note = f" bias={apply_bias}" if scaled else ""
    print(
        f"[bench] epilogue={epilogue}{bias_note} pattern={major_pattern} M={m} N={n} K={k} grid={grid} block={block} "
        f"threads={block[0]} smem={smem} {us:.1f} us/iter  {tops:.2f} TOPS(int8){torch_note}"
    )

    expected = _reference(A, B, epilogue, scale_a=scale_a, scale_b=scale_b, bias=bias).to("cuda")
    if scaled:
        diff = (C.float() - expected.float()).abs()
        tol = 2e-2 if epilogue == "scaled_bf16" else 5e-3
        if not bool((diff <= tol).all()):
            print(f"  [WARN] post-bench correctness FAILED (max_abs={float(diff.max())})")
    elif not torch.equal(C, expected):
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
        help="output store: i32 | i8 | scaled_bf16 | scaled_fp16",
    )
    p.add_argument(
        "--bias",
        action="store_true",
        help="with scaled_* epilogue: fuse per-column bias",
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
        short_m = _is_scaled_epilogue(args.epilogue) and (m % _cta_bm(args.warps_m, args.warp_atoms_m) != 0)
        a = torch.randint(-8, 8, (m, k), dtype=torch.int8)
        b = torch.randint(-8, 8, (n, k), dtype=torch.int8)
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
            apply_bias=args.bias,
            allow_dynamic_m=short_m,
        )
        c = torch.zeros(launcher.m_ceil, n, dtype=_OUT_DTYPE[args.epilogue])
        a_dev, b_dev = remap_gemm_tensors(a, b, args.major_pattern)
        if _is_scaled_epilogue(args.epilogue):
            sa = torch.ones(m, dtype=torch.float32)
            sb = torch.ones(n, dtype=torch.float32)
            if args.bias:
                bias = torch.zeros(n, dtype=torch.float32)
                launcher(a_dev, b_dev, sa, sb, c, bias)
            else:
                launcher(a_dev, b_dev, sa, sb, c)
        else:
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
        apply_bias=args.bias,
    )
    if not ok:
        return 1
    if _is_scaled_epilogue(args.epilogue) and (cm % _cta_bm(args.warps_m, args.warp_atoms_m) == 0):
        # Scaled epilogue can store a live M that is not a multiple of BM.
        ok = _check(
            100,
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
            apply_bias=args.bias,
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
            apply_bias=args.bias,
        )
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
