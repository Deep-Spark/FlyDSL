# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Correctness and performance driver for MR int8 BypassSlb Conv3D.

Run from the FlyDSL repository root:

    python examples/06-conv3d-iluvatar-mr-bypass-slb-i.py --check
    python examples/06-conv3d-iluvatar-mr-bypass-slb-i.py --bench

The full default check command:
    python examples/06-conv3d-iluvatar-mr-bypass-slb-i.py --check --n 1 --c 4 --d 16 --h 64 --w 64 \
    --k 64 --kt 3 --kh 7 --kw 7 --groups 1 --stride 2 --padding 1 3 3 --dilation 1 \
    --value-range 3 --seed 2026 --warmup 1 --iters 100

The measured FlyDSL time is kernel-only: NCDHW/weight packing and output
unpacking are deliberately excluded.
"""

import argparse
import os
import sys
import time

os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "iluvatar")
os.environ.setdefault("FLYDSL_RUNTIME_KIND", "iluvatar")
os.environ.setdefault("ARCH", "ivcore11")

import torch
import torch.nn.functional as F

import flydsl.compiler as flyc
import flydsl.expr as fx
from kernels.conv.iluvatar.mr.conv3d_implicit_bypass_slb_int8 import (
    BLOCK_K,
    BLOCK_M,
    BLOCK_N,
    BLOCK_THREADS,
    compile_conv3d_implicit_bypass_slb,
    prepare_conv3d_implicit_bypass_slb,
    unpack_conv3d_implicit_bypass_slb_output,
)


def _triple(value):
    if len(value) == 1:
        return (value[0],) * 3
    if len(value) != 3:
        raise ValueError("expected one or three values")
    return tuple(value)


def _make_tensors(args, *, device):
    torch.manual_seed(args.seed)
    x = torch.randint(
        -args.value_range,
        args.value_range + 1,
        (args.n, args.c, args.d, args.h, args.w),
        dtype=torch.int8,
        device=device,
    )
    weight = torch.randint(
        -args.value_range,
        args.value_range + 1,
        (args.k, args.c // args.groups, args.kt, args.kh, args.kw),
        dtype=torch.int8,
        device=device,
    )
    return x, weight


def _build(args, x, weight):
    stride = _triple(args.stride)
    padding = _triple(args.padding)
    dilation = _triple(args.dilation)
    xp, wp, yw, meta = prepare_conv3d_implicit_bypass_slb(
        x,
        weight,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=args.groups,
    )
    n, c_per_group, d, h, w, k_per_group, kt, kh, kw = meta["shape"]
    launch = compile_conv3d_implicit_bypass_slb(
        n,
        c_per_group,
        d,
        h,
        w,
        k_per_group,
        kt,
        kh,
        kw,
        *stride,
        *padding,
        *dilation,
        args.groups,
    )
    grid = (
        meta["n_padded"] // BLOCK_N,
        args.groups,
        meta["m_padded"] // BLOCK_M,
    )
    block = (BLOCK_THREADS, 1, 1)
    smem = BLOCK_N * BLOCK_K
    return launch, xp, wp, yw, meta, grid, block, smem


def _reference(args, x, weight):
    ref = F.conv3d(
        x.cpu().float(),
        weight.cpu().float(),
        stride=_triple(args.stride),
        padding=_triple(args.padding),
        dilation=_triple(args.dilation),
        groups=args.groups,
    )
    return ref.round().to(torch.int32).clamp(-128, 127).to(torch.int8)


def _logical_ops(args, meta):
    do, ho, wo = meta["output_shape"]
    reduction = (args.c // args.groups) * args.kt * args.kh * args.kw
    return 2.0 * args.n * args.k * do * ho * wo * reduction


def _padded_ops(args, meta):
    return (
        2.0
        * meta["m_padded"]
        * args.groups
        * meta["n_padded"]
        * meta["reduction_k_padded"]
    )


def _check(args):
    x, weight = _make_tensors(args, device="cuda")
    launch, xp, wp, yw, meta, grid, block, smem = _build(args, x, weight)
    stream = torch.cuda.Stream()
    launch(xp, wp, yw, stream)
    torch.cuda.synchronize()

    actual = unpack_conv3d_implicit_bypass_slb_output(yw, meta).cpu()
    expected = _reference(args, x, weight)
    ok = torch.equal(actual, expected)
    diff = (actual.to(torch.int32) - expected.to(torch.int32)).abs()
    print(
        f"[check] x={tuple(x.shape)} w={tuple(weight.shape)} "
        f"stride={_triple(args.stride)} padding={_triple(args.padding)} "
        f"dilation={_triple(args.dilation)} groups={args.groups} "
        f"grid={grid} block={block} smem={smem} ok={ok} "
        f"max_abs={int(diff.max())} mismatches={int((diff != 0).sum())}"
    )
    if not ok:
        mismatch = (diff != 0).nonzero()
        first = tuple(int(v) for v in mismatch[0])
        print(
            f"  first mismatch at {first}: "
            f"got={int(actual[first])}, expected={int(expected[first])}"
        )
    return ok


def _bench(args):
    x, weight = _make_tensors(args, device="cuda")
    launch, xp, wp, yw, meta, grid, block, smem = _build(args, x, weight)
    stream = torch.cuda.Stream()

    xp_arg = flyc.from_dlpack(xp)
    wp_arg = flyc.from_dlpack(wp)
    yw_arg = flyc.from_dlpack(yw)
    stream_arg = fx.Stream(stream)

    t0 = time.perf_counter()
    compiled = flyc.compile(launch, xp_arg, wp_arg, yw_arg, stream_arg)
    torch.cuda.synchronize()
    print(
        f"[compile] flyc.compile() took "
        f"{1e3 * (time.perf_counter() - t0):.1f} ms"
    )

    for _ in range(args.warmup):
        compiled(xp_arg, wp_arg, yw_arg, stream_arg)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record()
        for _ in range(args.iters):
            compiled(xp_arg, wp_arg, yw_arg, stream_arg)
        end.record()
    torch.cuda.synchronize()
    us = start.elapsed_time(end) * 1e3 / args.iters
    logical_tops = _logical_ops(args, meta) / (us * 1e-6) / 1e12
    padded_tops = _padded_ops(args, meta) / (us * 1e-6) / 1e12

    # PyTorch does not provide a general native int8 Conv3D reference here.
    # Float32 Conv3D is reported only as a latency baseline.
    xf = x.float()
    wf = weight.float()
    torch_us = None
    try:
        for _ in range(args.warmup):
            F.conv3d(
                xf,
                wf,
                stride=_triple(args.stride),
                padding=_triple(args.padding),
                dilation=_triple(args.dilation),
                groups=args.groups,
            )
        torch.cuda.synchronize()
        ts = torch.cuda.Event(enable_timing=True)
        te = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            ts.record()
            for _ in range(args.iters):
                F.conv3d(
                    xf,
                    wf,
                    stride=_triple(args.stride),
                    padding=_triple(args.padding),
                    dilation=_triple(args.dilation),
                    groups=args.groups,
                )
            te.record()
        torch.cuda.synchronize()
        torch_us = ts.elapsed_time(te) * 1e3 / args.iters
    except Exception as exc:  # noqa: BLE001
        print(
            f"  [info] torch float32 Conv3D baseline unavailable: "
            f"{repr(exc)[:120]}"
        )

    baseline = (
        f" (torch-f32 {torch_us:.1f} us)" if torch_us is not None else ""
    )
    print(
        f"[bench] x={tuple(x.shape)} w={tuple(weight.shape)} "
        f"groups={args.groups} grid={grid} block={block} smem={smem} "
        f"{us:.1f} us/iter {logical_tops:.2f} effective-TOPS "
        f"{padded_tops:.2f} padded-TOPS{baseline}"
    )

    actual = unpack_conv3d_implicit_bypass_slb_output(yw, meta).cpu()
    expected = _reference(args, x, weight)
    if not torch.equal(actual, expected):
        diff = (actual.to(torch.int32) - expected.to(torch.int32)).abs()
        print(
            f"  [WARN] post-bench correctness FAILED "
            f"(max_abs={int(diff.max())})"
        )
        return False
    return True


def _parse_args(argv):
    p = argparse.ArgumentParser(
        description="Iluvatar MR int8 implicit-GEMM Conv3D BypassSlb"
    )
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--c", type=int, default=4)
    p.add_argument("--d", type=int, default=16)
    p.add_argument("--h", type=int, default=64)
    p.add_argument("--w", type=int, default=64)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--kt", type=int, default=3)
    p.add_argument("--kh", type=int, default=7)
    p.add_argument("--kw", type=int, default=7)
    p.add_argument("--groups", type=int, default=1)
    p.add_argument("--stride", type=int, nargs="+", default=[2])
    p.add_argument("--padding", type=int, nargs="+", default=[1, 3, 3])
    p.add_argument("--dilation", type=int, nargs="+", default=[1])
    p.add_argument("--value-range", type=int, default=3)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--check", action="store_true", help="run correctness only")
    p.add_argument("--bench", action="store_true", help="run kernel-only benchmark")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=100)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv or sys.argv[1:])
    if args.groups <= 0 or args.c % args.groups or args.k % args.groups:
        raise ValueError("c and k must be divisible by groups")

    compile_only = os.environ.get(
        "COMPILE_ONLY", ""
    ).lower() in {"1", "true", "yes", "on"}
    if compile_only or not torch.cuda.is_available():
        os.environ["COMPILE_ONLY"] = "1"
        x, weight = _make_tensors(args, device="cpu")
        launch, xp, wp, yw, meta, grid, block, smem = _build(args, x, weight)
        launch(xp, wp, yw)
        print(
            f"Compiled MR int8 implicit Conv3D BypassSlb "
            f"(COMPILE_ONLY; grid={grid}, block={block}, smem={smem}, "
            f"Kpad={meta['reduction_k_padded']})."
        )
        return 0

    if args.check:
        return 0 if _check(args) else 1
    if args.bench:
        return 0 if _bench(args) else 1

    # A no-flag invocation runs a small, quick correctness check.
    args.n = 1
    args.c = max(args.groups * 4, min(args.c, args.groups * 8))
    args.d, args.h, args.w = 5, 8, 8
    args.k = max(args.groups, min(args.k, args.groups * 16))
    args.kt, args.kh, args.kw = 3, 3, 3
    args.stride = [1]
    args.padding = [1]
    return 0 if _check(args) else 1


if __name__ == "__main__":
    raise SystemExit(main())
