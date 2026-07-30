# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

# Companion Iluvatar MR tiled-copy teaching example. Update this path if renamed;
# doc/comments refer to it as "the tiled-copy teaching example".
_TILEDCOPY_TEACHING_EXAMPLE = "examples/02-tiledCopy-iluvatar-mr.py"

_DOC = """Iluvatar MR (ivcore11) tiledMma pipeline HGEMM: f16 / bf16 inputs.

Double-buffered shared-memory pipeline with async SME G2S
(``MRAsyncCpRow16b`` / ``MRAsyncCpCol``), ``make_sme_shared_layout``, Ki-deferred
S2R/MMA mainloop, and ``UniversalCopy32b`` for S2R (swizzled shared -> TCU fragment).

For the **teaching** TiledCopy variant of the same SME primitives (single 16x16
tile per warp, explicit ``cp_async_wait``, generic S2R to global), see
``{tiledcopy_teaching_example}``. The two examples deliberately choose different
APIs for the same hardware ops; the rationale is below.

Design choices vs the tiled-copy teaching example
--------------------------------------------------

**1. Global -> shared (async G2S)**

* **What we do here:** ``copy_atom_call`` in ``issue_stage``, looping over
  ``a_per_warp`` / ``b_per_warp`` SME bricks per warp. A/B use pattern-specific
  swizzle atoms; gmem is ``zipped_divide``'d into 16x32 f16 bricks; smem views use
  swizzled ``make_sme_shared_layout``. Only ``cp_async_commit_group`` is issued --
  no per-stage ``cp_async_wait_group``.
* **Why not the tiled-copy teaching style** (``make_tiled_copy_tv`` +
  ``partition_S/D`` + ``fx.copy`` with a ``(1,1)`` issuer)? A production CTA tile
  is ``256 x 256 x BK`` with tens of SME bricks per warp. A/B need **different**
  swizzle atoms and **explicit warp-level work distribution** (``atom_idx``,
  ``warp_a_start``). Wrapping each brick in a separate TiledCopy with
  ``(1,1)`` thread layout would repeat the same boilerplate without adding
  safety. ``copy_atom_call`` keeps the multi-brick pipeline readable and matches
  how real Iluvatar HGEMM kernels schedule SME issues.
* **Sync:** double-buffered K-pipeline with ``barrier`` between stages. Async
  loads of stage N+1 overlap with S2R/MMA on stage N; explicit ``wait_group``
  after every issue would serialize the pipeline and hurt throughput. The
  tiled-copy teaching example uses explicit wait because it has no compute stage
  to hide behind.

**2. Shared -> register (S2R)**

* **What we do here:** ``make_tiled_copy_A/B(copy_atom, tiled_mma)`` --
  S2R tiling is **derived from** ``MRMma``. Each ``_ki_load`` does
  ``partition_S(smem_tile)`` + ``retile(frag_A/B)`` into TCU operand fragments.
  Ki-deferred scheduling interleaves S2R and MMA within one K-tile so the last
  ki slice can overlap the next async issue.
* **Why not the tiled-copy teaching style** (``make_tiled_copy_tv`` with a
  hand-built 64-lane layout over physical smem)? That example targets **logical
  GMEM readback** after a NoSwizzle copy. **This example** feeds an **MRMma
  fragment** with a hardware-specific register layout. ``make_tiled_copy_A/B``
  encodes the smem swizzle (Row16b/Col) -> TCU operand mapping; a generic TV
  layout would not land data in the registers ``fx.gemm`` expects. Do not reuse
  the tiled-copy teaching S2R path when feeding MMA.

When to use this pattern vs the tiled-copy teaching example
------------------------------------------------------------

* **Use the tiled-copy teaching example** to learn SME async copy or build simple
  copy/elementwise kernels.
* **Use ``kernels.gemm.iluvatar.mr.hgemm``** for HGEMM and other compute-bound kernels
  that need multi-brick G2S, software pipelining, and MMA-coupled S2R.

**This file** is a check/bench harness around ``kernels.gemm.iluvatar.mr.hgemm``.
Epilogue modes, CTA presets (``--cta``), ``major_pattern``, and other tuning
parameters are documented in that module; CLI flags here mirror those kwargs
(``--epilogue both`` runs check/bench for ``no_c_read`` then ``read_c_accum``).

Run::

    # Correctness (small shape) and a single default bench:
    python examples/03-tiledMma-iluvatar-mr-pipeline-hgemm.py --check
    python examples/03-tiledMma-iluvatar-mr-pipeline-hgemm.py --bench

    # Peak-shape reference (Gate2-style contract), fp16 or bf16:
    python examples/03-tiledMma-iluvatar-mr-pipeline-hgemm.py --bench \\
      --dtype fp16 --m 4096 --n 4096 --k 4096 --cta 1024 --k-atoms 2 \\
      --epilogue no_c_read --epilogue-store shfl --major-pattern tn \\
      --warmup 1 --iters 100

    # Full sweep over layouts x k_atoms (the canonical tuning command; the best
    # of k_atoms in {{2,4}} per point is the reported FlyDSL config). Replace
    # 4096 with the target square size (or set M/N/K independently):
    for p in nt tn nn tt; do
      for ka in 2 4; do
        python examples/03-tiledMma-iluvatar-mr-pipeline-hgemm.py --bench \\
          --epilogue no_c_read --epilogue-store shfl --k-atoms "$ka" --cta 1024 \\
          --major-pattern "$p" --warmup 1 --iters 100 --m 4096 --n 4096 --k 4096
      done
    done
"""
__doc__ = _DOC.format(tiledcopy_teaching_example=_TILEDCOPY_TEACHING_EXAMPLE)

import argparse  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

os.environ.setdefault("FLYDSL_COMPILE_BACKEND", "iluvatar")
os.environ.setdefault("FLYDSL_RUNTIME_KIND", "iluvatar")
os.environ.setdefault("ARCH", "ivcore11")

import torch  # noqa: E402

import flydsl.compiler as flyc  # noqa: E402
import flydsl.expr as fx  # noqa: E402
from kernels.gemm.iluvatar.common import remap_gemm_tensors  # noqa: E402
from kernels.gemm.iluvatar.mr.hgemm import (  # noqa: E402
    ATOM_K_B16,
    DEFAULT_ELEM_DTYPE,
    DEFAULT_MAJOR_PATTERN,
    DEFAULT_SMEM_CAP_BYTES,
    DEFAULT_SWIZZLE_CTA,
    EPILOGUE_NO_C_READ,
    EPILOGUE_READ_C_ACCUM,
    EPILOGUE_STORE_SHFL,
    EPILOGUE_STORE_TILED,
    MAJOR_PATTERN_CHOICES,
    SWIZZLE_CTA_PRESETS,
    WARP_SIZE,
    SwizzleCtaPreset,
    _swizzle_atom_work_ok,
    _swizzle_cta_shape,
    compile_iluvatar_mr_hgemm,
)
from kernels.gemm.iluvatar.mr.hgemm_splitk import (  # noqa: E402
    DEFAULT_SPLIT_K_MODE,
    SPLIT_K_MODE_CHOICES,
    SPLIT_K_MODE_PARALLEL,
    compile_iluvatar_mr_hgemm_splitk,
)

# Harness-only default: run both epilogue modes unless --epilogue selects one.
EPILOGUE_BOTH = "both"
DEFAULT_EPILOGUE = EPILOGUE_BOTH
DEFAULT_K_ATOMS = 4  # BK = 64; matches SWIZZLE_CTA_PRESETS[*].default_k_atoms
DEFAULT_EPILOGUE_STORE = EPILOGUE_STORE_TILED

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


def _warn_if_card_busy() -> None:
    if shutil.which("ixsmi") is None:
        return
    try:
        out = subprocess.check_output(["ixsmi"], text=True, timeout=5, stderr=subprocess.DEVNULL)
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


def _reference(A, B, C_in=None):
    ab = A.to(torch.float32) @ B.to(torch.float32).T
    if C_in is None:
        return ab
    return ab + C_in.to(torch.float32)


def _epilogue_modes(epilogue: str) -> list[str]:
    if epilogue == EPILOGUE_BOTH:
        return [EPILOGUE_NO_C_READ, EPILOGUE_READ_C_ACCUM]
    return [epilogue]


def _epilogue_label(
    epilogue: str,
    *,
    epilogue_store: str = DEFAULT_EPILOGUE_STORE,
    dtype_name: str = "fp16",
) -> str:
    if epilogue == EPILOGUE_NO_C_READ:
        return f"no_c_read (D=A@B.T, {dtype_name}, no read C, store={epilogue_store})"
    return "read_c_accum (C=A@B.T+C, fp32, read C)"


def _make_c_tensor(
    m: int,
    n: int,
    epilogue: str,
    *,
    seed: int,
    device: str = "cuda",
    torch_dtype=torch.float16,
):
    if epilogue == EPILOGUE_NO_C_READ:
        # Poison so incomplete stores (and missing split-K C-zero) surface;
        # non-split overwrite and split-K launcher zero both clear it.
        C = torch.empty(m, n, dtype=torch_dtype, device=device)
        C.fill_(7777.0)
        return C
    torch.manual_seed(seed)
    return torch.randn(m, n, dtype=torch.float32, device=device)


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
    epilogue_store: str = DEFAULT_EPILOGUE_STORE,
    major_pattern: str = DEFAULT_MAJOR_PATTERN,
    elem_dtype=DEFAULT_ELEM_DTYPE,
    split_k: int = 1,
    split_k_mode: str = DEFAULT_SPLIT_K_MODE,
):
    if split_k > 1:
        launcher = compile_iluvatar_mr_hgemm_splitk(
            M=m,
            N=n,
            K=k,
            warps_m=warps_m,
            warps_n=warps_n,
            k_atoms=k_atoms,
            warp_atoms_m=warp_atoms_m,
            warp_atoms_n=warp_atoms_n,
            epilogue=epilogue,
            major_pattern=major_pattern,
            elem_dtype=elem_dtype,
            split_k=split_k,
            split_k_mode=split_k_mode,
        )
        # All Global Split-K modes use grid.z=split_k (serial turnstile / parallel / atomic).
        grid_z = split_k
    else:
        launcher = compile_iluvatar_mr_hgemm(
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
        grid_z = 1
    bm, bn, _bk, threads, smem = _swizzle_cta_shape(
        warps_m,
        warps_n,
        k_atoms,
        warp_atoms_m=warp_atoms_m,
        warp_atoms_n=warp_atoms_n,
    )
    grid = (m // bm, n // bn, grid_z)
    block = (threads, 1, 1)
    return launcher, grid, block, smem


def _gemm_flops(m: int, n: int, k: int) -> float:
    return 2.0 * float(m) * float(n) * float(k)


def _expected_result(A, B, C_in, epilogue: str):
    if epilogue == EPILOGUE_NO_C_READ:
        return _reference(A, B)
    return _reference(A, B, C_in)


def _compare_atol(k: int, k_atoms: int, *, dtype_name: str = "fp16") -> float:
    bk = ATOM_K_B16 * k_atoms
    scale = 2.0 if dtype_name in {"bf16", "bfloat16"} else 1.0
    return 2e-2 * scale * max(1.0, (k / bk) ** 0.5)


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
    epilogue_store: str = DEFAULT_EPILOGUE_STORE,
    major_pattern: str = DEFAULT_MAJOR_PATTERN,
    dtype_name: str = "fp16",
    seed: int = 0,
    split_k: int = 1,
    split_k_mode: str = DEFAULT_SPLIT_K_MODE,
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
        split_k=split_k,
        split_k_mode=split_k_mode,
    )
    a_dev, b_dev = remap_gemm_tensors(A, B, major_pattern)
    stream = torch.cuda.Stream()
    launcher(a_dev, b_dev, C, stream=stream)
    torch.cuda.synchronize()

    expected = _expected_result(A, B, C_in, epilogue)
    if epilogue == EPILOGUE_NO_C_READ:
        got = C.to(torch.float32)
    else:
        got = C
    diff = (got - expected).abs()
    # atomic / serial truncate through f16/bf16 each partition; parallel keeps
    # fp32 workspace then one trunc -- scale atol for low-precision reduce modes.
    atol = _compare_atol(k, k_atoms, dtype_name=dtype_name)
    if split_k > 1 and split_k_mode != SPLIT_K_MODE_PARALLEL:
        atol = atol * (max(1, split_k) ** 0.5)
    ok = torch.allclose(got, expected, atol=atol, rtol=2e-2)
    finite_ok = torch.isfinite(got).all().item()
    cta_note = (
        f" cta={warps_m}x{warps_n}warps"
        f" atoms={warp_atoms_m}x{warp_atoms_n}"
        f" threads={warps_m * warps_n * WARP_SIZE}"
    )
    if epilogue == EPILOGUE_NO_C_READ:
        store_note = f" store={split_k_mode}_splitk" if split_k > 1 else f" store={epilogue_store}"
    else:
        store_note = ""
    split_note = f" split_k={split_k} mode={split_k_mode}" if split_k > 1 else ""
    print(
        f"[check] dtype={dtype_name} epilogue={epilogue}{store_note}{split_note} pattern={major_pattern} "
        f"M={m} N={n} K={k}{cta_note} grid={grid} block={block} smem={smem} "
        f"ok={ok} finite={finite_ok} max_abs={diff.max().item():.3e} "
        f"mean_abs={diff.mean().item():.3e} atol={atol:.2e}"
    )
    if not ok:
        print(f"  C[0,0:4]      = {got[0, 0:4].tolist()}")
        print(f"  expect[0,0:4] = {expected[0, 0:4].tolist()}")
    return bool(ok and finite_ok)


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
    epilogue_store: str = DEFAULT_EPILOGUE_STORE,
    major_pattern: str = DEFAULT_MAJOR_PATTERN,
    dtype_name: str = "fp16",
    iters: int,
    warmup: int,
    split_k: int = 1,
    split_k_mode: str = DEFAULT_SPLIT_K_MODE,
) -> None:
    elem_dtype, torch_dtype = _resolve_dtype(dtype_name)
    print(f"[bench] === {_epilogue_label(epilogue, epilogue_store=epilogue_store, dtype_name=dtype_name)} ===")
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
        split_k=split_k,
        split_k_mode=split_k_mode,
    )
    a_dev, b_dev = remap_gemm_tensors(A, B, major_pattern)
    stream = torch.cuda.Stream()
    fxs = fx.Stream(stream)

    t0 = time.perf_counter()
    if split_k > 1:
        # Split-K launchers are host wrappers (serial/parallel/atomic), not @flyc.jit.
        compiled = launcher
        compiled(a_dev, b_dev, C, fxs)  # trigger JIT of inner kernels
        torch.cuda.synchronize()
        print(f"[compile] split-K host wrapper first-launch took {1e3 * (time.perf_counter() - t0):.1f} ms")
    else:
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

    print(
        f"[bench] dtype={dtype_name} epilogue={epilogue}"
        f"{f' store={epilogue_store}' if epilogue == EPILOGUE_NO_C_READ else ''} "
        f"pattern={major_pattern} k_atoms={k_atoms} M={m} N={n} K={k} "
        f"grid={grid} block={block} threads={block[0]} smem={smem} "
        f"{us:.1f} us/iter  {tflops:.2f} TFLOPS  "
        f"(torch {torch_us:.1f} us, {torch_tflops:.2f} TFLOPS)"
    )

    expected = _expected_result(A, B, C_in, epilogue)
    if epilogue == EPILOGUE_NO_C_READ:
        got = C.to(torch.float32)
    else:
        got = C
    atol = _compare_atol(k, k_atoms, dtype_name=dtype_name) * (max(1, split_k) ** 0.5)
    if not torch.allclose(got, expected, atol=atol, rtol=2e-2):
        diff = (got - expected).abs()
        print(f"  [WARN] post-bench correctness FAILED (max_abs={diff.max().item():.3e})")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Iluvatar ivcore11 tiledMma pipeline HGEMM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "canonical tuning sweep (best of k_atoms in {2,4} per point;\n"
            "replace 4096 with the target shape):\n"
            "  for p in nt tn nn tt; do\n"
            "    for ka in 2 4; do\n"
            "      python examples/03-tiledMma-iluvatar-mr-pipeline-hgemm.py --bench \\\n"
            '        --epilogue no_c_read --epilogue-store shfl --k-atoms "$ka" --cta 1024 \\\n'
            '        --major-pattern "$p" --warmup 1 --iters 100 --m 4096 --n 4096 --k 4096\n'
            "    done\n"
            "  done"
        ),
    )
    p.add_argument("--m", type=int, default=1024)
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--k", type=int, default=512)
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
        help="no_c_read=D=A@B.T f16/bf16 no read C; read_c_accum=C+=A@B.T fp32 read C; both=sequential",
    )
    p.add_argument(
        "--epilogue-store",
        choices=(EPILOGUE_STORE_TILED, EPILOGUE_STORE_SHFL),
        default=DEFAULT_EPILOGUE_STORE,
        help="no_c_read output store: tiled=trunc_f+UniversalCopy16b; shfl=shuffle+packed i32 store",
    )
    p.add_argument(
        "--major-pattern",
        choices=MAJOR_PATTERN_CHOICES,
        default=DEFAULT_MAJOR_PATTERN,
        help="CUTLASS 3.x BLAS global layout tag for A/B (see kernels.gemm.iluvatar.mr.hgemm)",
    )
    p.add_argument(
        "--cta",
        choices=sorted(SWIZZLE_CTA_PRESETS),
        default=DEFAULT_SWIZZLE_CTA,
        help="thread-block preset: 1024 (4x4 warps, 64x64/warp) or "
        "2048 (4x8 warps, 64x32/warp, same 256x256 CTA tile)",
    )
    p.add_argument("--warps-m", type=int, default=None, help="override preset warps_m")
    p.add_argument("--warps-n", type=int, default=None, help="override preset warps_n")
    p.add_argument("--warp-atoms-m", type=int, default=None, help="MMA atoms per warp in M")
    p.add_argument("--warp-atoms-n", type=int, default=None, help="MMA atoms per warp in N")
    p.add_argument("--k-atoms", type=int, default=DEFAULT_K_ATOMS, help="BK = 16 * k_atoms")
    p.add_argument(
        "--check-shape",
        nargs=3,
        type=int,
        metavar=("M", "N", "K"),
        default=[256, 256, 64],
        help="correctness shape (default 256 256 64)",
    )
    p.add_argument(
        "--split-k",
        type=int,
        default=1,
        help="Global Split-K slices (default 1). >1 requires no_c_read and "
        "K %% (split_k * bk) == 0; see --split-k-mode",
    )
    p.add_argument(
        "--split-k-mode",
        choices=list(SPLIT_K_MODE_CHOICES),
        default=DEFAULT_SPLIT_K_MODE,
        help="Split-K reduction: serial (default, ordered load-add-store), "
        "parallel (fp32 workspace + reduce kernel), or atomic (UniversalAtomicAdd)",
    )
    p.add_argument("--check", action="store_true", help="correctness only")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=30)
    return p.parse_args(argv)


def _finalize_swizzle_cta(args: argparse.Namespace) -> SwizzleCtaPreset:
    preset = SWIZZLE_CTA_PRESETS[args.cta]
    args.warps_m = preset.warps_m if args.warps_m is None else args.warps_m
    args.warps_n = preset.warps_n if args.warps_n is None else args.warps_n
    args.warp_atoms_m = preset.warp_atoms_m if args.warp_atoms_m is None else args.warp_atoms_m
    args.warp_atoms_n = preset.warp_atoms_n if args.warp_atoms_n is None else args.warp_atoms_n
    return preset


def _validate_shape(m: int, n: int, k: int, args: argparse.Namespace) -> str | None:
    bm, bn, bk, threads, smem_bytes = _swizzle_cta_shape(
        args.warps_m,
        args.warps_n,
        args.k_atoms,
        warp_atoms_m=args.warp_atoms_m,
        warp_atoms_n=args.warp_atoms_n,
    )
    if k % bk:
        return f"K must be a multiple of {bk} (16 * k_atoms)"
    if m % bm or n % bn:
        return f"M,N must be multiples of {bm}/{bn} for swizzle CTA"
    if not _swizzle_atom_work_ok(bm, bn, bk, args.warps_m, args.warps_n):
        return (
            f"SME brick count must divide evenly across {args.warps_m}x{args.warps_n} warps; "
            f"try larger k_atoms (current BK={bk})"
        )
    if args.split_k > 1 and k % (args.split_k * bk):
        return f"K must be a multiple of split_k * bk = {args.split_k * bk} for --split-k {args.split_k}"
    if smem_bytes > DEFAULT_SMEM_CAP_BYTES:
        return (
            f"CTA smem {smem_bytes} B exceeds device cap {DEFAULT_SMEM_CAP_BYTES} B "
            f"({bm}x{bn}x{bk}, {threads} threads); use smaller tile or k_atoms"
        )
    return None


def main(argv: list[str] | None = None) -> int:
    _warn_if_card_busy()
    args = _parse_args(argv or sys.argv[1:])
    preset = _finalize_swizzle_cta(args)
    if args.cta == "2048" and args.k_atoms < 4:
        print(
            "[WARN] 2048-thread CTA usually needs --k-atoms >= 4 for even SME work " "and smem within 128 KiB",
            file=sys.stderr,
        )
    elif preset.name == "2048":
        pass

    m, n, k = args.m, args.n, args.k
    cm, cn, ck = args.check_shape
    epilogues = _epilogue_modes(args.epilogue)
    if args.split_k > 1 and epilogues != [EPILOGUE_NO_C_READ]:
        print(
            f"[WARN] --split-k {args.split_k} requires the {EPILOGUE_NO_C_READ} epilogue; "
            "ignoring other epilogue modes",
            file=sys.stderr,
        )
        epilogues = [EPILOGUE_NO_C_READ]
    if args.epilogue_store == EPILOGUE_STORE_SHFL and EPILOGUE_READ_C_ACCUM in epilogues:
        print(
            "[WARN] --epilogue-store shfl applies to no_c_read only; read_c_accum still uses f32 tiled_copy",
            file=sys.stderr,
        )

    compile_only = os.environ.get("COMPILE_ONLY", "").lower() in {"1", "true", "yes", "on"}
    elem_dtype, torch_dtype = _resolve_dtype(args.dtype)
    if compile_only or not torch.cuda.is_available():
        os.environ["COMPILE_ONLY"] = "1"
        err = _validate_shape(m, n, k, args)
        if err:
            print(f"Error: {err}", file=sys.stderr)
            return 2
        a = torch.randn(m, k, dtype=torch_dtype, device="cpu")
        b = torch.randn(n, k, dtype=torch_dtype, device="cpu")
        for epilogue in epilogues:
            c = _make_c_tensor(m, n, epilogue, seed=0, device="cpu", torch_dtype=torch_dtype)
            launcher, grid, block, smem = _build_launcher(
                m,
                n,
                k,
                warps_m=args.warps_m,
                warps_n=args.warps_n,
                k_atoms=args.k_atoms,
                warp_atoms_m=args.warp_atoms_m,
                warp_atoms_n=args.warp_atoms_n,
                epilogue=epilogue,
                epilogue_store=args.epilogue_store,
                major_pattern=args.major_pattern,
                elem_dtype=elem_dtype,
                split_k=args.split_k,
                split_k_mode=args.split_k_mode,
            )
            a_dev, b_dev = remap_gemm_tensors(a, b, args.major_pattern)
            launcher(a_dev, b_dev, c)
            store_note = f", store={args.epilogue_store}" if epilogue == EPILOGUE_NO_C_READ else ""
            split_note = f", split_k={args.split_k}/{args.split_k_mode}" if args.split_k > 1 else ""
            print(
                f"Compiled tiledMma pipeline HGEMM (COMPILE_ONLY; dtype={args.dtype}, "
                f"epilogue={epilogue}{store_note}{split_note}, "
                f"pattern={args.major_pattern}, {m}x{n}x{k}, cta={args.cta}, "
                f"grid={grid}, block={block}, smem={smem})."
            )
        return 0

    if args.split_k > 1:
        check_bk = ATOM_K_B16 * args.k_atoms
        if ck % (args.split_k * check_bk):
            print(
                f"Error: --check-shape K={ck} must be a multiple of split_k * bk = "
                f"{args.split_k * check_bk} for --split-k {args.split_k}",
                file=sys.stderr,
            )
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
            split_k=args.split_k,
            split_k_mode=args.split_k_mode,
        )
        all_ok = all_ok and ok
    if not all_ok:
        return 1
    if args.check:
        return 0

    err = _validate_shape(m, n, k, args)
    if err:
        print(f"Error: {err}", file=sys.stderr)
        return 2

    if args.bench:
        for epilogue in epilogues:
            _bench(
                m,
                n,
                k,
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
                split_k=args.split_k,
                split_k_mode=args.split_k_mode,
            )
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
