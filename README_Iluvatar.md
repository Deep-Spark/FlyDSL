# FlyDSL on Iluvatar (FlyIXDL)

FlyDSL includes an **Iluvatar** compile and runtime backend targeting the **FlyIXDL**
dialect. Kernels are authored in the same Python layout DSL and `@flyc.kernel` /
`@flyc.jit` APIs as the ROCm path; lowering goes through `fly` → `FlyIXDL` → IXDL
LLVM IR → device fatbin.

This document covers Iluvatar-specific features, supported hardware, runnable
examples, and HGEMM performance reference numbers.

## Features

| Area | What is provided |
|------|------------------|
| **Dialect & lowering** | `FlyIXDL` dialect, `convert-fly-to-ixdl`, `gpu-to-ixdl`, `ixdl-attach-target` pipeline (`python/flydsl/compiler/backends/iluvatar.py`) |
| **Layout algebra** | Same Fly layout API as ROCm (`logical_divide`, `copy_atom_call`, `make_tiled_copy_*`, …) |
| **SME async copy** | `MRAsyncCpRow8b` / `Row16b` / `Col`, `make_sme_gmem_tensor`, `make_sme_shared_layout`, `cp_async_commit_group` / `cp_async_wait_group` (`python/flydsl/expr/ixdl/`) |
| **Pipeline sync** | `sl_waitmem`, `sl_pipebar_arrive`, `sl_pipebar_wait` for software-pipelined kernels |
| **Tensor core MMA** | `MRMma` atom (16×16×16 f16 TCU), MMA-coupled S2R via `make_tiled_copy_A/B` |
| **Production HGEMM** | `kernels.iluvatar_mr_hgemm` — double-buffered G2S, Ki-deferred S2R/MMA, configurable epilogue and major pattern |
| **JIT runtime** | `libfly_iluvatar_jit_runtime.so`, `FLYDSL_RUNTIME_KIND=iluvatar` |
| **Unit tests** | `tests/unit/test_iluvatar_*` (backend, runtime, binary pipeline, async copy & MMA) |

## Supported hardware

| Item | Details |
|------|---------|
| **Primary target** | **ivcore11** (default `ARCH`) — Iluvatar **BI-V150**, **BI-V150S**, **MR-100**, **MR-50** |
| **Future chips** | `ARCH=ivcore30` (and other ixdl chip strings) are accepted by the compile backend when the IXDL toolchain supports them |
| **Warp size** | 64 lanes |
| **Block shared memory** | 128 KiB per CTA (ivcore11 device property) |
| **Host API** | CUDA-compatible PyTorch tensors and streams (`torch.cuda.*`) with the Iluvatar driver stack |

> **Note:** The main FlyDSL README lists AMD ROCm platforms. Iluvatar is a separate
> backend; enable it at CMake configure time (see below).

## Build

Iluvatar is optional and off by default. **MLIR must come from [ixcc](https://github.com/IluvatarCorex/ixcc)** (Iluvatar LLVM fork with `IXDL` dialect), **not** from FlyDSL `scripts/build_llvm.sh` (upstream ROCm LLVM lacks `ixdl-attach-target` / fatbin lowering).

**Prerequisites**

1. **ixcc MLIR** — build ixcc with `mlir;clang;lld`, `MLIR_ENABLE_BINDINGS_PYTHON=ON`, then set `MLIR_DIR` to `…/lib/cmake/mlir` in the ixcc build or install tree.
2. **CoreX / CUDA-compatible toolkit** — `CUDAToolkit_ROOT` for `libfly_iluvatar_jit_runtime.so` and runtime driver loading.

```bash
export MLIR_DIR=~/sw_home/sdk/ixcc/build-flydsl/lib/cmake/mlir
export CUDAToolkit_ROOT=/path/to/corex

cmake -S . -B build-fly \
  -DFLYDSL_BACKENDS="iluvatar" \
  -DMLIR_DIR="${MLIR_DIR}" \
  -DCUDAToolkit_ROOT="${CUDAToolkit_ROOT}"
cmake --build build-fly -j$(nproc)
pip install -e .

# Sanity check: fly-opt must list IXDL passes
build-fly/bin/fly-opt --help | grep ixdl-attach-target
```

If not using editable install:

```bash
export PYTHONPATH="${PWD}/build-fly/python_packages:${PWD}:${PYTHONPATH}"
export LD_LIBRARY_PATH="${CUDAToolkit_ROOT}/lib64:${PWD}/build-fly/python_packages/flydsl/_mlir/_mlir_libs:${LD_LIBRARY_PATH}"
```

## Environment

| Variable | Typical value | Purpose |
|----------|---------------|---------|
| `FLYDSL_COMPILE_BACKEND` | `iluvatar` | Select Iluvatar compile pipeline |
| `FLYDSL_RUNTIME_KIND` | `iluvatar` | Select Iluvatar JIT runtime |
| `ARCH` | `ivcore11` | IXDL chip target (override per card generation) |
| `COMPILE_ONLY` | `1` | Compile without device execution (CI / no GPU) |
| `FLYDSL_RUNTIME_ENABLE_CACHE` | `0` | Disable disk cache while iterating on kernel or pass changes |

Iluvatar examples set the first three via `os.environ.setdefault(...)`.

## Examples

Start here after a successful Iluvatar build:

| Example | Purpose |
|---------|---------|
| [`examples/02-tiledCopy-iluvatar-mr.py`](examples/02-tiledCopy-iluvatar-mr.py) | **Teaching** TiledCopy + SME async G2S/S2R on a single 16×16 tile per warp; explicit `cp_async_wait`; good for layout/swizzle debugging |
| [`examples/03-tiledMma-iluvatar-mr-pipeline-hgemm.py`](examples/03-tiledMma-iluvatar-mr-pipeline-hgemm.py) | **Check / bench harness** for the pipelined HGEMM kernel (`--check`, `--bench`, CTA presets, epilogue modes) |

```bash
export FLYDSL_COMPILE_BACKEND=iluvatar
export FLYDSL_RUNTIME_KIND=iluvatar
export ARCH=ivcore11

# Correctness (small shapes)
python examples/03-tiledMma-iluvatar-mr-pipeline-hgemm.py --check

# Benchmark (defaults: 1024×1024×512)
python examples/03-tiledMma-iluvatar-mr-pipeline-hgemm.py --bench

# Peak-shape reference run (see performance table below)
python examples/03-tiledMma-iluvatar-mr-pipeline-hgemm.py --bench \
  --m 4096 --n 4096 --k 4096 --cta 1024 --k-rep 2 \
  --epilogue no_c_read --epilogue-store shfl
```

**Production import** (same algorithm, API-stable tuning kwargs):

```python
from kernels.iluvatar_mr_hgemm import compile_iluvatar_mr_hgemm

launch = compile_iluvatar_mr_hgemm(
    M=4096, N=4096, K=4096,
    major_pattern="nt",       # G2S layout tag for A/B (A then B: n=row SME, t=col SME)
    epilogue="no_c_read",      # D = A @ B.T, fp16, no C read
    epilogue_store="shfl",     # warp-shuffle epilogue (fastest for no_c_read)
    k_rep=2,                   # BK = 16 * k_rep = 32
)
launch(A, B, C, stream=torch.cuda.Stream())
```

See `kernels/iluvatar_mr_hgemm.py` module docstring for all tuning parameters
(`major_pattern`, CTA presets `1024` / `2048`, `read_c_accum` epilogue, etc.).

## HGEMM performance reference

Measured on **Iluvatar BI-V150S** (`ARCH=ivcore11`), using `kernels.iluvatar_mr_hgemm`
(modular G2S / S2R / epilogue helpers) with:

- CTA preset **1024** (16 warps × 64 lanes → **256×256** output tile per block)
- **`k_rep=2`** → **BK = 32**
- **`epilogue=no_c_read`**, **`epilogue_store=shfl`**
- ROCm-style K-loop: outer `fx.range` + inner `range_constexpr(K_LOOP_UNROLL=2)`
- JIT launch via `flyc.compile()` (same path as the example bench harness)

TFLOPS below are **medians of 3 runs** (warmup=15, iters=30 per run, CUDA events).
TFLOPS = `2·M·N·K / time`. Reproduce with::

    python examples/03-tiledMma-iluvatar-mr-pipeline-hgemm.py --bench \
      --epilogue no_c_read --epilogue-store shfl --k-rep 2 \
      --m <M> --n <N> --k <K> [--major-pattern nt]

### Square GEMM by `major_pattern` (fp16, `no_c_read` + `shfl`)

| Pattern | 1024³ TFLOPS | 2048³ TFLOPS | 4096³ TFLOPS |
|---------|-------------|-------------|-------------|
| `nn` | 68.9 | 94.8 | 99.5 |
| `tn` | 65.3 | 93.7 | **101.3** |
| `nt` | 74.5 | 94.0 | 101.0 |
| `tt` | 69.3 | 94.5 | **101.3** |

At **4096³**, `tn` / `tt` peak at **~101 TFLOPS**; `nt` is within **0.3 T**;
`nn` is **~99.5 T** (~1.5 T below peak). **2048³** is **~94–95 TFLOPS** across
patterns (`tn` ~93.7 T). **1024³** spans **~65–75 TFLOPS** (`nt` fastest;
short-kernel timing is sensitive to launch/JIT overhead — bench through
`flyc.compile()` as in example 03).

### `major_pattern` (G2S global layout tags)

Logical tensors are always `A(m,k)`, `B(n,k)`. Two-letter tags encode A then B
(`n` = NoTrans / row SME, `t` = Trans / col SME). The pattern selects how SME
global views map to those layouts (not the epilogue store mode):

| Pattern | A G2S / atom | B G2S / atom | Host layout (A, B) | Native kernel path? |
|---------|--------------|--------------|--------------------|---------------------|
| `nn` | row / `Row16b` | row / `Row16b` | `(m,k)`, `(k,n)` | host physical layout |
| `tn` | col / `Col` | row / `Row16b` | `(k,m)`, `(k,n)` | host physical layout |
| `nt` | row / `Row16b` | col / `Col` | `(m,k)`, `(n,k)` | **yes** (default) |
| `tt` | col / `Col` | col / `Col` | `(k,m)`, `(n,k)` | host physical layout |

Choose the pattern that matches your framework tensor layouts; peak TFLOPS at 4k are
similar across all four when host tensors use the expected physical layout.

### Epilogue modes

| Mode | Compute | Output dtype | Global C read | Typical use |
|------|---------|--------------|---------------|-------------|
| `no_c_read` | `D = A @ B.T` | fp16 | No | Inference GEMM, peak TFLOPS |
| `read_c_accum` | `C = A @ B.T + C` | fp32 | Yes | Training / accumulation |

`epilogue_store` applies to `no_c_read` only: **`shfl`** (default, fastest) or
**`tiled`** (`trunc_f` + `UniversalCopy16b`).

## Local correctness gate (no CI)

Use example **03** on a machine with an Iluvatar GPU. Default `--check-shape` is
`256 256 64`; default `--epilogue both` runs `no_c_read` and `read_c_accum`
sequentially for the chosen `--major-pattern`:

```bash
FLYDSL_COMPILE_BACKEND=iluvatar FLYDSL_RUNTIME_KIND=iluvatar \
  python examples/03-tiledMma-iluvatar-mr-pipeline-hgemm.py --check \
  --major-pattern nt
```

All four `major_pattern` values (`no_c_read` and `read_c_accum` each):

```bash
for p in nn tn nt tt; do
  FLYDSL_COMPILE_BACKEND=iluvatar FLYDSL_RUNTIME_KIND=iluvatar \
    python examples/03-tiledMma-iluvatar-mr-pipeline-hgemm.py --check \
    --major-pattern "$p" || exit 1
done
```

Optional shape override:

```bash
FLYDSL_COMPILE_BACKEND=iluvatar FLYDSL_RUNTIME_KIND=iluvatar \
  python examples/03-tiledMma-iluvatar-mr-pipeline-hgemm.py --check \
  --check-shape 256 256 64 --major-pattern nt
```

All four `major_pattern` values pass at the default check shape. Staged device unit
tests (`test_iluvatar_mr_*`) additionally cover G2S, S2R, MMA, epilogue, and
full-pipeline shapes including `k_rep=2` and multi-CTA cases.

Example 03 exits non-zero if any check fails, so it can be used as a manual
pre-bench / pre-commit gate on machines without CI.

### CTA presets

| Preset | Warps (M×N) | Warp tile | Block output tile | `k_rep` guidance |
|--------|-------------|-----------|-------------------|------------------|
| `1024` | 4×4 | 64×64 | 256×256 | `k_rep=2` for peak; `k_rep=4` (BK=64) for preset default smem |
| `2048` | 4×8 | 64×32 | 256×256 | Usually `k_rep ≥ 4` for even SME work within 128 KiB smem |

## Tests

Compile-only backend smoke (no GPU):

```bash
COMPILE_ONLY=1 FLYDSL_COMPILE_BACKEND=iluvatar FLYDSL_RUNTIME_KIND=iluvatar \
  python3 -m pytest tests/unit/test_iluvatar_compile_backend.py -v
```

With device (Iluvatar driver + CUDA-enabled PyTorch):

```bash
FLYDSL_COMPILE_BACKEND=iluvatar FLYDSL_RUNTIME_KIND=iluvatar \
  python3 -m pytest tests/unit/test_iluvatar_mr_async_cp_device.py \
                   tests/unit/test_iluvatar_mr_mma_pipeline_device.py \
                   tests/unit/test_iluvatar_jit_launch_smoke.py -v
```

## Known issues

| Area | Status | Notes |
|------|--------|-------|
| **Small-shape HGEMM perf** | Unstable / sub-peak | At **1024³** and smaller M/N/K, measured TFLOPS are far below **4096³** peak (~101 T) and vary with launch/JIT overhead. Bench through `flyc.compile()` (see example 03), not raw `@flyc.jit` launch. Treat small-shape numbers as indicative only. |
| **B8 / INT8 GEMM** | Not implemented | Low-level pieces exist (`MRAsyncCpRow8b`, `MRMma` i8 path in FlyIXDL), but there is no pipelined **INT8/B8 GEMM** kernel module comparable to `kernels.iluvatar_mr_hgemm`. |
| **Other production kernels** | ROCm-only today | The main FlyDSL `kernels/` portfolio (FP8/INT4 preshuffle GEMM, blockscale GEMM, MoE, paged/flash attention, LayerNorm/RMSNorm/Softmax, fused RoPE, all-reduce, quant, etc.) has **no Iluvatar port** yet. Only **f16 HGEMM** (`kernels.iluvatar_mr_hgemm`) and teaching/unit-test coverage are in tree. |
| **S2R register pressure** | Tuning item | Shared→register uses generic `UniversalCopy32b` tiling rather than a TCU-specialized load; SRF usage on large shapes is higher than ideal and may leave headroom on the table. |
| **Multi-GPU** | Not supported | No Iluvatar multi-device runtime or collective kernels (e.g. custom all-reduce). |
| **ivcore30+** | Bring-up incomplete | `ARCH=ivcore30` is accepted by the compile backend; device validation and kernel tuning on newer chips are ongoing. |
