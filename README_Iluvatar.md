# FlyDSL on Iluvatar (FlyIXDL)

FlyDSL includes an **Iluvatar** compile and runtime backend targeting the **FlyIXDL**
dialect. Kernels are authored in the same Python layout DSL and `@flyc.kernel` /
`@flyc.jit` APIs as the ROCm path; lowering goes through `fly` -> `FlyIXDL` -> IXDL
LLVM IR -> device fatbin.

This document covers Iluvatar-specific features, supported hardware, runnable
examples, and HGEMM / IGEMM performance vs hand-tuned kernels.

## Features

| Area | What is provided |
|------|------------------|
| **Dialect & lowering** | `FlyIXDL` dialect, `convert-fly-to-ixdl`, `gpu-to-ixdl`, `ixdl-attach-target` pipeline (`python/flydsl/compiler/backends/iluvatar.py`) |
| **Layout algebra** | Same Fly layout API as ROCm (`logical_divide`, `copy_atom_call`, `make_tiled_copy_*`, ...) |
| **SME async copy** | `MRAsyncCpRow8b` / `Row16b` / `Col`, `make_sme_gmem_tensor`, `make_sme_shared_layout` / `make_sme_shared_layout_k_spanning`, `cp_async_commit_group` / `cp_async_wait_group` (`python/flydsl/expr/ixdl/`) |
| **CQ SMEX G2S** | `CQSmexCp` (mtx/plain); `row_mask` / `col_mask` are arbitrary bit prefixes; SME gmem global row stride must be **16B-aligned** (`make_sme_gmem_tensor` fail-fasts; hardware otherwise silently truncates the low 4 bits) |
| **CQ SMEX matrix copy** | `CQSmexCp(layout="mtx")` G2S + `CQMtxLoadn` S2R for 8/16-bit CQ MMA A/B fragments |
| **Pipeline sync** | **MR:** `sl_waitmem` + `sl_pipebar_arrive` / `sl_pipebar_wait` (`pipe-bar`). **CQ:** `nbarrier_reach` / `nbarrier_wait` / `nbarrier_sync` (`named-bar` / `named-bar-sync`). Do not emit pipebar on CQ. See `python/flydsl/expr/ixdl/sync.py` |
| **Tensor core MMA** | `MRMma` -- **16x16x16 f16** and **16x16x32 i8->i32**; MMA-coupled S2R via `make_tiled_copy_A/B` |
| **Production HGEMM** | `kernels.gemm.iluvatar.mr.hgemm` -- double-buffered G2S, Ki-deferred S2R/MMA, configurable epilogue / major pattern |
| **Production IGEMM** | `kernels.gemm.iluvatar.mr.igemm` -- int8xint8 -> i32/i8/`scaled_bf16`/`scaled_fp16`, same MR SME pipeline helpers as HGEMM (`mr_gemm_*`) |
| **Flex-attention L3** | `compile_iluvatar_flex_attention` (dense / varlen / paged) + Torch entry `flydsl_flex_attn_func` in `kernels.attention.iluvatar`; vanilla bf16 `D in {128,256}` (no mods / no explicit `tile_config`) routes to `flash_attn_varlen_func` unless `FLYDSL_FLEX_FA_FASTPATH=0`; optional `tile_config` (`{32,64}x{32,64}`, default 64x64) + dense-only `autotune_iluvatar_flex_attention_tile`; causal / SWA / softcap; f16/bf16; D in {64,128,256}; GQA; dense alibi/score_bias; `score_mod=TracedScoreMod` on dense, varlen, and paged; dense `create_block_mask` / `mask_mod` / `has_block_mask`; varlen `create_block_masks_varlen` / `block_masks` / `mask_mod`; paged `create_block_masks_paged` / `block_masks` / `mask_mod` (dispatcher packs tables); dense `return_lse` + `compile_iluvatar_flex_attention_bwd` (MHA, D in {64,128}, correctness-first) |
| **MoE GEMM V1** | `kernels.moe.iluvatar.mr` -- sorted grouped GEMM, `int8` / `int8smooth`, f16/bf16/f32 out (A gather + B SME) |
| **GEMV V1** | `kernels.gemm.iluvatar.gemv` -- `F.linear`-aligned M=1, fp16/bf16, fp32 accum |
| **JIT runtime** | `libfly_iluvatar_jit_runtime.so`, `FLYDSL_RUNTIME_KIND=iluvatar` |
| **Unit / device tests** | `tests/kernels/test_iluvatar_*` (device kernels: G2S/S2R/MMA/epilogue/HGEMM stages, scaled IGEMM, GEMV, RMSNorm, LayerNorm, flex-attention, atomic CAS, split-K) + `tests/unit/test_iluvatar_*` (backend, runtime, JIT launch/binary smokes); select via `-m iluvatar_lower`. IGEMM i32/i8 also via example `--check` |
| **CI (optional)** | Iluvatar `ci-core` / `ci-device`, IX toolchain refresh, publish-image, perf-daily (see `.github/workflows/*iluvatar*`) |

### CQ shared-memory access paths

CQ has two incompatible shared-buffer contracts:

- `SmexMtx`: `CQSmexCp(layout="mtx")` writes the CQ matrix format and
  `CQMtxLoadn` reads it with `ixdl.mtx_loadn_*x2`. `loadn16` / `loadn64`
  only pair with the 16-row / 64-row G2S tile height -- both emit the same
  x2 load (64 bits/lane); a 64-row footprint needs multiple atom calls with
  different EmPart / slot bases. Row gather produces the CQ MMA A fragment;
  column gather produces B.
- `LegacySme`: MR-style SME copies and scalar/legacy S2R use the historical
  byte-swizzled shared layout.

A shared buffer must use one contract for both G2S and S2R. Do not write with
one path and read with the other. "Swizzle bypass" on `SmexMtx` means the
legacy byte-swizzle transform leaves shared indices unchanged; it does not mean
the data is unswizzled. SMEX writes the matrix-format mtx-index XOR mapping, and
the matrix-load instruction consumes that mapping in hardware.

### Kernel package layout

```text
kernels/gemm/iluvatar/
  common.py          # GemmLayout, WARP_SIZE, parse_major_pattern, ...
  epilogue.py        # HGEMM fp16 stores + IGEMM i32 / i8 / scaled f16|bf16 stores
  gemv.py            # GEMV V1
  mr/
    common.py        # MrOperandGeom, mr_stage_smem_ab, byte_perm, ...
    hgemm.py         # compile_iluvatar_mr_hgemm
    igemm.py         # compile_iluvatar_mr_igemm
    operand_copy.py  # mr_gemm_g2s_issue_* (dtype-generic)
    s2r.py           # mr_gemm_s2r_* (dtype-generic; i8 k-spanning)

kernels/moe/iluvatar/mr/
  moe_gemm.py        # compile_iluvatar_mr_moe_gemm (int8 / int8smooth V1)

kernels/attention/iluvatar/
  flex_attention.py       # compile_iluvatar_flex_attention (dense / varlen / paged; optional return_lse)
  flex_attention_bwd.py   # compile_iluvatar_flex_attention_bwd (dense MHA)
  flex_attn_interface.py  # flydsl_flex_attn_func + autotune_*_tile
  flex_attn_fa_backend.py # vanilla flydsl_flex_attn_func -> flash_attn_varlen_func
```

## Supported hardware

| Item | Details |
|------|---------|
| **Primary target** | **ivcore11** (default `ARCH`) -- Iluvatar **BI-V150**, **BI-V150S**, **MR-100**, **MR-50** |
| **Future chips** | `ARCH=ivcore30` (and other ixdl chip strings) are accepted by the compile backend when the IXDL toolchain supports them |
| **Warp size** | 64 lanes |
| **Block shared memory** | 128 KiB per CTA (ivcore11 device property) |
| **Host API** | CUDA-compatible PyTorch tensors and streams (`torch.cuda.*`) with the Iluvatar driver stack |

> **Note:** The main FlyDSL README lists AMD ROCm platforms. Iluvatar is a separate
> backend; enable it at CMake configure time (see below).

## Build

Iluvatar is optional and off by default. **MLIR must come from [ixcc](https://github.com/IluvatarCorex/ixcc)** (Iluvatar LLVM fork with `IXDL` dialect), **not** from FlyDSL `scripts/build_llvm.sh` (upstream ROCm LLVM lacks `ixdl-attach-target` / fatbin lowering).

**Prerequisites**

1. **ixcc MLIR** -- build ixcc with `mlir;clang;lld`, `MLIR_ENABLE_BINDINGS_PYTHON=ON`, then set `MLIR_DIR` to `.../lib/cmake/mlir` in the ixcc build or install tree.
2. **CoreX / CUDA-compatible toolkit** -- `CUDAToolkit_ROOT` for `libfly_iluvatar_jit_runtime.so` and runtime driver loading.

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
| `FLYDSL_ILUVATAR_RUN_FLEX_ATTN_PERF` | `1` | Opt-in flex-attention large-shape perf probe (`tests/kernels/test_iluvatar_flex_attention_device.py`) |
| `FLYDSL_FLEX_FA_FASTPATH` | `0` / `false` | Disable vanilla `flydsl_flex_attn_func` routing to `flash_attn_varlen_func` (default on) |
| `FLYDSL_ILUVATAR_SMOKE_BLOB_PATH` / `FLYDSL_ILUVATAR_SMOKE_KERNEL` | path / symbol | Blob + kernel name consumed by `tests/unit/test_iluvatar_runtime_smoke.py`; tests are skipped when unset |

Iluvatar examples set the first three via `os.environ.setdefault(...)`.

## Examples

Start here after a successful Iluvatar build:

| Example | Purpose |
|---------|---------|
| [`examples/02-tiledCopy-iluvatar-mr.py`](examples/02-tiledCopy-iluvatar-mr.py) | **Teaching** TiledCopy + SME async G2S/S2R on a single 16x16 tile per warp; explicit `cp_async_wait`; good for layout/swizzle debugging |
| [`examples/03-tiledMma-iluvatar-mr-pipeline-hgemm.py`](examples/03-tiledMma-iluvatar-mr-pipeline-hgemm.py) | **Check / bench** harness for pipelined **f16 HGEMM** (`--check`, `--bench`, CTA presets, epilogue / store modes) |
| [`examples/03-tiledMma-iluvatar-mr-pipeline-igemm.py`](examples/03-tiledMma-iluvatar-mr-pipeline-igemm.py) | **Check / bench** harness for pipelined **int8 IGEMM** (`--epilogue i32\|i8\|scaled_bf16\|scaled_fp16`, four `major_pattern`s; scaled also checks M not a multiple of BM) |

```bash
export FLYDSL_COMPILE_BACKEND=iluvatar
export FLYDSL_RUNTIME_KIND=iluvatar
export ARCH=ivcore11

# HGEMM correctness (small shapes)
python examples/03-tiledMma-iluvatar-mr-pipeline-hgemm.py --check

# IGEMM correctness (default check shape; try --epilogue i8 / scaled_bf16 / other patterns)
python examples/03-tiledMma-iluvatar-mr-pipeline-igemm.py --check --major-pattern tn --epilogue i32
python examples/03-tiledMma-iluvatar-mr-pipeline-igemm.py --check --major-pattern tn --epilogue scaled_bf16

# HGEMM peak-shape reference (Gate2-style contract -- see performance section)
python examples/03-tiledMma-iluvatar-mr-pipeline-hgemm.py --bench \
  --m 4096 --n 4096 --k 4096 --cta 1024 --k-atoms 2 \
  --epilogue no_c_read --epilogue-store shfl --major-pattern tn \
  --warmup 1 --iters 100
```

**Production imports:**

```python
import flydsl.expr as fx
from kernels.gemm.iluvatar.mr.hgemm import compile_iluvatar_mr_hgemm
from kernels.gemm.iluvatar.mr.igemm import compile_iluvatar_mr_igemm
from kernels.gemm.iluvatar.gemv import compile_iluvatar_gemv

launch_h = compile_iluvatar_mr_hgemm(
    M=4096, N=4096, K=4096,
    elem_dtype=fx.Float16,      # or fx.BFloat16; A/B (and no_c_read C) dtype
    major_pattern="tn",         # CUTLASS 3.x BLAS tag (default: A/B both K-major)
    epilogue="no_c_read",       # D = A @ B.T, f16/bf16, no C read
    epilogue_store="shfl",      # warp-shuffle epilogue (fastest for no_c_read)
    k_atoms=2,                  # BK = 16 * k_atoms = 32
)
launch_h(A, B, C, stream=torch.cuda.Stream())

launch_i = compile_iluvatar_mr_igemm(
    M=1024, N=1024, K=1024,
    major_pattern="tn",
    epilogue="i32",             # or "i8" / "scaled_bf16" / "scaled_fp16"
)
launch_i(A_i8, B_i8, C_i32, stream=torch.cuda.Stream())

launch_v = compile_iluvatar_gemv(N=4096, K=4096)
y = launch_v(x, w, bias=None)   # x: [K] or [1,K]; w: [N,K]; fp16/bf16
```

See module docstrings under `kernels/gemm/iluvatar/` for tuning parameters
(`major_pattern`, CTA presets `1024` / `2048`, epilogue modes, GEMV tile constraints, etc.).

## Performance reference

### Add a daily performance case

The `perf-daily-iluvatar` workflow is configuration-driven. Its suite lives in
`.github/perf-kernels-iluvatar.json`; adding a supported case does not require
editing the workflow.

For a pytest-based benchmark:

1. Add a deterministic test that prints one final metrics marker:

   ```python
   print(
       "PERF_CASE_JSON="
       + json.dumps(
           {
               "metrics": {
                   "4096x8192.f16": {
                       "latency_us": latency_us,
                       "torch_latency_us": torch_latency_us,
                   }
               }
           }
       )
   )
   ```

   Point names and metric names are stable artifact keys. Metric values must be
   numeric.
2. Add a `pytest_perf` case to the suite:

   ```json
   {
     "id": "my_perf_case",
     "type": "pytest_perf",
     "name": "my_perf_case",
     "entry": "tests/kernels/test_my_perf_case.py",
     "params": {
       "markers": "iluvatar_lower",
       "extra_args": ["-k", "test_my_perf_case"]
     },
     "gate": {
       "metric": "latency_us"
     }
   }
   ```

   For flex-attention, put the daily matrix under `params.perf_config`
   (`shape` + `cases`; optional `warmup` / `iters`). The `pytest_perf` handler
   writes that object to `.perf/<id>.perf_config.json` and sets
   `FLYDSL_PERF_CONFIG_PATH` to the **repo-relative** path (so the file is
   visible inside the perf container where the workspace is mounted at
   `/workspace`). The flex perf test reads the file when present; with no path
   it falls back to the same matrix hardcoded in the test (six
   ``compile``-backend groups plus vanilla causal bf16 ``backend: fa`` ->
   metric key ``causal.bf16.fa``). ``backend`` must be omitted (``compile``
   is the implicit default and never appears as an explicit value or as a
   metric-key suffix) or set to ``"fa"``. Prefer editing the json to add /
   drop daily groups or change shape, instead of hardcoding a new matrix in
   pytest.

Use a stable `id` so later runs can find the previous result in the
`perf-metrics` artifact. A missing baseline (for example, on the first run)
produces `n/a` deltas and is not an error. Benchmark, parser, or toolchain
failures invalidate the sample and fail the workflow; a measured regression is
reported in the summary but does not block. The suite-level
`defaults.delta_flat_pct` controls the summary threshold; `gate.metric` selects
the metric compared with the previous run.

Existing GEMM-style scripts can use the `bench_tflops` case type. New
non-GEMM tests should prefer the `PERF_CASE_JSON` contract above. A new parser
under `.github/scripts/perf_parsers/` is only needed for a genuinely different
execution or output model.

Measured on **Iluvatar BI-V150S** (`ARCH=ivcore11`). All four `major_pattern`
values (`nt` / `tn` / `nn` / `tt`) x square sizes. **Grey** = matched hand-tuned
(100%); **green** = FlyDSL % of that baseline, taking the **best FlyDSL config**
per point -- fp16/bf16 hGEMM best of `k_atoms in {2,4}` (`no_c_read`+`shfl`,
`cta=1024`) vs `mr_gemm_fp16_db_trad_opt4 -T=...` (bf16 has no hand-tuned binary,
so the same fp16 opt4 peak is the baseline); iGEMM `k_rep=2`, `i32`, `cta=1024`
vs `opt6` (4096^3 also vs `opt4`, take higher). Sampling: `warmup=1`,
`iters=100`, median of 5 runs.

<p align="center">
  <img src="./docs/images/iluvatar-flydsl-vs-handtuned-pct.png" alt="FlyDSL vs hand-tuned hGEMM/iGEMM" width="100%"/>
</p>

Reproduce (hGEMM: try both `k_atoms` and keep the better; add `--dtype bf16` for the middle panel):

```bash
for p in nt tn nn tt; do
  for ka in 2 4; do
    python examples/03-tiledMma-iluvatar-mr-pipeline-hgemm.py --bench \
      --epilogue no_c_read --epilogue-store shfl --k-atoms "$ka" --cta 1024 \
      --major-pattern "$p" --warmup 1 --iters 100 --m <M> --n <N> --k <K>
  done
  python examples/03-tiledMma-iluvatar-mr-pipeline-igemm.py --bench \
    --epilogue i32 --cta 1024 --k-rep 2 --major-pattern "$p" \
    --warmup 1 --iters 100 --m <M> --n <N> --k <K>
done
```

### `major_pattern` (CUTLASS 3.x BLAS layout tags)

Logical tensors are always `A(m,k)`, `B(n,k)` (CuTe `MxK * NxK`). Tags are opaque
BLAS names -- **do not** decode each letter as that operand's M/N/K major mode.
Major modes describe logical layouts; host shapes are `tensor.shape` after remap.
Internally kernels use ``GemmLayout`` (`a_mn_major` / `b_mn_major` in
``kernels/gemm/iluvatar/common.py``), not integer pattern ids.
Table order matches the [CuTe gemm tutorial](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/0x_gemm_tutorial.md):

| Pattern | A major | B major | Host `A.shape`, `B.shape` | Native path? |
|---------|---------|---------|-----------------------------|--------------|
| `nt` | M | N | `(k,m)`, `(k,n)` | host transpose |
| `tn` | K | K | `(m,k)`, `(n,k)` | **yes** (default) |
| `nn` | M | K | `(k,m)`, `(n,k)` | host transpose |
| `tt` | K | N | `(m,k)`, `(k,n)` | host transpose |

Choose the pattern that matches your framework tensor layouts; peak TFLOPS at 4k are
similar across all four when host tensors use the expected physical layout.

### Epilogue modes (HGEMM)

| Mode | Compute | Output dtype | Global C read | Typical use |
|------|---------|--------------|---------------|-------------|
| `no_c_read` | `D = A @ B.T` | f16 / bf16 (`elem_dtype`) | No | Inference GEMM, peak TFLOPS |
| `read_c_accum` | `C = A @ B.T + C` | fp32 | Yes | Training / accumulation |

`epilogue_store` applies to `no_c_read` only: **`shfl`** (default, fastest) or
**`tiled`** (`trunc_f` + `UniversalCopy16b`).

IGEMM uses `--epilogue i32` (direct int32 store), `i8` (packed store with
truncating cast; no quant scale), or `scaled_bf16` / `scaled_fp16`
(`D = acc * scale_a[m] * scale_b[n] [+ bias]` via PackSlb). Scaled kernels
accept a live M that is not a multiple of BM (`allow_dynamic_m`).

### Global Split-K (HGEMM)

Split-K lives in a **separate** module so the non-split-K golden path in
`kernels/gemm/iluvatar/mr/hgemm.py` stays untouched (same layout as ROCm
`hgemm_splitk.py` vs `hgemm`):

* Non-split-K: `compile_iluvatar_mr_hgemm(...)` -- `kernels/gemm/iluvatar/mr/hgemm.py`
* Split-K: `compile_iluvatar_mr_hgemm_splitk(..., split_k, split_k_mode=...)` --
  `kernels/gemm/iluvatar/mr/hgemm_splitk.py` (`split_k > 1`, `epilogue=no_c_read`,
  `K % (split_k * bk) == 0` with `bk = 16 * k_atoms`)

The example harness routes `--split-k 1` (default) to the golden path and
`--split-k N>1` to the split-K entry. Launch stays `(A, B, C, stream=...)` --
workspace / turnstile locks are allocated and cached inside the split-K wrapper.

| Mode | CUTLASS analogue | Reduction | Extra resources |
|---|---|---|---|
| **`serial`** (default) | `SplitKSerial` | Single `grid.z=split_k` launch; per-tile GMEM locks + `llvm.cmpxchg` turnstile order load-add-store into C (f32 add, trunc store). MMA across K-slices can overlap. | `i32[tiles_m * tiles_n]` locks |
| **`parallel`** | `GemmSplitKParallel` | Each slice writes fp32 partials to workspace; a reduce kernel sums along `split_k` into C | `fp32[split_k, M, N]` workspace |
| **`atomic`** | (non-CUTLASS) | Zero C, then scalar `UniversalAtomicAdd` (`f16`/`bf16`) with `grid.z=split_k` | none |

`serial` / `atomic` truncate through the output dtype each partition, so absolute
error grows ~`sqrt(split_k)`. `parallel` keeps fp32 until the final trunc.
Cosine similarity vs `A @ B.T` stays > 0.999 in all modes.

```bash
# serial (default) / parallel / atomic -- bk=32 at k_atoms=2
FLYDSL_COMPILE_BACKEND=iluvatar FLYDSL_RUNTIME_KIND=iluvatar \
  python examples/03-tiledMma-iluvatar-mr-pipeline-hgemm.py --check \
  --dtype bf16 --split-k 4 --split-k-mode serial --check-shape 256 256 2048 --k-atoms 2
FLYDSL_COMPILE_BACKEND=iluvatar FLYDSL_RUNTIME_KIND=iluvatar \
  python examples/03-tiledMma-iluvatar-mr-pipeline-hgemm.py --check \
  --dtype fp16 --split-k 4 --split-k-mode parallel --check-shape 256 256 2048 --k-atoms 2
```

Device kernel coverage: `tests/kernels/test_iluvatar_mr_splitk_hgemm.py`
(selected by `-m iluvatar_lower`).

## Local correctness gate (no CI)

Use the example **03** harnesses on a machine with an Iluvatar GPU.

**HGEMM** -- default `--check-shape` is `256 256 64`; default `--epilogue both` runs
`no_c_read` and `read_c_accum` for the chosen `--major-pattern`:

```bash
FLYDSL_COMPILE_BACKEND=iluvatar FLYDSL_RUNTIME_KIND=iluvatar \
  python examples/03-tiledMma-iluvatar-mr-pipeline-hgemm.py --check \
  --major-pattern tn
```

All four `major_pattern` values:

```bash
for p in nt tn nn tt; do
  FLYDSL_COMPILE_BACKEND=iluvatar FLYDSL_RUNTIME_KIND=iluvatar \
    python examples/03-tiledMma-iluvatar-mr-pipeline-hgemm.py --check \
    --major-pattern "$p" || exit 1
done
```

**IGEMM** -- four patterns x `{i32,i8}`, plus scaled PackSlb on `tn` (also
exercises M=100, which is not a multiple of BM):

```bash
for p in tn nt nn tt; do
  for e in i32 i8; do
    FLYDSL_COMPILE_BACKEND=iluvatar FLYDSL_RUNTIME_KIND=iluvatar \
      python examples/03-tiledMma-iluvatar-mr-pipeline-igemm.py --check \
      --major-pattern "$p" --epilogue "$e" || exit 1
  done
done
FLYDSL_COMPILE_BACKEND=iluvatar FLYDSL_RUNTIME_KIND=iluvatar \
  python examples/03-tiledMma-iluvatar-mr-pipeline-igemm.py --check \
  --major-pattern tn --epilogue scaled_bf16
```

Staged device kernel tests (`tests/kernels/test_iluvatar_mr_*`,
`tests/kernels/test_iluvatar_gemv.py`) cover G2S, S2R, MMA, epilogue, HGEMM
pipeline stages, and GEMV. Select them with the `iluvatar_lower` pytest
marker (see the Tests section below).

Example harnesses exit non-zero if any check fails, so they can be used as a manual
pre-bench / pre-commit gate on machines without CI.

### CTA presets (HGEMM / IGEMM)

| Preset | Warps (MxN) | Warp tile | Block output tile | Guidance |
|--------|-------------|-----------|-------------------|----------|
| `1024` | 4x4 | 64x64 | 256x256 | HGEMM peak: `k_atoms=2`; IGEMM uses `k_rep` (BK = 32 x k_rep) |
| `2048` | 4x8 | 64x32 | 256x256 | HGEMM usually `k_atoms >= 4` for even SME work within 128 KiB smem |

M/N must be multiples of the CTA tile (**256** for these presets); **128^3** is not a
valid default-CTA shape.

## Tests

Compile-only backend smoke (no GPU):

```bash
COMPILE_ONLY=1 FLYDSL_COMPILE_BACKEND=iluvatar FLYDSL_RUNTIME_KIND=iluvatar \
  python3 -m pytest tests/unit/test_iluvatar_compile_backend.py -v
```

With device (Iluvatar driver + CUDA-enabled PyTorch):

```bash
export FLYDSL_COMPILE_BACKEND=iluvatar FLYDSL_RUNTIME_KIND=iluvatar

python3 -m pytest \
  tests/kernels/test_iluvatar_*.py \
  tests/unit/test_iluvatar_*.py \
  -m iluvatar_lower -v
```

The `iluvatar_lower` marker matches every Iluvatar L1b/L2 test (kernel
tests under `tests/kernels/`, plus the JIT / binary / runtime smokes under
`tests/unit/`). The four backend-agnostic `test_iluvatar_*` files in
`tests/unit/` intentionally lack the marker so they stay runnable in
environments without an Iluvatar lowering stack; ci-core covers those.

MLIR FileCheck (needs `fly-opt` + `FileCheck` on `PATH`):

```bash
# e.g. tests/mlir/Conversion/ixdl_*.mlir
```

## Known issues

| Area | Status | Notes |
|------|--------|-------|
| **Small-shape HGEMM perf** | Sub-peak / noisy | Below **~2048^3**, TFLOPS are far from 4k peak and sensitive to launch/JIT and `k_atoms`. Quote **Gate2 contract** medians; treat smaller shapes as indicative. |
| **B8 / INT8 GEMM** | **Implemented (MR IGEMM)** | `kernels.gemm.iluvatar.mr.igemm` + example 03-igemm; i32 / i8 / **scaled_bf16|scaled_fp16** (PackSlb) epilogues; four major patterns. `allow_dynamic_m` / `dynamic_m` for a live M that is not a multiple of BM. |
| **GEMV** | V1 only | `kernels.gemm.iluvatar.gemv` -- M=1, strict N/K tile divisibility; not a general batched GEMV. |
| **Flex-attention** | L3 forward (dense / varlen / paged) via `compile_*` + `flydsl_flex_attn_func`; vanilla bf16 D in {128,256} uses the FA fast path; optional `tile_config` / dense `autotune_*_tile`; `score_mod` on dense+varlen+paged; dense/varlen/paged `mask_mod` / BlockMask sparse skip; dense `return_lse` + `compile_*_bwd` (MHA bring-up) | Not a full PyTorch flex_attention compiler (no dynamo / arbitrary gather). Bwd is dense-MHA correctness-first (not MMA-opt). Relative to Torch SDPA still has a gap on long sequences. |
| **Other production kernels** | Mostly ROCm-only | Broader FlyDSL portfolio (FP8/INT4 preshuffle GEMM, CDNA flash-attn family, all-reduce, ...) still has **no** full Iluvatar port beyond HGEMM / IGEMM / GEMV / MoE GEMM / flex-attention V1 + teaching/unit coverage. |
| **S2R register pressure** | Tuning item | Shared->register uses generic `UniversalCopy32b` tiling rather than a TCU-specialized load; SRF usage on large shapes is higher than ideal and may leave headroom on the table. |
| **Multi-GPU** | Not supported | No Iluvatar multi-device runtime or collective kernels (e.g. custom all-reduce). |
| **ivcore30+** | Bring-up incomplete | `ARCH=ivcore30` is accepted by the compile backend; device validation and kernel tuning on newer chips are ongoing. |
