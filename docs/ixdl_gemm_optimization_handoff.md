# IXDL GEMM Optimization Handoff

This note summarizes the current FlyDSL IXDL GEMM optimization work, the toolchain setup, and the current state of performance/debugging. It is intended as a handoff for another agent.

## Project And Toolchain

- FlyDSL repo: `/home/caokefan/dev/FlyDSL`
- Iluvatar software stack enable script: `~/sw_home/enable`
- ixcc build: `/home/caokefan/sw_home/sdk/ixcc/build`
- ixcc / MLIR source: `/home/caokefan/sw_home/sdk/ixcc/mlir`
- Iluvatar CUTLASS / ixmma reference: `/home/caokefan/sw_home/sdk/cutlass`

Useful CUTLASS reference files:

- `/home/caokefan/sw_home/sdk/cutlass/include/cute/arch/mma_ix11.hpp`
- `/home/caokefan/sw_home/sdk/cutlass/include/cute/atom/mma_traits_ix11.hpp`
- `/home/caokefan/sw_home/sdk/cutlass/include/cute/arch/copy_ix11_sme.hpp`
- `/home/caokefan/sw_home/sdk/cutlass/include/cute/atom/copy_traits_ix11_sme.hpp`

Enable the FlyDSL Python environment:

```bash
cd /home/caokefan/dev/FlyDSL
source .venv/bin/activate
export PYTHONPATH=/home/caokefan/dev/FlyDSL/build-fly/python_packages:$PYTHONPATH
```

Build everything:

```bash
cd /home/caokefan/dev/FlyDSL
ninja -C build-fly
```

Build only `fly-opt`:

```bash
ninja -C build-fly fly-opt
```

When changing C++ passes used from Python, rebuild the full project, not only `fly-opt`, so the Python shared libraries relink.

## Optimization Goal

The goal is to make FlyDSL's IX11 bf16 GEMM codegen closer to ixmma, especially by reducing ISA `sl_wait` instructions while preserving `slb_blkload`.

Current main FlyDSL benchmark:

- `examples/25-gemm-ixdl-bf16-swizzled-pipebar-ixmma-calcpipe.py`
- Config: `M=N=K=4096`, `warps=4x4`, `k_rep=2`, `copy_bits=32`

Benchmark command:

```bash
source .venv/bin/activate
PYTHONPATH=/home/caokefan/dev/FlyDSL/build-fly/python_packages:$PYTHONPATH \
python examples/25-gemm-ixdl-bf16-swizzled-pipebar-ixmma-calcpipe.py \
  --shape 4096 4096 4096 \
  --check-shape 256 256 128 \
  --iters 10 --warmup 3 \
  --warps 4 4 --k-rep 2 --copy-bits 32
```

Recent performance:

- `1518.6 us/iter`, `90.50 TFLOPS`
- `1559.2 us/iter`, `88.15 TFLOPS`

The machine warned that the GPU may have been busy, so treat this as roughly `88-90 TFLOPS`.

## ISA Status

Current FlyDSL full-shape ISA:

- total `sl_wait`: `31`
- hot-loop `sl_wait`: `8`
- `ml_slb_blkload_x1`: `24`
- `ml_slb_load_b32x1`: `72`
- `ml_matrix_mad`: `96`

ixmma reference:

- total `sl_wait`: `26`
- hot-loop `sl_wait`: `5`
- `ml_slb_blkload_x1`: `24`
- `ml_slb_load_b32x1`: `72`
- `ml_matrix_mad`: `96`

The instruction mix is mostly aligned. The remaining gap is mainly scheduling / wait insertion: FlyDSL has 8 hot-loop waits, ixmma has 5.

Useful ISA generation command:

```bash
llc -mtriple=bi-iluvatar-ilurt -mcpu=ivcore11 input.ll -o output.s
```

## Main Experiments So Far

- `25-gemm-...-ixmma-calcpipe.py`: current best FlyDSL mainline, modeled after ixmma's `CalcMMPerBLKK` pipeline.
- `26-...-ixmma-ssa.py`: explicit SSA MMAD / packed `<2 x i32>` fragment experiment. LLVM PHI became closer to ixmma, but `sl_wait` did not improve.
- `27` / `28`: S2R hoisting and im-outer MMAD ordering. LLVM IR changed, but `llc` still produced the same conservative wait pattern.
- `29`: loop-carried stage base offsets. Correct, but no `sl_wait` improvement.
- `30`: manual S2R with `ptr_load`. Correct after fixing swizzle math, but it destroyed `slb_blkload` recognition and regressed to more scalar SLB loads. Do not continue down this path.
- Shared GEP split experiment: splitting `base + lane` into two GEPs also destroyed `slb_blkload` after a full rebuild. This was reverted.

Important conclusion: keep the `UniversalCopy + blkload-friendly lowering` path. Handwritten S2R loads or changed GEP shape can easily break `slb_blkload`.

## Current Compiler Changes To Know About

Current retained optimization is in:

- `lib/Conversion/FlyToROCDL/FlyToROCDL.cpp`

Added local CSE for IXDL address helper ops after IX11 S2R-friendly load rewrites:

- CSE simple integer address arithmetic in the same block.
- CSE `ixdl.lane.id`.
- CSE identical `ixdl.readlane`.

Observed effect:

- `ixdl.lane.id`: `72 -> 2`
- `ixdl.readlane`: `72 -> 24`
- `arith.divsi`: `73 -> 25`
- `slb_blkload`: preserved at `24`
- `sl_wait`: unchanged at `31`

So this cleans up IR but does not close the wait-count gap.

## Backend-Independent Issues Found

### UniversalCopy64/128 swizzled S2R handling

`UniversalCopy64b/128b` for swizzled shared -> register S2R is not fully supported by the current FlyDSL pass pipeline.

Related files:

- `include/flydsl/Dialect/Fly/Utils/UniversalCopyUtils.h`
- `lib/Dialect/Fly/IR/FlyOps.cpp`
- `lib/Dialect/Fly/Transforms/LayoutLowering.cpp`
- `tests/mlir/Transforms/promote_regmem_to_vectorssa_universal_copy64_swizzled_s2r.mlir`
- `tests/mlir/Transforms/convert_fly_to_rocdl_universal_copy_strided.mlir`

Current handling adds verification / fast path / repro coverage so invalid cases fail explicitly rather than silently leaving bad IR such as `ub.poison`. This is not a full implementation of every swizzled `UniversalCopy64/128` S2R case.

### Not a pre-existing bug

`python/flydsl/expr/primitive.py::mma_atom_call_ssa` was added during experiments. A temporary argument-order issue in that new wrapper should not be counted as a pre-existing FlyDSL bug.

## Recommended Next Step

Do not blindly tune `k_rep`, and do not handwrite S2R loads unless you first prove `slb_blkload` is preserved.

The highest-value next investigation is backend-facing:

1. Compare FlyDSL and ixmma hot-loop MIR / ISA around S2R loads, MMADs, and `sl_wait`.
2. Determine why ixmma schedules the same rough instruction mix with 5 hot-loop waits while FlyDSL gets 8.
3. Keep checking that any IR-shape change preserves:
   - `ml_slb_blkload_x1 == 24`
   - `ml_slb_load_b32x1 == 72`
   - `ml_matrix_mad == 96`

