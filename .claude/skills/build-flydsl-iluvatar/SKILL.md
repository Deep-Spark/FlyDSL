---
name: build-flydsl-iluvatar
description: >
  Build the ixcc LLVM/MLIR dependency and FlyDSL with the Iluvatar backend,
  configure CoreX runtime and Python environment variables, and run Iluvatar
  compile-only, FileCheck, device, and example correctness tests. Use when
  setting up, rebuilding, testing, or troubleshooting FlyDSL for ivcore11,
  ivcore30, FlyIXDL, IXDL, or Iluvatar GPUs.
allowed-tools: Bash Read Grep Glob
---

# Build and Test FlyDSL for Iluvatar

Use this workflow for the optional Iluvatar backend. Do not use
`scripts/build_llvm.sh`: it builds upstream ROCm LLVM, which does not provide
the IXDL dialect, `ixdl-attach-target`, or Iluvatar code generation.

## 1. Define paths

Keep ixcc and FlyDSL in separate source and build directories. Adapt only the
three roots below:

```bash
export FLYDSL_ROOT=/path/to/FlyDSL
export IXCC_ROOT=/path/to/ixcc
export IXCC_BUILD="${IXCC_ROOT}/build-flydsl"
export FLY_BUILD="${FLYDSL_ROOT}/build-fly-iluvatar"
export FLY_BUILD_DIR="${FLY_BUILD}"
export COREX_ROOT=/path/to/corex
export JOBS="$(nproc)"
```

Use a dedicated FlyDSL build directory. A CMake cache configured for `rocdl`
must not be reused for `iluvatar`.

Activate the intended Python environment before configuring either project:

```bash
python3 -m pip install cmake ninja nanobind pybind11 numpy pytest
export PYTHON="$(command -v python3)"
export NANOBIND_DIR="$("${PYTHON}" -c \
  "import nanobind, os; print(os.path.join(os.path.dirname(nanobind.__file__), 'cmake'))")"
```

## 2. Build ixcc LLVM/MLIR

FlyDSL needs ixcc's MLIR libraries and CMake package, IXDL dialect, Iluvatar
LLVM target, Python-binding support, and `FileCheck`.

```bash
cmake -S "${IXCC_ROOT}/llvm" -B "${IXCC_BUILD}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="mlir;clang;lld" \
  -DLLVM_TARGETS_TO_BUILD="Iluvatar;X86" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DMLIR_BINDINGS_PYTHON_NB_DOMAIN=mlir \
  -DPython3_EXECUTABLE="${PYTHON}" \
  -Dnanobind_DIR="${NANOBIND_DIR}" \
  -DBUILD_SHARED_LIBS=OFF

cmake --build "${IXCC_BUILD}" -j"${JOBS}"
```

`X86` builds host-side tools; `Iluvatar` supplies device code generation.
`clang` and `lld` match the supported toolchain configuration even if a local
incremental FlyDSL build does not invoke their drivers directly.

Verify the required products before configuring FlyDSL:

```bash
test -f "${IXCC_BUILD}/lib/cmake/mlir/MLIRConfig.cmake"
test -x "${IXCC_BUILD}/bin/FileCheck"
"${IXCC_BUILD}/bin/llvm-config" --targets-built
"${IXCC_BUILD}/bin/mlir-opt" --show-dialects | grep -w ixdl
```

After ixcc source changes, rerun only:

```bash
cmake --build "${IXCC_BUILD}" -j"${JOBS}"
```

CMake/Ninja tracks TableGen, MLIR, translation, and Iluvatar target
dependencies. Reconfigure only when build options or Python environments
change.

## 3. Build FlyDSL with FlyIXDL

`CUDAToolkit_ROOT` must point to the CoreX CUDA-compatible toolkit containing
headers and the driver library needed by `libfly_iluvatar_jit_runtime.so`.

```bash
export MLIR_DIR="${IXCC_BUILD}/lib/cmake/mlir"
export CUDAToolkit_ROOT="${COREX_ROOT}"
export CUDA_HOME="${COREX_ROOT}"
export IXA_HOME="${COREX_ROOT}"

cmake -S "${FLYDSL_ROOT}" -B "${FLY_BUILD}" -G Ninja \
  -DFLYDSL_BACKENDS=iluvatar \
  -DMLIR_DIR="${MLIR_DIR}" \
  -DCUDAToolkit_ROOT="${CUDAToolkit_ROOT}" \
  -DPython3_EXECUTABLE="${PYTHON}" \
  -Dnanobind_DIR="${NANOBIND_DIR}"

cmake --build "${FLY_BUILD}" -j"${JOBS}"
```

Do not use `scripts/build.sh` for the first Iluvatar configure: its generic
path does not select `-DFLYDSL_BACKENDS=iluvatar`. It is safe to use
`cmake --build "${FLY_BUILD}"` for every incremental rebuild.

An editable install is optional:

```bash
cd "${FLYDSL_ROOT}"
python3 -m pip install -e .
```

Prefer the built package first during development so tests cannot accidentally
load bindings from another build tree.

## 4. Set the runtime environment

```bash
cd "${FLYDSL_ROOT}"

export PATH="${FLY_BUILD}/bin:${IXCC_BUILD}/bin:${PATH}"
export PYTHONPATH="${FLY_BUILD}/python_packages:${FLYDSL_ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${COREX_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${FLY_BUILD}/python_packages/flydsl/_mlir/_mlir_libs:${LD_LIBRARY_PATH}"

export FLYDSL_COMPILE_BACKEND=iluvatar
export FLYDSL_RUNTIME_KIND=iluvatar
export ARCH=ivcore11

export FLYDSL_ILUVATAR_FLY_OPT="${FLY_BUILD}/bin/fly-opt"
export FLYDSL_ILUVATAR_JIT_RUNTIME_LIB="${FLY_BUILD}/python_packages/flydsl/_mlir/_mlir_libs/libfly_iluvatar_jit_runtime.so"
```

Architecture selection:

- `ARCH=ivcore11`: MR / BI-V150 / MR-50 / MR-100 path.
- `ARCH=ivcore30`: CQ path, including FP8 and long-matrix MMA.
- Always pass the real chip. CQ must not rely on the default, which is MR.

Useful iteration variables:

```bash
export COMPILE_ONLY=1                  # compile without launching a GPU
export FLYDSL_RUNTIME_ENABLE_CACHE=0   # disable disk cache after C++ pass changes
export FLYDSL_DUMP_IR=1                # optional IR dumps
export FLYDSL_DUMP_DIR=/tmp/flydsl-ir
```

Unset `COMPILE_ONLY` before device correctness tests:

```bash
unset COMPILE_ONLY
```

## 5. Verify the build

```bash
test -x "${FLYDSL_ILUVATAR_FLY_OPT}"
test -f "${FLYDSL_ILUVATAR_JIT_RUNTIME_LIB}"
"${FLYDSL_ILUVATAR_FLY_OPT}" --help | grep ixdl-attach-target
python3 -c "import flydsl; import flydsl.expr.ixdl; print('FlyDSL Iluvatar import OK')"
```

If `ixdl-attach-target` is absent, FlyDSL was built against the wrong LLVM or
with the wrong backend cache. Check:

```bash
grep '^FLYDSL_BACKENDS:' "${FLY_BUILD}/CMakeCache.txt"
grep '^MLIR_DIR:' "${FLY_BUILD}/CMakeCache.txt"
```

## 6. Test ladder

Run the narrowest relevant tier first, then widen.

### Backend-agnostic Iluvatar unit tests

These do not require IXDL lowering or a GPU:

```bash
python3 -m pytest \
  tests/unit/test_iluvatar_backend_cmake.py \
  tests/unit/test_iluvatar_device_runtime.py \
  tests/unit/test_iluvatar_jit_runtime_resolution.py \
  -v
```

### Compile-only IXDL tests

```bash
COMPILE_ONLY=1 python3 -m pytest \
  tests/unit/test_iluvatar_binary_pipeline_smoke.py \
  tests/unit/test_iluvatar_compile_backend.py \
  -v
```

Set `ARCH=ivcore30` when the changed path is CQ-specific.

### MLIR FileCheck

Use ixcc's `FileCheck`, not an unrelated system LLVM:

```bash
"${FLYDSL_ILUVATAR_FLY_OPT}" \
  tests/mlir/Conversion/ixdl_cq_mma.mlir \
  --convert-fly-to-ixdl |
  "${IXCC_BUILD}/bin/FileCheck" \
  tests/mlir/Conversion/ixdl_cq_mma.mlir

"${FLYDSL_ILUVATAR_FLY_OPT}" \
  tests/mlir/Conversion/ixdl_cq_mma_invalid.mlir \
  --split-input-file --verify-diagnostics
```

For another `.mlir` file, execute its `// RUN:` command after replacing
`%fly-opt`, `FileCheck`, and `%s` with the paths above.

### Iluvatar device pytest

Requires an Iluvatar GPU, CoreX driver, and CUDA-compatible PyTorch:

```bash
unset COMPILE_ONLY
python3 -m pytest \
  tests/kernels/test_iluvatar_*.py \
  tests/unit/test_iluvatar_*.py \
  -m iluvatar_lower -v
```

The `iluvatar_lower` marker selects target-lowering and device tests. It does
not include the backend-agnostic unit tests listed earlier.

### Example correctness gates

MR:

```bash
export ARCH=ivcore11
python3 examples/03-tiledMma-iluvatar-mr-pipeline-hgemm.py --check --major-pattern tn
python3 examples/03-tiledMma-iluvatar-mr-pipeline-igemm.py --check --major-pattern tn --epilogue i32
```

CQ:

```bash
export ARCH=ivcore30
python3 examples/03-tiledMma-iluvatar-cq-mma-tile.py --check
python3 examples/03-tiledMma-iluvatar-cq-mma-tile.py --check --dtype s8 --mma-tile 32x32
```

## 7. Change-specific rebuild and test choices

- Python kernel only: no C++ rebuild; run the focused example/pytest.
- FlyIXDL C++, TableGen, conversion, or Python binding: rebuild FlyDSL, then
  run focused FileCheck and compile-only tests.
- ixcc IXDL dialect/translation or Iluvatar LLVM target: rebuild ixcc, rebuild
  FlyDSL, then run FileCheck, compile-only, and device tests.
- Runtime wrapper: rebuild FlyDSL and run JIT/runtime smoke plus one device
  launch.

## 8. Troubleshooting

- Wrong dialect/pass or undefined IXDL symbols: inspect `MLIR_DIR`; never point
  an Iluvatar build at ROCm/upstream LLVM.
- CMake still builds ROCDL: use a fresh dedicated `FLY_BUILD`, or explicitly
  reconfigure it with `-DFLYDSL_BACKENDS=iluvatar`.
- `Could NOT find CUDAToolkit`: set `CUDAToolkit_ROOT`, `CUDA_HOME`, and
  `IXA_HOME` to the CoreX toolkit root.
- Python imports stale bindings: put `${FLY_BUILD}/python_packages` first in
  `PYTHONPATH` and verify `flydsl.__file__`.
- Runtime `.so` load failure: check `LD_LIBRARY_PATH`, then run `ldd` on
  `libfly_iluvatar_jit_runtime.so`.
- Stale generated code after C++ changes: rebuild FlyDSL and set
  `FLYDSL_RUNTIME_ENABLE_CACHE=0`.
- Build OOM: reduce `JOBS`; do not delete working build trees as the first
  response.
