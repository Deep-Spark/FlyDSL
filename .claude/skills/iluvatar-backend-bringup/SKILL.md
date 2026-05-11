---
name: iluvatar-backend-bringup
description: >
  Plan and execute staged Iluvatar backend support in FlyDSL. Use when adding
  or reviewing Iluvatar hardware support, multi-backend plumbing, FlyToIXDL
  conversion, Iluvatar runtime kind pairing, JIT runtime wrappers, or backend
  CI/test tier rollout.
allowed-tools: Read Edit Bash Grep Glob Agent
---

# Iluvatar Backend Bring-Up

Use this skill to split in-tree Iluvatar support into independently reviewable
patches. Default to the in-tree path unless the user explicitly asks for an
out-of-tree prototype. Do not jump straight to GPU execution. Bring up the four
axes in order: Python compile backend, Python device runtime pairing,
MLIR/CMake backend stack, then native runtime/binary execution.

## Non-Negotiable Rule: Every Step Must Compile And Test

Each baby step must be independently reviewable and leave the tree in a
buildable, testable state.

For every patch:

- Add or update unit tests in the same patch.
- Run the narrowest meaningful verification command before moving on.
- Do not add dead skeleton code that is not wired into a compile path or test.
- Do not defer tests to a later patch unless the current patch is docs-only.
- If a phase is too large to compile/test in one patch, split it smaller.

Before starting any phase:

- Produce a phase-start checklist of what will be decided or changed.
- Include expected touched layers, tests, and completion criteria.
- Ask the user to confirm the checklist.
- Do not begin implementation for that phase until the user confirms.

Minimum completion criteria for any code patch:

- The edited code compiles or imports in the relevant build mode.
- New behavior has a focused L0/L1a/L1b/L2 test, matching the layer touched.
- Existing focused tests for the touched layer still pass.
- The final answer for that patch reports the exact verification command and result.

Compatibility invariant:

- Iluvatar support must be opt-in. Community users who do not know about
  Iluvatar and do not set Iluvatar-specific environment variables must keep the
  same default behavior as before.
- Default compile backend remains `rocm`.
- Default device runtime kind remains `rocm`.
- Adding Iluvatar files must not require Iluvatar tools, libraries, or hardware
  for default imports, default tests, or default builds.
- Add explicit tests for these defaults before adding opt-in Iluvatar behavior
  whenever possible.

## Design Baseline

The existing multi-backend work landed in stages:

- `a4586aa`: Python device runtime layer, `FLYDSL_RUNTIME_KIND`, compile/runtime pairing.
- `e6b8853`: per-backend conversion TableGen layout, starting with `FlyToROCDL`.
- `aeadac9` / `778ee34`: test tiering and backend matrix docs.
- `d2d7305`: CMake backend descriptors, `FOR_EACH_BACKEND`, per-backend CAPI registration, Python MLIR registration.

Dependency boundary:

- `a4586aa` is enough for the first Python-only L0 baby steps. Adding
  `IluvatarDeviceRuntime`, compile/runtime pairing, and Python backend selection
  should not require `FLYDSL_BACKENDS_TUPLE`, `FOR_EACH_BACKEND`, or backend
  CMake descriptors.
- `d2d7305` becomes mandatory at the first C++/MLIR in-tree skeleton step.
  From that point on, Iluvatar must use `cmake/backends/iluvatar.cmake`,
  backend global properties, `FLYDSL_BACKENDS`, and the
  `flydsl_register_iluvatar_{dialects,passes}` CAPI symbols consumed by
  `FOR_EACH_BACKEND`.
- `e6b8853` defines the conversion layout to mirror when `FlyToIXDL`
  appears: per-backend `Passes.td`, `Passes.h.inc`, and pass-inc-gen targets.
- `aeadac9` / `778ee34` define how to validate each layer: L0 for Python
  registry/pairing, L1b for target lowering, L2 for device execution.

Treat these as the architecture contract. Iluvatar support must satisfy both:

- Build-time backend stack selection: `FLYDSL_BACKENDS=rocdl;iluvatar`.
- Runtime/JIT pairing: `FLYDSL_COMPILE_BACKEND=iluvatar` and `FLYDSL_RUNTIME_KIND=iluvatar` unless deliberately mapped to an existing runtime.

## Two Support Modes

Upstream supports two ways to bring up a new hardware vendor:

### Out-of-tree backend

Use this for fast prototyping, private hardware enablement, or a vendor package
that should not modify upstream FlyDSL immediately.

Available hooks:

- Python compile backend entry point group: `flydsl.backends`.
- Direct Python registration: `register_backend("iluvatar", IluvatarBackend)`.
- Runtime pairing hooks: `register_compile_runtime_mapping()` and `register_device_runtime()`.

Limits:

- Out-of-tree Python hooks can select a backend and runtime kind, but cannot by
  themselves add C++ dialects, CAPI registration, `fly-opt` passes, or Python
  MLIR bindings to an already-built upstream wheel.
- If Iluvatar needs new MLIR dialect/conversion/native runtime libraries, the
  package must either ship a custom FlyDSL build or move those pieces in-tree.

Use out-of-tree first when the backend can reuse existing in-tree C++/runtime
plumbing, or when validating arch detection, env selection, cache keys, and
runtime pairing before committing to upstream file layout.

### In-tree backend

Use this for upstreamable full-stack support.

Required pieces:

- `cmake/backends/iluvatar.cmake` and `FLYDSL_BACKENDS` validation.
- `FlyIXDL` dialect and `FlyToIXDL` conversion trees.
- CAPI symbols consumed by `FOR_EACH_BACKEND`.
- Python MLIR dialect bindings and nanobind extension.
- Optional `lib/Runtime/Iluvatar/` native JIT runtime.

Default in-tree path:

1. Add stable backend/runtime naming and tests in-tree.
2. Add in-tree CMake/MLIR/CAPI skeletons.
3. Add binary/runtime execution support.

Only use out-of-tree first when explicitly requested or when upstream file
layout would block urgent private validation.

## Phase 0: Classify The Target

Before editing code, answer:

- Does Iluvatar reuse HIP/ROCm-compatible module load and launch semantics?
- What is the target arch string and wave/warp size vocabulary?
- Which upstream MLIR target dialect or binary attach path will be used?
- Can `mgpu*` symbols be implemented as a compatibility shim, or is a new launch ABI required?

Decision:

- If Iluvatar can mimic `mgpuModuleLoad`, `mgpuLaunchKernel`, `mgpuModuleUnload`, prefer a runtime shim first.
- If not, plan a separate ABI-generalization patch for `FlyLLVMTranslation.cpp` and `jit_executor.py`.

Current Iluvatar decisions:

- Iluvatar cannot use the ROCm-specific `mgpu*` implementation, but can provide
  a nearly identical runtime wrapper implementation backed by an Iluvatar stack
  that is CUDA-runtime compatible.
- Keep the FlyDSL JIT/runtime ABI shape initially: provide Iluvatar-owned
  implementations of the required `mgpu*` symbols instead of immediately
  generalizing the launch ABI.
- Architecture strings use `ivcoreXX`. Start with `ivcore11`.
- `ivcore11` is architecture family `middlerange` (`MR`).
- `ivcore30` is architecture family `conqueror` (`CQ`) and should be planned for
  later, but not implemented before `ivcore11` works.
- `GPUTarget.warp_size` is fixed at `64`.
- Iluvatar uses its own `llvm-project` / MLIR build. The required backend
  pipeline exists there and is broadly similar to NVGPU/AMDGPU style pipelines;
  defer exact pipeline details until Phase 4.
- Iluvatar MLIR target dialect names are `IXGPUDialect` and `IXDLDialect`.
  Treat them as the Iluvatar analogs of AMD `AMDGPUDialect` / `ROCDLDialect`
  and NVIDIA `NVGPUDialect` / `NVVMDialect`.
- Follow the existing FlyDSL naming pattern by mapping `FlyROCDL` to
  `FlyIXDL`: `FlyROCDLDialect.h` -> `FlyIXDLDialect.h`,
  `FlyROCDLDialect.cpp` -> `FlyIXDLDialect.cpp`, `FlyToROCDL` ->
  `FlyToIXDL`.
- Runtime naming is not finalized. The current tentative mapping is
  `lib/Runtime/ROCm` -> `lib/Runtime/Iluvatar` and
  `FlyRocmRuntimeWrappers.cpp` -> `FlyIluvatarRuntimeWrappers.cpp`, but confirm
  this before starting Phase 5.
- Iluvatar cluster support is not available for either currently planned
  architecture family: `ivcore11` / MR and `ivcore30` / CQ. Do not add an
  Iluvatar equivalent of `FlyROCDLClusterAttrPass` unless cluster support is
  explicitly introduced later.
- The in-tree end goal is full kernel execution and performance reporting for
  complex kernels such as GEMM and attention, reached only through compile/test
  baby steps.

## Phase 1: Python Backend And Runtime Pairing

Goal: make Python select Iluvatar consistently without initializing a GPU.

This phase should be done in-tree for the default Iluvatar plan.

Status: completed on `fujun.han/in-tree-iluvatar`.

Completed commits:

- Guardrail baseline: `test: preserve default backend behavior`
- Runtime pairing: `feat(runtime): add Iluvatar device runtime pairing`
- Compile backend metadata: `feat(compiler): add Iluvatar backend target metadata`

Touch points:

- `python/flydsl/compiler/backends/iluvatar.py`
- `python/flydsl/runtime/device_runtime/iluvatar.py`
- `python/flydsl/runtime/device_runtime/__init__.py`
- `python/flydsl/runtime/__init__.py`
- `python/flydsl/utils/env.py` docs/descriptions if needed
- `tests/unit/test_device_runtime.py`

Implement:

- `IluvatarBackend(BaseBackend)` with `supports_target`, `detect_target`, `make_target`, `pipeline_fragments`, `gpu_module_targets`, `native_lib_patterns`, `jit_runtime_lib_basenames`.
- `IluvatarDeviceRuntime(DeviceRuntime)` with at least `kind = "iluvatar"` and `device_count()`.
- Add `COMPILE_BACKEND_TO_RUNTIME_KIND["iluvatar"] = "iluvatar"` in-tree.
- Keep `register_compile_runtime_mapping()` / `register_device_runtime()` tests as extension coverage, but do not rely on them for the in-tree default.

Verify:

```bash
python3 -m pytest tests/unit/test_device_runtime.py -v
```

Expected tier: L0 backend-agnostic. This phase should not require target dialects or GPU execution.

Baby-step split:

1. Add `IluvatarDeviceRuntime` plus pairing tests; verify `test_device_runtime.py`.
2. Add `IluvatarBackend` with conservative compile-only stubs plus backend registry tests; verify the new backend tests.
3. Wire exports/docs only after tests prove selection and pairing behavior.

## Phase 2: CMake Backend Descriptor Skeleton

Goal: make Iluvatar a valid `FLYDSL_BACKENDS` entry and participate in registration.

This phase is in-tree unless the vendor package builds and distributes a custom
FlyDSL fork/wheel.

Touch points:

- `cmake/FlyDSLBackends.cmake`
- `cmake/backends/iluvatar.cmake`
- `include/flydsl/Backend/ForEachBackend.h` only if more than five backends are needed
- Aggregating `CMakeLists.txt` files under `include/flydsl`, `lib`, `lib/CAPI/Dialect`, `tools/fly-opt`, `python/mlir_flydsl`

Phase 2.1: Backend name and empty descriptor

Goal: make `iluvatar` a legal CMake backend name while keeping it opt-in and
not registering nonexistent C++ targets yet.

Status: completed locally in the current Iluvatar branch. This step changes
only `cmake/FlyDSLBackends.cmake`, adds comment-only
`cmake/backends/iluvatar.cmake`, and adds Iluvatar-specific CMake selection
tests in `tests/unit/test_backend_cmake_defaults.py`.

Implement:

- Add `iluvatar` to the allowed backend list.
- Keep `FLYDSL_BACKENDS` default as `rocdl`.
- Add `cmake/backends/iluvatar.cmake`.
- Keep the Iluvatar descriptor empty or comment-only in this baby step.
- Do not register nonexistent dialect, conversion, CAPI, runtime, Python, or
  stubgen targets yet.
- Keep ROCDL-specific gates (`FLYDSL_HAS_ROCDL`) separate from future Iluvatar
  gates (`FLYDSL_HAS_ILUVATAR`).

Tests:

- Keep the generic guardrail tests unchanged.
- Add Iluvatar-specific tests that assert:
  - `iluvatar` is a legal `FLYDSL_BACKENDS` value.
  - `cmake/backends/iluvatar.cmake` exists.
  - default `FLYDSL_BACKENDS` remains `rocdl`.
  - selecting only `iluvatar` does not select `rocdl` descriptor or ROCm runtime.

Verify:

```bash
python3 -m py_compile tests/unit/test_backend_cmake_defaults.py
PYTHONPATH="$PWD/python:$PWD" python3 -m pytest tests/unit/test_backend_cmake_defaults.py -v --confcutdir=tests/unit
```

Optional configure smoke when `MLIR_PATH` is usable:

```bash
cmake -S . -B /tmp/flydsl-cmake-iluvatar-smoke \
  -DMLIR_DIR="$MLIR_PATH/lib/cmake/mlir" \
  -DFLYDSL_BACKENDS=iluvatar
```

Baby-step split:

1. Add `iluvatar` to `FLYDSL_BACKENDS` validation with a descriptor that only references files introduced in the same patch.
2. Add a configure-level or CMake smoke test when practical; otherwise record the exact configure command in the patch verification.
3. Keep every descriptor property backed by an existing target or directory.

Phase 2.2: Empty in-tree CMake descriptor wiring

Goal: ensure `-DFLYDSL_BACKENDS=iluvatar` can configure without touching
ROCDL/HIP, while still not adding C++ targets.

Status: completed locally in the current Iluvatar branch. This step keeps the
production Iluvatar descriptor minimal and strengthens
`tests/unit/test_backend_cmake_defaults.py` so the mini CMake smoke includes
`lib/Runtime/CMakeLists.txt` and verifies that Iluvatar-only selection does not
enter `lib/Runtime/ROCm`.

Implement:

- Keep `cmake/backends/iluvatar.cmake` minimal.
- Add only descriptor properties that point to existing targets/directories.
- Do not add `FlyIXDL`, `FlyToIXDL`, runtime, or CAPI subdirectories yet.

Tests:

- CMake smoke for default build includes only ROCDL descriptor.
- CMake smoke for `FLYDSL_BACKENDS=iluvatar` includes only Iluvatar descriptor.
- Assert the Iluvatar-only smoke does not enter `lib/Runtime/ROCm`.

Completion criteria:

- Focused tests pass.
- Optional configure smoke passes.
- No HIP dependency is required for Iluvatar-only configure.

## Phase 3: MLIR Dialect And Conversion Skeleton

Goal: add a compile-only Iluvatar target dialect and conversion pass shell.

This phase is in-tree for upstream FlyDSL. Out-of-tree packages need their own
custom build and must still provide equivalent CAPI registration symbols.

Naming note: Iluvatar's MLIR target dialects are `IXGPUDialect` and
`IXDLDialect`, corresponding to AMD `AMDGPUDialect` / `ROCDLDialect` and
NVIDIA `NVGPUDialect` / `NVVMDialect`. In FlyDSL's backend-specific naming,
mirror `FlyROCDL` as `FlyIXDL`, not `FlyIluvatar`. Runtime file naming is a
separate Phase 5 decision and is currently only tentative.

Mirror the ROCDL layout:

- `include/flydsl/Dialect/FlyIXDL/`
- `lib/Dialect/FlyIXDL/`
- `include/flydsl/Conversion/FlyToIXDL/`
- `lib/Conversion/FlyToIXDL/`
- `include/flydsl-c/FlyIXDLDialect.h`
- `lib/CAPI/Dialect/FlyIXDL/`
- `lib/Bindings/Python/FlyIXDLExtension.cpp`
- `python/mlir_flydsl/dialects/FlyIXDL.td`
- `python/mlir_flydsl/dialects/fly_ixdl.py`

Required CAPI symbols:

```cpp
extern "C" void flydsl_register_iluvatar_dialects(MlirDialectRegistry);
extern "C" void flydsl_register_iluvatar_passes(void);
```

Guidelines:

- Reuse the target-neutral `fly` atom interfaces instead of duplicating them.
- Keep `include/flydsl/Conversion/Passes.h` from becoming a new global coupling point. Prefer per-backend pass registration through the Iluvatar CAPI wrapper.
- Start with a pass skeleton and FileCheck registration test before adding real lowering.

Verify:

```bash
bash scripts/build.sh -j$(nproc)
fly-opt --help | rg "ixdl|fly-to-ixdl|convert-fly-to-ixdl"
```

Expected tier: L1b target dialect compile coverage, no GPU execution.

Baby-step split:

Phase 3.1: `FlyIXDL` dialect skeleton

Status: completed locally in the current Iluvatar branch. This step adds a
minimal `FlyIXDL` dialect declaration/definition, CAPI registration through
`flydsl_register_iluvatar_dialects`, and descriptor wiring for the dialect and
CAPI targets. It intentionally adds no ops, conversion pass, Python MLIR
binding, runtime, atoms, or lowering. Full configure is still blocked in the
current environment before backend targets are reached because Python cannot
import `nanobind`.

- Add `include/flydsl/Dialect/FlyIXDL/`.
- Add `lib/Dialect/FlyIXDL/`.
- Add `include/flydsl-c/FlyIXDLDialect.h`.
- Add `lib/CAPI/Dialect/FlyIXDL/`.
- Add `flydsl_register_iluvatar_dialects`.
- Register only dialect subdirs in `cmake/backends/iluvatar.cmake`.
- Test CMake build and dialect registration/import.
- Do not add ops, conversion, runtime, atoms, or Python MLIR binding yet.

Phase 3.2: Python MLIR binding skeleton

Status: completed locally in the current Iluvatar branch. This step adds the
minimal `FlyIXDL` Python dialect binding files, a no-op nanobind extension
module, `FLYDSL_HAS_ILUVATAR`-gated Python CMake wiring, and opt-in stubgen
module registration. Full configure is still blocked in the current environment
because Python cannot import `nanobind`.

- Add `python/mlir_flydsl/dialects/FlyIXDL.td`.
- Add `python/mlir_flydsl/dialects/fly_ixdl.py`.
- Add minimal `lib/Bindings/Python/FlyIXDLExtension.cpp`.
- Add `FLYDSL_HAS_ILUVATAR` gated Python CMake wiring.
- Test Python import and stub generation when practical.
- Ensure default ROCDL Python bindings remain unchanged.

Phase 3.3: `FlyToIXDL` conversion pass skeleton

Status: completed locally in the current Iluvatar branch. This step adds the
`FlyToIXDL` TableGen pass skeleton, a no-op `convert-fly-to-ixdl` pass,
Iluvatar CAPI pass registration, and opt-in conversion descriptor wiring. Full
configure is still blocked in the current environment because Python cannot
import `nanobind`.

- Add `include/flydsl/Conversion/FlyToIXDL/Passes.td`.
- Add `include/flydsl/Conversion/FlyToIXDL/FlyToIXDL.h`.
- Add `lib/Conversion/FlyToIXDL/FlyToIXDL.cpp`.
- Register `flydsl_register_iluvatar_passes`.
- Register conversion subdirs in `cmake/backends/iluvatar.cmake`.
- Test `fly-opt --help | rg "ixdl|convert-fly-to-ixdl"`.
- Add a no-op pass or FileCheck smoke.

## Phase 4: Backend Pipeline And Binary Codegen

Goal: make `IluvatarBackend.pipeline_fragments()` produce a coherent end-to-end MLIR pipeline.

Touch points:

- `python/flydsl/compiler/backends/iluvatar.py`
- `python/flydsl/compiler/jit_function.py` if external `link_libs` need a non-ROCDL attach pass
- `python/flydsl/compiler/jit_executor.py` only if runtime library resolution or explicit module loading differs

Guidelines:

- Do not hard-code Iluvatar into generic code unless the generic API is insufficient.
- Generalize `link_libs` handling if the attach pass is not `rocdl-attach-target`.
- Include Iluvatar native libraries in `native_lib_patterns()` so JIT cache invalidates correctly.
- Return Iluvatar JIT runtime libraries from `jit_runtime_lib_basenames()` in load order; the first library must export `mgpuModuleUnload` unless `GpuJitModule` is generalized.

Verify:

```bash
COMPILE_ONLY=1 FLYDSL_COMPILE_BACKEND=iluvatar FLYDSL_RUNTIME_KIND=iluvatar python3 -m pytest <focused compile-only test> -v
```

Expected tier: L1b. This phase may generate a device binary but should still be testable without launching kernels.

Baby-step split:

Phase 4.1: Pipeline parse skeleton

Status: completed locally in the current Iluvatar branch. This step replaces
the Iluvatar pipeline `NotImplementedError` with a minimal Python-level pass
fragment list that reaches `convert-fly-to-ixdl` and deliberately excludes
target attach, binary codegen, and runtime loading.

- Replace `NotImplementedError` in `IluvatarBackend.pipeline_fragments()` with
  a minimal pipeline that only uses passes available after Phase 3.
- Do not attach a binary target yet.
- Test that fragments parse and a minimal compile-only module reaches the
  expected boundary.
- Verify default ROCm pipeline stays unchanged.

Phase 4.2a: IXDL compile-only GPU lowering pipeline

Status: completed locally in the current Iluvatar branch. This step adds the
upstream Iluvatar `convert-gpu-to-ixdl` GPU-module lowering fragment to the
Python pipeline, and keeps `gpu-module-to-binary` plus target attach out of
scope. It leaves `gpu_module_targets()` unchanged; target attribute syntax is
confirmed separately before binary attach/codegen is enabled.

- Add `gpu.module(..., convert-gpu-to-ixdl{...})` after `convert-fly-to-ixdl`.
- Keep this compile-only; no binary attach and no kernel launch.
- Do not change `gpu_module_targets()` in this baby step.
- Test that the Iluvatar pipeline includes `convert-gpu-to-ixdl`, excludes
  target attach, excludes `gpu-module-to-binary`, and excludes ROCm-only
  cluster lowering.

Phase 4.2b: Target attribute and attach path

Status: completed locally in the current Iluvatar branch. This step aligns the
Iluvatar backend with the real ixcc `#ixdl.target<...>` attribute and
`ixdl-attach-target` pass, then enables the backend pipeline through
`gpu-module-to-binary{format=fatbin}`. It also generalizes external bitcode
`link_libs` injection from ROCDL-only attach passes to backend attach-target
passes.

- Confirm Iluvatar target attribute syntax from the Iluvatar MLIR build.
- Update `gpu_module_targets()`.
- Add Iluvatar attach/codegen pass fragment.
- Keep this compile-only; no kernel launch.
- Test generated IR or pass output contains expected Iluvatar target marker.

Phase 4.3: Cache and runtime library metadata

Status: completed locally in the current Iluvatar branch. This step adds
Python unit coverage for the existing Iluvatar runtime/cache metadata and keeps
native runtime implementation out of scope.

- Ensure `native_lib_patterns()` fingerprints Iluvatar compiler/runtime pieces.
- Ensure `jit_runtime_lib_basenames()` names the future Iluvatar runtime libs in
  the intended load order.
- Test cache key inputs or metadata without requiring the runtime library to
  exist unless runtime resolution is explicitly invoked.

## Phase 5: Native Runtime Path

Goal: load modules and launch kernels on Iluvatar hardware.

If using `mgpu*` compatibility:

- Add `lib/Runtime/Iluvatar/`.
- Build a JIT runtime shared library that exports the same symbols currently provided by `lib/Runtime/ROCm/FlyRocmRuntimeWrappers.cpp`.
- Add it from `lib/Runtime/CMakeLists.txt` when `"iluvatar" IN_LIST FLYDSL_BACKENDS`.
- Copy the library into `flydsl/_mlir/_mlir_libs` through `python/mlir_flydsl/CMakeLists.txt` or backend CMake properties.

If not using `mgpu*` compatibility:

- Generalize `lib/Dialect/Fly/IR/FlyLLVMTranslation.cpp` so module load and launch symbol names are selected by backend.
- Generalize `python/flydsl/compiler/jit_executor.py` so explicit module load/unload is not ROCm-specific.
- Update error messages to say backend/runtime instead of ROCm/HIP.

Verify:

```bash
FLYDSL_COMPILE_BACKEND=iluvatar FLYDSL_RUNTIME_KIND=iluvatar python3 -m pytest <single L2 smoke test> -v
```

Expected tier: L2 device execution.

Baby-step split:

Phase 5.1: Native runtime library skeleton

Status: completed locally in the current Iluvatar branch. This step adds the
`lib/Runtime/Iluvatar` skeleton with a CUDA-compatible
`FlyIluvatarRuntimeWrappers.cpp`, builds `libfly_iluvatar_jit_runtime.so` as an
opt-in CMake target, and wires the target into Python package output when
selected.

- Add `lib/Runtime/Iluvatar/`.
- Add `FlyIluvatarRuntimeWrappers.cpp`.
- Build `libfly_iluvatar_jit_runtime.so`.
- Export the required `mgpu*` symbols with stubs or minimal implementations.
- Add it from `lib/Runtime/CMakeLists.txt` only when `"iluvatar" IN_LIST FLYDSL_BACKENDS`.
- Test symbol export and CMake build.

Phase 5.2: Module load/unload smoke

Phase 5.2a: Runtime symbol contract

Status: completed locally in the current Iluvatar branch. This step adds
static unit coverage for the Iluvatar runtime wrapper `mgpu*` ABI contract,
including module load/unload, function lookup, launch, stream/event, memory, and
host registration symbols. It continues to assert that cluster launch is not
exported for Iluvatar.

- Implement `mgpuModuleLoad` and `mgpuModuleUnload`.
- Keep current FlyDSL explicit module ABI shape.
- Test load/unload on a smallest possible module or runtime-provided smoke.

Phase 5.2b: Configure/build smoke

Status: completed locally in the current Iluvatar branch. This step configures a
standalone smoke build with `FLYDSL_BACKENDS=iluvatar`, the ixcc MLIR build, and
the CUDA-compatible CoreX toolkit, then builds only `FlyIluvatarJitRuntime`.

Verified command shape:

```bash
cmake -S /home/peter/sw_home/sdk/FlyDSL -B /tmp/flydsl-iluvatar-runtime-smoke \
  -DMLIR_DIR=/home/peter/sw_home/sdk/ixcc/build/lib/cmake/mlir \
  -DFLYDSL_BACKENDS=iluvatar \
  -DCUDAToolkit_ROOT=/home/peter/sw_home/local/corex \
  -DPython3_EXECUTABLE=/home/peter/sw_home/sdk/FlyDSL/.venv/bin/python
cmake --build /tmp/flydsl-iluvatar-runtime-smoke --target FlyIluvatarJitRuntime -j8
```

Notes:

- Use the repo-local `.venv` for this smoke. It already contains `nanobind`
  2.12.0, and CMake finds its package config under
  `.venv/lib/python3.12/site-packages/nanobind/cmake`.
- Do not use the system `python3` for this smoke unless it has the same Python
  build dependencies available; system Python may be PEP 668 managed and may not
  expose `nanobind`.
- The build produced `libfly_iluvatar_jit_runtime.so` under
  `python_packages/flydsl/_mlir/_mlir_libs/`.
- `nm -D` confirmed exported symbols including `mgpuModuleLoad`,
  `mgpuModuleLoadJIT`, `mgpuLaunchKernel`, `mgpuMemAlloc`, and
  `mgpuSetDefaultDevice`.

Phase 5.2c: Module load/unload smoke

Status: completed locally in the current Iluvatar branch. This step adds an
opt-in L2 pytest smoke that `dlopen`s the built Iluvatar JIT runtime shared
library and calls `mgpuModuleLoad` / `mgpuModuleUnload` on a caller-provided
Iluvatar module blob. The test is skipped by default unless both required env
vars are provided, so default ROCm users do not need Iluvatar hardware,
libraries, or blobs.

Test env:

- `FLYDSL_ILUVATAR_JIT_RUNTIME_LIB`: path to `libfly_iluvatar_jit_runtime.so`.
- `FLYDSL_ILUVATAR_SMOKE_BLOB`: path to a loadable Iluvatar cubin/fatbin blob.
- `LD_LIBRARY_PATH`: must include the CUDA-compatible CoreX driver library path
  when the runtime shared library depends on `libcuda.so.1`.

Verified command shape:

```bash
LD_LIBRARY_PATH=/home/peter/sw_home/local/corex/lib64:$LD_LIBRARY_PATH \
  FLYDSL_ILUVATAR_JIT_RUNTIME_LIB=/tmp/flydsl-iluvatar-runtime-smoke/python_packages/flydsl/_mlir/_mlir_libs/libfly_iluvatar_jit_runtime.so \
  FLYDSL_ILUVATAR_SMOKE_BLOB=/tmp/add_ivcore11.cubin \
  /home/peter/sw_home/sdk/FlyDSL/.venv/bin/python -m pytest \
    tests/unit/test_iluvatar_runtime_smoke.py -q --confcutdir=tests/unit
```

Default verification:

```bash
python3 -m pytest tests/unit/test_iluvatar_runtime_smoke.py -q --confcutdir=tests/unit
```

The default run should skip the test when the Iluvatar-specific env vars are not
set.

Phase 5.3: Kernel launch smoke

- Implement `mgpuModuleGetFunction`, `mgpuLaunchKernel`, and minimal stream handling.
- Add smallest possible L2 launch test.
- Use `FLYDSL_COMPILE_BACKEND=iluvatar` and `FLYDSL_RUNTIME_KIND=iluvatar`.

Phase 5.3a: Function lookup smoke

Status: completed locally in the current Iluvatar branch. This step extends the
opt-in Iluvatar runtime smoke so it loads a module and calls
`mgpuModuleGetFunction` for a caller-provided kernel symbol, without launching
the kernel. The test is skipped by default unless the runtime library, module
blob, and kernel name env vars are all provided.

Additional test env:

- `FLYDSL_ILUVATAR_SMOKE_KERNEL`: kernel symbol to look up in the loaded module.

Verified command shape:

```bash
LD_LIBRARY_PATH=/home/peter/sw_home/local/corex/lib64:$LD_LIBRARY_PATH \
  FLYDSL_ILUVATAR_JIT_RUNTIME_LIB=/tmp/flydsl-iluvatar-runtime-smoke/python_packages/flydsl/_mlir/_mlir_libs/libfly_iluvatar_jit_runtime.so \
  FLYDSL_ILUVATAR_SMOKE_BLOB=/tmp/add_ivcore11.cubin \
  FLYDSL_ILUVATAR_SMOKE_KERNEL=add_kernel \
  /home/peter/sw_home/sdk/FlyDSL/.venv/bin/python -m pytest \
    tests/unit/test_iluvatar_runtime_smoke.py -q --confcutdir=tests/unit
```

## Phase 6: Atoms And Kernels

Goal: add real Iluvatar copy/MMA atoms and port kernels.

Use `/add-target-atom-op` for atom type work. Each target-specific atom must implement the `Fly_MmaOpTypeInterface`, `Fly_CopyOpTypeInterface`, and optional `Fly_StatefulOpTypeInterface` contracts.

Start with:

- One simple copy atom.
- One minimal MMA/tensor-core atom if available.
- One FileCheck lowering test.
- One tiny end-to-end kernel smoke test.

Only then port larger kernels.

Baby-step split:

Phase 6.1: First copy atom

- Add one copy atom type declaration.
- Add verifier/layout tests.
- Add lowering FileCheck.
- Add tiny copy kernel smoke.

Phase 6.2: First MMA/tensor-core atom

- Add one MMA atom type declaration.
- Implement interface methods and layout tests.
- Add lowering FileCheck.
- Add tiny MMA/GEMM smoke.

Phase 6.3+: Production kernels

- Port one production kernel at a time.
- Start with minimal GEMM, then production GEMM, then attention.
- Add focused correctness tests and performance reporting only after correctness
  is stable.

## Test Matrix

Use the existing tier markers:

- L0: registry, env, runtime pairing, backend selection.
- L1a: portable Fly and upstream dialect tests.
- L1b: `FlyToIXDL` lowering and binary codegen without execution.
- L2: Iluvatar device execution.

Add backend-specific markers only when tests genuinely assume Iluvatar lowering or runtime. Do not mark generic tests as Iluvatar-specific.

## Patch Split

Prefer this review sequence:

1. In-tree Python runtime/backend pairing and tests.
2. CMake backend descriptor plus empty skeletons.
3. Dialect/CAPI/Python MLIR registration.
4. Conversion pass skeleton and FileCheck.
5. Backend pipeline and compile-only smoke.
6. Runtime library shim or launch ABI generalization.
7. First atoms and minimal kernel.
8. Production kernel ports and L2 coverage.

Each patch should have a focused verification command and should avoid mixing unrelated ROCDL cleanup.

Do not merge two adjacent steps just because they are small if doing so would
make failures harder to localize. The preferred patch size is the smallest unit
that compiles, tests, and proves one new invariant.
