# 基于 ixcc LLVM 22.x 搭建 FlyDSL 开发环境

本文说明如何从 **ixcc SDK**（LLVM/MLIR **22.1.x**，见 ixcc 源码中 `cmake/Modules/LLVMVersion.cmake`）编译 MLIR 安装前缀，再编译 **FlyDSL**，以及如何配置环境并**成功运行测试**（pytest：`tests/unit`、`tests/system` 等；FileCheck：`tests/mlir`）。

> **说明：** 历史上曾有 **`tests/pyir/`** 目录；当前主线在测试重构后该目录可能为空或已移除，IR/编译向覆盖主要在 **`tests/unit`**、**`tests/system`** 与 **`tests/mlir`**。请以仓库内 [`tests/README.md`](../tests/README.md) 为准。

适用于：已有一份 ixcc 目录布局（与 upstream `llvm-project` 类似：`llvm/`、`mlir/`、`clang/` 等为同级子目录），并希望与其他开发者共用同一套依赖搭建流程。

### 与仓库版本对齐

下文中的脚本路径与源码改动说明以 **你本地的 FlyDSL `HEAD`** 为准。拉取他人分支或旧快照时，请先 **`git pull`**，并确认已包含 **第 4 节**所列改动（或等价合并），否则在 ixcc LLVM 22.x 上可能无法编译或通过 FileCheck。

---

## 1. 前置条件

### 1.1 系统与工具

- Linux（文档基于常见 x86_64 环境）。
- **CMake** ≥ 3.20、**Ninja**（推荐）或 Make、**C/C++ 编译器**（GCC 或 Clang）。
- **Python** 3.10+（与 MLIR Python 绑定一致即可）。
- 磁盘空间：完整编译 LLVM+MLIR+FlyDSL 通常需要 **数十 GB**；LLVM 单独编译约 **30–90 分钟**（视 CPU 与磁盘而定）。

### 1.2 ixcc 目录布局

确认存在：

- `${IXCC_ROOT}/llvm/CMakeLists.txt`（CMake 源码根为 `llvm/`，不是 ixcc 仓库根）。
- 同级存在 `mlir/`、`clang/` 等（标准 llvm-project 布局）。

例如：

```text
${IXCC_ROOT}/
  cmake/
  llvm/
  mlir/
  clang/
  ...
```

### 1.3 FlyDSL 仓库与 Python 虚拟环境

在 FlyDSL 仓库根目录创建虚拟环境并安装构建依赖（与官方构建脚本一致）：

```bash
cd /path/to/FlyDSL
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install 'nanobind>=2.0' 'pybind11>=2.10' numpy cmake pytest
```

后续 **`scripts/build_ixcc_llvm.sh`** 若检测到 `.venv`，会自动 `source` 该环境以使用同一套 Python/nanobind。

---

## 2. 从 ixcc 源码构建并安装 LLVM/MLIR（22.x）

仓库提供脚本：**[`scripts/build_ixcc_llvm.sh`](../scripts/build_ixcc_llvm.sh)**。

作用概要：

- **删除** `${IXCC_ROOT}/build`（可用 `IXCC_BUILD_DIR` 覆盖构建目录），避免沿用旧 CMake 缓存。
- 使用 **`${IXCC_ROOT}/llvm`** 作为 `-S`，配置 **`mlir` + `clang`**，启用 **MLIR Python 绑定（nanobind）**，安装到 **`${IXCC_ROOT}/mlir_install`**（可用 `IXCC_MLIR_INSTALL` 覆盖）。

### 2.1 标准命令

```bash
export IXCC_ROOT=/path/to/ixcc          # 按需修改
cd /path/to/FlyDSL
source .venv/bin/activate

bash scripts/build_ixcc_llvm.sh -j$(nproc)
```

成功后安装前缀默认为：

```text
${IXCC_ROOT}/mlir_install
```

其中应存在 **`lib/cmake/mlir`** 与 **`lib/cmake/llvm`**（以及 `bin/mlir-tblgen`、`bin/FileCheck` 等）。

### 2.2 常用环境变量

| 变量 | 含义 | 默认 |
|------|------|------|
| `IXCC_ROOT` | ixcc 仓库根目录 | `$HOME/sw_home/sdk/ixcc` |
| `IXCC_BUILD_DIR` | LLVM 构建目录（会先被删除再新建） | `$IXCC_ROOT/build` |
| `IXCC_MLIR_INSTALL` | `cmake --install` 前缀 | `$IXCC_ROOT/mlir_install` |
| `IXCC_MLIR_ENABLE_ROCM_RUNNER` | 是否开启需要 HIP 的 ROCm runner | `OFF`；若本机已配 ROCm 且提供 `hipConfig.cmake`，可设为 `1` |
| `IXCC_LLVM_TARGETS` | `LLVM_TARGETS_TO_BUILD` | `X86;NVPTX` |

### 2.3 ixcc 源码/脚本层面的已知规避（脚本已默认设置）

1. **`MLIR_ENABLE_ROCM_RUNNER`**：在未安装 ROCm HIP 开发包时保持 **`OFF`**，否则 CMake 会找不到 **`hipConfig.cmake`**。需要 ROCm 时再打开。
2. **`LLVM_TARGETS_TO_BUILD`**：部分 ixcc 树曾在 **AMDGPU** TableGen 上出现 **重复类定义** 错误。默认仅 **`X86;NVPTX`**；若 ixcc 已修复 AMDGPU，可例如：
   ```bash
   export IXCC_LLVM_TARGETS="X86;NVPTX;AMDGPU"
   ```
3. **`LLVM_TOOL_DYNAMIC_COMPILE_BUILD=OFF`**：ixcc 自带的 **`tools/dynamic-compile`** 依赖 **Iluvatar** 目标；在未构建该目标时会链接失败，脚本默认关闭该工具（FlyDSL 不依赖它）。

### 2.4 版本确认

安装完成后可检查：

```bash
grep -E 'LLVM_VERSION|set\(LLVM_VERSION' "$IXCC_MLIR_INSTALL/lib/cmake/mlir/MLIRConfig.cmake" | head -5
grep LLVM_PACKAGE_VERSION "$IXCC_MLIR_INSTALL/lib/cmake/llvm/LLVMConfig.cmake" | head -3
```

应与 ixcc **22.x** 源码版本一致（例如 **22.1.0git**）。

---

## 3. 构建 FlyDSL

### 3.1 指向 MLIR 安装前缀

```bash
export MLIR_PATH=/path/to/ixcc/mlir_install   # 与上一节 IXCC_MLIR_INSTALL 一致
```

**不要**把 `MLIR_PATH` 指到仅有源码、没有 **`lib/cmake/mlir`** 的目录。

### 3.2 `find_package(LLVM)` 版本与路径

[`scripts/build.sh`](../scripts/build.sh) 已传入：

- `-DMLIR_DIR="${MLIR_PATH}/lib/cmake/mlir"`
- `-DLLVM_DIR="${MLIR_PATH}/lib/cmake/llvm"`

避免系统里其它 LLVM（例如 `/usr/lib/llvm-*`）抢走 **`find_package(LLVM)`**，导致版本不匹配。

### 3.3 CMake：`MLIR_MAIN_SRC_DIR`（可选）

仓库根 [`CMakeLists.txt`](../CMakeLists.txt) 中：若 `find_package(MLIR)` 提供了 **`MLIR_MAIN_SRC_DIR`**，会把 **`${MLIR_MAIN_SRC_DIR}/cmake/modules`** 置于 **`CMAKE_MODULE_PATH` 前面**，以便使用完整的 **`AddMLIR.cmake`**（含 **`add_mlir_generic_tablegen_target`** 等宏）。

使用 **安装树** `mlir_install` 时，通常安装目录内的 **`lib/cmake/mlir/AddMLIR.cmake`** 已包含完整宏；使用 **仅构建树** 且遇到 TableGen 宏缺失时，可优先检查 MLIR 源码路径是否一致。

### 3.4 执行构建

```bash
cd /path/to/FlyDSL
source .venv/bin/activate
export MLIR_PATH=/path/to/ixcc/mlir_install
export FLY_BUILD_DIR=/path/to/FlyDSL/build-fly    # 可选，默认仓库下 build-fly

bash scripts/build.sh -j$(nproc)
```

产物：

- **`${FLY_BUILD_DIR}/python_packages/`**：Python 包（含 `_mlir` 扩展与 `flydsl`）。
- **`${FLY_BUILD_DIR}/bin/fly-opt`**：命令行优化/调试工具（FileCheck 测试需要）。

构建完成后，CMake 会将 **`python/flydsl`** 拷贝至 **`${FLY_BUILD_DIR}/python_packages/flydsl`**。若你**仅修改 Python 源码**而未重新跑 **`build.sh`**，测试仍可能加载旧的 **`build-fly`** 副本；可 **`bash scripts/build.sh`** 或手动 **`cp python/flydsl/... build-fly/python_packages/flydsl/...`**，或使用 **`pip install -e .`**（开发时常用，可直接编辑仓库内 **`python/flydsl`**）。

可选安装（可编辑模式）：

```bash
pip install -e .
```

---

## 4. 本仓库为适配 MLIR 22.x（ixcc）已做的改动

以下改动针对 **MLIR 22** 中 GPU/ROCDL API、诊断与打印格式的变化。若你的分支缺少条目，在 ixcc + `mlir_install` 上可能出现 **CMake 失败**、**C++ 编译错误**、**Python 导入失败**或 **FileCheck 不匹配**。

### 4.1 CMake / 构建

| 位置 | 改动说明 |
|------|----------|
| [`CMakeLists.txt`](../CMakeLists.txt) | `find_package(MLIR)` 后，若存在 **`MLIR_MAIN_SRC_DIR`**，将其 **`cmake/modules`** 前置加入 **`CMAKE_MODULE_PATH`**。 |
| [`scripts/build.sh`](../scripts/build.sh) | **`-DLLVM_DIR="${MLIR_PATH}/lib/cmake/llvm"`**，避免误用系统 LLVM。 |
| [`python/mlir_flydsl/CMakeLists.txt`](../python/mlir_flydsl/CMakeLists.txt) | **`FLY_BUILD_JIT_RUNTIME`**（默认 ON）：若 **`find_package(hip)`** 失败（无 `/opt/rocm` 等），**跳过** **`FlyJitRuntime`**（`libfly_jit_runtime.so`），使无 ROCm 的机器仍能编译 Python 绑定；需要 GPU JIT 时安装 ROCm/HIP 并重新配置。也可 **`cmake -DFLY_BUILD_JIT_RUNTIME=OFF`**（若你的 CMake 入口支持传入该缓存变量）。 |

### 4.2 C++（Fly 方言 / lowering）

| 位置 | 改动说明 |
|------|----------|
| [`lib/Dialect/Fly/IR/FlyOps.cpp`](../lib/Dialect/Fly/IR/FlyOps.cpp) | **`emitOptionalError`** 中 **`AddressSpace`** 改为 **`static_cast<int32_t>(addrSpace)`**（MLIR 22 诊断流无法直接打印该枚举）。 |
| [`lib/Dialect/FlyROCDL/CDNA3/MmaAtom.cpp`](../lib/Dialect/FlyROCDL/CDNA3/MmaAtom.cpp) | **`DISPATCH_MFMA_SSA`**：`mfma_*::create` 改为 **`(builder, loc, accTy, ValueRange{a,b,c,z,z,z})`**，立即数用 **`arith::ConstantIntOp`**，对齐 ROCDL **`Variadic` 操作数**形式。 |
| [`lib/Dialect/FlyROCDL/CDNA4/MmaAtom.cpp`](../lib/Dialect/FlyROCDL/CDNA4/MmaAtom.cpp) | **`mfma_scale_*`**：`create` 改为 **`ValueRange`**（含 **`vCbsz`、`vBlgp`、`vOpsel*`** 等 **`arith.constant`**），对齐 **`mfma.scale`** 新签名。 |
| [`lib/Conversion/FlyToROCDL/FlyToROCDL.cpp`](../lib/Conversion/FlyToROCDL/FlyToROCDL.cpp) | 同类 MFMA **`create`** / **`ValueRange`** 调整（若与本分支一致则已包含）。 |

### 4.3 Python

| 位置 | 改动说明 |
|------|----------|
| [`python/flydsl/expr/rocdl/__init__.py`](../python/flydsl/expr/rocdl/__init__.py) | **`wave_id`**、cluster 系列、**`s_wait_asynccnt`**、**`readfirstlane`** 等通过 **`globals().get(..., None)`** 获取：部分 ixcc **ROCDL Python 绑定**未导出符号，避免导入期 **`NameError`**；调用时再 **`AttributeError`**。 |
| [`python/flydsl/compiler/kernel_function.py`](../python/flydsl/compiler/kernel_function.py) | **`gpu.LaunchFuncOp`** 的 MLIR 22 绑定**不再接受** **`cluster_size=`** 关键字；改为 **`_gpu_launch_func_cluster_kwargs()`** 按 **`inspect.signature`** 仅在支持时传入 cluster 相关参数（无 cluster 时不要传未声明关键字）。 |

### 4.4 FileCheck（MLIR 文本）

| 位置 | 改动说明 |
|------|----------|
| [`tests/mlir/Conversion/mma_atom_stateful.mlir`](../tests/mlir/Conversion/mma_atom_stateful.mlir) | MLIR 22 中 **`rocdl.mfma.scale*.`** 打印为 **SSA 操作数**（如 **`%c0_i32`**），不再在行内印 **`0, 0, 2`** 等字面量；CHECK 已放宽或增加 **`arith.constant`** 的 **`CHECK-DAG`**。 |

### 4.5 新增脚本

| 脚本 | 用途 |
|------|------|
| [`scripts/build_ixcc_llvm.sh`](../scripts/build_ixcc_llvm.sh) | 一键清理并重建 ixcc → **`mlir_install`**（第 2 节）。 |
| [`scripts/run_mlir_filecheck.sh`](../scripts/run_mlir_filecheck.sh) | **仅**运行 **`tests/mlir`** FileCheck；失败只结束脚本进程，不会关掉交互式 shell（见 §7.4）。 |

### 4.6 改动涉及的文件与代码区域（便于 diff / 评审）

下列路径均为相对于 **FlyDSL 仓库根**。行号会随分支漂移，请在本地用符号名或注释搜索核对。

#### 汇总：变更文件清单

| 文件路径 |
|----------|
| [`CMakeLists.txt`](../CMakeLists.txt) |
| [`scripts/build.sh`](../scripts/build.sh) |
| [`python/mlir_flydsl/CMakeLists.txt`](../python/mlir_flydsl/CMakeLists.txt) |
| [`lib/Dialect/Fly/IR/FlyOps.cpp`](../lib/Dialect/Fly/IR/FlyOps.cpp) |
| [`lib/Dialect/FlyROCDL/CDNA3/MmaAtom.cpp`](../lib/Dialect/FlyROCDL/CDNA3/MmaAtom.cpp) |
| [`lib/Dialect/FlyROCDL/CDNA4/MmaAtom.cpp`](../lib/Dialect/FlyROCDL/CDNA4/MmaAtom.cpp) |
| [`lib/Conversion/FlyToROCDL/FlyToROCDL.cpp`](../lib/Conversion/FlyToROCDL/FlyToROCDL.cpp)（若主线已含 MLIR 22 MFMA 对齐则才有对应 diff） |
| [`python/flydsl/expr/rocdl/__init__.py`](../python/flydsl/expr/rocdl/__init__.py) |
| [`python/flydsl/compiler/kernel_function.py`](../python/flydsl/compiler/kernel_function.py) |
| [`tests/mlir/Conversion/mma_atom_stateful.mlir`](../tests/mlir/Conversion/mma_atom_stateful.mlir) |
| [`scripts/run_mlir_filecheck.sh`](../scripts/run_mlir_filecheck.sh) |
| [`docs/ixcc_llvm22_flydsl_dev_setup.md`](../docs/ixcc_llvm22_flydsl_dev_setup.md)、[`docs/ixcc_llvm22_flydsl_dev_setup_confluence.wiki`](../docs/ixcc_llvm22_flydsl_dev_setup_confluence.wiki)（文档维护） |

#### 分文件说明

- **`CMakeLists.txt`（仓库根）**  
  - **`find_package(MLIR)`** 之后：若存在 **`MLIR_MAIN_SRC_DIR`**，把 **`${MLIR_MAIN_SRC_DIR}/cmake/modules`** 前置加入 **`CMAKE_MODULE_PATH`**。

- **`scripts/build.sh`**  
  - 调用 **`cmake`** 时传入 **`-DLLVM_DIR="${MLIR_PATH}/lib/cmake/llvm"`**（与 **`-DMLIR_DIR`** 成对）。

- **`python/mlir_flydsl/CMakeLists.txt`**  
  - 选项 **`FLY_BUILD_JIT_RUNTIME`**（默认 ON）。  
  - **`FlyJitRuntime`** 段：**`find_package(hip CONFIG QUIET PATHS …)`**；仅当 **`hip_FOUND`** 时 **`add_library(FlyJitRuntime …)`**、**`add_dependencies(FlyPythonCAPI FlyJitRuntime)`**；否则打印跳过信息。

- **`lib/Dialect/Fly/IR/FlyOps.cpp`**  
  - **`FLY_INFER_RETURN_TYPES(PtrToIntOp)`**：**`default:`** 分支里 **`emitOptionalError(..., addrSpace)`** 改为 **`static_cast<int32_t>(addrSpace)`**（**`AddressSpace`** 枚举不能直接流入诊断流）。

- **`lib/Dialect/FlyROCDL/CDNA3/MmaAtom.cpp`**  
  - **`MmaOpCDNA3_MFMAType::emitAtomCallSSA`** 中的宏 **`DISPATCH_MFMA_SSA`**：每个分支内用 **`arith::ConstantIntOp`** 生成三个 **`i32` 0**，**`ROCDL::mfma_*::create(builder, loc, accTy, ValueRange{a,b,c,z,z,z})`**。  
  - 文件头部增加 **`#include "mlir/Dialect/Arith/IR/Arith.h"`**（若尚未包含）。

- **`lib/Dialect/FlyROCDL/CDNA4/MmaAtom.cpp`**  
  - **`MmaOpCDNA4_MFMAScaleType::emitAtomCallSSA`** 中 **`mfma_scale_f32_16x16x128_f8f6f4` / `mfma_scale_f32_32x32x64_f8f6f4`**：**`create(builder, loc, accTy, ValueRange scaleOperands)`**，操作数顺序为 **`a, b, c, vCbsz, vBlgp, vOpselA, scaleA, vOpselB, scaleB`**（**`v*`** 由 **`arith::ConstantIntOp`** 得到）。

- **`lib/Conversion/FlyToROCDL/FlyToROCDL.cpp`**  
  - 与 ROCDL **`mfma` / `mfma.scale`** 的 **`create`** 签名、**`ValueRange`** 操作数对齐相关的 lowering（以当前分支 **git diff** 为准）。

- **`python/flydsl/expr/rocdl/__init__.py`**  
  - 模块级：**`_ods_wave_id`**、**`_ods_cluster_workgroup_id_*`**、**`_ods_cluster_load_async_to_lds_*`**、**`_ods_s_wait_asynccnt`**、**`_ods_readfirstlane`** 使用 **`globals().get("…", None)`**。  
  - 函数 **`wave_id`**、**`cluster_workgroup_id_*`**、**`cluster_load_async_to_lds`**、**`s_wait_asynccnt`**、**`readfirstlane`**：在对应 **`_ods_* is None`** 时 **`raise AttributeError`**。

- **`python/flydsl/compiler/kernel_function.py`**  
  - 新增 **`_gpu_launch_func_cluster_kwargs(...)`**（与 **`gpu.LaunchFuncOp.__init__`** 的 **`inspect.signature`** 对齐）。  
  - **`KernelLauncher.launch`**：调用 **`gpu.LaunchFuncOp(..., **_gpu_launch_func_cluster_kwargs(cluster_size))`**，**不再**向构造函数传入未支持的 **`cluster_size=`** 关键字。

- **`tests/mlir/Conversion/mma_atom_stateful.mlir`**  
  - 函数 **`test_mma_scale_atom_call_16x16x128`**、**`test_mma_scale_atom_call_32x32x64_mixed`**、**`test_mma_scale_atom_call_with_opsel`** 下方的 **`// CHECK` / `// CHECK-DAG`**：匹配 MLIR 22 打印出的 **`rocdl.mfma.scale`** SSA 形式（**`arith.constant`**），而非旧版行内整数字面量。

- **`scripts/run_mlir_filecheck.sh`**  
  - 新增脚本：遍历 **`tests/mlir/**/*.mlir`**，执行首行 **`// RUN:`**，汇总失败数后 **`exit`**。

---

## 5. 运行环境变量（Python / 动态库）

在运行测试或示例前（路径按你的 **`FLY_BUILD_DIR`** 修改）：

```bash
export FLY_BUILD_DIR=/path/to/FlyDSL/build-fly
export PYTHONPATH="${FLY_BUILD_DIR}/python_packages:/path/to/FlyDSL:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${FLY_BUILD_DIR}/python_packages/flydsl/_mlir/_mlir_libs:${LD_LIBRARY_PATH:-}"
export FLYDSL_RUN_QUANT=1
```

可选：JIT 调试时关闭缓存：

```bash
export FLYDSL_RUNTIME_ENABLE_CACHE=0
```

---

## 6. 运行测试

测试分层说明见 [`tests/README.md`](../tests/README.md)（L0/L1a/L1b/L2 与 pytest marker）。以下命令均假设：已在仓库根目录执行 **第 5 节**环境变量，且 **`build-fly/bin/fly-opt`** 已存在。

### 6.1 Pytest：unit / system（编译与 IR 相关子集示例）

```bash
cd /path/to/FlyDSL
source .venv/bin/activate

python3 -m pytest tests/unit/ tests/system/ \
  -m "l0_backend_agnostic or l1a_compile_no_target_dialect or l1b_target_dialect" \
  -v --tb=short
```

更多目录（含 GPU 内核）：

```bash
python3 -m pytest tests/kernels/ tests/unit/ tests/system/ tests/python/examples/ \
  -m "not large_shape" -v --tb=short
```

### 6.2 Pytest：`tests/conftest.py` 与 `build-fly`

[`tests/conftest.py`](../tests/conftest.py) 若发现 **`build-fly/python_packages`**，会将其 **插入 `sys.path` 首位**。修改 **`python/flydsl`** 后若未重新 **`bash scripts/build.sh`**，测试仍可能加载旧拷贝；可重编、手动 **`cp`** 同步，或使用 **`pip install -e .`**。

### 6.3 FileCheck：`tests/mlir`（推荐）

```bash
export FLY_BUILD_DIR=/path/to/FlyDSL/build-fly
export PATH="${MLIR_PATH}/bin:${PATH}"
bash scripts/run_mlir_filecheck.sh
```

**`FileCheck`** 必须与 **`MLIR_PATH`** 同属 ixcc **`mlir_install`**（版本一致）。找不到时可 **`export PATH="${MLIR_PATH}/bin:${PATH}"`**。

### 6.4 FileCheck：手工循环（慎用 `exit 1`）

在**交互式**终端粘贴含 **`exit 1`** 的失败分支时，可能直接**退出当前 shell**。优先使用 **§6.3**；若手写循环，请用 **`bash scripts/run_mlir_filecheck.sh`** 或 **`bash -c '...'`** 包裹。

### 6.5 全量测试

```bash
FLY_BUILD_DIR=/path/to/FlyDSL/build-fly bash scripts/run_tests.sh
```

### 6.6 端到端示例（路径替换为你的机器）

```bash
export IXCC_ROOT="${HOME}/sw_home/sdk/ixcc"
export MLIR_PATH="${IXCC_ROOT}/mlir_install"

cd /path/to/FlyDSL
source .venv/bin/activate
export MLIR_PATH
bash scripts/build.sh -j"$(nproc)"

export FLY_BUILD_DIR=/path/to/FlyDSL/build-fly
export PYTHONPATH="${FLY_BUILD_DIR}/python_packages:${PWD}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${FLY_BUILD_DIR}/python_packages/flydsl/_mlir/_mlir_libs:${LD_LIBRARY_PATH:-}"
export FLYDSL_RUN_QUANT=1
export PATH="${MLIR_PATH}/bin:${PATH}"

bash scripts/run_mlir_filecheck.sh
python3 -m pytest tests/unit/ tests/system/ \
  -m "l0_backend_agnostic or l1a_compile_no_target_dialect or l1b_target_dialect" \
  -v --tb=short
```

---

## 7. 故障排查

### 7.1 `find_package(LLVM)` 找到错误版本

确保设置 **`MLIR_PATH`** 为 ixcc 的 **`mlir_install`**，且使用已更新的 **`scripts/build.sh`**（含 **`-DLLVM_DIR=...`**）。不要在同一环境中混用系统 LLVM 与 ixcc LLVM。

### 7.2 ixcc **构建树** CMake 里出现他人机器路径（例如 `/home/other/...`）

若 SDK 从别处拷贝，**`LLVMConfig.cmake`** 可能仍包含旧绝对路径，导致 **`include(...LLVMExports.cmake)`** 失败。需要在本机统一替换为当前路径（示例，谨慎操作）：

```bash
grep -rl '/home/olduser/sw_home/sdk/ixcc' /path/to/ixcc/build/lib/cmake \
  | xargs sed -i 's|/home/olduser|'"$HOME"'|g'
```

优先使用 **`mlir_install`**（安装前缀），路径一般由 CMake 按前缀生成，更可移植。

### 7.3 AMDGPU / ROCm

默认 **`build_ixcc_llvm.sh`** 不编 **AMDGPU**、不开 **ROCm runner**，以降低环境与源码缺陷概率。若需完整 GPU 后端与 ROCm，需在 ixcc 源码修复 AMDGPU TableGen、安装 HIP/ROCm，并相应调整 **`IXCC_LLVM_TARGETS`** 与 **`IXCC_MLIR_ENABLE_ROCM_RUNNER`**。

构建 FlyDSL 时若已安装 ROCm/HIP（**`find_package(hip)`** 成功），将生成 **`libfly_jit_runtime.so`**；否则在包含 **§4.1** 改动的分支上会 **跳过 FlyJitRuntime**，属预期行为。

### 7.4 Python 导入与 `gpu.LaunchFuncOp`

- **`NameError: wave_id is not defined`**（**`flydsl.expr.rocdl`**）：当前 MLIR 的 ROCDL Python 绑定可能未导出部分 intrinsic；需 **§4.3** 的 **`globals().get`** 写法，并确保 **`build-fly`** 与源码同步（见 **§6.2**）。
- **`LaunchFuncOp.__init__() got an unexpected keyword argument 'cluster_size'`**：MLIR 22 绑定不再接受该关键字；需 **§4.3** 的 **`kernel_function`** 修复。

### 7.5 FileCheck 与 `rocdl.mfma.scale` 打印格式

MLIR 22 将 **`mfma.scale`** 的操作数印为 **SSA**（**`arith.constant`**），旧 CHECK 中的行内整数 **`0, 0, 2`** 会失败；属 **IR 文本格式**变化，不是硬件后端错误。需 **§4.4** 的测试更新或等价放宽 CHECK。

---

## 8. 快速检查清单

- [ ] **`${MLIR_PATH}/lib/cmake/mlir`** 与 **`${MLIR_PATH}/lib/cmake/llvm`** 存在（**`MLIR_PATH`** 一般为 **`${IXCC_ROOT}/mlir_install`**）。
- [ ] **`bash scripts/build.sh`** 成功，**`${FLY_BUILD_DIR}/bin/fly-opt`** 存在。
- [ ] 运行 pytest / 导入 **`flydsl`** 前已设置 **`PYTHONPATH`**（含 **`python_packages`**）、**`LD_LIBRARY_PATH`**（**`_mlir_libs`**）。
- [ ] FileCheck：**`PATH`** 含 **`${MLIR_PATH}/bin`**，且 **`bash scripts/run_mlir_filecheck.sh`** 通过。
- [ ] Pytest：至少 **`tests/unit`** + **`tests/system`** 的目标 marker 子集可通过（见 **§6.1**）。
- [ ] 分支包含 **第 4 节**所列适配 MLIR 22 的改动（或等价合并）。

---

## 9. 参考脚本路径（仓库内）

| 脚本 | 用途 |
|------|------|
| [`scripts/build_ixcc_llvm.sh`](../scripts/build_ixcc_llvm.sh) | 清理并编译安装 ixcc LLVM/MLIR → **`mlir_install`** |
| [`scripts/build.sh`](../scripts/build.sh) | 配置并编译 FlyDSL（需 **`MLIR_PATH`**） |
| [`scripts/run_mlir_filecheck.sh`](../scripts/run_mlir_filecheck.sh) | **仅** **`tests/mlir`** FileCheck |
| [`scripts/run_tests.sh`](../scripts/run_tests.sh) | 全量测试（pytest 多目录 + FileCheck） |

以上为基于 ixcc **LLVM 22.x** 搭建 FlyDSL、构建并通过测试的推荐流程。
