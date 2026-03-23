# FlyDSL → Iluvatar (ixcc) 完整移植手册

> 本文档记录了将 FlyDSL GPU kernel 编译后端从 AMD ROCm 移植到 Iluvatar ixcc 的**全部流程**，
> 包括每一步的改动原因、遇到的错误及其修复方法。
> 以 `vectorAdd` kernel 为首个验证目标，**已完成端到端验证（编译 + 运行 + 结果正确）**。

---

## 目录

1. [项目概览与架构](#overview)
2. [关键路径与环境变量](#paths-and-env)
3. [阶段一 - 编译 ixcc MLIR](#stage1-build-ixcc)
4. [阶段二 - 编译 FlyDSL](#stage2-build-flydsl)
5. [阶段三 - 代码改动 FlyDSL 侧](#stage3-flydsl-changes)
6. [阶段四 - 代码改动 ixcc 侧](#stage4-ixcc-changes)
7. [阶段五 - Runtime 层替换](#stage5-runtime)
8. [阶段六 - Python 环境配置](#stage6-python-env)
9. [阶段七 - 端到端运行验证](#stage7-e2e-test)
10. [错误排查手册](#troubleshooting)
11. [MLIR Pass Pipeline 详解](#mlir-pipeline)
12. [附录](#appendix)

---

<a id="overview"></a>

## 项目概览与架构

### FlyDSL 编译流水线

```
Python DSL code (vectorAdd.py)
        │
        ▼
   AST Rewriter ──→ MLIR IR (Fly dialect + GPU dialect)
        │
        ▼
   MLIR Pass Pipeline (13 passes)
        │
        ├── fly-canonicalize / fly-layout-lowering  (FlyDSL 自有, 平台无关)
        ├── convert-fly-to-rocdl                     (类型 lowering, 见下文说明)
        ├── convert-gpu-to-ixdl                      (GPU intrinsics → IXDL intrinsics)
        ├── gpu-to-llvm / convert-*-to-llvm          (LLVM IR 生成)
        └── gpu-module-to-binary                     (调用 ixcc 后端生成 ELF binary)
        │
        ▼
   JIT Executor (加载 binary, 启动 kernel)
```

### 关键概念

| 概念 | 说明 |
|---|---|
| **FlyDSL** | Python GPU kernel DSL，基于 Layout Algebra，JIT 编译 |
| **Fly dialect** | FlyDSL 自有 MLIR dialect（`fly.ptr`, `fly.copy_atom_call` 等） |
| **FlyROCDL dialect** | FlyDSL 中 AMD CDNA 专用的 dialect（MFMA 指令映射） |
| **convert-fly-to-rocdl pass** | **名字有误导性**——实际上它不只处理 ROCDL，更关键的是做 Fly dialect 的通用类型 lowering（`!fly.ptr` → `!llvm.ptr` 等），host 和 device 代码都需要 |
| **IXDL dialect** | ixcc 中 Iluvatar 硬件的 MLIR dialect |
| **convert-gpu-to-ixdl pass** | ixcc 提供的 GPU → IXDL 转换 pass |
| **`#ixdl.target`** | Iluvatar GPU 的 target attribute，替代 `#rocdl.target` |
| **Iluvatar triple** | `bi-iluvatar-ilurt` |
| **chip 名称** | `ivcore11` |

---

<a id="paths-and-env"></a>

## 关键路径与环境变量

### 路径

| 路径 | 说明 |
|---|---|
| `/home/wenhui-liang/flydsl` | FlyDSL 项目根目录 |
| `/home/wenhui-liang/flydsl/build-fly` | upstream MLIR 的 build（保留不动） |
| `/home/wenhui-liang/flydsl/build-ixcc` | ixcc MLIR build 目录 |
| `/home/wenhui-liang/ilu/ixcc` | Iluvatar LLVM 仓库（基于 LLVM 22） |
| `/home/wenhui-liang/ilu/ixcc/build` | ixcc 编译产物 |
| `/home/wenhui-liang/flydsl/lib/Runtime/FlyCorexRuntimeWrappers.cpp` | Iluvatar COREX runtime wrapper（新建） |
| `/home/wenhui-liang/flydsl/lib/Runtime/FlyRocmRuntimeWrappers.cpp` | 原始 ROCm/HIP runtime wrapper（保留不动） |
| `/home/wenhui-liang/sw_home/local/corex` | Iluvatar COREX SDK 安装目录 |
| `/home/wenhui-liang/test/vectorAdd.py` | 平台无关的 vectorAdd 测试脚本 |
| `~/.flydsl/debug/<kernelName>_N/` | `FLYDSL_DUMP_IR=1` 时各 pass 的 IR 输出目录 |

### 环境变量

| 变量 | 值 | 作用 |
|---|---|---|
| `PYTHONPATH` | `/home/wenhui-liang/flydsl/build-ixcc/python_packages` | 指向 ixcc build 的 Python 包 |
| `FLYDSL_GPU_ARCH` / `ARCH` | `ivcore11` | 绕过 ROCm 设备检测，直接指定 chip |
| `COMPILE_ONLY` | `1` | 只编译不执行（无需硬件） |
| `FLYDSL_RUNTIME_ENABLE_CACHE` | `0` | 禁用编译缓存（调试时必须） |
| `FLYDSL_DUMP_IR` | `1` | 每个 pass 后 dump IR 到文件 |
| `PATH` | 必须将 `ixcc/build/bin` 放在最前面 | 让 `ld.lld` 使用 ixcc 自带版本 |
| `LD_LIBRARY_PATH` | 包含 `/home/wenhui-liang/sw_home/local/corex/lib64` | 运行时加载 `libcuda.so`、`libcudart.so` 等 COREX 库 |
| `IXA_PATH` | **不要设置** | 设置后会触发 libdevice 路径查找失败 |

### 验证编译的完整命令（仅编译，不需要硬件）

```bash
source /home/wenhui-liang/flydsl/.venv/bin/activate

PATH=/home/wenhui-liang/ilu/ixcc/build/bin:$PATH \
COMPILE_ONLY=1 \
ARCH=ivcore11 \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
PYTHONPATH=/home/wenhui-liang/flydsl/build-ixcc/python_packages \
python /home/wenhui-liang/test/vectorAdd.py
```

### 端到端运行的完整命令（需要 Iluvatar 硬件）

```bash
source /home/wenhui-liang/flydsl/.venv/bin/activate

PATH=/home/wenhui-liang/ilu/ixcc/build/bin:$PATH \
ARCH=ivcore11 \
PYTHONPATH=/home/wenhui-liang/flydsl/build-ixcc/python_packages \
LD_LIBRARY_PATH=/home/wenhui-liang/sw_home/local/corex/lib64:$LD_LIBRARY_PATH \
python /home/wenhui-liang/test/vectorAdd.py
```

**预期输出**:
```
[FlyCorex] mgpuModuleLoad: data=0x..., size=2768
[FlyCorex] cuModuleLoadData OK: module=0x...
[FlyCorex] mgpuModuleGetFunction: module=0x..., name='vectorAddKernel_0'
[FlyCorex] cuModuleGetFunction OK: func=0x...
Result correct: True
```

---

<a id="stage1-build-ixcc"></a>

## 阶段一 - 编译 ixcc MLIR

ixcc 使用 `make config` 封装 cmake。标准构建**不含 Python bindings**，但 FlyDSL 必须有。

### 构建步骤

```bash
# 步骤 1: ixcc 标准 config（用 Release 避免 BUILD_SHARED_LIBS=on 问题）
cd /home/wenhui-liang/ilu/ixcc
make config TARGET_DIR=./build BUILD_TYPE=Release \
    CUDA_PATH=/home/wenhui-liang/sw_home/local/corex \
    CPUS=192 BINUTILS_INCDIR=/usr/include ENABLE_Z3= PUBLIC=1 ENABLE_MLIR=1

# 步骤 2: 追加 Python binding 参数
cd /home/wenhui-liang/ilu/ixcc/build
cmake . \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DMLIR_BINDINGS_PYTHON_NB_DOMAIN=mlir \
    -DPython3_EXECUTABLE=$(which python3) \
    -DBUILD_SHARED_LIBS=OFF

# 步骤 3: 编译
cd /home/wenhui-liang/ilu/ixcc
make build TARGET_DIR=./build CPUS=192
```

**关键参数说明：**
- `MLIR_BINDINGS_PYTHON_NB_DOMAIN=mlir` — 必须与 FlyDSL CMakeLists.txt 一致（写死为 `mlir`）
- `BUILD_SHARED_LIBS=OFF` — Debug 模式默认共享库，FlyDSL 需要静态链接
- `CUDA_PATH` 指向 COREX SDK（`IluvatarSerializer` 运行时用来找 `ld.lld` 等工具）

**编译完成后 cmake 路径：**
```
MLIR_DIR: /home/wenhui-liang/ilu/ixcc/build/lib/cmake/mlir
LLVM_DIR: /home/wenhui-liang/ilu/ixcc/build/lib/cmake/llvm
```

---

<a id="stage2-build-flydsl"></a>

## 阶段二 - 编译 FlyDSL

```bash
cd /home/wenhui-liang/flydsl
mkdir -p build-ixcc && cd build-ixcc

cmake .. \
    -DMLIR_DIR=/home/wenhui-liang/ilu/ixcc/build/lib/cmake/mlir \
    -DLLVM_DIR=/home/wenhui-liang/ilu/ixcc/build/lib/cmake/llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DFLYDSL_ENABLE_ROCM=OFF \
    -DFLYDSL_ENABLE_COREX=ON \
    -DBUILD_SHARED_LIBS=OFF

cmake --build . -j$(nproc)
```

注意：
- `-DFLYDSL_ENABLE_ROCM=OFF` 跳过 HIP runtime wrapper 的编译（机器上没有 ROCm）
- `-DFLYDSL_ENABLE_COREX=ON` 启用 Iluvatar COREX runtime wrapper（基于 CUDA Driver API）
- 编完后 Python 包在 `build-ixcc/python_packages/`
- 如果修改了 `python/flydsl/` 下的 `.py` 文件，需要手动复制到 `build-ixcc/python_packages/flydsl/`（或重新 cmake build）

### Python 包隔离机制

FlyDSL 的 Python binding 完全自包含：
- 包前缀：`flydsl._mlir`（由 `MLIR_PYTHON_PACKAGE_PREFIX="_mlir"` 控制）
- 所有 MLIR 符号静态链接进 `libFlyPythonCAPI.so`，用 version script 控制导出
- `build-fly/` 和 `build-ixcc/` 产物完全隔离，用 `PYTHONPATH` 切换

`python/flydsl/_mlir` 是一个 symlink → build 目录。切换 build 有两种方式：

```bash
# 方式一：更新 symlink
ln -sfn ../../build-ixcc/python_packages/flydsl/_mlir \
        /home/wenhui-liang/flydsl/python/flydsl/_mlir

# 方式二：用 PYTHONPATH（推荐，不动源码目录）
PYTHONPATH=/home/wenhui-liang/flydsl/build-ixcc/python_packages python ...
```

---

<a id="stage3-flydsl-changes"></a>

## 阶段三 - 代码改动 FlyDSL 侧

### 核心发现：`convert-fly-to-rocdl` 必须保留

**这是整个移植过程中最关键的认知。**

`convert-fly-to-rocdl` 这个 pass 的名字有误导性。它实际做了两件事：

1. **平台无关的 Fly dialect 类型 lowering**：`!fly.ptr` → `!llvm.ptr`，`fly.copy_atom_call` → 标准 LLVM/MLIR ops
2. **AMD 专用的 MFMA 指令 lowering**：将 `fly.mma_atom_call` 映射到 AMD MFMA intrinsics

对于 `vectorAdd` 这类不使用 MFMA 的 kernel，只需要第 1 部分。我们的策略是：**保留 pass，stub 掉 MFMA 部分**。

### 改动 1：`convert-fly-to-rocdl` 的作用域

`convert-fly-to-rocdl` 在 pipeline 中必须是模块级别的（而非嵌套在 `gpu.module(...)` 中），这样它能同时处理 host 代码和 device 代码中的 `!fly.ptr` 类型。

如果把它放进 `gpu.module(...)` 中，只有 kernel 侧的类型被 lower，host 侧的 `gpu.launch_func` 仍持有 `!fly.ptr` 类型参数，与 kernel 侧已经 lower 为 `!llvm.ptr` 的签名不匹配，报错：

```
'gpu.launch_func' op type of function argument 0 does not match
```

**最终正确的位置**：模块级别，在 `gpu.module(...)` 之前。

### 改动 2：架构自动分辨与 pipeline 分派

`jit_function.py` 通过 chip 名称前缀自动区分 AMD 和 Iluvatar 平台，同一份代码兼容两种后端：

```python
# python/flydsl/compiler/jit_function.py

def _is_iluvatar_arch(chip: str) -> bool:
    return chip.startswith("ivcore")

def _gpu_target_attr(chip: str) -> str:
    if _is_iluvatar_arch(chip):
        return f'#ixdl.target<chip = "{chip}">'
    return f'#rocdl.target<chip = "{chip}">'
```

`_pipeline_fragments()` 根据平台选择不同的 GPU lowering pass 和 binary 格式：

| 平台判断 | `ivcore*` (Iluvatar) | `gfx*` (AMD ROCm) |
|---|---|---|
| GPU lowering | `convert-gpu-to-ixdl{index-bitwidth=32}` | `convert-gpu-to-rocdl{chipset=... runtime=HIP ...}` |
| attach target | 不需要（target 在 `create_gpu_module` 中设置） | `rocdl-attach-target{...}` |
| binary format | `binary`（ELF） | `fatbin`（多架构 fat binary） |
| target attribute | `#ixdl.target<chip = "ivcore11">` | `#rocdl.target<chip = "gfx942">` |

公共部分（`convert-fly-to-rocdl`、`canonicalize`、`gpu-to-llvm`、`convert-*-to-llvm` 等）两个平台完全共享。

通过 `ARCH` 环境变量指定架构（如 `ARCH=ivcore11` 或 `ARCH=gfx942`），无需修改代码即可切换平台。

### 改动 3：stub 掉 MFMA lowering（C++ 层）

**文件**: `lib/Conversion/FlyToROCDL/FlyToROCDL.cpp`

ixcc 的 ROCDL ops header 修改了 `MfmaOp::create()` 的 API 签名（参数数量/类型变化），
导致 `emitMfma` 模板编译失败。由于 `vectorAdd` 不使用 MFMA，直接 stub 掉：

```diff
   template <typename MfmaOp>
   LogicalResult emitMfma(MmaAtomCall op, ConversionPatternRewriter &rewriter, Location loc,
                          Type abTyA, Type abTyB, VectorType accTy, Value aPtr, Value bPtr,
                          Value cPtr, Value dPtr) const {
-    Value a = LLVM::LoadOp::create(rewriter, loc, abTyA, aPtr);
-    Value b = LLVM::LoadOp::create(rewriter, loc, abTyB, bPtr);
-    Value c = LLVM::LoadOp::create(rewriter, loc, accTy, cPtr);
-    auto zeroAttr = rewriter.getI32IntegerAttr(0);
-    Value res =
-        MfmaOp::create(rewriter, loc, accTy, a, b, c, zeroAttr, zeroAttr, zeroAttr).getResult();
-    LLVM::StoreOp::create(rewriter, loc, res, dPtr);
-    rewriter.eraseOp(op);
-    return success();
+    return op.emitOpError("MFMA lowering not available in Iluvatar port"), failure();
   }
```

> **如果后续需要支持 MFMA kernel**，需要根据 ixcc 版本的 `MfmaOp::create()` 签名重写此函数。

### 改动 4：FlyOps.cpp 的 Diagnostic API 修复

ixcc 的 `Diagnostics.h` 模板约束更严格，不允许直接把自定义枚举传给 diagnostic：

```cpp
// lib/Dialect/Fly/IR/FlyOps.cpp 约 1614 行
// 原来:
return emitOptionalError(location,
                         "MmaMakeFragmentOp: invalid operand_id value: ",
                         operandId);
// 改成:
return emitOptionalError(location,
                         "MmaMakeFragmentOp: invalid operand_id value: ",
                         static_cast<int>(operandId));
```

> 如果后续遇到类似 `no matching function for call to 'adl_begin'` 错误，统一用 `static_cast<int>(...)` 修复。

---

<a id="stage4-ixcc-changes"></a>

## 阶段四 - 代码改动 ixcc 侧

### Bug fix 1 - GPUReturnOpLowering 缺失

**文件**: `mlir/lib/Conversion/GPUToIXDL/LowerGpuOpsToIXDLOps.cpp`

ixcc 的 `convert-gpu-to-ixdl` pass 的 `populateGpuToIXDLConversionPatterns` 函数中，
`GPUReturnOpLowering` 没有被注册到 patterns 中，导致 `gpu.return` 操作无法被 legalize：

```
error: failed to legalize operation 'gpu.return' that was explicitly marked illegal
```

修复：

```diff
-  patterns.add<GPULaneIdOpToIXDL, GPUSubgroupIdOpToIXDL, GPUShuffleOpLowering>(
-      converter);
+  patterns.add<GPULaneIdOpToIXDL, GPUSubgroupIdOpToIXDL, GPUShuffleOpLowering,
+               GPUReturnOpLowering>(converter);
```

修复后需要重新编译 ixcc 中的对应 target：

```bash
cd /home/wenhui-liang/ilu/ixcc/build
cmake --build . --target MLIRGPUToIXDLTransforms -j$(nproc)
```

然后重新编译 FlyDSL（因为它静态链接了 ixcc 的库）：

```bash
cd /home/wenhui-liang/flydsl/build-ixcc
cmake --build . -j$(nproc)
```

<a id="bugfix2-callingconv"></a>

### Bug fix 2 - CallingConv ILUVATAR_KERNEL 未设置

**文件**: `mlir/lib/Target/LLVMIR/Dialect/IXDL/IXDLToLLVMIRTranslation.cpp`

**这是阻塞端到端运行的最后一个 bug。** 症状：编译成功、ELF 生成成功、`cuModuleLoadData` 成功，但 `cuModuleGetFunction` 返回 `CUDA_ERROR_NOT_FOUND`，无法找到 kernel 函数。

**根因分析链路**：

1. `GPUFuncOpLowering` 将 `gpu.func` 降级为 `llvm.func`，附加 `ixdl.kernel` 属性
2. `IXDLToLLVMIRTranslation` 翻译到 LLVM IR 时，只向 `nvvm.annotations` metadata 注册了 `(func, "kernel", 1)`
3. **缺失：没有调用 `llvmFunc->setCallingConv(llvm::CallingConv::ILUVATAR_KERNEL)`**
4. Iluvatar LLVM 后端通过 `isEntryFunctionCC()` 判断 kernel（检查 calling convention），返回 `false`
5. `IluvatarAsmPrinter` 不发出 `.iluvatar_kernel` 指令
6. ELF `.note` section 中 `iluvatar.kernels` 为空 msgpack 数组（`\x90`）
7. COREX runtime `cuModuleGetFunction` 无法索引 kernel → `CUDA_ERROR_NOT_FOUND`

**对比 NVVM 翻译层**（NVVM 没有此问题）：

| 翻译层 | 属性 | 设置 calling conv | 设置 metadata |
|---|---|---|---|
| `NVVMToLLVMIRTranslation` | `nvvm.kernel` | `CallingConv::PTX_Kernel` | 无 |
| `IXDLToLLVMIRTranslation`（修复前） | `ixdl.kernel` | **缺失** | `nvvm.annotations` |
| `IXDLToLLVMIRTranslation`（修复后） | `ixdl.kernel` | `CallingConv::ILUVATAR_KERNEL` | `nvvm.annotations` |

**Iluvatar 后端两条 kernel 识别路径**：

- `isEntryFunctionCC(F.getCallingConv())` — 检查 `CallingConv::ILUVATAR_KERNEL`（值为 128）→ 决定 ELF symbol type `STT_ILUVATAR_KERNEL` 和 `.note` 中 kernel 列表
- `isKernelFunction(F)` — 检查函数属性 `"kernel"` = 1（来自 `nvvm.annotations`）→ 仅用于获取 `ReqNTIDx` 等参数

修复（仅增加一行）：

```diff
     } else if (attribute.getName() ==
                IXDL::IXDLDialect::getKernelFuncAttrName()) {
+      llvmFunc->setCallingConv(llvm::CallingConv::ILUVATAR_KERNEL);
       llvm::Metadata *llvmMetadataKernel[] = {
           llvm::ValueAsMetadata::get(llvmFunc),
           llvm::MDString::get(llvmContext, "kernel"),
```

修复后重新编译 ixcc 和 FlyDSL：

```bash
cd /home/wenhui-liang/ilu/ixcc/build
ninja -j$(nproc) MLIRIXDLToLLVMIRTranslation

cd /home/wenhui-liang/flydsl/build-ixcc
make -j$(nproc)
```

**验证**：修复后 ELF `.note` section 中 `iluvatar.kernels` 从空数组 `\x90` 变为包含 kernel 信息的完整 msgpack：
```
iluvatar.kernels → [{.name: "vectorAddKernel_0", .symbol: "vectorAddKernel_0.kd", .args: [...], ...}]
```

---

<a id="stage5-runtime"></a>

## 阶段五 - Runtime 层替换

### 概述

MLIR 的 `gpu-to-llvm` pass 将 GPU host ops（如 `gpu.launch_func`、`gpu.alloc`）lowering 为
对 `mgpu*` C ABI 函数的调用。这些函数需要由 runtime wrapper 库实现。

原始 FlyDSL 使用 `FlyRocmRuntimeWrappers.cpp`（基于 HIP API）。Iluvatar COREX SDK 提供
CUDA Driver API 兼容层，因此我们创建了 `FlyCorexRuntimeWrappers.cpp`（基于 CUDA Driver API）。

### 改动 1：创建 `FlyCorexRuntimeWrappers.cpp`

**文件**: `lib/Runtime/FlyCorexRuntimeWrappers.cpp`（新建）

基于 MLIR upstream 的 `CudaRuntimeWrappers.cpp` 改写，导出 `mgpu*` C ABI 接口：

| mgpu 函数 | CUDA Driver API 调用 | 作用 |
|---|---|---|
| `mgpuModuleLoad` | `cuModuleLoadData` | 加载 ELF binary 为 CUmodule |
| `mgpuModuleGetFunction` | `cuModuleGetFunction` | 获取 kernel 函数句柄 |
| `mgpuLaunchKernel` | `cuLaunchKernel` | 启动 kernel |
| `mgpuStreamCreate/Destroy/Sync` | `cuStreamCreate/Destroy/Synchronize` | stream 管理 |
| `mgpuStreamWaitEvent` | `cuStreamWaitEvent` | stream 等待 event |
| `mgpuEventCreate/Destroy/Sync/Record` | `cuEvent*` | event 管理 |
| `mgpuMemAlloc/Free` | `cuMemAlloc/cuMemFree` | 显存分配/释放 |
| `mgpuMemcpy` | `cuMemcpyAsync` | 异步内存拷贝 |
| `mgpuMemset32/16` | `cuMemsetD32Async/D16Async` | 显存置零 |
| `mgpuMemHostRegister/Unregister` | `cuMemHostRegister/Unregister` | 主机内存注册 |
| `mgpuSetDefaultDevice` | 设置线程局部 `defaultDevice` | 选择 GPU 设备 |

上下文管理使用 `ScopedContext` RAII 类，首次使用时通过 `cuInit` + `cuDevicePrimaryCtxRetain`
初始化，之后每次通过 `cuCtxPushCurrent` / `cuCtxPopCurrent` 管理。

> 当前 wrapper 中 `mgpuModuleLoad` 和 `mgpuModuleGetFunction` 保留了 debug print
> （输出到 stderr），便于调试。生产环境可以去除。

### 改动 2：`CMakeLists.txt` 条件编译

**文件**: `python/mlir_flydsl/CMakeLists.txt`

添加了两个 cmake option，支持在 ROCm 和 COREX 之间选择 runtime：

```cmake
option(FLYDSL_ENABLE_ROCM "Enable ROCm/HIP JIT runtime" ON)
option(FLYDSL_ENABLE_COREX "Enable Iluvatar COREX (CUDA Driver API) JIT runtime" OFF)
```

构建优先级：COREX > ROCm > 无 GPU runtime。逻辑：

```
if FLYDSL_ENABLE_COREX:
    搜索 COREX SDK ($COREX_PATH, $CUDA_PATH, ~/sw_home/local/corex, /usr/local/corex)
    找到 cuda.h + libcuda.so → 编译 FlyCorexRuntimeWrappers.cpp, 链接 libcuda.so
if FLYDSL_ENABLE_ROCM and 未构建:
    搜索 ROCm (/opt/rocm*)
    找到 hip → 编译 FlyRocmRuntimeWrappers.cpp, 链接 hip::host hip::amdhip64
if 都未构建:
    add_definitions(-DFLYDSL_DISABLE_ROCM)  # 无 GPU runtime
```

**FlyDSL 编译命令更新**：

```bash
cd /home/wenhui-liang/flydsl/build-ixcc
cmake .. \
    -DMLIR_DIR=/home/wenhui-liang/ilu/ixcc/build/lib/cmake/mlir \
    -DLLVM_DIR=/home/wenhui-liang/ilu/ixcc/build/lib/cmake/llvm \
    -DCMAKE_BUILD_TYPE=Release \
    -DFLYDSL_ENABLE_ROCM=OFF \
    -DFLYDSL_ENABLE_COREX=ON \
    -DBUILD_SHARED_LIBS=OFF

cmake --build . -j$(nproc)
```

编译产物 `libfly_jit_runtime.so` 输出到 `build-ixcc/python_packages/flydsl/_mlir/_mlir_libs/`。

---

<a id="stage6-python-env"></a>

## 阶段六 - Python 环境配置

### Iluvatar PyTorch 安装

Iluvatar 有自己的 PyTorch 发行版（与 COREX SDK 版本匹配）。标准 pip 安装的 NVIDIA PyTorch 不兼容。

```bash
# 创建 Python 3.12 虚拟环境
cd /home/wenhui-liang/flydsl
uv venv --python 3.12 .venv
source .venv/bin/activate

# 安装 Iluvatar PyTorch（需要从 Iluvatar 内部源获取 whl）
uv pip install /path/to/torch-xxx+corex.yyy-cpXXX-linux_x86_64.whl

# 如果遇到 NumPy 版本不兼容（NumPy 2.x vs PyTorch 需要 NumPy 1.x）
uv pip install "numpy<2"
```

### 环境变量

PyTorch 运行时需要 COREX SDK 的共享库：

```bash
export LD_LIBRARY_PATH=/home/wenhui-liang/sw_home/local/corex/lib64:$LD_LIBRARY_PATH
```

### 验证 PyTorch + Iluvatar GPU

```python
import torch
print(torch.cuda.is_available())       # True
print(torch.cuda.device_count())       # >= 1
print(torch.cuda.get_device_name(0))   # 应显示 Iluvatar 设备名称

a = torch.randn(100, device='cuda')
b = torch.randn(100, device='cuda')
c = a + b
print("GPU compute OK:", c.device)     # cuda:0
```

> **注意**：如果 `torch.cuda.is_available()` 返回 `False` 并报 "unsupported display driver / cuda
> driver combination"，通常是 KMD（Kernel Module Driver）版本与 COREX SDK 版本不匹配，需要同步更新。

---

<a id="stage7-e2e-test"></a>

## 阶段七 - 端到端运行验证

### 编译测试（不需要 Iluvatar 硬件）

```bash
source /home/wenhui-liang/flydsl/.venv/bin/activate

PATH=/home/wenhui-liang/ilu/ixcc/build/bin:$PATH \
COMPILE_ONLY=1 \
ARCH=ivcore11 \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
PYTHONPATH=/home/wenhui-liang/flydsl/build-ixcc/python_packages \
python /home/wenhui-liang/test/vectorAdd.py
```

**预期**: exit code 0，输出 `[flydsl] COMPILE_ONLY=1, compilation succeeded (arch=ivcore11)`。

### 端到端运行（需要 Iluvatar 硬件）

```bash
source /home/wenhui-liang/flydsl/.venv/bin/activate

PATH=/home/wenhui-liang/ilu/ixcc/build/bin:$PATH \
ARCH=ivcore11 \
PYTHONPATH=/home/wenhui-liang/flydsl/build-ixcc/python_packages \
LD_LIBRARY_PATH=/home/wenhui-liang/sw_home/local/corex/lib64:$LD_LIBRARY_PATH \
python /home/wenhui-liang/test/vectorAdd.py
```

**预期输出**:
```
[FlyCorex] mgpuModuleLoad: data=0x..., size=2768
[FlyCorex] cuModuleLoadData OK: module=0x...
[FlyCorex] mgpuModuleGetFunction: module=0x..., name='vectorAddKernel_0'
[FlyCorex] cuModuleGetFunction OK: func=0x...
A: [1.0, 8.0, ...]
B: [6.0, 8.0, ...]
C: [7.0, 16.0, ...]
Expected: [7.0, 16.0, ...]
Result correct: True
```

### 调试 IR Dump

```bash
PATH=/home/wenhui-liang/ilu/ixcc/build/bin:$PATH \
COMPILE_ONLY=1 \
ARCH=ivcore11 \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
FLYDSL_DUMP_IR=1 \
PYTHONPATH=/home/wenhui-liang/flydsl/build-ixcc/python_packages \
python /home/wenhui-liang/test/vectorAdd.py
```

IR dump 文件在 `~/.flydsl/debug/vectorAddKernel_0/` 下，每个 pass 一个 `.mlir` 文件：

```
00_origin.mlir
01_gpu_kernel_outlining.mlir
02_fly_canonicalize.mlir
03_fly_layout_lowering.mlir
04_convert_fly_to_rocdl.mlir      ← 注意此处名字含 rocdl，但做的是通用类型 lowering
05_canonicalize.mlir
06_convert_scf_to_cf_cse_convert_gpu_to_ixdl.mlir
07_convert_scf_to_cf.mlir
08_convert_cf_to_llvm.mlir
09_gpu_to_llvm.mlir
10_convert_arith_to_llvm.mlir
11_convert_func_to_llvm.mlir
12_reconcile_unrealized_casts.mlir
13_gpu_module_to_binary.mlir      ← 含嵌入的 ELF binary
14_final_isa.s                    ← 最终的 ISA 汇编
```

合并为单个带 pass 名称标题的文件：

```bash
cd ~/.flydsl/debug/vectorAddKernel_0
for f in $(ls *.mlir | sort); do
    name=$(echo "$f" | sed 's/\.mlir$//')
    echo "// -----// IR Dump After $name //-----"
    cat "$f"
    echo
done > /home/wenhui-liang/test/vectorAdd_all_passes.mlir
```

### vectorAdd.py 测试脚本说明

`/home/wenhui-liang/test/vectorAdd.py` 是平台无关版本，与原始 `examples/01-vectorAdd.py` 的区别：
- 不使用 `fx.rocdl.make_buffer_tensor`、`fx.rocdl.BufferCopy32b`（AMD buffer 语义专用）
- 使用 `fx.UniversalCopy32b()`（通用 copy atom）
- `fx.slice(tA, (None, bid))` 之后必须有 `fx.logical_divide(tA, fx.make_layout(1, 1))`，否则后续 `(None, tid)` slice 会报 rank 不匹配

---

<a id="troubleshooting"></a>

## 错误排查手册

### 错误 1: `failed to legalize operation 'gpu.func' that was explicitly marked illegal`

**原因**: `convert-fly-to-rocdl` pass 被禁用或从 pipeline 中移除，
导致 `!fly.ptr` 类型没有被 lower 为 `!llvm.ptr`。后续的 `convert-gpu-to-ixdl` 看到
不认识的 `!fly.ptr` 类型，无法处理 `gpu.func` 的函数签名。

**修复**: 保留 FlyROCDL 编译和 `convert-fly-to-rocdl` pass——它负责平台无关的类型 lowering，不能移除。

---

### 错误 2: `MfmaOp::create()` API 编译失败

**原因**: ixcc 的 ROCDL ops header 修改了 `create()` 方法签名（参数数量/类型不同于 upstream），
FlyDSL 的 `FlyToROCDL.cpp` 中的 `emitMfma` 模板无法编译。

**修复**: Stub 掉 `emitMfma` 函数体，返回 failure。`vectorAdd` 不使用 MFMA 操作，所以 stub 不影响功能。

---

### 错误 3: `ModuleNotFoundError: No module named 'flydsl._mlir._mlir_libs._fly_rocdl'`

**原因**: `python/flydsl/expr/__init__.py` 中 `from . import rocdl` 如果被改成 `try-except ImportError`，
会静默吞掉真正的导入错误，导致 FlyROCDL 模块无法加载。

**修复**: 确保使用直接 import：
```python
from . import arith, vector, gpu, buffer_ops, rocdl
```

---

### 错误 4: `'gpu.launch_func' op type of function argument 0 does not match`

**原因**: `convert-fly-to-rocdl` 被放在 `gpu.module(...)` 内部执行，只 lower 了 kernel 侧的
`!fly.ptr` 类型。host 侧的 `gpu.launch_func` 仍然持有 `!fly.ptr` 参数，与 kernel 的 `!llvm.ptr`
签名不匹配。

**修复**: 将 `convert-fly-to-rocdl` 移到 `gpu.module(...)` **外面**，在模块级别运行，同时 lower host 和 device 代码。

这是**最容易犯的错误**，因为直觉上会认为 "rocdl 相关的 pass 应该只在 GPU module 中运行"。

---

### 错误 5: `failed to legalize operation 'gpu.return' that was explicitly marked illegal`

**原因**: ixcc 的 `convert-gpu-to-ixdl` pass 注册 lowering patterns 时遗漏了 `GPUReturnOpLowering`。

**修复**: 在 ixcc 源码 `LowerGpuOpsToIXDLOps.cpp` 中的 `populateGpuToIXDLConversionPatterns` 里
添加 `GPUReturnOpLowering`。

---

### 错误 6: ld.lld unsupported ABI version

见下方 [错误 9](#error9)。

---

### 错误 7: LibDevice path does not exist

见下方 [错误 10](#error10)。

---

### 错误 8: `FLYDSL_PRINT_AFTER_ALL=1` 不产生输出

**原因**: `FLYDSL_PRINT_AFTER_ALL=1` 通过 C++ `llvm::errs()` 输出到 stderr，但 Python 的
`sys.stderr` 重定向无法可靠捕获 C++ 层的 stderr 输出（尤其是 compilation 提前失败或通过
pipe/tail 访问时）。

**解决办法**: 使用 `FLYDSL_DUMP_IR=1` 替代，它将每个 pass 的 IR 写入独立文件。
然后用 shell 命令合并成 "IR Dump After" 格式的单文件。

---

<a id="error9"></a>

### 错误 9: ld.lld unsupported ABI version

**原因**: 系统自带的 `ld.lld`（来自系统 LLVM 包）不认识 Iluvatar 的 ELF ABI 版本。
`gpu-module-to-binary` pass 内部调用 `ld.lld` 做链接，ixcc 的 `findTool` 优先从 `PATH` 查找。

**修复**: 将 ixcc 的 `build/bin` 放在 `PATH` 最前面：
```bash
PATH=/home/wenhui-liang/ilu/ixcc/build/bin:$PATH
```

---

<a id="error10"></a>

### 错误 10: LibDevice path does not exist

**原因**: 设置了 `IXA_PATH` 环境变量指向 ixcc 编译目录，触发了 `appendStandardLibs` 中的
libdevice 查找逻辑。`vectorAdd` 不需要 libdevice（不调用数学函数如 sin/cos/sqrt）。

**修复**: **不设置** `IXA_PATH` 环境变量。当 `toolkitPath` 为空时，`appendStandardLibs` 跳过
libdevice 加载。

> 如果后续 kernel 确实需要 libdevice（使用了 math 函数），需要找到 Iluvatar SDK 中的
> `libdevice.10.bc` 的实际位置并正确设置 `IXA_PATH`。

---

### 错误 11: `cuModuleGetFunction` 返回 `CUDA_ERROR_NOT_FOUND`

**原因**: ixcc 的 `IXDLToLLVMIRTranslation` 在翻译 `ixdl.kernel` 属性时，只向
`nvvm.annotations` metadata 注册了 kernel 标记，但**没有设置 `CallingConv::ILUVATAR_KERNEL`**。
Iluvatar 后端通过 `isEntryFunctionCC()` 判断 kernel（检查 calling convention 而非 metadata），
导致 kernel 函数未被识别为 entry function，ELF `.note` section 中 `iluvatar.kernels` 为空。

**修复**: 在 `IXDLToLLVMIRTranslation.cpp` 中处理 `ixdl.kernel` 属性时增加一行：
```cpp
llvmFunc->setCallingConv(llvm::CallingConv::ILUVATAR_KERNEL);
```

详见 [阶段四 Bug fix 2](#bugfix2-callingconv)。

---

### 错误 12: PyTorch `NumPy 1.x cannot be run in NumPy 2.x`

**原因**: Iluvatar PyTorch 编译时链接的是 NumPy 1.x ABI，但系统安装了 NumPy 2.x。

**修复**: 降级 NumPy：
```bash
uv pip install "numpy<2"
```

---

### 错误 13: `OSError: libcudart.so.10.2: cannot open shared object file`

**原因**: PyTorch 导入时需要 COREX SDK 的共享库，但 `LD_LIBRARY_PATH` 未包含 COREX 路径。

**修复**:
```bash
export LD_LIBRARY_PATH=/home/wenhui-liang/sw_home/local/corex/lib64:$LD_LIBRARY_PATH
```

---

### 错误 14: `CUDA initialization: Unexpected error ... unsupported display driver / cuda driver combination`

**原因**: Iluvatar KMD（Kernel Module Driver）版本与 COREX SDK 版本不匹配。

**修复**: 同步更新 KMD 和 COREX SDK 到匹配的版本。

---

<a id="mlir-pipeline"></a>

## MLIR Pass Pipeline 详解

完整的 13 阶段变换流程（以 `vectorAdd` kernel 为例）：

| # | Pass | 作用 | 平台相关 |
|---|---|---|---|
| 0 | 原始 IR | FlyDSL Python DSL 生成的 Fly dialect IR | — |
| 1 | `gpu-kernel-outlining` | 把 `gpu.launch` 提取为独立的 `gpu.func` + `gpu.launch_func` | 无关 |
| 2 | `fly-canonicalize` | Fly dialect 规范化 | 无关 |
| 3 | `fly-layout-lowering` | Layout algebra lowering（slice/divide/compose → 具体索引计算） | 无关 |
| 4 | `convert-fly-to-rocdl` | `!fly.ptr` → `!llvm.ptr`，`fly.copy_atom_call` → load/store | **名字误导，实际平台无关** |
| 5 | `canonicalize` | 标准 MLIR 规范化（消除冗余、常量折叠） | 无关 |
| 6 | `convert-scf-to-cf` + `cse` + `convert-gpu-to-ixdl` | SCF → CF 控制流转换 + GPU intrinsics → IXDL intrinsics（`gpu.thread_id` → `ixdl.threadid.*`） | **IXDL 专用** |
| 7 | `convert-scf-to-cf` | host 侧剩余 SCF → CF | 无关 |
| 8 | `convert-cf-to-llvm` | CF dialect → LLVM dialect | 无关 |
| 9 | `gpu-to-llvm` | GPU host ops（`gpu.launch_func`, `gpu.alloc` 等）→ LLVM function calls | 无关 |
| 10 | `convert-arith-to-llvm` | Arith ops → LLVM ops | 无关 |
| 11 | `convert-func-to-llvm` | func.func → llvm.func | 无关 |
| 12 | `reconcile-unrealized-casts` | 消除类型转换中间节点 | 无关 |
| 13 | `gpu-module-to-binary` | 调用 `#ixdl.target` 序列化器，生成 ELF binary 嵌入 module | **IXDL 序列化** |

### 关键变换节点

**Pass 4 (`convert-fly-to-rocdl`) 之前 → 之后**:
```
之前: func.func @vectorAdd(%arg0: !fly.ptr<f32>, ...)
之后: func.func @vectorAdd(%arg0: !llvm.ptr, ...)
```

**Pass 6 (`convert-gpu-to-ixdl`) 之前 → 之后**:
```
之前: %tid = gpu.thread_id x
之后: %tid = ixdl.threadid.x : index
```

**Pass 13 (`gpu-module-to-binary`) 之后**:
```
gpu.binary @kernels [#gpu.object<#ixdl.target<chip = "ivcore11">, bin = "...ELF binary...">]
```

---

<a id="appendix"></a>

## 附录

### A. ixcc 仓库关键组件

| 组件 | 位置 | 状态 |
|---|---|---|
| IXDL dialect | `mlir/include/mlir/Dialect/LLVMIR/IXDLOps.td` | 已有 |
| `convert-gpu-to-ixdl` pass | `mlir/lib/Conversion/GPUToIXDL/` | 已有（已修复 GPUReturnOpLowering） |
| `#ixdl.target` + 序列化 | `mlir/lib/Target/LLVM/IXDL/Target.cpp` | 已有 |
| IXDL → LLVM IR 翻译 | `mlir/lib/Target/LLVMIR/Dialect/IXDL/` | 已有（已修复 CallingConv） |
| `ixdl-attach-target` pass | — | 不存在（需手动设置 target attribute） |
| Iluvatar LLVM backend | `llvm/lib/Target/Iluvatar/` | 已有 |
| `CallingConv::ILUVATAR_KERNEL` | `llvm/include/llvm/IR/CallingConv.h` (值 = 128) | 已有 |

### B. Iluvatar 硬件参数（来自 IREE `mod2.mlir`）

| 参数 | 值 |
|---|---|
| `arch` | `ivcore11` |
| `subgroup_size_choices` | `[64]`（wavefront size = 64，类 AMD CDNA） |
| `max_workgroup_sizes` | `[4096, 1024, 1024]` |
| `max_thread_count_per_workgroup` | `4096` |
| `max_workgroup_memory_bytes` | `131072` (128KB) |

### C. 当前文件改动清单

**FlyDSL 仓库** (`/home/wenhui-liang/flydsl`)：

| 文件 | 改动类型 | 说明 |
|---|---|---|
| `python/flydsl/compiler/jit_function.py` | 修改 | 架构自动分辨（`_is_iluvatar_arch`/`_gpu_target_attr`）+ pipeline 按平台分派 |
| `lib/Conversion/FlyToROCDL/FlyToROCDL.cpp` | 修改 | stub emitMfma（避免 ixcc API 不兼容） |
| `lib/Runtime/FlyCorexRuntimeWrappers.cpp` | **新建** | Iluvatar COREX runtime wrapper（CUDA Driver API） |
| `python/flydsl/utils/env.py` | 修改 | `ARCH` 环境变量说明增加 `ivcore11` |
| `python/mlir_flydsl/CMakeLists.txt` | 修改 | 添加 COREX/ROCm 条件编译的 runtime 构建逻辑 |
| `lib/Dialect/Fly/IR/FlyOps.cpp` | 修改 | `static_cast<int>` 修复 ixcc Diagnostics API 兼容 |
| `scripts/build_llvm.sh` | 修改 | 构建脚本更新 |
| `docs/iluvatar-port-guide.md` | **新建** | 完整移植文档 |

**ixcc 仓库** (`/home/wenhui-liang/ilu/ixcc`)：

| 文件 | 改动类型 | 说明 |
|---|---|---|
| `mlir/lib/Conversion/GPUToIXDL/LowerGpuOpsToIXDLOps.cpp` | bug fix | 添加 `GPUReturnOpLowering` |
| `mlir/lib/Target/LLVMIR/Dialect/IXDL/IXDLToLLVMIRTranslation.cpp` | **bug fix** | 添加 `setCallingConv(ILUVATAR_KERNEL)`，修复 kernel 无法被 runtime 找到 |

### D. 修改 Python 文件后的快速同步

由于 build 目录中的 `.py` 文件是从源码拷贝的，修改源码后需要同步：

```bash
# 方法 1: 手动拷贝
cp python/flydsl/compiler/jit_function.py \
   build-ixcc/python_packages/flydsl/compiler/jit_function.py
cp python/flydsl/expr/__init__.py \
   build-ixcc/python_packages/flydsl/expr/__init__.py

# 方法 2: 重新 build（会触发 CopyFlyPythonSources target）
cd build-ixcc && cmake --build . -j$(nproc)
```

### E. 常见问题

**Q: 为什么不把 `convert-fly-to-rocdl` 改名为 `convert-fly-to-llvm`？**
A: 可以考虑。当前保持原名是为了最小化改动、避免引入新的 pass 注册名称。后续如果要正式化，
建议将通用类型 lowering 部分抽取为独立 pass。

**Q: 为什么 `convert-gpu-to-ixdl` 不需要 `chipset` 参数？**
A: 当前 `convert-gpu-to-ixdl` 的实现不区分 chip 版本，所有 Iluvatar GPU 使用相同的 intrinsics
映射。chipset 信息在 `#ixdl.target<chip = "ivcore11">` 中传递给序列化器。

**Q: `format=binary` 和 `format=fatbin` 的区别？**
A: `fatbin` 是 NVIDIA 专用格式（多架构 fat binary）。`binary`/`bin` 是通用 ELF 格式，
IXDL 的 `linkToBinary` 生成此格式。

**Q: 需要 libdevice 吗？**
A: `vectorAdd` 不需要（不使用 sin/cos/sqrt 等数学函数）。如果后续 kernel 使用了数学函数，
需要找到 Iluvatar SDK 中 `libdevice.10.bc` 的位置并正确设置 `IXA_PATH`。
