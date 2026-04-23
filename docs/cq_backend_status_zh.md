# Iluvatar CQ 后端（`fly_cq`）实现状态

本文档汇总 **FlyCQ** 方言与 **FlyToCQ** 转换链的当前落地情况、主要代码改动位置，以及相对路线图（M1→M6）的推进程度。便于评审、交接与后续迭代时对照。

**约定**：工程简称 **CQ**；MLIR 方言名 **`fly_cq`**；C++ 命名空间 **`mlir::fly_cq`**；目录 / CMake / Python 模块 **`FlyCQ`** / **`MLIRFlyCQDialect`** / **`_mlirDialectsFlyCQ`**。

---

## 1. 路线图位置（对照规划 M1→M6）

| 阶段 | 名称 | 当前状态 |
|------|------|----------|
| **M1** | IR 垂直切片 | **基本完成**：`FlyCQ` 注册、**无状态** MMA + Copy 占位类型及 `emitAtomCall` / `emitAtomCallSSA`、`--convert-fly-to-cq`、`fly-opt` 接线、**FileCheck** 回归（MMA + Copy）。 |
| **M2** | Dialect + 转换完备 | **未开始**：无 CQ **Stateful** Copy/MMA、无 **类型全集**、**FlyToCQ** 与 **FlyToROCDL** 的 pattern **未做逐项 parity**、无对称大规模 MLIR 矩阵。 |
| **M3–M6** | 产物出口 / 厂商链 / 真机 / 生产化 | **未开始**：依赖厂商工具链、硬件与产品决策。 |

**结论**：当前处于 **M1 可合并、可测的垂直切片** 阶段；在「与 FlyROCDL 工程完备度对齐」意义上，整体约在 **M1 末 / M2 初** 边界（M2 的 Stateful 与全量对齐尚未做）。

---

## 2. 已实现功能一览

### 2.1 方言（TableGen + C++）

- **入口与基类**（`include/flydsl/Dialect/FlyCQ/IR/Dialect.td`）  
  - `FlyCQ_Dialect`：`dependentDialects` 含 **LLVM**、**vector**（占位 lowering 会用到）。  
  - **`FlyCQCL_MmaOp`**、**`FlyCQCL_CopyOp`**：分别挂 **`Fly_MmaOpTypeInterface`**、**`Fly_CopyOpTypeInterface`**（及 **`Fly_MayStaticTypeInterface`**）。

- **Atom 聚合**（`include/flydsl/Dialect/FlyCQ/IR/Atom.td`）  
  - 占位枚举 **`FlyCQ_PlaceholderKind`**（供 Python 等生成绑定）。  
  - `include`：**`CopyAtom.td`**、**`MmaAtom.td`**。

- **MMA 占位**（`include/flydsl/Dialect/FlyCQ/IR/MmaAtom.td`）  
  - **`!fly_cq.matmul_f32<MNK, (ty,ty)->ty>`**，MNK 使用与 ROCDL 一致的 **`custom<MNKDimensionList>`**（避免无空格的 `16x16x4` 解析问题）。  
  - 校验：当前占位为 **16×16×4、f32**。

- **Copy 占位（垂直切片）**（`include/flydsl/Dialect/FlyCQ/IR/CopyAtom.td`）  
  - **`!fly_cq.scalar_mem<bitSize>`**，无状态；校验暂为 **`bitSize == 32`**。  
  - 语义：**寄存器侧 `fly.memref` → LLVM `load`/`store`**，非 buffer/LDS 路径。

- **C++ 实现**  
  - `lib/Dialect/FlyCQ/Dialect.cpp`：注册类型、加载 **LLVM / Vector** 方言（满足生成的 `Dialect.cpp.inc` 中对 `loadDialect` 的引用）。  
  - `lib/Dialect/FlyCQ/CQ/MmaAtom.cpp`：`emitAtomCall` / `emitAtomCallSSA` → **`vector.broadcast` + `arith.mulf` + `arith.addf`** 等。  
  - `lib/Dialect/FlyCQ/CQ/CopyAtom.cpp`：`emit*` → **`llvm.load` / `llvm.store`**；带 predicate 路径使用 **`scf.if`**（与 ROCDL 形态对齐，便于后续扩测）。  
  - `include/flydsl/Dialect/FlyCQ/IR/Dialect.h`：包含 **`AtomEnums.h.inc`**，保证枚举生成与 `.cpp.inc` 一致。

### 2.2 转换与工具

- **`lib/Conversion/FlyToCQ/FlyToCQ.cpp`**  
  - 与 **FlyToROCDL** 同构：`ConversionTarget`、`FlyTypeConverter`、`copy_atom_call` / `mma_atom_call` 等 pattern；**`BufferFatPtr`** 仍复用 **`FlyROCDL/Utils/BufferFatPtr.h`**（CQ 目录下无独立 Utils）。  
  - Pass：**`--convert-fly-to-cq`**（`include/flydsl/Conversion/FlyToCQ/Passes.td`）。  
  - **注意**：合法集中仍包含 **ROCDL**（与 buffer 路径等一致），并非「纯 LLVM-only」窄合法集。

- **`include/flydsl/Conversion/Passes.h`**  
  - 在包含 **`FlyToCQ/Passes.h.inc`** 之前**再次** `#define GEN_PASS_REGISTRATION`：因 **`FlyToROCDL/Passes.h.inc`** 末尾会 `#undef GEN_PASS_REGISTRATION`，否则 **CQ 的 `registerFlyToCQConversionPass` 不会被生成**。

- **`tools/fly-opt/fly-opt.cpp`**  
  - 注册 **`FlyCQDialect`** 与 **`registerFlyToCQConversionPass()`**。

### 2.3 C API 与 Python

- **C API**：`include/flydsl-c/FlyCQDialect.h`，`lib/CAPI/Dialect/FlyCQ/FlyCQDialect.cpp`（方言注册 + **`mlirRegisterFlyToCQConversionPass`**）。  
- **Python MLIR 绑定**：`python/mlir_flydsl/dialects/FlyCQ.td`、`fly_cq.py`、`FlyRegisterEverything.cpp` 中注册 dialect / pass；`lib/Bindings/Python/FlyCQExtension.cpp` 暴露 **`MmaOpCQ_MatmulF32Type::get`**、**`CopyOpCQ_ScalarMemType::get`**。  
- **DSL 薄封装**：`python/flydsl/expr/cq.py` — **`MatmulF32()`**、**`ScalarMem32()`**；`python/flydsl/expr/__init__.py` 导出子模块 **`cq`**。

### 2.4 构建与测试脚本

- **`python/mlir_flydsl/CMakeLists.txt`**：**`find_package(hip QUIET …)`**；仅在找到 HIP 时构建 **`FlyJitRuntime`**，否则跳过并打印 STATUS（便于无 ROCm 环境完成 **fly-opt** 与 MLIR 测试配置）。  
- **`scripts/run_tests.sh`**：增加 **`%fly-cq-opt`** → 与 **`%fly-opt`** 相同，指向 **`${FLY_BUILD_DIR}/bin/fly-opt`**。

### 2.5 MLIR 回归用例

| 文件 | 内容 |
|------|------|
| `tests/mlir/Conversion/mma_atom_cq.mlir` | 与 `mma_atom.mlir` 同管道，`--convert-fly-to-cq`，**`!fly_cq.matmul_f32<16x16x4, (f32,f32)->f32>`**。 |
| `tests/mlir/Conversion/copy_atom_cq.mlir` | **寄存器 memref** 上 **`!fly_cq.scalar_mem<32>`** 的 **`copy_atom_call`**，检查 **无状态 atom** 与 **load/store**。 |

---

## 3. 刻意未覆盖 / 已知边界

- **无 Stateful CQ 类型**（无 `Fly_StatefulOpTypeInterface` 的 CQ Copy/MMA），因此 **不**覆盖 `copy_atom_stateful.mlir` 等 ROCDL 有状态路径的 CQ 镜像。  
- **`scalar_mem`** 不处理 **`buffer_desc` / shared / LDS** 等；复杂 kernel 仍会在 CQ 路径上早于本切片即失败，除非继续补 **FlyToCQ parity** 与更多 atom。  
- **真机 / 厂商 LLVM**：未接入；占位 lowering 面向 **当前 FlyDSL 所链接的上游 MLIR/LLVM**。  
- **双 LLVM 版本**：若未来 CQ 工具链与主线 LLVM 不兼容，需按规划在 **M3** 选择 **外接二段式** 或 **双构建 `fly-opt-cq`** 等方案（见原规划文档「不同 LLVM 版本」节）。

---

## 4. 如何验证（简令）

```bash
# 构建 fly-opt（示例构建目录）
cmake --build "${FLY_BUILD_DIR:-build-fly}" --target fly-opt -j"$(nproc)"

# CQ 相关 MLIR + FileCheck（与 run_tests.sh 中逻辑一致）
FLY_OPT="${FLY_BUILD_DIR:-build-fly}/bin/fly-opt"
MLIR_DIR="$(grep '^MLIR_DIR:' "${FLY_BUILD_DIR:-build-fly}/CMakeCache.txt" | cut -d= -f2)"
FILECHECK="${MLIR_DIR%/lib/cmake/mlir}/bin/FileCheck"

for t in mma_atom_cq copy_atom_cq; do
  "${FLY_OPT}" "tests/mlir/Conversion/${t}.mlir" \
    --fly-rewrite-func-signature --fly-canonicalize --fly-layout-lowering \
    --convert-fly-to-cq | "${FILECHECK}" "tests/mlir/Conversion/${t}.mlir" && echo "PASS ${t}"
done
```

全量脚本：`bash scripts/run_tests.sh`（含 pytest 与其它 MLIR 用例）。

---

## 5. 建议的下一步（与输入依赖）

1. **补 Stateful 垂直切片**（若要对齐 `copy_atom_stateful.mlir` / 部分 ROCDL kernel）：需 **`getConvertedType` / `getDefaultState` / `setAtomState`** 与 TableGen **`FlyCQCL_StatefulCopyOp`** 等。  
2. **FlyToCQ vs FlyToROCDL diff 清单**：按真实 kernel 会触达的 op 排序补 pattern。  
3. **厂商输入**：ISA、上游 MLIR 方言名、`emitAtomCall` 目标 op 形态、工具链与 **M3 交接格式**。

---

## 6. 文档维护

- **本文路径**：`docs/cq_backend_status_zh.md`。  
- 建议在合入较大 CQ 变更时 **更新本节日期与表格状态**；路线图全文仍以团队内 **CQ / FlyDSL 后端规划** 为准。

**最后更新**：以提交时的仓库状态为准（自动生成说明可写进 commit message 指向本文件）。
