//===- FlyIluvatarRuntimeWrappers.cpp - Iluvatar runtime wrappers ---------===//
//
// Derived from LLVM Project: mlir/lib/ExecutionEngine/CudaRuntimeWrappers.cpp
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Thin CUDA-compatible Iluvatar runtime wrappers for MLIR ExecutionEngine JIT.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "cuda.h"

#define CUDA_REPORT_IF_ERROR(expr)                                                                 \
  [](CUresult result) {                                                                            \
    if (!result)                                                                                   \
      return;                                                                                      \
    const char *name = nullptr;                                                                    \
    cuGetErrorName(result, &name);                                                                 \
    if (!name)                                                                                     \
      name = "<unknown>";                                                                          \
    fprintf(stderr, "'%s' failed with '%s'\n", #expr, name);                                       \
  }(expr)

thread_local static int32_t defaultDevice = 0;

static CUdevice getDefaultCuDevice() {
  CUdevice device;
  CUDA_REPORT_IF_ERROR(cuDeviceGet(&device, /*ordinal=*/defaultDevice));
  return device;
}

static void ensureCudaInitialized() {
  static const bool initialized = [] {
    CUDA_REPORT_IF_ERROR(cuInit(/*flags=*/0));
    return true;
  }();
  (void)initialized;
}

// Make a CUDA context current for this scope. Does not move a CUmodule or
// CUfunction onto that context -- those stay bound to the context that
// loaded them. The fallback primary is retained once, from the first
// defaultDevice seen in this process.
class ScopedContext {
  CUcontext currentContext = nullptr;
  bool ownsPush = false;

public:
  ScopedContext() {
    ensureCudaInitialized();
    CUresult status = cuCtxGetCurrent(&currentContext);
    if (status != CUDA_SUCCESS) {
      CUDA_REPORT_IF_ERROR(status);
      currentContext = nullptr;
    }
    if (currentContext != nullptr)
      return;

    static CUcontext primary = [] {
      CUcontext ctx = nullptr;
      CUDA_REPORT_IF_ERROR(cuDevicePrimaryCtxRetain(&ctx, getDefaultCuDevice()));
      return ctx;
    }();
    CUDA_REPORT_IF_ERROR(cuCtxPushCurrent(primary));
    currentContext = primary;
    ownsPush = true;
  }

  ~ScopedContext() {
    if (ownsPush)
      CUDA_REPORT_IF_ERROR(cuCtxPopCurrent(nullptr));
  }

  // The context every call in this scope runs against, so callers that also
  // need it do not pay for a second cuCtxGetCurrent.
  CUcontext context() const { return currentContext; }
};

// Cubin/PTX bytes plus the module loaded from them into each CUDA context that
// used them. A CUmodule, and every CUfunction resolved from it, only works in
// the context that created it, while the JIT host stub holds a single handle
// for all callers -- so a context that has not seen this binary loads its own
// copy instead of reusing a handle that does not belong to it.
struct LoadedGpuBinary {
  std::vector<char> blob;
  int jitOptLevel = -1;
  std::mutex mutex;
  std::unordered_map<CUcontext, CUmodule> modules;

  CUmodule loadInCurrentContext() {
    CUmodule module = nullptr;
    if (jitOptLevel >= 0) {
      char jitErrorBuffer[4096] = {0};
      CUjit_option jitOptions[] = {CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
                                   CU_JIT_OPTIMIZATION_LEVEL};
      void *jitOptionsVals[] = {jitErrorBuffer, reinterpret_cast<void *>(sizeof(jitErrorBuffer)),
                                reinterpret_cast<void *>(jitOptLevel)};
      CUresult result = cuModuleLoadDataEx(&module, blob.data(), 3, jitOptions, jitOptionsVals);
      if (result) {
        fprintf(stderr, "JIT compilation failed with: '%s'\n", jitErrorBuffer);
        CUDA_REPORT_IF_ERROR(result);
      }
      return module;
    }
    CUDA_REPORT_IF_ERROR(cuModuleLoadData(&module, blob.data()));
    return module;
  }

  // `context` must be the current one, since loading targets whatever context
  // is current.
  CUmodule moduleFor(CUcontext context) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = modules.find(context);
    if (it != modules.end())
      return it->second;
    CUmodule module = loadInCurrentContext();
    if (module)
      modules.emplace(context, module);
    return module;
  }
};

// Bumped whenever a binary is unloaded, so memo entries that may name a
// CUfunction from a freed module stop matching. Allocators reuse addresses,
// so the memo cannot rely on the binary pointer alone staying meaningful.
static std::atomic<uint64_t> unloadGeneration{1};

// The JIT host stub re-resolves the kernel on every launch, and walking the
// shared table costs a chain of dependent loads that miss cache when a launch
// touches it just once. A direct-mapped thread-local memo answers the repeat
// case out of a single line, without locking, and is trusted only while it
// still agrees with the current context and generation.
struct FunctionMemoEntry {
  CUmodule binary = nullptr;
  const char *name = nullptr;
  CUcontext context = nullptr;
  CUfunction function = nullptr;
  uint64_t generation = 0;
};

static FunctionMemoEntry &functionMemoSlot(CUmodule binary, const char *name) {
  constexpr size_t numSlots = 64;
  static thread_local FunctionMemoEntry memo[numSlots];
  uintptr_t mixed =
      (reinterpret_cast<uintptr_t>(binary) >> 4) ^ (reinterpret_cast<uintptr_t>(name) >> 3);
  return memo[mixed & (numSlots - 1)];
}

static CUmodule makeLoadedGpuBinary(CUcontext context, const char *data, size_t size,
                                    int jitOptLevel) {
  if (!context || !data || size == 0)
    return nullptr;
  auto *binary = new LoadedGpuBinary();
  binary->blob.assign(data, data + size);
  binary->jitOptLevel = jitOptLevel;
  if (!binary->moduleFor(context)) {
    delete binary;
    return nullptr;
  }
  return reinterpret_cast<CUmodule>(binary);
}

extern "C" CUmodule mgpuModuleLoad(void *data, size_t gpuBlobSize) {
  ScopedContext scopedContext;
  return makeLoadedGpuBinary(scopedContext.context(), static_cast<const char *>(data), gpuBlobSize,
                             /*jitOptLevel=*/-1);
}

extern "C" CUmodule mgpuModuleLoadJIT(void *data, int optLevel) {
  ScopedContext scopedContext;
  if (!data)
    return nullptr;
  const char *text = static_cast<const char *>(data);
  return makeLoadedGpuBinary(scopedContext.context(), text, std::strlen(text) + 1, optLevel);
}

extern "C" void mgpuModuleUnload(CUmodule module) {
  if (!module)
    return;
  auto *binary = reinterpret_cast<LoadedGpuBinary *>(module);
  std::unordered_map<CUcontext, CUmodule> modules;
  {
    std::lock_guard<std::mutex> lock(binary->mutex);
    modules.swap(binary->modules);
  }
  unloadGeneration.fetch_add(1, std::memory_order_relaxed);
  CUcontext current = nullptr;
  CUDA_REPORT_IF_ERROR(cuCtxGetCurrent(&current));
  for (const auto &entry : modules) {
    bool pushed = false;
    if (current != entry.first) {
      CUresult status = cuCtxPushCurrent(entry.first);
      if (status != CUDA_SUCCESS) {
        CUDA_REPORT_IF_ERROR(status);
        continue;
      }
      pushed = true;
    }
    CUDA_REPORT_IF_ERROR(cuModuleUnload(entry.second));
    if (pushed)
      CUDA_REPORT_IF_ERROR(cuCtxPopCurrent(nullptr));
  }
  delete binary;
}

extern "C" CUfunction mgpuModuleGetFunction(CUmodule module, const char *name) {
  if (!module || !name)
    return nullptr;
  ScopedContext scopedContext;
  CUcontext context = scopedContext.context();
  uint64_t generation = unloadGeneration.load(std::memory_order_relaxed);
  FunctionMemoEntry &slot = functionMemoSlot(module, name);
  if (slot.binary == module && slot.name == name && slot.context == context &&
      slot.generation == generation)
    return slot.function;

  auto *binary = reinterpret_cast<LoadedGpuBinary *>(module);
  CUmodule cuModule = binary->moduleFor(context);
  if (!cuModule)
    return nullptr;
  CUfunction function = nullptr;
  CUDA_REPORT_IF_ERROR(cuModuleGetFunction(&function, cuModule, name));
  if (function)
    slot = {module, name, context, function, generation};
  return function;
}

extern "C" void mgpuLaunchKernel(CUfunction function, intptr_t gridX, intptr_t gridY,
                                 intptr_t gridZ, intptr_t blockX, intptr_t blockY, intptr_t blockZ,
                                 int32_t smem, CUstream stream, void **params, void **extra,
                                 size_t /*paramsCount*/) {
  ScopedContext scopedContext;
  CUDA_REPORT_IF_ERROR(cuLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ, smem,
                                      stream, params, extra));
}

extern "C" CUstream mgpuStreamCreate() {
  ScopedContext scopedContext;
  CUstream stream = nullptr;
  CUDA_REPORT_IF_ERROR(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
  return stream;
}

extern "C" void mgpuStreamDestroy(CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuStreamDestroy(stream));
}

extern "C" void mgpuStreamSynchronize(CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuStreamSynchronize(stream));
}

extern "C" void mgpuStreamWaitEvent(CUstream stream, CUevent event) {
  CUDA_REPORT_IF_ERROR(cuStreamWaitEvent(stream, event, /*flags=*/0));
}

extern "C" CUevent mgpuEventCreate() {
  ScopedContext scopedContext;
  CUevent event = nullptr;
  CUDA_REPORT_IF_ERROR(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
  return event;
}

extern "C" void mgpuEventDestroy(CUevent event) { CUDA_REPORT_IF_ERROR(cuEventDestroy(event)); }

extern "C" void mgpuEventSynchronize(CUevent event) {
  CUDA_REPORT_IF_ERROR(cuEventSynchronize(event));
}

extern "C" void mgpuEventRecord(CUevent event, CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuEventRecord(event, stream));
}

extern "C" void *mgpuMemAlloc(uint64_t sizeBytes, CUstream /*stream*/, bool isHostShared) {
  ScopedContext scopedContext;
  CUdeviceptr ptr = 0;
  if (sizeBytes == 0)
    return reinterpret_cast<void *>(ptr);
  if (isHostShared) {
    CUDA_REPORT_IF_ERROR(cuMemAllocManaged(&ptr, sizeBytes, CU_MEM_ATTACH_GLOBAL));
    return reinterpret_cast<void *>(ptr);
  }
  CUDA_REPORT_IF_ERROR(cuMemAlloc(&ptr, sizeBytes));
  return reinterpret_cast<void *>(ptr);
}

extern "C" void mgpuMemFree(void *ptr, CUstream /*stream*/) {
  CUDA_REPORT_IF_ERROR(cuMemFree(reinterpret_cast<CUdeviceptr>(ptr)));
}

extern "C" void mgpuMemcpy(void *dst, void *src, size_t sizeBytes, CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuMemcpyAsync(reinterpret_cast<CUdeviceptr>(dst),
                                     reinterpret_cast<CUdeviceptr>(src), sizeBytes, stream));
}

extern "C" void mgpuMemset32(void *dst, int value, size_t count, CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(dst),
                                        static_cast<unsigned int>(value), count, stream));
}

extern "C" void mgpuMemset16(void *dst, int shortValue, size_t count, CUstream stream) {
  CUDA_REPORT_IF_ERROR(cuMemsetD16Async(reinterpret_cast<CUdeviceptr>(dst),
                                        static_cast<unsigned short>(shortValue), count, stream));
}

extern "C" void mgpuMemHostRegister(void *ptr, uint64_t sizeBytes) {
  ScopedContext scopedContext;
  CUDA_REPORT_IF_ERROR(cuMemHostRegister(ptr, sizeBytes, /*flags=*/0));
}

extern "C" void mgpuMemHostRegisterMemRef(int64_t rank, StridedMemRefType<char, 1> *descriptor,
                                          int64_t elementSizeBytes) {
  int64_t *sizes = descriptor->sizes;
  int64_t *strides = sizes + rank;

  std::vector<int64_t> denseStrides(static_cast<size_t>(rank));
  if (rank > 0) {
    denseStrides[static_cast<size_t>(rank - 1)] = sizes[rank - 1];
    for (int64_t i = rank - 2; i >= 0; --i)
      denseStrides[static_cast<size_t>(i)] = sizes[i] * denseStrides[static_cast<size_t>(i + 1)];
  }
  auto sizeBytes = (rank > 0 ? denseStrides[0] : 1) * elementSizeBytes;

  for (int64_t i = 0; i < rank - 1; ++i)
    denseStrides[static_cast<size_t>(i)] = denseStrides[static_cast<size_t>(i + 1)];
  if (rank > 0)
    denseStrides[static_cast<size_t>(rank - 1)] = 1;

  for (int64_t i = 0; i < rank; ++i)
    assert(strides[i] == denseStrides[static_cast<size_t>(i)]);

  auto ptr = descriptor->data + descriptor->offset * elementSizeBytes;
  mgpuMemHostRegister(ptr, sizeBytes);
}

extern "C" void mgpuMemHostUnregister(void *ptr) {
  ScopedContext scopedContext;
  CUDA_REPORT_IF_ERROR(cuMemHostUnregister(ptr));
}

extern "C" void mgpuMemHostUnregisterMemRef(int64_t /*rank*/,
                                            StridedMemRefType<char, 1> *descriptor,
                                            int64_t elementSizeBytes) {
  auto ptr = descriptor->data + descriptor->offset * elementSizeBytes;
  mgpuMemHostUnregister(ptr);
}

template <typename T> static void mgpuMemGetDevicePointer(T *hostPtr, T **devicePtr) {
  ScopedContext scopedContext;
  CUdeviceptr ptr = 0;
  CUDA_REPORT_IF_ERROR(cuMemHostGetDevicePointer(&ptr, hostPtr, /*flags=*/0));
  *devicePtr = reinterpret_cast<T *>(ptr);
}

extern "C" StridedMemRefType<float, 1> mgpuMemGetDeviceMemRef1dFloat(float * /*allocated*/,
                                                                     float *aligned, int64_t offset,
                                                                     int64_t size, int64_t stride) {
  float *devicePtr = nullptr;
  mgpuMemGetDevicePointer(aligned, &devicePtr);
  return {devicePtr, devicePtr, offset, {size}, {stride}};
}

extern "C" StridedMemRefType<int32_t, 1> mgpuMemGetDeviceMemRef1dInt32(int32_t * /*allocated*/,
                                                                       int32_t *aligned,
                                                                       int64_t offset, int64_t size,
                                                                       int64_t stride) {
  int32_t *devicePtr = nullptr;
  mgpuMemGetDevicePointer(aligned, &devicePtr);
  return {devicePtr, devicePtr, offset, {size}, {stride}};
}

extern "C" void mgpuSetDefaultDevice(int32_t device) {
  // Thread-local only. The no-current-context fallback in ScopedContext is a
  // process-static retain of whichever defaultDevice was first observed.
  defaultDevice = device;
}
