//===- FlyIluvatarAotRuntime.cpp - Self-contained Iluvatar AOT runtime ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// CUDA-driver module lifecycle and launch support embedded into exported AOT
// objects. This intentionally uses only the CUDA driver API, libc and pthread
// so a final application or shared library has no FlyDSL runtime dependency.
//
// A CUmodule, and every CUfunction resolved from it, only works in the context
// that loaded it, so loaded modules are keyed by CUcontext rather than by
// device ordinal. Launches resolve the context the same way loads do, which
// makes "init, load, launch" on one thread land on a single context.
//
//===----------------------------------------------------------------------===//

#include "FlyIluvatarRuntimeError.h"

#include <cstdint>
#include <cstdlib>
#include <pthread.h>

enum : int32_t {
  FLYDSL_AOT_ERR_NOT_INITIALIZED = -1,
  FLYDSL_AOT_ERR_NOT_LOADED = -2,
};

namespace {

struct AotFunction {
  const char *name;
  CUfunction function;
  AotFunction *next;
};

struct AotContextModule {
  CUcontext context;
  CUmodule module;
  AotFunction *functions;
  AotContextModule *next;
};

struct AotModule {
  const void *binary;
  pthread_mutex_t mutex;
  AotContextModule *contexts;
};

// Guards every state slot: exclusive to create or destroy an AotModule,
// shared to use one. Embedded runtime symbols are localized into each export,
// so independently exported objects do not share this lock.
pthread_rwlock_t aotStateLock = PTHREAD_RWLOCK_INITIALIZER;

pthread_once_t cuInitOnce = PTHREAD_ONCE_INIT;
CUresult cuInitStatus = CUDA_SUCCESS;

void runCuInit() { cuInitStatus = cuInit(/*flags=*/0); }

CUresult ensureCudaInitialized() {
  pthread_once(&cuInitOnce, runCuInit);
  return cuInitStatus;
}

// Primary contexts are retained on first use and never released: the exported
// object cannot know whether the host application also drives them, and
// releasing one it still uses would tear down its allocations.
CUcontext primaryContextFor(CUdevice device, CUresult *status) {
  enum : int { kMaxDevices = 64 };
  static CUcontext cache[kMaxDevices];
  static pthread_mutex_t cacheMutex = PTHREAD_MUTEX_INITIALIZER;

  if (device < 0 || device >= kMaxDevices) {
    *status = CUDA_ERROR_INVALID_DEVICE;
    return nullptr;
  }
  pthread_mutex_lock(&cacheMutex);
  CUcontext context = cache[device];
  if (!context) {
    CUresult err = cuDevicePrimaryCtxRetain(&context, device);
    if (err) {
      *status = err;
      context = nullptr;
    } else {
      cache[device] = context;
    }
  }
  pthread_mutex_unlock(&cacheMutex);
  return context;
}

// Make a context current for this scope. A negative device keeps the caller's
// current context when there is one, so an application that already selected a
// device keeps its own context; otherwise the primary context of device 0 is
// used, which is what the generated *_call_auto helper relies on.
class ScopedContext {
  CUcontext active = nullptr;
  bool ownsPush = false;
  CUresult status = CUDA_SUCCESS;

public:
  explicit ScopedContext(int32_t device) {
    status = ensureCudaInitialized();
    if (status)
      return;

    CUcontext current = nullptr;
    status = cuCtxGetCurrent(&current);
    if (status)
      return;
    if (current) {
      CUdevice currentDevice = -1;
      if (device < 0 ||
          (cuCtxGetDevice(&currentDevice) == CUDA_SUCCESS && currentDevice == device)) {
        active = current;
        return;
      }
    }

    CUcontext primary = primaryContextFor(device < 0 ? 0 : device, &status);
    if (!primary)
      return;
    status = cuCtxPushCurrent(primary);
    if (status)
      return;
    active = primary;
    ownsPush = true;
  }

  ~ScopedContext() {
    if (ownsPush)
      CUDA_REPORT_IF_ERROR(cuCtxPopCurrent(nullptr));
  }

  ScopedContext(const ScopedContext &) = delete;
  ScopedContext &operator=(const ScopedContext &) = delete;

  CUcontext context() const { return active; }
  // CUDA_SUCCESS with a null context cannot happen; callers report this
  // whenever context() is null.
  CUresult error() const { return status ? status : CUDA_ERROR_UNKNOWN; }
};

AotContextModule *findContext(AotModule *aot, CUcontext context) {
  for (AotContextModule *entry = aot->contexts; entry; entry = entry->next)
    if (entry->context == context)
      return entry;
  return nullptr;
}

CUfunction getFunction(AotModule *aot, CUcontext context, const char *name) {
  pthread_mutex_lock(&aot->mutex);
  AotContextModule *entry = findContext(aot, context);
  if (!entry) {
    pthread_mutex_unlock(&aot->mutex);
    flydslRecordRuntimeError(FLYDSL_AOT_ERR_NOT_LOADED);
    return nullptr;
  }
  for (AotFunction *cached = entry->functions; cached; cached = cached->next) {
    // Kernel names are internal constants in the exported object, so pointer
    // identity is stable and avoids copying or comparing their contents.
    if (cached->name == name) {
      CUfunction function = cached->function;
      pthread_mutex_unlock(&aot->mutex);
      return function;
    }
  }

  CUfunction function = nullptr;
  if (CUresult err = cuModuleGetFunction(&function, entry->module, name)) {
    pthread_mutex_unlock(&aot->mutex);
    flydslRecordRuntimeError(static_cast<int32_t>(err));
    return nullptr;
  }
  if (auto *cached = static_cast<AotFunction *>(malloc(sizeof(AotFunction)))) {
    cached->name = name;
    cached->function = function;
    cached->next = entry->functions;
    entry->functions = cached;
  }
  pthread_mutex_unlock(&aot->mutex);
  return function;
}

void launchKernel(CUfunction function, intptr_t gridX, intptr_t gridY, intptr_t gridZ,
                  intptr_t blockX, intptr_t blockY, intptr_t blockZ, int32_t smem, CUstream stream,
                  void **params, void **extra) {
  if (!function)
    return;
  CUDA_REPORT_IF_ERROR(cuLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ, smem,
                                      stream, params, extra));
}

void freeFunctions(AotFunction *function) {
  while (function) {
    AotFunction *next = function->next;
    free(function);
    function = next;
  }
}

} // namespace

extern "C" FLYDSL_RUNTIME_API int32_t flydslAotModuleInit(AotModule **slot, const void *binary) {
  pthread_rwlock_wrlock(&aotStateLock);
  if (!*slot) {
    auto *aot = static_cast<AotModule *>(calloc(1, sizeof(AotModule)));
    if (!aot) {
      pthread_rwlock_unlock(&aotStateLock);
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    if (pthread_mutex_init(&aot->mutex, nullptr)) {
      free(aot);
      pthread_rwlock_unlock(&aotStateLock);
      return CUDA_ERROR_UNKNOWN;
    }
    aot->binary = binary;
    *slot = aot;
  }
  pthread_rwlock_unlock(&aotStateLock);
  return 0;
}

// Load the module on `device`, or on the current context's device when
// `device` < 0.
extern "C" FLYDSL_RUNTIME_API int32_t flydslAotModuleLoad(AotModule **slot, int32_t device) {
  pthread_rwlock_rdlock(&aotStateLock);
  AotModule *aot = *slot;
  if (!aot) {
    pthread_rwlock_unlock(&aotStateLock);
    return FLYDSL_AOT_ERR_NOT_INITIALIZED;
  }

  ScopedContext scopedContext(device);
  if (!scopedContext.context()) {
    pthread_rwlock_unlock(&aotStateLock);
    return scopedContext.error();
  }

  pthread_mutex_lock(&aot->mutex);
  if (findContext(aot, scopedContext.context())) {
    pthread_mutex_unlock(&aot->mutex);
    pthread_rwlock_unlock(&aotStateLock);
    return 0;
  }
  auto *entry = static_cast<AotContextModule *>(calloc(1, sizeof(AotContextModule)));
  if (!entry) {
    pthread_mutex_unlock(&aot->mutex);
    pthread_rwlock_unlock(&aotStateLock);
    return CUDA_ERROR_OUT_OF_MEMORY;
  }

  // The embedded binary is a constant in the exported object, so the driver
  // can parse it in place; its length is not part of the lifecycle ABI.
  CUmodule module = nullptr;
  CUresult err = cuModuleLoadData(&module, aot->binary);
  if (err) {
    free(entry);
    pthread_mutex_unlock(&aot->mutex);
    pthread_rwlock_unlock(&aotStateLock);
    return err;
  }

  entry->context = scopedContext.context();
  entry->module = module;
  entry->next = aot->contexts;
  aot->contexts = entry;
  pthread_mutex_unlock(&aot->mutex);
  pthread_rwlock_unlock(&aotStateLock);
  return 0;
}

// Unload the module from every context it was loaded on and release the state.
extern "C" FLYDSL_RUNTIME_API int32_t flydslAotModuleUnload(AotModule **slot) {
  pthread_rwlock_wrlock(&aotStateLock);
  AotModule *aot = *slot;
  *slot = nullptr;
  if (!aot) {
    pthread_rwlock_unlock(&aotStateLock);
    return 0;
  }

  int32_t status = 0;
  CUcontext current = nullptr;
  (void)cuCtxGetCurrent(&current);
  AotContextModule *entry = aot->contexts;
  while (entry) {
    bool pushed = false;
    CUresult err = CUDA_SUCCESS;
    if (entry->context != current) {
      err = cuCtxPushCurrent(entry->context);
      pushed = err == CUDA_SUCCESS;
    }
    if (!err) {
      // The state lock prevents new host submissions, but previously submitted
      // kernels may still be executing asynchronously.
      err = cuCtxSynchronize();
      CUresult unloadErr = cuModuleUnload(entry->module);
      if (!err)
        err = unloadErr;
    }
    if (pushed)
      (void)cuCtxPopCurrent(nullptr);
    if (err && !status)
      status = static_cast<int32_t>(err);
    freeFunctions(entry->functions);
    AotContextModule *next = entry->next;
    free(entry);
    entry = next;
  }
  pthread_mutex_destroy(&aot->mutex);
  free(aot);
  pthread_rwlock_unlock(&aotStateLock);
  return status;
}

extern "C" FLYDSL_RUNTIME_API void
flydslAotModuleLaunchKernel(AotModule **slot, const char *name, intptr_t gridX, intptr_t gridY,
                            intptr_t gridZ, intptr_t blockX, intptr_t blockY, intptr_t blockZ,
                            int32_t smem, CUstream stream, void **params, void **extra,
                            size_t /*paramsCount*/) {
  pthread_rwlock_rdlock(&aotStateLock);
  AotModule *aot = *slot;
  if (!aot) {
    flydslRecordRuntimeError(FLYDSL_AOT_ERR_NOT_INITIALIZED);
    pthread_rwlock_unlock(&aotStateLock);
    return;
  }
  ScopedContext scopedContext(/*device=*/-1);
  if (!scopedContext.context()) {
    flydslRecordRuntimeError(static_cast<int32_t>(scopedContext.error()));
    pthread_rwlock_unlock(&aotStateLock);
    return;
  }
  CUfunction function = getFunction(aot, scopedContext.context(), name);
  launchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ, smem, stream, params, extra);
  pthread_rwlock_unlock(&aotStateLock);
}

extern "C" FLYDSL_RUNTIME_API void
flydslAotModuleLaunchClusterKernel(AotModule **slot, const char *name, intptr_t clusterX,
                                   intptr_t clusterY, intptr_t clusterZ, intptr_t gridX,
                                   intptr_t gridY, intptr_t gridZ, intptr_t blockX, intptr_t blockY,
                                   intptr_t blockZ, int32_t smem, CUstream stream, void **params,
                                   void **extra, size_t /*paramsCount*/) {
  // Iluvatar has no thread-block clusters. A degenerate cluster is an ordinary
  // launch; anything larger would silently drop the requested geometry.
  if (clusterX > 1 || clusterY > 1 || clusterZ > 1) {
    fprintf(stderr,
            "[FlyDSL AOT] cluster=(%ld,%ld,%ld) requested but this target has no cluster "
            "launch.\n",
            static_cast<long>(clusterX), static_cast<long>(clusterY), static_cast<long>(clusterZ));
    flydslRecordRuntimeError(static_cast<int32_t>(CUDA_ERROR_NOT_SUPPORTED));
    return;
  }
  flydslAotModuleLaunchKernel(slot, name, gridX, gridY, gridZ, blockX, blockY, blockZ, smem, stream,
                              params, extra, /*paramsCount=*/0);
}
