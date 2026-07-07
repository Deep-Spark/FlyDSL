#!/usr/bin/env bash
set -euo pipefail

if ! command -v docker >/dev/null 2>&1; then
  echo "::error::docker is required on the self-hosted runner"
  exit 1
fi

: "${CI_DEVICE_IMAGE:?CI_DEVICE_IMAGE is required}"
: "${COREX_ROOT:?COREX_ROOT is required}"

if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ "$#" -eq 0 ]]; then
  echo "::error::missing benchmark command to run inside container"
  exit 1
fi

WORKSPACE="${GITHUB_WORKSPACE:-$(pwd)}"
CI_DEVICE_PRIVILEGED="${CI_DEVICE_PRIVILEGED:-1}"
CI_DEVICE_RUN_AS_HOST_USER="${CI_DEVICE_RUN_AS_HOST_USER:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PERF_LOCK_WAIT_SECONDS="${PERF_LOCK_WAIT_SECONDS:-900}"
PERF_GPU_IDLE_RETRIES="${PERF_GPU_IDLE_RETRIES:-9}"
PERF_GPU_IDLE_SLEEP_SECONDS="${PERF_GPU_IDLE_SLEEP_SECONDS:-10}"
PERF_CACHE_ROOT="${PERF_CACHE_ROOT:-/workspace/.flydsl}"
PERF_CACHE_DIR="${PERF_CACHE_DIR:-${PERF_CACHE_ROOT}/cache}"

if [[ ! -d "${WORKSPACE}" ]]; then
  echo "::error::workspace does not exist: ${WORKSPACE}"
  exit 1
fi

if [[ ! -d "${COREX_ROOT}" ]]; then
  echo "::error::COREX_ROOT does not exist: ${COREX_ROOT}"
  exit 1
fi

selected_gpu="${CUDA_VISIBLE_DEVICES%%,*}"
selected_gpu="${selected_gpu//[[:space:]]/}"
if [[ -z "${selected_gpu}" ]]; then
  selected_gpu="0"
fi

lock_file="/tmp/flydsl-perf-gpu-${selected_gpu}.lock"
if command -v flock >/dev/null 2>&1; then
  exec 9>"${lock_file}"
  if ! flock -w "${PERF_LOCK_WAIT_SECONDS}" 9; then
    echo "::warning::timed out waiting for perf lock ${lock_file}; continue without lock"
  fi
else
  echo "::warning::flock is unavailable on runner; skip host lock"
fi

is_gpu_busy_ixsmi() {
  if ! command -v ixsmi >/dev/null 2>&1; then
    return 1
  fi
  local out=""
  out="$(ixsmi 2>/dev/null || true)"
  if [[ -z "${out}" ]]; then
    return 1
  fi
  while IFS= read -r line; do
    local ls="${line#"${line%%[![:space:]]*}"}"
    if [[ "${ls}" == \|* ]] && [[ "${ls}" == *"MiB"* ]]; then
      if [[ "${ls}" == *"python"* ]] || ([[ "${ls}" == *"MiB /"* ]] && [[ "${ls}" != *"0MiB /"* ]]); then
        return 0
      fi
    fi
  done <<<"${out}"
  return 1
}

for ((attempt=1; attempt<=PERF_GPU_IDLE_RETRIES; attempt++)); do
  if ! is_gpu_busy_ixsmi; then
    break
  fi
  if (( attempt == PERF_GPU_IDLE_RETRIES )); then
    echo "::warning::GPU appears busy after ${PERF_GPU_IDLE_RETRIES} checks; continue anyway"
    break
  fi
  echo "::warning::GPU appears busy (attempt ${attempt}/${PERF_GPU_IDLE_RETRIES}); sleep ${PERF_GPU_IDLE_SLEEP_SECONDS}s"
  sleep "${PERF_GPU_IDLE_SLEEP_SECONDS}"
done

ixcc_root=""
if [[ -n "${IXCC_MLIR_CMAKE:-}" && ! -f "${IXCC_MLIR_CMAKE}/MLIRConfig.cmake" ]]; then
  echo "::error::IXCC_MLIR_CMAKE missing MLIRConfig.cmake: ${IXCC_MLIR_CMAKE}"
  exit 1
fi
if [[ -n "${IXCC_MLIR_CMAKE:-}" ]]; then
  ixcc_root="$(cd "${IXCC_MLIR_CMAKE}/../../../.." && pwd)"
  if [[ ! -d "${ixcc_root}" ]]; then
    echo "::error::derived IXCC root does not exist: ${ixcc_root}"
    exit 1
  fi
fi

host_mpi_lib_path=""
host_mpi_lib_dir=""
if command -v ldconfig >/dev/null 2>&1; then
  host_mpi_lib_path="$(ldconfig -p 2>/dev/null | awk '/libmpi\.so\.40/ { p=$NF } END { if (p != "") print p }')"
  if [[ -n "${host_mpi_lib_path}" && -f "${host_mpi_lib_path}" ]]; then
    host_mpi_lib_dir="$(dirname "${host_mpi_lib_path}")"
  fi
fi

docker_args=(
  --rm
  --network host
  --ipc host
  -v "${WORKSPACE}:/workspace"
  -v "${COREX_ROOT}:${COREX_ROOT}:ro"
  -w /workspace
  -e COREX_ROOT="${COREX_ROOT}"
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
  -e PYTHONUNBUFFERED=1
  -e HOME=/workspace
  -e XDG_CACHE_HOME="${PERF_CACHE_ROOT}"
  -e FLYDSL_RUNTIME_CACHE_DIR="${PERF_CACHE_DIR}"
)

if [[ "${CI_DEVICE_PRIVILEGED}" == "1" ]]; then
  docker_args+=(--privileged)
fi

if [[ "${CI_DEVICE_RUN_AS_HOST_USER}" == "1" ]]; then
  docker_args+=(-u "$(id -u):$(id -g)")
fi

if [[ -n "${host_mpi_lib_dir}" ]]; then
  docker_args+=(
    -v "${host_mpi_lib_dir}:${host_mpi_lib_dir}:ro"
    -e HOST_MPI_LIB_DIR="${host_mpi_lib_dir}"
  )
fi

if [[ -n "${IXCC_WORKING_ROOT:-}" && -d "${IXCC_WORKING_ROOT}" ]]; then
  docker_args+=(-v "${IXCC_WORKING_ROOT}:${IXCC_WORKING_ROOT}:ro" -e IXCC_WORKING_ROOT="${IXCC_WORKING_ROOT}")
fi
if [[ -n "${IXSDK_WORKING_ROOT:-}" && -d "${IXSDK_WORKING_ROOT}" ]]; then
  docker_args+=(-v "${IXSDK_WORKING_ROOT}:${IXSDK_WORKING_ROOT}:ro" -e IXSDK_WORKING_ROOT="${IXSDK_WORKING_ROOT}")
fi
if [[ -n "${IXCC_MLIR_CMAKE:-}" && -d "${IXCC_MLIR_CMAKE}" ]]; then
  docker_args+=(-v "${IXCC_MLIR_CMAKE}:${IXCC_MLIR_CMAKE}:ro" -e IXCC_MLIR_CMAKE="${IXCC_MLIR_CMAKE}")
fi
if [[ -n "${ixcc_root}" ]]; then
  docker_args+=(-v "${ixcc_root}:${ixcc_root}:ro" -e IXCC_ROOT="${ixcc_root}")
fi

docker run "${docker_args[@]}" "${CI_DEVICE_IMAGE}" bash -lc '
  set -euo pipefail
  # Perf trend must execute kernels; force-disable compile-only mode in case
  # runner/image environment accidentally exports COMPILE_ONLY=1.
  export COMPILE_ONLY=0
  mkdir -p "${XDG_CACHE_HOME}" "${FLYDSL_RUNTIME_CACHE_DIR}"
  if [[ ! -w "${XDG_CACHE_HOME}" || ! -w "${FLYDSL_RUNTIME_CACHE_DIR}" ]]; then
    echo "::error::cache directories are not writable: XDG_CACHE_HOME=${XDG_CACHE_HOME}, FLYDSL_RUNTIME_CACHE_DIR=${FLYDSL_RUNTIME_CACHE_DIR}" >&2
    exit 1
  fi
  export PATH="${COREX_ROOT}/bin:${PATH}"
  if [[ -n "${HOST_MPI_LIB_DIR:-}" ]]; then
    export LD_LIBRARY_PATH="${COREX_ROOT}/lib64:${COREX_ROOT}/lib:${HOST_MPI_LIB_DIR}:${LD_LIBRARY_PATH:-}"
  else
    export LD_LIBRARY_PATH="${COREX_ROOT}/lib64:${COREX_ROOT}/lib:${LD_LIBRARY_PATH:-}"
  fi
  export PYTHONPATH="/workspace/build-fly/python_packages:/workspace/python:/workspace:${PYTHONPATH:-}"
  python3 - <<'PY'
import sys

import torch

ok = bool(torch.cuda.is_available()) and torch.cuda.device_count() > 0
if not ok:
    print(
        f"::error::GPU is required for perf trend, but torch reports "
        f"cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
  exec "$@"
' perf-cmd "$@"
