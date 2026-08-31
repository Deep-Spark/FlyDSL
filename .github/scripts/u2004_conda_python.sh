#!/usr/bin/env bash
# Shared Ubuntu 20.04 Python/compiler setup for ixcc and wheel builds.
set -euo pipefail

u2004_conda_prefix() {
  local ver="$1"
  local base="${U2004_CONDA_BASE:-/tmp/flydsl-u2004-conda}"
  echo "${base}/cp${ver//./}"
}

u2004_prepare_conda_base() {
  local base="${U2004_CONDA_BASE:-/tmp/flydsl-u2004-conda}"
  if [[ -e "${base}" && ! -d "${base}" ]]; then
    echo "U2004_CONDA_BASE exists but is not a directory: ${base}" >&2
    exit 1
  fi
  mkdir -p "${base}"
  if [[ ! -w "${base}" ]]; then
    echo "U2004_CONDA_BASE is not writable by uid $(id -u): ${base}" >&2
    echo "Remove the root-owned directory or chown it to the CI runner user." >&2
    exit 1
  fi
}

u2004_python_root() {
  local py="$1"
  cd "$(dirname "${py}")/.." && pwd
}

u2004_ensure_conda_python() {
  local ver="$1"
  local prefix
  prefix="$(u2004_conda_prefix "${ver}")"
  u2004_prepare_conda_base
  if [[ ! -x "${prefix}/bin/python" ]]; then
    mkdir -p "$(dirname "${prefix}")"
    export HOME="${U2004_CONDA_HOME:-$(dirname "${prefix}")/.conda-home}"
    mkdir -p "${HOME}"
    /usr/local/conda/bin/conda create -y -p "${prefix}" \
      python="${ver}" gxx_linux-64 gcc_linux-64 zlib >&2
  fi
  echo "${prefix}/bin/python"
}

u2004_ensure_python_packages() {
  local py="$1"
  if ! "${py}" -c "import nanobind" >/dev/null 2>&1; then
    "${py}" -m pip install -U pip "nanobind>=2.9,<3" >&2
  fi
}

u2004_resolve_python() {
  local ver="$1"
  local py=""
  if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN}" ]]; then
    py="${PYTHON_BIN}"
  elif [[ -x "/usr/local/conda/envs/py${ver}/bin/python" ]]; then
    py="/usr/local/conda/envs/py${ver}/bin/python"
  elif [[ -x "/opt/conda/envs/py${ver}/bin/python" ]]; then
    py="/opt/conda/envs/py${ver}/bin/python"
  elif command -v "python${ver}" >/dev/null 2>&1; then
    py="$(command -v "python${ver}")"
  elif [[ "${ver}" == "3.12" ]]; then
    py="$(u2004_ensure_conda_python "${ver}")"
  elif command -v python3 >/dev/null 2>&1; then
    py="$(command -v python3)"
  else
    echo "cannot find python ${ver} in container" >&2
    exit 1
  fi
  u2004_ensure_python_packages "${py}"
  echo "${py}"
}

u2004_isolate_python_runtime() {
  local py="$1"
  local py_bin py_root
  py_bin="$(cd "$(dirname "${py}")" && pwd)"
  py_root="$(u2004_python_root "${py}")"
  export PATH="${py_bin}:/usr/local/bin:/usr/bin:/bin"
  export PYTHONHOME="${py_root}"
  unset PYTHONPATH PYTHONUSERBASE CONDA_PREFIX CONDA_DEFAULT_ENV _CONDA_EXE || true
  # The corex-base-20.04 image hard-codes CPATH/_COREX_PY_INC to the py3.10
  # headers even after we activate a py3.12 conda env. GCC still ranks CPATH
  # ahead of the compile-line -isystem, so <Python.h> silently resolves to
  # the py3.10 tree. The 32-byte PyHeapTypeObject drift then corrupts
  # ht_qualname during nb_type_new and crashes at import (ASLR-sensitive).
  unset CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH _COREX_PY_INC || true
  export LD_LIBRARY_PATH="${py_root}/lib:${COREX_ROOT:+${COREX_ROOT}/lib64:}${LD_LIBRARY_PATH:-}"
}

u2004_export_python_runtime() {
  u2004_isolate_python_runtime "$1"
}

u2004_cmake_compiler_args() {
  :
}

u2004_cmake_python_args() {
  local py="$1"
  local root
  root="$(u2004_python_root "${py}")"
  printf '%s\n' \
    "-DPython3_EXECUTABLE=${py}" \
    "-DPython3_ROOT_DIR=${root}" \
    "-DPython_EXECUTABLE=${py}" \
    "-DPython_ROOT_DIR=${root}" \
    "-DPython3_FIND_STRATEGY=LOCATION" \
    "-DPython_FIND_STRATEGY=LOCATION"
}

u2004_validate_ixcc_python_bindings() {
  local ver="$1"
  local ixcc_root="$2"
  local want="${ver//./}"
  shopt -s nullglob
  local bad=()
  local found=0
  for so in "${ixcc_root}"/build/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs/*.so; do
    [[ -f "${so}" ]] || continue
    base="$(basename "${so}")"
    if [[ "${base}" == *cpython-${want}* ]]; then
      found=1
    elif [[ "${base}" == *cpython-* ]]; then
      bad+=("${base}")
    fi
  done
  if [[ ${#bad[@]} -gt 0 ]]; then
    echo "ixcc MLIR bindings use unexpected Python tag (wanted cpython-${want}):" >&2
    printf '  %s\n' "${bad[@]}" >&2
    return 1
  fi
  if [[ "${found}" != "1" ]]; then
    echo "ixcc MLIR bindings missing cpython-${want} artifacts under ${ixcc_root}/build" >&2
    return 1
  fi
  return 0
}
