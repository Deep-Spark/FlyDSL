# SPDX-License-Identifier: Apache-2.0
# FlyDSL Iluvatar runtime / test environment.
#
# Usage (from any directory):
#   source /path/to/FlyDSL/env.sh
#   source .venv/bin/activate   # if using the repo venv
#
# Covers: L0 unit tests, binary pipeline smoke, minimal_iluvatar_kernel*.py

# Repo root = directory containing this file.
if [[ -n "${BASH_VERSION:-}" ]]; then
  FLYDSL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  FLYDSL_ROOT="$(cd "$(dirname "${(%):-%x}")" && pwd)"
else
  FLYDSL_ROOT="$(cd "$(dirname "$0")" && pwd)"
fi

# Vendor toolchain (override before sourcing if your paths differ).
export IXCC_MLIR_CMAKE="${IXCC_MLIR_CMAKE:-/home/wcyx/sw_home/sdk/ixcc/build/lib/cmake/mlir}"
export COREX_ROOT="${COREX_ROOT:-/home/wcyx/sw_home/local/corex}"

# CoreX bin first so fly-opt / JIT use Iluvatar ld.lld (not system lld).
export PATH="${COREX_ROOT}/bin:${PATH}"

# CoreX lib64 first, then FlyDSL JIT / MLIR shared libs.
export LD_LIBRARY_PATH="${COREX_ROOT}/lib64:${FLYDSL_ROOT}/build-fly/python_packages/flydsl/_mlir/_mlir_libs:${LD_LIBRARY_PATH:-}"

# Python imports (pytest --confcutdir=tests/unit still benefits).
export PYTHONPATH="${FLYDSL_ROOT}/build-fly/python_packages:${FLYDSL_ROOT}:${PYTHONPATH:-}"

# Iluvatar compile / runtime pairing.
export FLYDSL_COMPILE_BACKEND="${FLYDSL_COMPILE_BACKEND:-iluvatar}"
export FLYDSL_RUNTIME_KIND="${FLYDSL_RUNTIME_KIND:-iluvatar}"
export ARCH="${ARCH:-ivcore11}"
export FLYDSL_RUNTIME_ENABLE_CACHE="${FLYDSL_RUNTIME_ENABLE_CACHE:-0}"
export FLYDSL_PYTHON_PACKAGES="${FLYDSL_ROOT}/build-fly/python_packages"

# L1b opt-in smoke: tests/unit/test_iluvatar_binary_pipeline_smoke.py
export FLYDSL_ILUVATAR_FLY_OPT="${FLYDSL_ILUVATAR_FLY_OPT:-${FLYDSL_ROOT}/build-fly/bin/fly-opt}"

unset LD_PRELOAD

if [[ -n "${BASH_VERSION:-}" ]]; then
  hash -r 2>/dev/null || true
fi
