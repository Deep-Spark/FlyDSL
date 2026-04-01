#!/bin/bash
# ---------------------------------------------------------------------------
# Activate a FlyDSL backend by setting MLIR_PATH and FLY_BUILD_DIR.
#
# Usage:  source scripts/activate_backend.sh <backend> [build_llvm_args...]
#
# Examples:
#   source scripts/activate_backend.sh rocdl        # ROCDL backend
#   source scripts/activate_backend.sh iluvatar      # Iluvatar backend
#   source scripts/activate_backend.sh rocdl -j32    # pass -j32 to build_llvm.sh
#
# LLVM install directories are cached under:
#   <cache_base>/<commit>/mlir_install/
#
# Cache location (checked in order):
#   1. $LLVM_CACHE_DIR          — explicit override
#   2. /opt/flydsl/llvm-cache   — shared across all users on the machine
#   3. ~/.cache/flydsl/llvm     — per-user fallback
#
# To set up the shared cache (one-time, needs sudo):
#   sudo mkdir -p /opt/flydsl/llvm-cache
#   sudo chmod 2777 /opt/flydsl/llvm-cache
#
# If a cached install exists for the backend's LLVM commit, it is reused
# immediately (no build). Otherwise build_llvm.sh is invoked once and the
# result is stored in the cache for future activations.
#
# After sourcing, the following environment variables are set:
#   MLIR_PATH       — points to the cached LLVM/MLIR install
#   FLY_BUILD_DIR   — backend-specific FlyDSL build directory
#   FLY_BACKEND     — the active backend name
# ---------------------------------------------------------------------------

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Error: this script must be sourced, not executed." >&2
    echo "Usage: source $0 <backend>" >&2
    exit 1
fi

_fly_backend="${1:?Usage: source scripts/activate_backend.sh <backend>}"
shift  # remaining args forwarded to build_llvm.sh on cache miss

_fly_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_fly_repo_root="$(cd "${_fly_script_dir}/.." && pwd)"

# Resolve hash file: rocdl uses llvm-hash.txt, others use llvm-hash-<backend>.txt
if [ "$_fly_backend" = "rocdl" ]; then
    _fly_hash_file="${_fly_repo_root}/thirdparty/llvm-hash.txt"
else
    _fly_hash_file="${_fly_repo_root}/thirdparty/llvm-hash-${_fly_backend}.txt"
fi

if [ ! -f "$_fly_hash_file" ]; then
    echo "Error: LLVM hash file not found: $_fly_hash_file" >&2
    echo "Create it with the desired LLVM commit SHA (40 hex chars)." >&2
    return 1
fi

_fly_commit=$(tr -d '[:space:]' < "$_fly_hash_file")
if [ -n "${LLVM_CACHE_DIR:-}" ]; then
    _fly_cache_base="$LLVM_CACHE_DIR"
elif [ -d "/opt/flydsl/llvm-cache" ] && [ -w "/opt/flydsl/llvm-cache" ]; then
    _fly_cache_base="/opt/flydsl/llvm-cache"
else
    _fly_cache_base="$HOME/.cache/flydsl/llvm"
fi
_fly_cached_install="${_fly_cache_base}/${_fly_commit}/mlir_install"

echo "=============================================="
echo "Activating backend: $_fly_backend"
echo "  LLVM commit:  ${_fly_commit:0:12}…"
echo "  Hash file:    $_fly_hash_file"
echo "  Cache dir:    ${_fly_cache_base}/${_fly_commit:0:12}…"
echo "=============================================="

if [ -d "${_fly_cached_install}/lib/cmake/mlir" ]; then
    echo "✓ Cache hit — reusing existing LLVM install"
else
    echo "✗ Cache miss — building LLVM (this may take 30+ minutes)…"
    echo ""
    mkdir -p "${_fly_cache_base}/${_fly_commit}"

    (
        export LLVM_COMMIT="$_fly_commit"
        export LLVM_INSTALL_DIR="$_fly_cached_install"
        export LLVM_INSTALL_TGZ="${_fly_cache_base}/${_fly_commit}/mlir_install.tgz"
        bash "${_fly_repo_root}/scripts/build_llvm.sh" "$@"
    )

    if [ ! -d "${_fly_cached_install}/lib/cmake/mlir" ]; then
        echo "Error: LLVM build did not produce expected install at:" >&2
        echo "  ${_fly_cached_install}" >&2
        return 1
    fi
    echo "✓ LLVM built and cached successfully"
fi

export MLIR_PATH="$_fly_cached_install"
export FLY_BUILD_DIR="${_fly_repo_root}/build-fly-${_fly_backend}"
export FLY_BACKEND="$_fly_backend"

echo ""
echo "Environment ready:"
echo "  MLIR_PATH      = ${MLIR_PATH}"
echo "  FLY_BUILD_DIR  = ${FLY_BUILD_DIR}"
echo "  FLY_BACKEND    = ${FLY_BACKEND}"
echo ""
echo "Next steps:"
echo "  pip install -e . --use-pep517          # build FlyDSL"
echo "  python3 -m pytest tests/pyir/ -v       # run pyir tests"
echo "  RUN_MLIR_TESTS_ONLY=1 bash scripts/run_tests.sh  # run mlir tests"
echo "=============================================="

unset _fly_backend _fly_script_dir _fly_repo_root _fly_hash_file
unset _fly_commit _fly_cache_base _fly_cached_install
