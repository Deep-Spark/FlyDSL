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
# Cache layout:
#   <cache_base>/
#   ├── src/llvm-project/              # Shared LLVM source (all commits)
#   └── <commit_short>-<param_hash>/
#       ├── mlir_install/              # Installed MLIR/LLVM
#       ├── mlir_install.tgz           # Tarball (CI compatible)
#       └── meta.env                   # Build parameters record
#
# The <param_hash> ensures that the same LLVM commit built with different
# flags (e.g. X86 vs X86;AMDGPU) gets separate cache entries.
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

# ── Resolve hash file ─────────────────────────────────────────────────
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

# ── Backend-specific build parameters (overridable via env) ───────────
case "$_fly_backend" in
    rocdl)
        LLVM_TARGETS="${LLVM_TARGETS:-X86;NVPTX;AMDGPU}"
        MLIR_ROCM_RUNNER="${MLIR_ROCM_RUNNER:-ON}"
        ;;
    iluvatar)
        LLVM_TARGETS="${LLVM_TARGETS:-X86;NVPTX}"
        MLIR_ROCM_RUNNER="${MLIR_ROCM_RUNNER:-OFF}"
        ;;
    *)
        LLVM_TARGETS="${LLVM_TARGETS:-X86}"
        MLIR_ROCM_RUNNER="${MLIR_ROCM_RUNNER:-OFF}"
        ;;
esac
export LLVM_TARGETS MLIR_ROCM_RUNNER

# ── Compute param-aware cache key ─────────────────────────────────────
_fly_param_str="TARGETS=${LLVM_TARGETS};ROCM=${MLIR_ROCM_RUNNER}"
_fly_param_hash=$(echo -n "$_fly_param_str" | sha256sum | cut -c1-8)
_fly_cache_key="${_fly_commit:0:12}-${_fly_param_hash}"

# ── Resolve cache base directory ──────────────────────────────────────
if [ -n "${LLVM_CACHE_DIR:-}" ]; then
    _fly_cache_base="$LLVM_CACHE_DIR"
elif [ -d "/opt/flydsl/llvm-cache" ] && [ -w "/opt/flydsl/llvm-cache" ]; then
    _fly_cache_base="/opt/flydsl/llvm-cache"
else
    _fly_cache_base="$HOME/.cache/flydsl/llvm"
fi
_fly_cached_install="${_fly_cache_base}/${_fly_cache_key}/mlir_install"

echo "=============================================="
echo "Activating backend: $_fly_backend"
echo "  LLVM commit:  ${_fly_commit:0:12}…"
echo "  Build params: ${_fly_param_str}"
echo "  Cache key:    ${_fly_cache_key}"
echo "  Cache dir:    ${_fly_cache_base}/${_fly_cache_key}"
echo "=============================================="

# ── Check cache & display metadata ────────────────────────────────────
if [ -d "${_fly_cached_install}/lib/cmake/mlir" ]; then
    echo "✓ Cache hit — reusing existing LLVM install"
    if [ -f "${_fly_cache_base}/${_fly_cache_key}/meta.env" ]; then
        echo "  Build metadata:"
        while IFS='=' read -r key value; do
            [[ -z "$key" || "$key" == \#* ]] && continue
            printf "    %-20s = %s\n" "$key" "$value"
        done < "${_fly_cache_base}/${_fly_cache_key}/meta.env"
    fi
else
    echo "✗ Cache miss — building LLVM (this may take 30+ minutes)…"
    echo ""
    mkdir -p "${_fly_cache_base}/${_fly_cache_key}"

    (
        export LLVM_COMMIT="$_fly_commit"
        # Shared source repo: all backends share one clone
        export LLVM_SRC_CACHE="${_fly_cache_base}/src/llvm-project"
        export LLVM_BUILD_DIR="${_fly_cache_base}/${_fly_cache_key}/build"
        export LLVM_INSTALL_DIR="$_fly_cached_install"
        export LLVM_INSTALL_TGZ="${_fly_cache_base}/${_fly_cache_key}/mlir_install.tgz"
        bash "${_fly_repo_root}/scripts/build_llvm.sh" "$@"
    )

    if [ ! -d "${_fly_cached_install}/lib/cmake/mlir" ]; then
        echo "Error: LLVM build did not produce expected install at:" >&2
        echo "  ${_fly_cached_install}" >&2
        return 1
    fi
    echo "✓ LLVM built and cached successfully"
fi

# ── Set environment variables ─────────────────────────────────────────
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
echo "  python3 -m pytest tests/unit/ -v       # run unit tests"
echo "  RUN_MLIR_TESTS_ONLY=1 bash scripts/run_tests.sh  # run mlir tests"
echo "=============================================="

# ── Cleanup local variables ───────────────────────────────────────────
unset _fly_backend _fly_script_dir _fly_repo_root _fly_hash_file
unset _fly_commit _fly_cache_base _fly_cached_install
unset _fly_param_str _fly_param_hash _fly_cache_key
