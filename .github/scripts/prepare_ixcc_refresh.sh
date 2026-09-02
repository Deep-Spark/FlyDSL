#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
#
# Refresh a single IXCC working tree on the runner: fetch, MLIR-only diff
# gate, then (if warranted) checkout + sw_home/build.sh -r ixcc --host under
# an exclusive flock so that concurrent wheel builds do not read half-written
# .so files.
#
# This is the IXCC-only successor to prepare_ix_toolchains_daily.sh. It runs
# on the host runner as the flydsl user; it does not enter docker.
#
# Two callers today:
#   - ixcc-refresh.yaml     (*/30 min, matrix internal + external)
#   - prepare_ix_toolchains_daily.sh (thin wrapper, keeps perf-daily-iluvatar
#     .yml working while we migrate it in a follow-up PR)
#
# Contract:
#   - Order of operations is deliberate. `git fetch` happens first, then the
#     MLIR-gate diff is computed against the last-built commit (stamp file)
#     WITHOUT touching the working tree. Only if the gate says "rebuild" do
#     we take the flock and do `git checkout && git pull && build.sh`.
#     This preserves the tree/binary invariant when we skip: HEAD stays on
#     the last-built commit, so ci-device / perf-daily still see a
#     consistent source-vs-build pair.
#   - `--mlir-gate GLOB1,GLOB2,...` narrows the "does this commit range
#     matter" question. When unset, every commit triggers a rebuild
#     (matches the pre-30-min behavior).
#   - The build.sh call is wrapped in `flock -x -w TIMEOUT`. Wheel builds
#     that read from ${ixcc_root}/build take a shared lock on the same
#     path (see _build_wheel_persistent_body.sh in a later PR).

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  prepare_ixcc_refresh.sh --ixcc-root DIR [options]

Required:
  --ixcc-root DIR          IXCC working repository root.

Options:
  --ixcc-branch REF        Remote branch to track; passed to `git fetch
                           origin REF` and `git pull --ff-only origin REF`.
                           Default: working
  --mlir-gate GLOBS        Comma-separated pathspecs; skip build when the
                           commit range built-stamp..origin/REF touches
                           none of them. Empty (default) = always build
                           when HEAD differs.
  --flock-file PATH        Exclusive lock file wrapping the build. Default:
                           <ixcc-root>/.build.lock
  --flock-timeout SECS     How long to wait for the lock. Default: 3600.
  --ixcc-build-marker PATH File that must exist for a build to be considered
                           valid. Default:
                             <ixcc-root>/build/lib/cmake/mlir/MLIRConfig.cmake
  --ixcc-commit-stamp PATH Records the commit last built here. Default:
                             <ixcc-root>/.flydsl_ixcc_build_commit
  --force-rebuild          Skip the gate; rebuild unconditionally.
  --no-flock               Skip the flock wrapper (for local dry-runs).
  --dry-run                Print the decision and stop; do not checkout,
                           pull, invoke build.sh, or update the stamp.
                           Useful for validating the gate logic without
                           touching the tree or spending a rebuild.
  -h, --help               Show help.

Env fallbacks (only used when the matching flag is omitted):
  IXCC_WORKING_ROOT, IXCC_BRANCH, IXCC_MLIR_GATE, IXCC_FLOCK_FILE,
  IXCC_BUILD_MARKER, IXCC_COMMIT_STAMP.

Outputs (GitHub Actions):
  ixcc_commit=<short>     Short SHA at the end of the run.
  ixcc_built=<true|false> Whether we actually ran build.sh this call.
  ixcc_skip_reason=<str>  "no-op-mlir-gate" | "already-latest" | "" when built.
EOF
}

IXCC_ROOT="${IXCC_WORKING_ROOT:-}"
IXCC_BRANCH="${IXCC_BRANCH:-working}"
IXCC_MLIR_GATE="${IXCC_MLIR_GATE:-}"
IXCC_FLOCK_FILE="${IXCC_FLOCK_FILE:-}"
IXCC_FLOCK_TIMEOUT=3600
IXCC_BUILD_MARKER="${IXCC_BUILD_MARKER:-}"
IXCC_COMMIT_STAMP="${IXCC_COMMIT_STAMP:-}"
FORCE_REBUILD=0
USE_FLOCK=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ixcc-root)          IXCC_ROOT="${2:?}"; shift 2 ;;
    --ixcc-branch)        IXCC_BRANCH="${2:?}"; shift 2 ;;
    --mlir-gate)          IXCC_MLIR_GATE="${2:?}"; shift 2 ;;
    --flock-file)         IXCC_FLOCK_FILE="${2:?}"; shift 2 ;;
    --flock-timeout)      IXCC_FLOCK_TIMEOUT="${2:?}"; shift 2 ;;
    --ixcc-build-marker)  IXCC_BUILD_MARKER="${2:?}"; shift 2 ;;
    --ixcc-commit-stamp)  IXCC_COMMIT_STAMP="${2:?}"; shift 2 ;;
    --force-rebuild)      FORCE_REBUILD=1; shift ;;
    --no-flock)           USE_FLOCK=0; shift ;;
    --dry-run)            DRY_RUN=1; shift ;;
    -h|--help)            usage; exit 0 ;;
    *)  echo "::error::unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

command -v git >/dev/null || { echo "::error::git not found"; exit 1; }

if [[ -z "${IXCC_ROOT}" ]]; then
  echo "::error::--ixcc-root (or IXCC_WORKING_ROOT) is required" >&2
  exit 2
fi
if [[ ! -d "${IXCC_ROOT}/.git" ]]; then
  echo "::error::IXCC root is not a git repository: ${IXCC_ROOT}" >&2
  exit 1
fi

# sw_home layout: <SW_HOME>/sdk/{ixcc,ixcc-external,...}. `enable` and
# `build.sh` live at SW_HOME. Derive SW_HOME so a caller passing a
# non-default ixcc-external tree still finds the right sw_home.
SW_HOME="$(cd "${IXCC_ROOT}/../.." && pwd)"
if [[ ! -f "${SW_HOME}/enable" ]]; then
  echo "::error::sw_home enable script not found: ${SW_HOME}/enable" >&2
  exit 1
fi
if [[ ! -x "${SW_HOME}/build.sh" ]]; then
  echo "::error::sw_home build.sh not found or not executable: ${SW_HOME}/build.sh" >&2
  exit 1
fi

# Self-hosted runners can end up owning the tree as a different uid than
# the runner user. Silence git's dubious-ownership check up front so the
# fetch/rev-parse below don't fail before we've had a chance to log why.
git config --global --add safe.directory "${IXCC_ROOT}" || true

[[ -n "${IXCC_BUILD_MARKER}" ]] || IXCC_BUILD_MARKER="${IXCC_ROOT}/build/lib/cmake/mlir/MLIRConfig.cmake"
[[ -n "${IXCC_COMMIT_STAMP}" ]] || IXCC_COMMIT_STAMP="${IXCC_ROOT}/.flydsl_ixcc_build_commit"
[[ -n "${IXCC_FLOCK_FILE}"   ]] || IXCC_FLOCK_FILE="${IXCC_ROOT}/.build.lock"

read_stamp() {
  [[ -f "$1" ]] && tr -d '[:space:]' < "$1" || true
}
write_stamp() {
  mkdir -p "$(dirname "$1")"
  printf '%s\n' "$2" > "$1"
}

echo "[ixcc-refresh] root=${IXCC_ROOT} branch=${IXCC_BRANCH}"
echo "[ixcc-refresh] fetching origin/${IXCC_BRANCH}"
git -C "${IXCC_ROOT}" fetch --quiet origin "${IXCC_BRANCH}"

remote_head="$(git -C "${IXCC_ROOT}" rev-parse "origin/${IXCC_BRANCH}")"
stamp_commit="$(read_stamp "${IXCC_COMMIT_STAMP}")"
build_marker_ok=0
[[ -e "${IXCC_BUILD_MARKER}" ]] && build_marker_ok=1

echo "[ixcc-refresh] remote_head=${remote_head:0:12}"
echo "[ixcc-refresh] stamp_commit=${stamp_commit:0:12} (build_marker_present=${build_marker_ok})"

# Decide whether the tree needs a rebuild.
#
#   force            -> yes
#   no stamp yet     -> yes (fresh clone or lost state)
#   marker missing   -> yes (someone rm -rf'd the build dir)
#   stamp == remote  -> no (idempotent tick)
#   stamp != remote  -> ask the MLIR gate: any commit in stamp..remote touch
#                       an MLIR path? if the gate is empty, treat as yes.
needs_build=0
skip_reason=""
if [[ "${FORCE_REBUILD}" == "1" ]]; then
  needs_build=1
  echo "[ixcc-refresh] --force-rebuild set; rebuilding"
elif [[ -z "${stamp_commit}" ]]; then
  needs_build=1
  echo "[ixcc-refresh] no build stamp; rebuilding (first-run)"
elif [[ "${build_marker_ok}" == "0" ]]; then
  needs_build=1
  echo "[ixcc-refresh] build marker missing (${IXCC_BUILD_MARKER}); rebuilding"
elif [[ "${stamp_commit}" == "${remote_head}" ]]; then
  needs_build=0
  skip_reason="already-latest"
  echo "[ixcc-refresh] stamp == remote; skip"
else
  # Ahead commits exist. Apply MLIR gate.
  if [[ -z "${IXCC_MLIR_GATE}" ]]; then
    needs_build=1
    echo "[ixcc-refresh] MLIR gate not set; any commit triggers rebuild"
  else
    # Split on commas -> pathspec array. Trailing slashes optional; git
    # treats "llvm/mlir" as "match everything under llvm/mlir/".
    IFS=',' read -r -a mlir_paths <<< "${IXCC_MLIR_GATE}"
    # Strip whitespace on each element.
    for i in "${!mlir_paths[@]}"; do
      mlir_paths[i]="${mlir_paths[i]# }"; mlir_paths[i]="${mlir_paths[i]% }"
    done
    echo "[ixcc-refresh] MLIR gate diff ${stamp_commit:0:12}..${remote_head:0:12} -- ${mlir_paths[*]}"
    # Note: stamp_commit may not exist locally if someone force-pushed the
    # branch. Guard with a merge-base check; if stamp isn't reachable, fall
    # back to "rebuild" and let the next tick converge.
    if ! git -C "${IXCC_ROOT}" cat-file -e "${stamp_commit}^{commit}" 2>/dev/null; then
      needs_build=1
      echo "[ixcc-refresh] stamp commit ${stamp_commit:0:12} not present locally (force-push?); rebuilding"
    else
      touched="$(git -C "${IXCC_ROOT}" diff --name-only \
                   "${stamp_commit}..${remote_head}" -- "${mlir_paths[@]}" 2>/dev/null || true)"
      if [[ -z "${touched}" ]]; then
        needs_build=0
        skip_reason="no-op-mlir-gate"
        ahead_n="$(git -C "${IXCC_ROOT}" rev-list --count \
                     "${stamp_commit}..${remote_head}" 2>/dev/null || echo 0)"
        echo "[ixcc-refresh] gate matched 0 files across ${ahead_n} ahead commit(s); skip"
      else
        needs_build=1
        touched_n="$(printf '%s\n' "${touched}" | wc -l)"
        echo "[ixcc-refresh] gate matched ${touched_n} file(s); rebuilding"
        printf '%s\n' "${touched}" | head -10 | sed 's/^/[ixcc-refresh]   /'
        [[ "${touched_n}" -gt 10 ]] && echo "[ixcc-refresh]   ... (${touched_n} total)"
      fi
    fi
  fi
fi

if [[ "${needs_build}" == "0" ]]; then
  short_head="$(git -C "${IXCC_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    {
      echo "ixcc_commit=${short_head}"
      echo "ixcc_built=false"
      echo "ixcc_skip_reason=${skip_reason}"
    } >> "${GITHUB_OUTPUT}"
  fi
  echo "[ixcc-refresh] ixcc_commit=${short_head} ixcc_built=false ixcc_skip_reason=${skip_reason}"
  exit 0
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  short_head="$(git -C "${IXCC_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  remote_short="${remote_head:0:12}"
  echo "[ixcc-refresh] --dry-run: would checkout+pull ${IXCC_BRANCH} and run sw_home/build.sh; not touching tree"
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    {
      echo "ixcc_commit=${short_head}"
      echo "ixcc_built=false"
      echo "ixcc_skip_reason=dry-run-would-build (target=${remote_short})"
    } >> "${GITHUB_OUTPUT}"
  fi
  exit 0
fi

do_build() {
  # Inside the flocked region: sync the tree to the remote head, source
  # sw_home/enable to set CPATH/CMAKE_ARGS to the requested python, and
  # invoke the vendor build entry point.
  git -C "${IXCC_ROOT}" checkout "${IXCC_BRANCH}"
  git -C "${IXCC_ROOT}" pull --ff-only origin "${IXCC_BRANCH}"
  (
    cd "${SW_HOME}"
    set +u
    # shellcheck disable=SC1091
    source "${SW_HOME}/enable"
    set -u
    ./build.sh -r ixcc --host
  )
}

echo "[ixcc-refresh] taking build lock: ${IXCC_FLOCK_FILE} (timeout=${IXCC_FLOCK_TIMEOUT}s)"
if [[ "${USE_FLOCK}" == "1" ]]; then
  command -v flock >/dev/null || {
    echo "::error::flock is required unless --no-flock is passed"; exit 1;
  }
  mkdir -p "$(dirname "${IXCC_FLOCK_FILE}")"
  # Open the lock file on FD 200 so the subshell inherits it and flock
  # releases automatically on exit. Timeout guards against a stuck holder.
  (
    flock -x -w "${IXCC_FLOCK_TIMEOUT}" 200 || {
      echo "::error::could not acquire ${IXCC_FLOCK_FILE} within ${IXCC_FLOCK_TIMEOUT}s"
      exit 1
    }
    do_build
  ) 200>"${IXCC_FLOCK_FILE}"
else
  echo "[ixcc-refresh] --no-flock: running build without lock"
  do_build
fi

built_commit="$(git -C "${IXCC_ROOT}" rev-parse HEAD)"
built_short="$(git -C "${IXCC_ROOT}" rev-parse --short HEAD)"
write_stamp "${IXCC_COMMIT_STAMP}" "${built_commit}"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  {
    echo "ixcc_commit=${built_short}"
    echo "ixcc_built=true"
    echo "ixcc_skip_reason="
  } >> "${GITHUB_OUTPUT}"
fi
echo "[ixcc-refresh] built ixcc_commit=${built_short} (full=${built_commit})"
