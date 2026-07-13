#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  prepare_ix_toolchains_daily.sh [options]

Options:
  --ixcc-root <path>       IXCC working repository root.
  --ixsdk-root <path>      IXSDK working repository root.
  --state-file <path>      State file for daily build marker.
                           Default: /var/tmp/flydsl-ix-toolchain-daily.state
  --ixcc-build-marker <path>
                           Marker file that must exist for IXCC build.
                           Default: <ixcc-root>/build/lib/cmake/mlir/MLIRConfig.cmake
  --ixsdk-install-marker <path>
                           Optional marker file that must exist for IXSDK install.
                           Default: empty (disabled)
  --ixcc-commit-stamp <path>
                           File that records IXCC commit last built locally.
                           Default: <ixcc-root>/.flydsl_ixcc_build_commit
  --ixsdk-commit-stamp <path>
                           File that records IXSDK commit last installed locally.
                           Default: <ixsdk-root>/.flydsl_ixsdk_install_commit
  --force-rebuild          Force rebuild even if already latest.
  -h, --help               Show help.
EOF
}

IXCC_ROOT="${IXCC_WORKING_ROOT:-/home/flydsl/sw_home/sdk/ixcc}"
IXSDK_ROOT="${IXSDK_WORKING_ROOT:-/home/flydsl/sw_home/sdk/ixsdk}"
STATE_FILE="${IX_TOOLCHAIN_DAILY_STATE_FILE:-/var/tmp/flydsl-ix-toolchain-daily.state}"
IXCC_BUILD_MARKER="${IXCC_BUILD_MARKER:-}"
IXSDK_INSTALL_MARKER="${IXSDK_INSTALL_MARKER:-}"
IXCC_COMMIT_STAMP="${IXCC_COMMIT_STAMP:-}"
IXSDK_COMMIT_STAMP="${IXSDK_COMMIT_STAMP:-}"
FORCE_REBUILD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ixcc-root)
      IXCC_ROOT="${2:?missing value for --ixcc-root}"
      shift 2
      ;;
    --ixsdk-root)
      IXSDK_ROOT="${2:?missing value for --ixsdk-root}"
      shift 2
      ;;
    --state-file)
      STATE_FILE="${2:?missing value for --state-file}"
      shift 2
      ;;
    --ixcc-build-marker)
      IXCC_BUILD_MARKER="${2:?missing value for --ixcc-build-marker}"
      shift 2
      ;;
    --ixsdk-install-marker)
      IXSDK_INSTALL_MARKER="${2:?missing value for --ixsdk-install-marker}"
      shift 2
      ;;
    --ixcc-commit-stamp)
      IXCC_COMMIT_STAMP="${2:?missing value for --ixcc-commit-stamp}"
      shift 2
      ;;
    --ixsdk-commit-stamp)
      IXSDK_COMMIT_STAMP="${2:?missing value for --ixsdk-commit-stamp}"
      shift 2
      ;;
    --force-rebuild)
      FORCE_REBUILD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "::error::missing required command: $1"
    exit 1
  }
}

require_cmd git

if [[ ! -d "${IXCC_ROOT}" ]]; then
  echo "::error::IXCC root not found: ${IXCC_ROOT}"
  exit 1
fi
if [[ ! -d "${IXSDK_ROOT}" ]]; then
  echo "::error::IXSDK root not found: ${IXSDK_ROOT}"
  exit 1
fi
if [[ ! -d "${IXCC_ROOT}/.git" ]]; then
  echo "::error::IXCC root is not a git repository: ${IXCC_ROOT}"
  exit 1
fi
if [[ ! -d "${IXSDK_ROOT}/.git" ]]; then
  echo "::error::IXSDK root is not a git repository: ${IXSDK_ROOT}"
  exit 1
fi

# IXCC build must run from sw_home root after sourcing enable.
SW_HOME="$(cd "${IXCC_ROOT}/../.." && pwd)"
if [[ ! -f "${SW_HOME}/enable" ]]; then
  echo "::error::sw_home enable script not found: ${SW_HOME}/enable"
  exit 1
fi
if [[ ! -x "${SW_HOME}/build.sh" ]]; then
  echo "::error::sw_home build.sh not found or not executable: ${SW_HOME}/build.sh"
  exit 1
fi

# Self-hosted runners can mount toolchain repos with ownership different from
# the runner user. Mark them as safe to prevent "dubious ownership" failures.
git config --global --add safe.directory "${IXCC_ROOT}" || true
git config --global --add safe.directory "${IXSDK_ROOT}" || true

if [[ -z "${IXCC_BUILD_MARKER}" ]]; then
  IXCC_BUILD_MARKER="${IXCC_ROOT}/build/lib/cmake/mlir/MLIRConfig.cmake"
fi
if [[ -z "${IXCC_COMMIT_STAMP}" ]]; then
  IXCC_COMMIT_STAMP="${IXCC_ROOT}/.flydsl_ixcc_build_commit"
fi
if [[ -z "${IXSDK_COMMIT_STAMP}" ]]; then
  IXSDK_COMMIT_STAMP="${IXSDK_ROOT}/.flydsl_ixsdk_install_commit"
fi

read_commit_stamp() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    tr -d '[:space:]' < "${path}"
  fi
}

write_commit_stamp() {
  local path="$1"
  local commit="$2"
  mkdir -p "$(dirname "${path}")"
  printf '%s\n' "${commit}" > "${path}"
}

has_local_artifact_for_commit() {
  local commit="$1"
  local stamp_path="$2"
  local marker_path="${3:-}"
  local stamp_commit
  stamp_commit="$(read_commit_stamp "${stamp_path}")"
  if [[ -z "${stamp_commit}" || "${stamp_commit}" != "${commit}" ]]; then
    return 1
  fi
  if [[ -n "${marker_path}" && ! -e "${marker_path}" ]]; then
    return 1
  fi
  return 0
}

write_state() {
  local date_utc="$1"
  local ixcc_commit="$2"
  local ixsdk_commit="$3"
  mkdir -p "$(dirname "${STATE_FILE}")"
  {
    echo "date_utc=${date_utc}"
    echo "ixcc_commit=${ixcc_commit}"
    echo "ixsdk_commit=${ixsdk_commit}"
    echo "updated_at_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  } > "${STATE_FILE}"
}

today_utc="$(date -u +%Y-%m-%d)"
ixcc_commit=""
ixsdk_commit=""
updated_today="false"

echo "[ix-toolchain] checking IXCC working @ ${IXCC_ROOT}"
git -C "${IXCC_ROOT}" fetch origin working
git -C "${IXCC_ROOT}" checkout working
ixcc_local_head="$(git -C "${IXCC_ROOT}" rev-parse HEAD)"
ixcc_remote_head="$(git -C "${IXCC_ROOT}" rev-parse origin/working)"
ixcc_needs_build=0
if [[ "${FORCE_REBUILD}" == "1" || "${ixcc_local_head}" != "${ixcc_remote_head}" ]]; then
  ixcc_needs_build=1
elif ! has_local_artifact_for_commit "${ixcc_local_head}" "${IXCC_COMMIT_STAMP}" "${IXCC_BUILD_MARKER}"; then
  echo "[ix-toolchain] IXCC commit unchanged but local build marker/stamp missing; rebuilding"
  ixcc_needs_build=1
fi
if [[ "${ixcc_needs_build}" == "1" ]]; then
  echo "[ix-toolchain] IXCC rebuild required, syncing working branch"
  git -C "${IXCC_ROOT}" pull --ff-only origin working
  echo "[ix-toolchain] building IXCC from sw_home (source enable && ./build.sh -r ixcc --host)"
  (
    cd "${SW_HOME}"
    # sw_home/enable expects interactive shell vars (e.g. PS1). In CI with
    # `set -u`, temporarily relax nounset while sourcing.
    set +u
    # shellcheck disable=SC1091
    source "${SW_HOME}/enable"
    set -u
    ./build.sh -r ixcc --host
  )
  ixcc_commit="$(git -C "${IXCC_ROOT}" rev-parse --short HEAD)"
  write_commit_stamp "${IXCC_COMMIT_STAMP}" "${ixcc_commit}"
  updated_today="true"
else
  echo "[ix-toolchain] IXCC already latest on working; skip build"
  ixcc_commit="$(git -C "${IXCC_ROOT}" rev-parse --short HEAD)"
fi

echo "[ix-toolchain] checking IXSDK working @ ${IXSDK_ROOT}"
git -C "${IXSDK_ROOT}" fetch origin working
git -C "${IXSDK_ROOT}" checkout working
ixsdk_local_head="$(git -C "${IXSDK_ROOT}" rev-parse HEAD)"
ixsdk_remote_head="$(git -C "${IXSDK_ROOT}" rev-parse origin/working)"
ixsdk_needs_install=0
if [[ "${FORCE_REBUILD}" == "1" || "${ixsdk_local_head}" != "${ixsdk_remote_head}" ]]; then
  ixsdk_needs_install=1
elif ! has_local_artifact_for_commit "${ixsdk_local_head}" "${IXSDK_COMMIT_STAMP}" "${IXSDK_INSTALL_MARKER}"; then
  echo "[ix-toolchain] IXSDK commit unchanged but local install marker/stamp missing; reinstalling"
  ixsdk_needs_install=1
fi
if [[ "${ixsdk_needs_install}" == "1" ]]; then
  echo "[ix-toolchain] IXSDK reinstall required, syncing working branch"
  git -C "${IXSDK_ROOT}" pull --ff-only origin working
  echo "[ix-toolchain] installing IXSDK (make install)"
  (cd "${IXSDK_ROOT}" && make install)
  ixsdk_commit="$(git -C "${IXSDK_ROOT}" rev-parse --short HEAD)"
  write_commit_stamp "${IXSDK_COMMIT_STAMP}" "${ixsdk_commit}"
  updated_today="true"
else
  echo "[ix-toolchain] IXSDK already latest on working; skip install"
  ixsdk_commit="$(git -C "${IXSDK_ROOT}" rev-parse --short HEAD)"
fi

write_state "${today_utc}" "${ixcc_commit}" "${ixsdk_commit}"

if [[ -z "${ixcc_commit}" ]]; then
  ixcc_commit="$(git -C "${IXCC_ROOT}" rev-parse --short HEAD 2>/dev/null || true)"
fi
if [[ -z "${ixsdk_commit}" ]]; then
  ixsdk_commit="$(git -C "${IXSDK_ROOT}" rev-parse --short HEAD 2>/dev/null || true)"
fi

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  {
    echo "ixcc_commit=${ixcc_commit:-unknown}"
    echo "ixsdk_commit=${ixsdk_commit:-unknown}"
    echo "updated_today=${updated_today}"
  } >> "${GITHUB_OUTPUT}"
fi

echo "[ix-toolchain] ixcc_commit=${ixcc_commit:-unknown}"
echo "[ix-toolchain] ixsdk_commit=${ixsdk_commit:-unknown}"
echo "[ix-toolchain] updated_today=${updated_today}"
