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
  --force-rebuild          Force rebuild even if already latest.
  -h, --help               Show help.
EOF
}

IXCC_ROOT="${IXCC_WORKING_ROOT:-/home/flydsl/sw_home/sdk/ixcc}"
IXSDK_ROOT="${IXSDK_WORKING_ROOT:-/home/flydsl/sw_home/sdk/ixsdk}"
STATE_FILE="${IX_TOOLCHAIN_DAILY_STATE_FILE:-/var/tmp/flydsl-ix-toolchain-daily.state}"
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
require_cmd rg

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
if [[ "${FORCE_REBUILD}" == "1" || "${ixcc_local_head}" != "${ixcc_remote_head}" ]]; then
  echo "[ix-toolchain] IXCC has updates (or force rebuild), syncing working branch"
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
  updated_today="true"
else
  echo "[ix-toolchain] IXCC already latest on working; skip build"
fi
ixcc_commit="$(git -C "${IXCC_ROOT}" rev-parse --short HEAD)"

echo "[ix-toolchain] checking IXSDK working @ ${IXSDK_ROOT}"
git -C "${IXSDK_ROOT}" fetch origin working
git -C "${IXSDK_ROOT}" checkout working
ixsdk_local_head="$(git -C "${IXSDK_ROOT}" rev-parse HEAD)"
ixsdk_remote_head="$(git -C "${IXSDK_ROOT}" rev-parse origin/working)"
if [[ "${FORCE_REBUILD}" == "1" || "${ixsdk_local_head}" != "${ixsdk_remote_head}" ]]; then
  echo "[ix-toolchain] IXSDK has updates (or force rebuild), syncing working branch"
  git -C "${IXSDK_ROOT}" pull --ff-only origin working
  echo "[ix-toolchain] installing IXSDK (make install)"
  (cd "${IXSDK_ROOT}" && make install)
  updated_today="true"
else
  echo "[ix-toolchain] IXSDK already latest on working; skip install"
fi
ixsdk_commit="$(git -C "${IXSDK_ROOT}" rev-parse --short HEAD)"

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
