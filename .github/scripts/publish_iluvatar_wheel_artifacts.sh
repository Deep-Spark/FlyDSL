#!/usr/bin/env bash
# Copy promoted Iluvatar wheels into the HTTP artifact dir using real PEP 427
# names only. A PEP 503 simple index keeps the CI URL stable; pip still needs
# --pre because CI wheels use a .dev version.
set -euo pipefail

SRC_DIR="${1:?usage: publish_iluvatar_wheel_artifacts.sh <src-dir> [channel-tag]}"
CHANNEL_TAG="${2:-ci}"
DEST_DIR="${FLYDSL_ILUVATAR_WHEEL_ARTIFACT_DIR:-/srv/artifacts/wheels/flydsl}"
SITE_ROOT="$(cd "${DEST_DIR}/../.." && pwd)"
SIMPLE_DIR="${SITE_ROOT}/simple/flydsl"

if [[ ! -d "${DEST_DIR}" ]]; then
  echo "artifact dir missing (create and chown to the runner user): ${DEST_DIR}" >&2
  exit 1
fi

shopt -s nullglob
src_wheels=("${SRC_DIR}"/flydsl-*.whl)
if [[ ${#src_wheels[@]} -eq 0 ]]; then
  echo "No flydsl wheels in ${SRC_DIR}" >&2
  exit 1
fi

copied=()
for whl in "${src_wheels[@]}"; do
  base="$(basename "${whl}")"
  if [[ "${base}" == flydsl-0+iluvatar.* ]]; then
    continue
  fi
  if [[ "${base}" == flydsl-iluvatar-*-manylinux_x86_64.whl ]]; then
    continue
  fi
  if [[ ! "${base}" =~ -(cp[0-9]+)-(cp[0-9]+|abi3|none)-([A-Za-z0-9_]+)\.whl$ ]]; then
    continue
  fi
  py="${BASH_REMATCH[1]}"
  abi="${BASH_REMATCH[2]}"
  plat="${BASH_REMATCH[3]}"
  suffix="${py}-${abi}-${plat}.whl"

  for old in "${DEST_DIR}"/flydsl-*-"${suffix}"; do
    rm -f "${old}"
  done

  cp -f "${whl}" "${DEST_DIR}/${base}"
  copied+=("${base}")
done

if [[ ${#copied[@]} -eq 0 ]]; then
  echo "No PEP 427 flydsl wheels to publish from ${SRC_DIR}" >&2
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${script_dir}/write_iluvatar_wheel_http_index.sh" "${DEST_DIR}"

mkdir -p "${SIMPLE_DIR}"
{
  echo '<!DOCTYPE html><html><body>'
  for base in "${copied[@]}"; do
    echo "<a href=\"../../wheels/flydsl/${base}\">${base}</a><br/>"
  done
  echo '</body></html>'
} > "${SIMPLE_DIR}/index.html"
{
  echo '<!DOCTYPE html><html><body><a href="flydsl/">flydsl</a></body></html>'
} > "${SITE_ROOT}/simple/index.html"

{
  echo "updated_utc=$(date -u +%FT%TZ)"
  echo "channel=${CHANNEL_TAG}"
  echo "github_run_id=${GITHUB_RUN_ID:-}"
  echo "wheels=${copied[*]}"
} > "${DEST_DIR}/CURRENT.txt"

echo "Published artifacts to ${DEST_DIR}"
printf '  %s\n' "${copied[@]}"
ls -l "${DEST_DIR}"
