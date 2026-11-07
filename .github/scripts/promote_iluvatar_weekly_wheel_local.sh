#!/usr/bin/env bash
# Atomically replace the local channel wheel dir after tests pass. The channel
# is whatever subdir of ${WHEEL_ROOT} the caller chose: `internal`, `latest`,
# and the legacy `ci` are all valid. This script does not decide the channel
# name; publishers do (weekly-whl-iluvatar.yaml derives it from ixcc_variant).
#
# The parent-level index.html is regenerated from whatever subdirs actually
# exist under ${WHEEL_ROOT}, so channels appear as they are populated -- no
# hardcoded list to keep in sync with the publisher yaml.
set -euo pipefail

INCOMING_DIR="${1:?usage: promote_iluvatar_weekly_wheel_local.sh <incoming-dir> <dest-dir>}"
DEST_DIR="${2:?usage: promote_iluvatar_weekly_wheel_local.sh <incoming-dir> <dest-dir>}"

shopt -s nullglob
incoming_wheels=("${INCOMING_DIR}"/flydsl-*.whl)
if [[ ${#incoming_wheels[@]} -eq 0 ]]; then
  echo "No staged wheels in ${INCOMING_DIR}" >&2
  exit 1
fi

parent="$(dirname "${DEST_DIR}")"
mkdir -p "${parent}"
tmp_dir="${DEST_DIR}.tmp.${RANDOM}"
old_dir="${DEST_DIR}.old"
rm -rf "${tmp_dir}" "${old_dir}"
cp -a "${INCOMING_DIR}" "${tmp_dir}"
{
  echo "promoted_utc=$(date -u +%FT%TZ)"
  echo "github_run_id=${GITHUB_RUN_ID:-}"
} >> "${tmp_dir}/CURRENT.txt"

if [[ -d "${DEST_DIR}" ]]; then
  mv "${DEST_DIR}" "${old_dir}"
fi
mv "${tmp_dir}" "${DEST_DIR}"
rm -rf "${old_dir}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${script_dir}/write_iluvatar_wheel_http_index.sh" "${DEST_DIR}"

# Regenerate the parent index by scanning for populated channel dirs. Anything
# with a CURRENT.txt is treated as a wheel channel. Ordering: `latest` first
# (external release), then everything else alphabetically, so the most-used
# public URL stays at the top of the page.
{
  echo '<!doctype html><meta charset="utf-8"><title>flydsl iluvatar wheels</title><ul>'
  mapfile -t channel_dirs < <(
    find "${parent}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
      | while read -r name; do
          [[ -f "${parent}/${name}/CURRENT.txt" ]] && echo "${name}"
        done \
      | sort
  )
  if [[ " ${channel_dirs[*]} " == *" latest "* ]]; then
    echo "<li><a href=\"latest/\">latest/</a></li>"
  fi
  for d in "${channel_dirs[@]}"; do
    [[ "${d}" == "latest" ]] && continue
    echo "<li><a href=\"${d}/\">${d}/</a></li>"
  done
  echo '</ul>'
} > "${parent}/index.html"

echo "Promoted ${INCOMING_DIR} -> ${DEST_DIR}"
ls -l "${DEST_DIR}"
