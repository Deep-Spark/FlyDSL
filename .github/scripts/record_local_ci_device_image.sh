#!/usr/bin/env bash
set -euo pipefail

# Record a mapping from registry digest -> local image config id / preferred tag.
#
# Usage:
#   record_local_ci_device_image.sh <desired_ref> [local_ref]
#
# desired_ref should preferably be repo@sha256:...
# local_ref is an optional preferred local runnable tag/id.

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -lt 1 ]]; then
  echo "usage: $0 <desired_image_ref> [local_ref]" >&2
  exit 1
fi

desired_ref="$1"
local_ref="${2:-}"
map_file="${FLYDSL_CI_IMAGE_MAP_FILE:-${HOME}/.cache/flydsl/ci-device-image-map.tsv}"

mkdir -p "$(dirname "${map_file}")"

digest=""
if [[ "${desired_ref}" == *@sha256:* ]]; then
  digest="${desired_ref##*@}"
elif [[ "${desired_ref}" == sha256:* ]]; then
  digest="${desired_ref}"
fi

inspect_ref="${local_ref:-${desired_ref}}"
if ! docker image inspect "${inspect_ref}" >/dev/null 2>&1; then
  # Fall back to desired ref if local_ref missing.
  inspect_ref="${desired_ref}"
fi
if ! docker image inspect "${inspect_ref}" >/dev/null 2>&1; then
  echo "::warning::cannot record map; image not found locally: ${inspect_ref}" >&2
  exit 0
fi

config_id="$(docker image inspect "${inspect_ref}" --format '{{.Id}}')"
if [[ -z "${digest}" ]]; then
  # Try RepoDigests first.
  digest="$(docker image inspect "${inspect_ref}" --format '{{if .RepoDigests}}{{index .RepoDigests 0}}{{end}}' 2>/dev/null || true)"
  if [[ "${digest}" == *@sha256:* ]]; then
    digest="${digest##*@}"
  else
    digest=""
  fi
fi

if [[ -z "${digest}" ]]; then
  echo "::warning::cannot record map without digest for ${inspect_ref}" >&2
  exit 0
fi

preferred_local="${local_ref:-${inspect_ref}}"
tmp="$(mktemp)"
if [[ -f "${map_file}" ]]; then
  awk -F '\t' -v d="${digest}" 'NF>=2 && $1!=d {print}' "${map_file}" >"${tmp}" || true
else
  : >"${tmp}"
fi
printf '%s\t%s\t%s\n' "${digest}" "${config_id}" "${preferred_local}" >>"${tmp}"
mv "${tmp}" "${map_file}"
echo "recorded local image map: ${digest} -> ${config_id} (${preferred_local})"
