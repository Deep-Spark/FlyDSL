#!/usr/bin/env bash
set -euo pipefail

# Resolve a desired CI device image ref to a locally runnable docker reference
# without pulling layers when local content already matches.
#
# Usage:
#   resolve_local_ci_device_image.sh <desired_ref>
#
# Exit codes:
#   0  local content hit; prints runnable local ref on stdout
#   2  miss; caller should pull desired_ref
#   1  usage/runtime error

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -lt 1 ]]; then
  echo "usage: $0 <desired_image_ref>" >&2
  exit 1
fi

desired_ref="$1"
map_file="${FLYDSL_CI_IMAGE_MAP_FILE:-${HOME}/.cache/flydsl/ci-device-image-map.tsv}"

log() {
  printf '[resolve-local-image] %s\n' "$*" >&2
}

exact_hit() {
  local ref="$1"
  if docker image inspect "${ref}" >/dev/null 2>&1; then
    printf '%s\n' "${ref}"
    return 0
  fi
  return 1
}

find_local_by_config_id() {
  local config_id="$1"
  if [[ -z "${config_id}" ]]; then
    return 1
  fi
  if docker image inspect "${config_id}" >/dev/null 2>&1; then
    printf '%s\n' "${config_id}"
    return 0
  fi
  return 1
}

lookup_map() {
  local digest="$1"
  local d cfg localref
  if [[ ! -f "${map_file}" ]]; then
    return 1
  fi
  while IFS=$'\t' read -r d cfg localref || [[ -n "${d:-}" ]]; do
    [[ -z "${d:-}" || "${d}" == \#* ]] && continue
    if [[ "${d}" == "${digest}" ]]; then
      if find_local_by_config_id "${cfg}"; then
        return 0
      fi
      if [[ -n "${localref:-}" ]] && docker image inspect "${localref}" >/dev/null 2>&1; then
        printf '%s\n' "${localref}"
        return 0
      fi
    fi
  done <"${map_file}"
  return 1
}

config_id_from_remote_ref() {
  local ref="$1"
  IMAGE_REF="${ref}" python3 - <<'PY'
import json
import os
import subprocess
import sys

ref = os.environ["IMAGE_REF"]

def inspect_raw(image_ref: str):
    try:
        out = subprocess.run(
            ["docker", "buildx", "imagetools", "inspect", image_ref, "--raw"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout
    except Exception:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None

root = inspect_raw(ref)
if not isinstance(root, dict):
    sys.exit(1)

if isinstance(root.get("config"), dict) and isinstance(root["config"].get("digest"), str):
    print(root["config"]["digest"])
    sys.exit(0)

manifests = root.get("manifests")
if not isinstance(manifests, list):
    sys.exit(1)

repo = ref.split("@", 1)[0]
for m in manifests:
    if not isinstance(m, dict):
        continue
    platform = m.get("platform")
    if isinstance(platform, dict):
        if platform.get("os") == "unknown" and platform.get("architecture") == "unknown":
            continue
        if platform.get("os") == "linux" and platform.get("architecture") in ("amd64", "x86_64"):
            digest = m.get("digest")
            if isinstance(digest, str) and digest.startswith("sha256:"):
                child = inspect_raw(f"{repo}@{digest}")
                if isinstance(child, dict) and isinstance(child.get("config"), dict):
                    cfg = child["config"].get("digest")
                    if isinstance(cfg, str) and cfg.startswith("sha256:"):
                        print(cfg)
                        sys.exit(0)

# Fallback: first non-attestation child with config.
for m in manifests:
    if not isinstance(m, dict):
        continue
    platform = m.get("platform")
    if isinstance(platform, dict) and platform.get("os") == "unknown" and platform.get("architecture") == "unknown":
        continue
    digest = m.get("digest")
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        continue
    child = inspect_raw(f"{repo}@{digest}")
    if isinstance(child, dict) and isinstance(child.get("config"), dict):
        cfg = child["config"].get("digest")
        if isinstance(cfg, str) and cfg.startswith("sha256:"):
            print(cfg)
            sys.exit(0)

sys.exit(1)
PY
}

if ! command -v docker >/dev/null 2>&1; then
  log "docker is required"
  exit 1
fi

if exact_hit "${desired_ref}"; then
  log "exact local ref hit: ${desired_ref}"
  exit 0
fi

digest=""
repo="${desired_ref%%[:@]*}"
if [[ "${desired_ref}" == *@sha256:* ]]; then
  digest="${desired_ref##*@}"
elif [[ "${desired_ref}" == sha256:* ]]; then
  digest="${desired_ref}"
fi

# Publish action tags local images as repo:sha256-<hex> for the registry digest.
if [[ -n "${repo}" && "${digest}" == sha256:* ]]; then
  digest_tag="${repo}:sha256-${digest#sha256:}"
  if exact_hit "${digest_tag}"; then
    log "local digest-tag hit: ${digest_tag}"
    exit 0
  fi
fi

if [[ -n "${digest}" ]]; then
  if lookup_map "${digest}"; then
    log "map hit for ${digest}"
    exit 0
  fi

  # Metadata-only match: compare remote config digest to local image IDs.
  # This does not pull image layers.
  if config_id="$(config_id_from_remote_ref "${desired_ref}" 2>/dev/null)"; then
    if find_local_by_config_id "${config_id}"; then
      log "config-digest hit: ${digest} -> ${config_id}"
      exit 0
    fi
  fi
fi

# Common local aliases produced by publish dual-exporter flow.
if [[ -n "${repo}" ]]; then
  for candidate in "${repo}:runner-local" "${repo}:stable"; do
    if docker image inspect "${candidate}" >/dev/null 2>&1; then
      # Only accept alias when desired is a mutable tag of same repo, not a digest.
      if [[ -z "${digest}" ]]; then
        log "local alias hit: ${candidate}"
        printf '%s\n' "${candidate}"
        exit 0
      fi
    fi
  done
fi

log "local miss for ${desired_ref}"
exit 2
