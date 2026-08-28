#!/usr/bin/env bash
# Create/replace the Iluvatar weekly GitHub Release with a stable asset name.
set -euo pipefail

DIST_DIR="${1:?usage: publish_iluvatar_weekly_release.sh <dist-dir> [source-ref] [channel]}"
SOURCE_REF="${2:-iluvatar}"
CHANNEL="${3:-weekly}"
REPO="${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"

if ! command -v gh >/dev/null 2>&1; then
  echo "gh is required" >&2
  exit 1
fi

mapfile -t wheels < <(find "${DIST_DIR}" -type f -name 'flydsl-*.whl' ! -name 'flydsl-iluvatar-*-manylinux_x86_64.whl' | sort)
if [[ ${#wheels[@]} -eq 0 ]]; then
  echo "No flydsl wheels found under ${DIST_DIR}" >&2
  exit 1
fi

stable_dir="${DIST_DIR}/stable-alias"
mkdir -p "${stable_dir}"
assets=()
stable_names=()
for whl in "${wheels[@]}"; do
  assets+=("${whl}")
  base="$(basename "${whl}")"
  if [[ "${base}" =~ -cp([0-9]+)- ]]; then
    stable_name="flydsl-iluvatar-cp${BASH_REMATCH[1]}-manylinux_x86_64.whl"
    cp -f "${whl}" "${stable_dir}/${stable_name}"
    assets+=("${stable_dir}/${stable_name}")
    stable_names+=("${stable_name}")
  fi
done
if [[ ${#stable_names[@]} -eq 0 ]]; then
  echo "Could not derive cp tag from wheel names: ${wheels[*]}" >&2
  exit 1
fi

flydsl_commit=""
manifest="$(find "${DIST_DIR}" -type f -name 'build-manifest.json' | head -n 1 || true)"
if [[ -n "${manifest}" ]]; then
  flydsl_commit="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("flydsl_commit",""))' "${manifest}")"
fi

if [[ "${CHANNEL}" == "ci" ]]; then
  tag="iluvatar-ci"
  title="Iluvatar CI wheel"
  kind_blurb="CI wheel for downstream CI (Ubuntu 20.04 / cp312 by default). Smoke-tested only; not a full-function/perf release."
  extra_flags=(--prerelease)
  install_prefix="${GITHUB_SERVER_URL}/${REPO}/releases/download/${tag}"
else
  tag="iluvatar-weekly-$(date -u +%F)"
  title="Iluvatar weekly wheel (${tag#iluvatar-weekly-})"
  kind_blurb="Weekly wheels: Ubuntu 20.04 (default cp312; optional cp310) and Ubuntu 24.04 cp312. u2404 cp312 runs the fuller device suite."
  extra_flags=(--latest)
  install_prefix="${GITHUB_SERVER_URL}/${REPO}/releases/latest/download"
fi
notes="$(mktemp)"
{
  echo "${kind_blurb}"
  echo
  echo "- Channel: \`${CHANNEL}\`"
  echo "- Source ref: \`${SOURCE_REF}\`"
  if [[ -n "${flydsl_commit}" ]]; then
    echo "- FlyDSL commit: \`${flydsl_commit}\`"
  fi
  echo "- Workflow run: ${GITHUB_SERVER_URL}/${REPO}/actions/runs/${GITHUB_RUN_ID}"
  echo
  echo "Original wheels:"
  for whl in "${wheels[@]}"; do
    echo "- \`$(basename "${whl}")\`"
  done
  echo
  echo "Install:"
  echo
  echo '```bash'
  for name in "${stable_names[@]}"; do
    echo "pip install ${install_prefix}/${name}"
  done
  echo '```'
} > "${notes}"

if gh release view "${tag}" --repo "${REPO}" >/dev/null 2>&1; then
  gh release delete "${tag}" --repo "${REPO}" --yes --cleanup-tag
fi

gh release create "${tag}" \
  --repo "${REPO}" \
  --title "${title}" \
  --notes-file "${notes}" \
  "${extra_flags[@]}" \
  "${assets[@]}"

echo "Published ${tag}"
printf 'Stable assets: %s\n' "${stable_names[*]}"
