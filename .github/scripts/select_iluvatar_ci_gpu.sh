#!/usr/bin/env bash
# Select one idle Iluvatar GPU from the CI allow-list.
#
# Prints a single GPU index to stdout. Intended for:
#   export CUDA_VISIBLE_DEVICES="$(bash .github/scripts/select_iluvatar_ci_gpu.sh)"
#
# Env:
#   FLYDSL_CI_GPU_POOL   comma-separated physical indices (optional; script has a default)
#   CUDA_VISIBLE_DEVICES if already set to a single index in the pool, keep it
#   COREX_ROOT           used to locate ixsmi when not on PATH
#
# Idle metric: lowest memory.used MiB among the pool (via ixsmi --query-gpu).
# Fallback if ixsmi unavailable: first index in the pool.

set -euo pipefail

POOL_CSV="${FLYDSL_CI_GPU_POOL:-0,1,3}"

if [[ -n "${COREX_ROOT:-}" && -x "${COREX_ROOT}/bin/ixsmi" ]]; then
  export PATH="${COREX_ROOT}/bin:${PATH}"
  export LD_LIBRARY_PATH="${COREX_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
fi

IFS=',' read -r -a POOL <<<"${POOL_CSV}"
declare -A ALLOWED=()
ORDERED=()
for raw in "${POOL[@]}"; do
  idx="${raw//[[:space:]]/}"
  [[ -z "${idx}" ]] && continue
  if [[ ! "${idx}" =~ ^[0-9]+$ ]]; then
    echo "::error::invalid GPU index in FLYDSL_CI_GPU_POOL: ${idx}" >&2
    exit 1
  fi
  ALLOWED["${idx}"]=1
  ORDERED+=("${idx}")
done

if [[ "${#ORDERED[@]}" -eq 0 ]]; then
  echo "::error::FLYDSL_CI_GPU_POOL is empty" >&2
  exit 1
fi

# Honor an explicit single-device override when it is inside the pool.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  existing="${CUDA_VISIBLE_DEVICES%%,*}"
  existing="${existing//[[:space:]]/}"
  if [[ -n "${ALLOWED[${existing}]+x}" ]]; then
    echo "${existing}"
    exit 0
  fi
  echo "::warning::CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} is outside pool [${POOL_CSV}]; re-selecting" >&2
fi

pick_first() {
  echo "${ORDERED[0]}"
}

if ! command -v ixsmi >/dev/null 2>&1; then
  echo "::warning::ixsmi not found; falling back to GPU ${ORDERED[0]}" >&2
  pick_first
  exit 0
fi

csv="$(ixsmi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null || true)"
if [[ -z "${csv}" ]]; then
  echo "::warning::ixsmi query failed; falling back to GPU ${ORDERED[0]}" >&2
  pick_first
  exit 0
fi

best_idx=""
best_used=""
while IFS=',' read -r idx used; do
  idx="${idx//[[:space:]]/}"
  used="${used//[[:space:]]/}"
  [[ -z "${idx}" || -z "${used}" ]] && continue
  [[ -z "${ALLOWED[${idx}]+x}" ]] && continue
  if [[ -z "${best_idx}" ]] || (( used < best_used )); then
    best_idx="${idx}"
    best_used="${used}"
  fi
done <<<"${csv}"

if [[ -z "${best_idx}" ]]; then
  echo "::warning::no pool GPU appeared in ixsmi output; falling back to GPU ${ORDERED[0]}" >&2
  pick_first
  exit 0
fi

echo "::notice::selected CI GPU ${best_idx} from pool [${POOL_CSV}] (memory.used=${best_used} MiB)" >&2
echo "${best_idx}"
