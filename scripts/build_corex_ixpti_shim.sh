#!/usr/bin/env bash
# Build libixpti_shim.so: export ixpti* jumping to cupti* in CoreX libcupti.
set -euo pipefail
# 默认仅使用 ~/sw_home/local/corex（勿指向 /usr/local/corex-*）
COREX_LIB="${COREX_LIB:-${HOME}/sw_home/local/corex/lib64}"
OUT="${1:-$(cd "$(dirname "$0")/.." && pwd)/build-fly/libixpti_shim.so}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

SYMS=(
  ActivityDisable ActivityEnable ActivityFlushAll ActivityGetNextRecord
  ActivityGetNumDroppedRecords ActivityPopExternalCorrelationId
  ActivityPushExternalCorrelationId ActivityRegisterCallbacks ActivitySetAttribute
  EnableCallback EnableDomain Finalize GetResultString GetVersion Subscribe Unsubscribe
)

{
  echo '.text'
  for s in "${SYMS[@]}"; do
    echo ".globl ixpti${s}"
    echo "ixpti${s}:"
    echo "    jmp cupti${s}@PLT"
  done
} >"${TMP}/ixpti_shim.S"

gcc -shared -fPIC -o "${OUT}" "${TMP}/ixpti_shim.S" \
  -L"${COREX_LIB}" -Wl,--no-as-needed -lcupti -Wl,-rpath,"${COREX_LIB}"
echo "Built ${OUT}"
nm -D "${OUT}" | rg ' ixptiActivityEnable$' || true
