#!/usr/bin/env bash
# Serve /srv/artifacts only (read-only). Does not mount /home/flydsl.
set -euo pipefail

WHEEL_DIR="${FLYDSL_ILUVATAR_WHEEL_ARTIFACT_DIR:-/srv/artifacts/wheels/flydsl}"
ROOT="$(cd "${WHEEL_DIR}/../.." && pwd)"
PORT="${FLYDSL_ILUVATAR_WHEEL_HTTP_PORT:-8080}"
NAME="${FLYDSL_ILUVATAR_WHEEL_HTTP_NAME:-flydsl-iluvatar-wheel-http}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF="${SCRIPT_DIR}/iluvatar_wheel_http.conf"

if [[ ! -d "${WHEEL_DIR}" ]]; then
  echo "artifact dir missing: ${WHEEL_DIR}" >&2
  exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required" >&2
  exit 1
fi

docker rm -f "${NAME}" >/dev/null 2>&1 || true
docker run -d \
  --name "${NAME}" \
  --restart unless-stopped \
  -p "${PORT}:80" \
  -v "${ROOT}:/usr/share/nginx/html:ro" \
  -v "${CONF}:/etc/nginx/conf.d/default.conf:ro" \
  nginx:latest

echo "Serving ${ROOT} on port ${PORT} (read-only)"
echo "CI: python3 -m pip install --pre --no-deps flydsl -i http://<host>:${PORT}/simple"
echo "or: python3 -m pip install --pre --no-deps flydsl --find-links http://<host>:${PORT}/wheels/flydsl/ --no-index"
