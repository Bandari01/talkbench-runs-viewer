#!/usr/bin/env bash
set -euo pipefail

viewer_port="${VIEWER_PORT:-8377}"
viewer_host="127.0.0.1"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "cloudflared is required. Install it with: brew install cloudflared" >&2
  exit 1
fi

uv sync
VIEWER_READ_ONLY=true VIEWER_ALLOW_SYNC=true \
  uv run uvicorn app.main:app --host "$viewer_host" --port "$viewer_port" &
viewer_pid=$!

cleanup() {
  kill "$viewer_pid" 2>/dev/null || true
  wait "$viewer_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in {1..50}; do
  if curl --silent --fail "http://${viewer_host}:${viewer_port}/api/config" >/dev/null; then
    break
  fi
  if ! kill -0 "$viewer_pid" 2>/dev/null; then
    wait "$viewer_pid"
  fi
  sleep 0.2
done

if ! curl --silent --fail "http://${viewer_host}:${viewer_port}/api/config" >/dev/null; then
  echo "viewer did not start on port ${viewer_port}" >&2
  exit 1
fi

cloudflared tunnel --no-autoupdate --url "http://${viewer_host}:${viewer_port}"
