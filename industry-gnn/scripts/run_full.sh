#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test "${1:-}" = "--confirm-full-run" || { echo "Pass --confirm-full-run after Stage A-C pass." >&2; exit 2; }
"$ROOT/.venv/bin/python" "$ROOT/run_local.py" full --confirm-full-run
