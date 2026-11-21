#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test "$#" -gt 0 || { echo "Usage: $0 MODEL [MODEL ...]" >&2; exit 2; }
"$ROOT/.venv/bin/python" "$ROOT/run_local.py" train --mode FULL_RUN --confirm-full-run --models "$@"
