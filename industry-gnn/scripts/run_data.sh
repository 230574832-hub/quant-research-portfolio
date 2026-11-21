#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/.venv/bin/python" "$ROOT/run_local.py" data --mode FULL_RUN
