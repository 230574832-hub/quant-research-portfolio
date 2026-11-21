#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found; install/update the NVIDIA driver first." >&2; exit 2; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
command -v python3.11 >/dev/null || { echo "Python 3.11 is required." >&2; exit 2; }
python3.11 -c 'import sys; assert sys.version_info[:2] == (3, 11); print(sys.version)'
test -x .venv/bin/python || python3.11 -m venv .venv
PYTHON="$PROJECT_ROOT/.venv/bin/python"
"$PYTHON" -m pip install --upgrade 'pip==25.2' 'setuptools==80.9.0' 'wheel==0.45.1'
"$PYTHON" -m pip install 'torch==2.12.1' --index-url https://download.pytorch.org/whl/cu126
"$PYTHON" -m pip install -r requirements-local.txt
"$PYTHON" -m pip install -e .
"$PYTHON" -m pip check
"$PYTHON" scripts/check_gpu.py --output environment_preflight.json
echo "Environment ready. Next: ./scripts/run_preflight.sh"
