"""Install the pinned GPU stack into the Conda interpreter selected by VS Code.

Open this file in VS Code and click the top-right "Run Python File" button.
Do not run it with the Conda base environment.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(*arguments: str) -> None:
    print("\n>", " ".join(arguments), flush=True)
    subprocess.check_call(list(arguments), cwd=PROJECT_ROOT)


conda_prefix = os.environ.get("CONDA_PREFIX")
conda_name = os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.prefix).name
is_conda = bool(conda_prefix) or (Path(sys.prefix) / "conda-meta").is_dir()
if not is_conda:
    raise RuntimeError(
        "The selected VS Code interpreter is not a Conda environment. Select the 'industry-gnn' interpreter first."
    )
if conda_name.lower() in {"base", "miniconda3", "anaconda3"}:
    raise RuntimeError("Refusing to install into Conda base. Select or create the 'industry-gnn' environment.")
if sys.version_info[:2] != (3, 11):
    raise RuntimeError(f"Python 3.11 is required; selected interpreter is {sys.version.split()[0]}.")

print("Selected Conda environment:", conda_name)
print("Python:", sys.version.split()[0])
print("Installing into:", Path(sys.executable).name)

run(sys.executable, "-m", "pip", "install", "--upgrade", "pip==25.2", "setuptools==80.9.0", "wheel==0.45.1")
run(
    sys.executable, "-m", "pip", "install", "torch==2.12.1",
    "--index-url", "https://download.pytorch.org/whl/cu126",
)
run(sys.executable, "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements.txt"))
run(sys.executable, "-m", "pip", "install", "-e", str(PROJECT_ROOT))
run(sys.executable, "-m", "pip", "check")
run(sys.executable, str(PROJECT_ROOT / "scripts" / "check_gpu.py"), "--output", str(PROJECT_ROOT / "environment_preflight.json"))

print("\nSETUP PASS. Select this same environment as the notebook kernel, then run the Local notebook.")
