$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw "nvidia-smi was not found. Install/update the NVIDIA driver before creating the research environment."
}
$Driver = (& nvidia-smi --query-gpu=driver_version --format=csv,noheader | Select-Object -First 1).Trim()
if ([version]$Driver -lt [version]"528.33") {
    throw "NVIDIA driver $Driver is below the CUDA 12.x compatibility floor (528.33 on Windows). Update it first."
}
Write-Host "NVIDIA driver: $Driver"
& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher 'py' was not found. Install 64-bit Python 3.11 from python.org."
}
& py -3.11 -c "import sys; assert sys.version_info[:2] == (3, 11); print(sys.version)"
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & py -3.11 -m venv .venv
}
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $Python -m pip install --upgrade "pip==25.2" "setuptools==80.9.0" "wheel==0.45.1"
& $Python -m pip install "torch==2.12.1" --index-url https://download.pytorch.org/whl/cu126
& $Python -m pip install -r requirements-local.txt
& $Python -m pip install -e .
& $Python -m pip check
& $Python scripts\check_gpu.py --output environment_preflight.json
Write-Host "Environment ready. Next: .\scripts\run_preflight.ps1"
