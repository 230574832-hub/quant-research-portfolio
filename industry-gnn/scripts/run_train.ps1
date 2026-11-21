param([Parameter(Mandatory=$true)][string[]]$Models)
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
& "$ProjectRoot\.venv\Scripts\python.exe" "$ProjectRoot\run_local.py" train --mode FULL_RUN --confirm-full-run --models $Models
