param([switch]$ConfirmFullRun)
$ErrorActionPreference = "Stop"
if (-not $ConfirmFullRun) { throw "Re-run with -ConfirmFullRun after Stage A-C pass." }
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
& "$ProjectRoot\.venv\Scripts\python.exe" "$ProjectRoot\run_local.py" full --confirm-full-run
