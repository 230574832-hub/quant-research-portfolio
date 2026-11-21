$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
& "$ProjectRoot\.venv\Scripts\python.exe" "$ProjectRoot\run_local.py" preflight
