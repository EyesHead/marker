# Keep this wrapper and ScriptOllama.py beside the PDF/Output folders.
$ErrorActionPreference = 'Stop'
$launcher = Join-Path $PSScriptRoot 'ScriptOllama.py'
if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
    Write-Error 'Missing ScriptOllama.py. Download it beside this wrapper (see README).'
    exit 1
}
if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 $launcher @args
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    & python $launcher @args
} else {
    Write-Error 'Python 3.10 or newer is required. Install Python and enable the Python launcher or PATH.'
    exit 1
}
exit $LASTEXITCODE
