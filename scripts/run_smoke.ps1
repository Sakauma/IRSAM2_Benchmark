param(
  [string]$Config = "configs/benchmark_smoke.yaml",
  [string]$Baseline = "bbox_rect"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = (Join-Path $root "src")
python main.py run baseline --config $Config --baseline $Baseline
