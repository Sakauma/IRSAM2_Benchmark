# Author: Egor Izmaylov
#
# 这个脚本用于在 Windows / PowerShell 环境下快速执行一轮 smoke baseline。
# 默认读取 smoke 配置，并把 src 目录注入到 PYTHONPATH，避免用户先手工配置环境。

param(
  [string]$Config = "configs/benchmark_smoke.yaml",
  [string]$Baseline = "bbox_rect"
)

$ErrorActionPreference = "Stop"

# 以脚本所在目录的父目录作为项目根目录，避免从任意位置调用时路径出错。
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# 让 python main.py 可以直接解析到 src 包。
$env:PYTHONPATH = (Join-Path $root "src")
python main.py run baseline --config $Config --baseline $Baseline
