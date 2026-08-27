param(
    [string]$Python = "python",
    [switch]$WithModel,
    [string]$ModelPath = "models/Qwen2.5-1.5B",
    [string]$AdapterPath = "models/lora_adapter"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "[1/5] Python environment"
& $Python --version

Write-Host "[2/5] Compile EdgeOps modules"
& $Python -m compileall -q edgeops edgeops_cli.py edgeops_api.py scripts

Write-Host "[3/5] Build structured SFT seed data"
& $Python scripts/build_edgeops_sft.py

Write-Host "[4/5] Run no-model safety and tool checks"
& $Python edgeops_cli.py --query "AMR-07 为什么停止运行？" --compact
& $Python edgeops_cli.py --query "让 AMR-07 立即移动到 B2" --compact
& $Python edgeops_cli.py --query "FORKLIFT-12 在哪里？" --compact

Write-Host "[5/5] Optional model-backed RAG check"
if ($WithModel) {
    & $Python edgeops_cli.py `
        --query "AMR 电量低于 20% 时应该怎么处理？" `
        --with-model `
        --model-path $ModelPath `
        --adapter-path $AdapterPath `
        --quantization int4
} else {
    Write-Host "Skipped. Use -WithModel after placing the model files."
}

Write-Host "EdgeOps verification completed."
