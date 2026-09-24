# Sobe a API de inferencia.
#
#   powershell -ExecutionPolicy Bypass -File scripts\serve_api.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\serve_api.ps1 -Mode real
#
# mock = responde sem modelo treinado (modo do dia 1, para o dashboard)
# real = carrega os checkpoints da cascata

param(
    [ValidateSet("mock", "real")]
    [string]$Mode = "mock",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$env:ASTRO_API_MODE = $Mode

Write-Host "API em modo '$Mode' na porta $Port" -ForegroundColor Cyan
Write-Host "Documentacao interativa: http://127.0.0.1:$Port/docs" -ForegroundColor DarkGray
Write-Host ""

& ".venv\Scripts\python.exe" -m uvicorn astro_classifier.api.main:app --reload --port $Port
