# Liga a API do classificador (FastAPI do colega) para o site Cassyn usar.
# Uso (na pasta site\dashboard):  npm run dev:classificador
#                                  npm run dev:classificador -- -Modo real
#
# Estrutura esperada:
#   D:\cassyn\classificador\          <- repositório do colega (git clone); o site fica em site\
#   D:\cassyn\modelos\checkpoints\   <- pacote dos modelos, FORA do repositório (como o colega recomenda)
# Outra pasta de modelos:  npm run dev:classificador -- -Modo real -Modelos E:\meus-modelos
#
# - Modo mock: respostas simuladas, não precisa dos modelos (bom para testar o site).
# - Modo real: usa object_best.pt, galaxy_best.pt, nebula_best.pt e ood_threshold.json.
# - Escuta só em 127.0.0.1 (a API não fica aberta para a rede) e sem --reload.
# - Não altera nenhum arquivo do repositório.
param(
  [ValidateSet('mock', 'real')][string]$Modo = 'mock',
  [int]$Porta = 8000,
  [string]$Modelos = ''
)
$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
if (-not $Modelos) { $Modelos = Join-Path (Split-Path $repo -Parent) 'modelos' }
$checkpoints = Join-Path $Modelos 'checkpoints'
$py = Join-Path $repo '.venv\Scripts\python.exe'

if (-not (Test-Path (Join-Path $repo 'src\astro_classifier'))) {
  Write-Host "[x] Não achei o código do colega (src\astro_classifier) em $repo" -ForegroundColor Red
  Write-Host "    A pasta site precisa estar dentro do git clone do repositório do colega." -ForegroundColor Yellow
  exit 1
}
if (-not (Test-Path $py)) {
  Write-Host "[x] O ambiente Python do classificador ainda não existe ($py)." -ForegroundColor Red
  Write-Host "    Na pasta $repo rode: powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1" -ForegroundColor Yellow
  exit 1
}
if ($Modo -eq 'real') {
  $faltando = @('object_best.pt', 'galaxy_best.pt', 'nebula_best.pt', 'ood_threshold.json') |
    Where-Object { -not (Test-Path (Join-Path $checkpoints $_)) }
  if ($faltando) {
    Write-Host "[x] Faltam arquivos do pacote dos modelos em $checkpoints :" -ForegroundColor Red
    $faltando | ForEach-Object { Write-Host "    - $_" }
    exit 1
  }
}

$env:ASTRO_API_MODE = $Modo
$env:ASTRO_DATA_ROOT = $Modelos
Write-Host "Classificador em modo $Modo, em http://127.0.0.1:$Porta (documentação em /docs)" -ForegroundColor Cyan
Write-Host "Modelos: $checkpoints"
Push-Location $repo
try {
  & $py -m uvicorn astro_classifier.api.main:app --app-dir src --host 127.0.0.1 --port $Porta
} finally {
  Pop-Location
}
