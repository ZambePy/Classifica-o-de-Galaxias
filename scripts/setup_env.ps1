# Prepara o ambiente de desenvolvimento no Windows.
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
#
# O que faz:
#   1. cria o venv em .venv
#   2. instala PyTorch com suporte CUDA (indice separado - por isso nao esta
#      no requirements.txt)
#   3. instala o resto das dependencias
#   4. instala o projeto em modo editavel
#   5. cria o .env e a arvore de dados FORA do OneDrive

$ErrorActionPreference = "Stop"

# Build CUDA do PyTorch. A GTX 1660 Ti (Turing, compute 7.5) roda qualquer
# build cu12x. Se der erro de driver, troque para cu124 ou use "cpu".
$TorchIndex = "https://download.pytorch.org/whl/cu128"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "==> Criando ambiente virtual em .venv" -ForegroundColor Cyan
if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

Write-Host "==> Atualizando pip" -ForegroundColor Cyan
& $Python -m pip install --upgrade pip --quiet

Write-Host "==> Instalando PyTorch (CUDA)" -ForegroundColor Cyan
Write-Host "    indice: $TorchIndex"
Write-Host "    isto baixa ~2.5 GB e demora. Va tomar um cafe." -ForegroundColor DarkGray
& $Python -m pip install torch torchvision --index-url $TorchIndex

Write-Host "==> Instalando dependencias do projeto" -ForegroundColor Cyan
& $Python -m pip install -r requirements-dev.txt

Write-Host "==> Instalando o pacote em modo editavel" -ForegroundColor Cyan
& $Python -m pip install -e . --no-deps

if (-not (Test-Path ".env")) {
    Write-Host "==> Criando .env a partir de .env.example" -ForegroundColor Cyan
    Copy-Item ".env.example" ".env"
}

Write-Host "==> Criando arvore de dados" -ForegroundColor Cyan
& $Python -c "from astro_classifier.paths import get_paths; p = get_paths().ensure(); print('    dados em:', p.root)"

Write-Host "`n==> Verificando a GPU" -ForegroundColor Cyan
& $Python -c "import torch; print('    torch', torch.__version__); print('    CUDA disponivel:', torch.cuda.is_available()); print('    GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'nenhuma')"

Write-Host "`nPronto." -ForegroundColor Green
Write-Host "Ative o ambiente com:  .\.venv\Scripts\Activate.ps1"
Write-Host "Suba a API mock com:   uvicorn astro_classifier.api.main:app --reload"
