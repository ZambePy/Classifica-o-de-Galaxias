# Confere se a máquina está pronta para integrar o site Cassyn com o classificador do colega.
# Uso (na pasta site\dashboard):  npm run verificar
# Só lê: não instala, não apaga e não muda nada.
#
# Estrutura esperada:
#   D:\cassyn\classificador\          <- git clone do repositório do colega (abra o Claude Code aqui)
#     site\                           <- o nosso site (vai para o repositório no commit)
#     CLAUDE.md, .claude\, AUDITORIA.md, integracao\   <- só locais (escondidos do git)
#   D:\cassyn\modelos\checkpoints\    <- pacote dos modelos, fora do repositório
param([string]$Modelos = '')
$ErrorActionPreference = 'Continue'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$site = Join-Path $repo 'site'
$dash = Join-Path $site 'dashboard'
if (-not $Modelos) { $Modelos = Join-Path (Split-Path $repo -Parent) 'modelos' }
$checkpoints = Join-Path $Modelos 'checkpoints'
$script:erros = 0
$script:avisos = 0

function Ok($t) { Write-Host "[ok] $t" -ForegroundColor Green }
function Falha($t, $dica) {
  Write-Host "[x]  $t" -ForegroundColor Red
  if ($dica) { Write-Host "     $dica" -ForegroundColor Yellow }
  $script:erros++
}
function Aviso($t) { Write-Host "[!]  $t" -ForegroundColor Yellow; $script:avisos++ }
function Titulo($t) { Write-Host ''; Write-Host $t -ForegroundColor Cyan }

Write-Host 'Cassyn - verificação do ambiente de integração' -ForegroundColor Cyan
Write-Host "Repositório: $repo"
if ($repo -match 'OneDrive|Dropbox|Google Drive') {
  Aviso 'A pasta está dentro de uma nuvem (OneDrive/Dropbox). Prefira uma pasta fora da nuvem, como D:\cassyn'
}

Titulo '1. Pastas'
if ((Test-Path (Join-Path $repo '.git')) -and (Test-Path (Join-Path $repo 'src\astro_classifier'))) { Ok 'Git clone do repositório do colega' }
else { Falha 'A pasta site não está dentro do git clone do colega' 'A pasta site precisa ficar dentro da pasta classificador (o git clone).' }
if (Test-Path (Join-Path $dash 'package.json')) { Ok 'site\dashboard' } else { Falha 'site\dashboard não encontrado' }
if (Test-Path (Join-Path $repo 'CLAUDE.md')) { Ok 'CLAUDE.md (regras para o Claude Code)' } else { Falha 'CLAUDE.md não está na raiz do repositório' }
if (Test-Path (Join-Path $repo '.claude\settings.json')) { Ok '.claude\settings.json (trava de commit)' } else { Falha '.claude\settings.json não está na raiz do repositório' 'Sem ele a trava de commit não funciona.' }

Titulo '2. Node.js'
$node = Get-Command node -ErrorAction SilentlyContinue
if ($node) {
  $v = (& node --version).TrimStart('v')
  if ([version]$v -ge [version]'22.9.0') { Ok "Node $v" } else { Falha "Node $v é antigo" 'Instale o Node 22 LTS ou mais novo (nodejs.org).' }
} else { Falha 'Node.js não instalado' 'Instale o Node 22 LTS (nodejs.org).' }
if (Test-Path (Join-Path $dash 'node_modules')) { Ok 'Dependências do site instaladas' } else { Aviso 'Dependências do site faltando: rode "npm install" na pasta site\dashboard' }

Titulo '3. Python do classificador'
$pyRepo = Join-Path $repo '.venv\Scripts\python.exe'
if (Test-Path $pyRepo) {
  $v = (& $pyRepo -c "import sys; print('%d.%d.%d' % sys.version_info[:3])") 2>$null
  if ($v -and [version]$v -ge [version]'3.10.0') { Ok "Python $v (.venv)" } else { Falha "Python $v no .venv (precisa 3.10+)" }
  $libs = (& $pyRepo -c "import torch, torchvision, fastapi, uvicorn; print(torch.__version__, 'GPU' if torch.cuda.is_available() else 'CPU')") 2>$null
  if ($LASTEXITCODE -eq 0 -and $libs) { Ok "PyTorch e FastAPI instalados ($libs)" }
  else { Falha 'Faltam bibliotecas no .venv (torch, torchvision, fastapi ou uvicorn)' "Na pasta $repo rode: powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1" }
} else {
  $sys = Get-Command py -ErrorAction SilentlyContinue
  if (-not $sys) { $sys = Get-Command python -ErrorAction SilentlyContinue }
  if ($sys) { Aviso "Python encontrado ($($sys.Source)), mas o .venv do classificador ainda não existe" } else { Falha 'Python não instalado' 'Instale o Python 3.11 (python.org), marcando "Add to PATH".' }
  Falha 'Ambiente do classificador não criado' "Na pasta $repo rode: powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1"
}

Titulo '4. Pacote dos modelos'
Write-Host "     Pasta: $checkpoints"
foreach ($f in 'object_best.pt', 'galaxy_best.pt', 'nebula_best.pt', 'ood_threshold.json') {
  $p = Join-Path $checkpoints $f
  if (Test-Path $p) {
    $mb = [math]::Round((Get-Item $p).Length / 1MB, 1)
    Ok "$f ($mb MB)"
  } else { Falha "$f não encontrado" "Coloque o pacote do colega em $checkpoints" }
}
$ood = Join-Path $checkpoints 'ood_threshold.json'
if (Test-Path $ood) {
  try { $null = Get-Content $ood -Raw | ConvertFrom-Json; Ok 'ood_threshold.json é um JSON válido' } catch { Falha 'ood_threshold.json está corrompido' }
}

Titulo '5. Portas'
foreach ($porta in 3001, 5173, 8000) {
  $uso = Get-NetTCPConnection -LocalPort $porta -State Listen -ErrorAction SilentlyContinue
  if (-not $uso) { Ok "Porta $porta livre"; continue }
  if ($porta -eq 8000) {
    try {
      $h = Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 3
      Ok "Porta 8000 já é o classificador (modo $($h.mode), contrato $($h.contract_version))"
    } catch { Aviso 'Porta 8000 ocupada por outro programa' }
  } else { Aviso "Porta $porta ocupada (talvez o site já esteja rodando)" }
}

Titulo '6. Git (só leitura)'
if ((Test-Path (Join-Path $repo '.git')) -and (Get-Command git -ErrorAction SilentlyContinue)) {
  # arquivos do colega que foram alterados (a pasta site é nossa; o README.md só ganha a seção "Como executar")
  $mudou = & git -C $repo status --porcelain --untracked-files=no | Where-Object { $_ -notmatch '^.{3}"?(site/|README\.md$)' }
  if ($mudou) { Aviso 'Há arquivos do colega alterados (o combinado é não mexer neles):'; $mudou | ForEach-Object { Write-Host "     $_" } }
  else { Ok 'Nenhum arquivo do colega foi alterado' }
  # arquivos só locais precisam estar escondidos do git
  $visiveis = & git -C $repo status --porcelain --untracked-files=all -- CLAUDE.md .claude AUDITORIA.md integracao site/branding site/dashboard/client/public/logos
  if ($visiveis) { Falha 'Arquivos que não vão para o repositório aparecem para o git (CLAUDE.md, .claude, AUDITORIA.md, integracao ou logos)' 'Rode de novo o bloco "Esconder do git" em integracao\LEIA-ME.md.' }
  else { Ok 'Arquivos locais, marca e logos estão escondidos do git' }
} else { Aviso 'Não consegui checar o git (git não instalado?)' }

Write-Host ''
if ($script:erros -eq 0) {
  Write-Host "Pronto para integrar ($script:avisos aviso(s))." -ForegroundColor Green
  exit 0
}
Write-Host "Faltam $script:erros item(ns) e há $script:avisos aviso(s). Corrija os itens [x] acima." -ForegroundColor Red
exit 1
