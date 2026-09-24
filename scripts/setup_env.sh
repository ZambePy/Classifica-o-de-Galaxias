#!/usr/bin/env bash
# Equivalente do setup_env.ps1 para Linux/macOS (util se o projeto rodar em
# Colab, num servidor ou na maquina do seu colega).
#
#   bash scripts/setup_env.sh
set -euo pipefail

TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Criando ambiente virtual em .venv"
[ -d .venv ] || python3 -m venv .venv
PYTHON=".venv/bin/python"

echo "==> Atualizando pip"
"$PYTHON" -m pip install --upgrade pip --quiet

echo "==> Instalando PyTorch (indice: $TORCH_INDEX)"
"$PYTHON" -m pip install torch torchvision --index-url "$TORCH_INDEX"

echo "==> Instalando dependencias"
"$PYTHON" -m pip install -r requirements-dev.txt

echo "==> Instalando o pacote em modo editavel"
"$PYTHON" -m pip install -e . --no-deps

[ -f .env ] || { echo "==> Criando .env"; cp .env.example .env; }

echo "==> Criando arvore de dados"
"$PYTHON" -c "from astro_classifier.paths import get_paths; p = get_paths().ensure(); print('    dados em:', p.root)"

echo "==> Verificando a GPU"
"$PYTHON" -c "import torch; print('    torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"

echo
echo "Pronto. Ative com: source .venv/bin/activate"
