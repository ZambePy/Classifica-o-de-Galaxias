"""Preenche a subpasta `non_astronomical` da classe `other`.

    # padrao: STL-10, baixado automaticamente
    python scripts/fetch_non_astronomical.py

    # ou a partir de uma pasta sua (fotos proprias, banco de imagens...)
    python scripts/fetch_non_astronomical.py --from-dir "C:/minhas_fotos" --n 800

POR QUE ESTA CLASSE EXISTE

As outras subpastas de `other` (aglomerados, campos vazios, campos
estelares) sao todas RECORTES DE CEU. Elas ensinam o modelo a separar tipos
de ceu. Nenhuma ensina que uma selfie nao e astronomia - e e selfie, print
de tela e foto de cachorro que as pessoas vao subir no dashboard.

POR QUE STL-10 E NAO CIFAR-10

CIFAR-10 seria menor e mais rapido, mas suas imagens tem 32x32 pixels.
Ampliadas para 224 viram borroes, e o modelo aprenderia o atalho "imagem
borrada = other" em vez de aprender o conteudo. STL-10 tem 96x96, que
ampliado para 224 continua nitido o bastante para nao entregar a resposta
pela resolucao.

Ainda assim, ATENCAO METODOLOGICA: fotos naturais sao coloridas e claras,
recortes de ceu sao escuros. Essa diferenca e grande e o modelo vai
aproveita-la. Nao conclua dai que ele "aprendeu a reconhecer astronomia" -
o teste honesto e a matriz de confusao entre galaxy e nebula, nao contra
esta classe. Vale um paragrafo no trabalho.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image

from astro_classifier.paths import get_paths

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def from_stl10(destino: Path, n: int, cache: Path) -> int:
    """Baixa o STL-10 e grava `n` imagens como JPG."""
    from torchvision.datasets import STL10

    print(f"Baixando STL-10 para {cache} (~2,5 GB na primeira vez)...")
    # O split 'test' tem 8000 imagens rotuladas e basta de sobra.
    dataset = STL10(root=str(cache), split="test", download=True)
    print(f"  {len(dataset)} imagens disponiveis, gravando {min(n, len(dataset))}")

    gravadas = 0
    for i in range(min(n, len(dataset))):
        imagem, _ = dataset[i]
        imagem.convert("RGB").save(destino / f"stl10_{i:05d}.jpg", quality=92)
        gravadas += 1
    return gravadas


def from_directory(origem: Path, destino: Path, n: int) -> int:
    """Copia imagens de uma pasta qualquer, validando cada uma."""
    if not origem.exists():
        raise FileNotFoundError(f"pasta nao encontrada: {origem}")

    candidatas = [p for p in origem.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES]
    print(f"  {len(candidatas)} imagens encontradas em {origem}")

    gravadas = 0
    for caminho in candidatas:
        if gravadas >= n:
            break
        try:
            with Image.open(caminho) as img:
                img.verify()  # arquivo corrompido nao entra no dataset
        except (OSError, ValueError):
            continue
        shutil.copy2(caminho, destino / f"dir_{gravadas:05d}{caminho.suffix.lower()}")
        gravadas += 1
    return gravadas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-dir", default=None, help="usa esta pasta em vez do STL-10")
    parser.add_argument("--n", type=int, default=600, help="quantas imagens gravar")
    parser.add_argument("--cache", default=None, help="onde guardar o download (padrao: raw/_cache)")
    args = parser.parse_args()

    paths = get_paths().ensure()
    destino = paths.raw_other / "non_astronomical"
    destino.mkdir(parents=True, exist_ok=True)

    existentes = [p for p in destino.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    if existentes:
        print(f"{len(existentes)} imagens ja existem em {destino}. Apague-as para refazer.")
        return 0

    cache = Path(args.cache) if args.cache else paths.raw / "_cache"
    cache.mkdir(parents=True, exist_ok=True)

    try:
        if args.from_dir:
            gravadas = from_directory(Path(args.from_dir), destino, args.n)
        else:
            gravadas = from_stl10(destino, args.n, cache)
    except Exception as exc:  # noqa: BLE001 - a mensagem importa mais que o tipo
        print(f"[ERRO] {exc}")
        print(
            "\nAlternativa sem download: aponte uma pasta sua com\n"
            '  python scripts/fetch_non_astronomical.py --from-dir "C:/caminho/fotos"'
        )
        return 1

    print(f"\n{gravadas} imagens gravadas em {destino}")
    print("Regenere os splits do nivel 1: python scripts/make_splits.py --level object ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
