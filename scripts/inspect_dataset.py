"""Monta grades de imagens por classe, para olhar antes de treinar.

    # todas as classes encontradas em raw/
    python scripts/inspect_dataset.py

    # so as nebulosas
    python scripts/inspect_dataset.py --class-dir nebulae

    # a partir de um CSV de indice (Galaxy Zoo)
    python scripts/inspect_dataset.py --from-index galaxies_index.csv

POR QUE ISTO NAO E OPCIONAL

A coleta por coordenada e cega: ela baixa o pedaco de ceu que o catalogo
aponta, sem verificar se o objeto aparece. Os modos de falha sao silenciosos
e todos produzem uma imagem valida, que o treino aceita sem reclamar:

  - recorte VAZIO       objeto fraco demais para o levantamento usado
  - campo ERRADO        fov grande demais (objeto virou um ponto) ou
                        pequeno demais (so o miolo, sem contexto)
  - objeto DESLOCADO    coordenada do catalogo imprecisa
  - listra/artefato     borda de placa fotografica do DSS

Treinar em cima disso produz um modelo que parece funcionar e nao funciona.
Descobrir com uma grade de 50 imagens custa dois minutos; descobrir depois
custa um trabalho inteiro.

O que procurar em cada grade: o objeto aparece? esta centralizado? o tamanho
esta razoavel? as imagens de uma mesma classe se parecem entre si mais do
que com as de outra classe?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from astro_classifier.paths import get_paths

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def build_grid(
    image_paths: list[Path],
    titulo: str,
    out_path: Path,
    cols: int = 10,
    thumb_px: int = 128,
) -> Path:
    """Grade com miniaturas. Imagens ilegiveis viram um quadro vermelho em
    vez de derrubar o script - saber que existem e parte do diagnostico."""
    rows = int(np.ceil(len(image_paths) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.5))
    axes = np.atleast_1d(axes).ravel()

    quebradas = 0
    for ax, path in zip(axes, image_paths, strict=False):
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img.thumbnail((thumb_px, thumb_px))
                ax.imshow(img)
        except OSError:
            ax.set_facecolor("#c0392b")
            quebradas += 1
        ax.set_title(path.stem[:16], fontsize=5)
        ax.axis("off")

    for ax in axes[len(image_paths) :]:
        ax.axis("off")

    sufixo = f"  ({quebradas} ilegiveis)" if quebradas else ""
    fig.suptitle(f"{titulo} - {len(image_paths)} amostras{sufixo}", fontsize=12)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def sample(paths: list[Path], n: int, seed: int) -> list[Path]:
    rng = np.random.default_rng(seed)
    if len(paths) <= n:
        return sorted(paths)
    idx = rng.choice(len(paths), size=n, replace=False)
    return sorted(paths[i] for i in idx)


def collect_from_folders(root: Path, class_dir: str | None) -> dict[str, list[Path]]:
    """Agrupa imagens por pasta de classe, no padrao raw/<grupo>/<classe>/."""
    base = root / class_dir if class_dir else root
    if not base.exists():
        print(f"Pasta nao encontrada: {base}")
        return {}

    grupos: dict[str, list[Path]] = {}
    for path in base.rglob("*"):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        grupos.setdefault(path.parent.name, []).append(path)
    return grupos


def collect_from_index(csv_path: Path, catalogs_dir: Path, raw_root: Path) -> dict[str, list[Path]]:
    resolved = csv_path if csv_path.is_absolute() or csv_path.exists() else catalogs_dir / csv_path
    if not resolved.exists():
        print(f"Indice nao encontrado: {resolved}")
        return {}

    df = pd.read_csv(resolved)
    grupos: dict[str, list[Path]] = {}
    for label, parte in df.groupby("label"):
        grupos[str(label)] = [raw_root / p for p in parte["path"]]
    return grupos


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--class-dir",
        default=None,
        help="subpasta de raw/ a inspecionar (nebulae, other, galaxies). "
        "Omitido: varre tudo.",
    )
    parser.add_argument("--from-index", default=None, metavar="CSV", help="inspeciona a partir de um indice")
    parser.add_argument("--n", type=int, default=50, help="amostras por classe (padrao 50)")
    parser.add_argument("--cols", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="pasta de saida (padrao: runs/inspection)")
    args = parser.parse_args()

    paths = get_paths().ensure()
    out_dir = Path(args.out) if args.out else paths.runs / "inspection"

    if args.from_index:
        grupos = collect_from_index(Path(args.from_index), paths.catalogs, paths.raw)
    else:
        grupos = collect_from_folders(paths.raw, args.class_dir)

    if not grupos:
        print("Nenhuma imagem encontrada para inspecionar.")
        return 1

    print(f"Gerando grades em {out_dir}\n")
    for classe in sorted(grupos):
        arquivos = grupos[classe]
        existentes = [p for p in arquivos if p.exists()]
        faltando = len(arquivos) - len(existentes)

        if not existentes:
            print(f"  [aviso] {classe}: nenhuma imagem existe em disco")
            continue

        escolhidas = sample(existentes, args.n, args.seed)
        destino = build_grid(escolhidas, classe, out_dir / f"{classe}.png", cols=args.cols)

        aviso = f"  [{faltando} listadas mas ausentes]" if faltando else ""
        print(f"  {classe:<22} {len(existentes):>6} imagens -> {destino.name}{aviso}")

    print(f"\nAbra as grades em: {out_dir}")
    print(
        "\nO que procurar: o objeto aparece? esta centralizado? o tamanho esta\n"
        "razoavel? as imagens de uma classe se parecem entre si mais do que\n"
        "com as de outra? Se muitos recortes vierem vazios, ajuste o campo de\n"
        "visao daquela classe em FOV_BY_LABEL (data/catalogs.py) e recolete."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
