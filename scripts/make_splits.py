"""Gera os CSVs de train/val/test para um nivel da cascata.

    # Nivel 1 (galaxy/nebula/other), com teto para nao afogar as nebulosas
    python scripts/make_splits.py --level object --cap-per-class 4000

    # Nivel 2a (morfologia de galaxia)
    python scripts/make_splits.py --level galaxy

    # Nivel 2b (tipo de nebulosa)
    python scripts/make_splits.py --level nebula

O script varre ASTRO_DATA_ROOT/raw/ procurando imagens organizadas em
subpastas por rotulo, filtra o que pertence ao nivel pedido, e escreve os
tres CSVs em ASTRO_DATA_ROOT/splits/.

Sobre --cap-per-class: no nivel 1 havera dezenas de milhares de galaxias
contra alguns milhares de nebulosas. Sem um teto, o modelo aprende que
"quase tudo e galaxia" e acerta muito por motivo errado. Limitar a classe
majoritaria e mais eficaz do que so ajustar pesos de loss.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from astro_classifier.data.splits import make_splits
from astro_classifier.paths import get_paths
from astro_classifier.taxonomy import Level

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# Como uma pasta do disco vira um rotulo do nivel 1. Nebulosas de qualquer
# tipo sao "nebula"; aglomerados e campos vazios sao "other".
OBJECT_FOLDER_MAP = {
    "spiral": "galaxy",
    "elliptical": "galaxy",
    "irregular": "galaxy",
    "emission": "nebula",
    "reflection": "nebula",
    "planetary": "nebula",
    "supernova_remnant": "nebula",
    "globular_cluster": "other",
    "empty_field": "other",
    "star_field": "other",
    "non_astronomical": "other",
}


def scan_images(raw_root: Path) -> pd.DataFrame:
    """Varre raw/ e devolve (path relativo, pasta de origem, fonte)."""
    records = []
    for image_path in raw_root.rglob("*"):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rel = image_path.relative_to(raw_root)
        if len(rel.parts) < 2:
            continue  # imagem solta na raiz, sem pasta de rotulo
        records.append(
            {
                "path": str(rel).replace("\\", "/"),
                "folder": rel.parts[-2],
                "source": rel.parts[0],
            }
        )
    return pd.DataFrame(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--level", required=True, choices=[lvl.value for lvl in Level])
    parser.add_argument("--cap-per-class", type=int, default=0, help="teto por classe (0 = sem teto)")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    level = Level(args.level)
    paths = get_paths().ensure()

    df = scan_images(paths.raw)
    if df.empty:
        print(f"Nenhuma imagem encontrada em {paths.raw}.")
        print("Baixe os dados primeiro (scripts/build_nebula_dataset.py, Galaxy Zoo).")
        return 1

    print(f"{len(df)} imagens encontradas em {paths.raw}")

    # Traduz a pasta de origem no rotulo do nivel pedido.
    if level is Level.OBJECT:
        df["label"] = df["folder"].map(OBJECT_FOLDER_MAP)
    else:
        df["label"] = df["folder"]

    unmapped = df[df["label"].isna()]["folder"].unique()
    if len(unmapped) > 0:
        print(f"  [aviso] pastas ignoradas (sem mapeamento): {list(unmapped)}")
    df = df.dropna(subset=["label"])

    df = df[df["label"].isin(level.classes)]
    if df.empty:
        print(f"Nenhuma imagem pertence as classes do nivel '{level.value}': {level.classes}")
        return 1

    print(f"apos filtrar para o nivel '{level.value}': {df['label'].value_counts().to_dict()}")

    if args.cap_per_class > 0:
        df = (
            df.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), args.cap_per_class), random_state=args.seed))
            .reset_index(drop=True)
        )
        print(f"apos teto de {args.cap_per_class}/classe: {df['label'].value_counts().to_dict()}")

    print()
    make_splits(
        df[["path", "label", "source"]],
        level=level,
        out_dir=paths.splits,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    print(f"\nSplits gravados em {paths.splits}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
