"""Converte os rotulos probabilisticos do Galaxy Zoo em classes morfologicas.

    python scripts/prepare_galaxy_zoo.py \
        --solutions C:/astro-data/raw/galaxy_zoo/training_solutions_rev1.csv \
        --images    C:/astro-data/raw/galaxy_zoo/images_training_rev1

O Galaxy Zoo nao rotula "esta galaxia e espiral". Ele da 37 colunas com a
fracao de voluntarios que respondeu cada item da arvore de decisao. Traduzir
isso em tres classes e uma DECISAO METODOLOGICA - nao um detalhe tecnico - e
ela precisa aparecer no seu texto academico. O mapeamento usado aqui:

    Class1.1  galaxia lisa, sem features          -> elliptical
    Class1.2  tem disco / features                -> spiral
    Class1.3  estrela ou artefato                 -> descartada (nao e galaxia)
    Class6.1  "tem algo estranho"                 -> irregular (ver abaixo)

E o limiar de confianca: so aceitamos a galaxia se a fracao de votos da
classe vencedora passar de `--min-vote`. Galaxias ambiguas (onde os humanos
nao concordaram) sao DESCARTADAS em vez de forcadas numa classe. Isso reduz
o dataset mas aumenta muito a qualidade do rotulo.

LIMITACAO HONESTA: "irregular" nao tem pergunta propria no Galaxy Zoo 2. A
aproximacao por Class6.1 (odd) mistura galaxias genuinamente irregulares com
fusoes, aneis e artefatos. Se o F1 de irregular ficar baixo, essa e a
primeira suspeita - e vale dizer isso no trabalho em vez de esconder.

CAMINHO MAIS RICO (Fase 2): treinar com soft labels, usando as proprias
probabilidades como alvo (KL-divergence em vez de cross-entropy). A
incerteza humana vira parte do treino. E uma pergunta de pesquisa de
verdade: "treinar com a distribuicao de votos melhora o desempenho em
relacao a rotular pelo voto majoritario?"
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

from astro_classifier.paths import get_paths

# Colunas do training_solutions_rev1.csv do Galaxy Zoo (competicao Kaggle).
COL_SMOOTH = "Class1.1"
COL_FEATURES = "Class1.2"
COL_ARTIFACT = "Class1.3"
COL_ODD = "Class6.1"

REQUIRED = [COL_SMOOTH, COL_FEATURES, COL_ARTIFACT, COL_ODD]


def assign_label(row: pd.Series, min_vote: float, odd_threshold: float) -> str | None:
    """Voto majoritario com limiar. Devolve None quando a galaxia e ambigua."""
    if row[COL_ARTIFACT] > 0.5:
        return None  # estrela ou artefato: nao e galaxia

    # "Estranha" com folga vira irregular, mesmo que tenha disco.
    if row[COL_ODD] >= odd_threshold:
        return "irregular"

    if row[COL_SMOOTH] >= min_vote:
        return "elliptical"
    if row[COL_FEATURES] >= min_vote:
        return "spiral"

    return None  # voluntarios nao concordaram o bastante


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--solutions", required=True, help="training_solutions_rev1.csv")
    parser.add_argument("--images", required=True, help="pasta com as imagens .jpg")
    parser.add_argument(
        "--min-vote",
        type=float,
        default=0.7,
        help="fracao minima de votos para aceitar o rotulo (padrao 0.7)",
    )
    parser.add_argument(
        "--odd-threshold",
        type=float,
        default=0.8,
        help="fracao de votos em Class6.1 para classificar como irregular",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="copia as imagens para raw/galaxies/<classe>/ (usa mais disco, "
        "mas deixa a estrutura igual a das nebulosas)",
    )
    args = parser.parse_args()

    solutions_path = Path(args.solutions)
    images_dir = Path(args.images)

    if not solutions_path.exists():
        print(f"CSV nao encontrado: {solutions_path}")
        print("Baixe em: https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data")
        return 1
    if not images_dir.exists():
        print(f"Pasta de imagens nao encontrada: {images_dir}")
        return 1

    df = pd.read_csv(solutions_path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        print(f"Colunas ausentes no CSV: {missing}")
        print(f"Colunas encontradas: {list(df.columns)[:10]}...")
        return 1

    print(f"{len(df)} galaxias no CSV de rotulos")
    df["label"] = df.apply(assign_label, axis=1, args=(args.min_vote, args.odd_threshold))

    descartadas = df["label"].isna().sum()
    df = df.dropna(subset=["label"])
    print(f"{descartadas} descartadas por ambiguidade ou por serem artefato")
    print(f"{len(df)} galaxias rotuladas: {df['label'].value_counts().to_dict()}\n")

    paths = get_paths().ensure()
    galaxies_dir = paths.raw / "galaxies"

    records = []
    faltando = 0

    for _, row in df.iterrows():
        galaxy_id = int(row["GalaxyID"])
        source = images_dir / f"{galaxy_id}.jpg"
        if not source.exists():
            faltando += 1
            continue

        if args.copy:
            dest_dir = galaxies_dir / row["label"]
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{galaxy_id}.jpg"
            if not dest.exists():
                shutil.copy2(source, dest)
            rel = str(dest.relative_to(paths.raw)).replace("\\", "/")
        else:
            rel = str(source.relative_to(paths.raw)).replace("\\", "/")

        records.append({"path": rel, "label": row["label"], "source": "GalaxyZoo2"})

    if faltando:
        print(f"[aviso] {faltando} imagens listadas no CSV nao existem em {images_dir}")

    index = pd.DataFrame(records)
    index_path = paths.catalogs / "galaxies_index.csv"
    index.to_csv(index_path, index=False)

    print(f"Indice salvo em {index_path} ({len(index)} imagens)")
    if not args.copy:
        print(
            "\nAs imagens continuam onde estavam (sem --copy). Para que\n"
            "make_splits.py as encontre pela pasta de classe, rode com --copy\n"
            "ou use o indice diretamente."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
