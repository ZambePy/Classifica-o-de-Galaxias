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


def soft_label_probs(row: pd.Series) -> dict[str, float]:
    """Distribuicao de votos por classe, normalizada, para treino com soft labels.

    O Galaxy Zoo tem 37 colunas; as tres que importam para a nossa taxonomia:

        Class1.1  lisa, sem features        -> elliptical
        Class1.2  tem disco / features      -> spiral
        Class6.1  "tem algo estranho"       -> irregular

    Elas nao somam 1 (Class1.1+1.2+1.3 somam; Class6.1 e uma pergunta
    separada da arvore), entao normalizamos as tres entre si. Isso PRESERVA
    a proporcao relativa dos votos, que e o que o soft label precisa, sem
    fingir uma semantica probabilistica que o catalogo nao tem.

    E uma simplificacao, e deve ser declarada como tal no texto: a arvore de
    decisao do Galaxy Zoo tem estrutura condicional que estamos achatando.
    """
    bruto = {
        "p_elliptical": max(float(row[COL_SMOOTH]), 0.0),
        "p_spiral": max(float(row[COL_FEATURES]), 0.0),
        "p_irregular": max(float(row[COL_ODD]), 0.0),
    }
    total = sum(bruto.values())
    if total <= 0:
        # Ninguem votou em nada: distribuicao uniforme e o alvo honesto.
        return dict.fromkeys(bruto, 1.0 / len(bruto))
    return {k: round(v / total, 6) for k, v in bruto.items()}


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

        # Guarda TAMBEM a distribuicao de votos, nao so o rotulo vencedor.
        # E ela que permite treinar com soft labels depois - a informacao
        # mais rica do Galaxy Zoo e justamente o quanto os humanos
        # concordaram, e reduzir tudo a um rotulo joga isso fora.
        registro = {"path": rel, "label": row["label"], "source": "GalaxyZoo2"}
        registro.update(soft_label_probs(row))
        records.append(registro)

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
