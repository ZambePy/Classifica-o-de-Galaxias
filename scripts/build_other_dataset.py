"""Monta a classe `other` - o que NAO e galaxia nem nebulosa.

Sem esta classe, o nivel 1 vira um classificador binario e o sistema e
obrigado a chamar tudo de galaxia ou nebulosa. Uma foto da Lua vira
"nebulosa planetaria, 97%". A classe `other` e o que permite recusar.

    # inspecione antes de baixar muito
    python scripts/build_other_dataset.py --limit 10

    # coleta completa
    python scripts/build_other_dataset.py

Tres fontes, todas pela mesma pipeline de recorte das nebulosas:

    globular_cluster  catalogo de Harris - aglomerados globulares. Sao
                      objetos extensos e brilhantes, faceis de confundir com
                      galaxias elipticas. Exatamente o caso dificil que
                      interessa.

    empty_field       ceu vazio, sorteado em alta latitude galactica.

    star_field        campos densos de estrelas, no plano galactico.

A quarta subpasta, `non_astronomical/`, NAO e automatizavel: sao fotos do
dia a dia (pessoas, objetos, paisagens). Coloque as imagens voce mesmo - o
script cria a pasta e avisa. Sem ela o modelo aprende a separar tipos de
ceu, mas nao aprende que uma selfie nao e astronomia.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from astro_classifier.data.catalogs import GLOBULAR_CLUSTERS, inspect_catalog, query_catalog
from astro_classifier.data.cutouts import SURVEY_DSS2, download_cutouts
from astro_classifier.data.sky_sampling import sample_empty_fields, sample_star_fields
from astro_classifier.paths import get_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="so imprime as primeiras linhas do catalogo de aglomerados e sai",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="maximo de objetos por classe (-1 = todos os aglomerados do catalogo)",
    )
    parser.add_argument(
        "--n-empty",
        type=int,
        default=300,
        help="quantos campos vazios sortear",
    )
    parser.add_argument(
        "--n-star",
        type=int,
        default=300,
        help="quantos campos estelares sortear",
    )
    parser.add_argument("--size", type=int, default=256, help="lado do recorte em pixels")
    parser.add_argument("--pause", type=float, default=0.5, help="segundos entre requisicoes")
    parser.add_argument("--survey", default=SURVEY_DSS2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.inspect:
        inspect_catalog(GLOBULAR_CLUSTERS)
        return 0

    paths = get_paths().ensure()
    out_dir = paths.raw_other
    print(f"Dados vao para: {out_dir}\n")

    n_empty = args.n_empty if args.limit < 0 else min(args.n_empty, args.limit)
    n_star = args.n_star if args.limit < 0 else min(args.n_star, args.limit)

    grupos: list[pd.DataFrame] = []

    # 1. Aglomerados globulares (catalogo real)
    print(f"[globular_cluster] consultando {GLOBULAR_CLUSTERS.vizier_id}")
    print(f"  {GLOBULAR_CLUSTERS.citation}")
    try:
        grupos.append(query_catalog(GLOBULAR_CLUSTERS, limit=args.limit))
    except Exception as exc:  # noqa: BLE001 - catalogo fora do ar nao pode matar o resto
        print(f"  [ERRO] {exc}")
        print("  pulando aglomerados. Rode --inspect para diagnosticar.\n")

    # 2. e 3. Campos sorteados (sem catalogo)
    print(f"[empty_field] sorteando {n_empty} direcoes em alta latitude galactica")
    grupos.append(sample_empty_fields(n_empty, seed=args.seed))

    print(f"[star_field] sorteando {n_star} direcoes no plano galactico")
    grupos.append(sample_star_fields(n_star, seed=args.seed + 1))

    todos = pd.concat(grupos, ignore_index=True)
    print(f"\n{len(todos)} recortes a baixar: {todos['label'].value_counts().to_dict()}\n")

    salvos = download_cutouts(
        todos.to_dict("records"),
        out_dir=out_dir,
        survey=args.survey,
        size_px=args.size,
        pause=args.pause,
        progress=print,
    )

    if not salvos:
        print("Nenhuma imagem baixada. Verifique a conexao.")
        return 1

    index = pd.DataFrame(salvos)
    index_path = paths.catalogs / "other_index.csv"
    index.to_csv(index_path, index=False)

    print(f"\nIndice salvo em {index_path}")
    print(f"Total: {len(index)} imagens")
    print(index["label"].value_counts().to_string())

    # A quarta subpasta e trabalho manual - criamos e avisamos.
    manual = out_dir / "non_astronomical"
    manual.mkdir(parents=True, exist_ok=True)
    existentes = len(list(manual.glob("*.jpg"))) + len(list(manual.glob("*.png")))

    print("\n" + "-" * 68)
    if existentes == 0:
        print(f"FALTA VOCE: coloque fotos do dia a dia em\n  {manual}")
        print(
            "\nAlgumas centenas bastam - pessoas, objetos, paisagens, prints de\n"
            "tela. Sem elas o modelo aprende a distinguir tipos de ceu, mas nao\n"
            "aprende que uma selfie nao e astronomia. E e selfie que as pessoas\n"
            "vao subir no dashboard."
        )
    else:
        print(f"{existentes} imagens ja em non_astronomical/ - nada a fazer ali.")
    print("-" * 68)
    print(
        "\nDepois: inspecione visualmente antes de treinar\n"
        "  python scripts/inspect_dataset.py --class-dir other"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
