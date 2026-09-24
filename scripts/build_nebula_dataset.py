"""Monta o dataset de nebulosas a partir de catalogos astronomicos.

Este e o script mais importante do projeto - ele resolve o problema que nao
tem solucao pronta. Consulta os catalogos no VizieR, pega as coordenadas e
baixa um recorte do ceu para cada objeto.

USO RECOMENDADO NA PRIMEIRA VEZ:

    # 1. Confira o que os catalogos devolvem ANTES de baixar milhares de imagens
    python scripts/build_nebula_dataset.py --inspect

    # 2. Teste com poucos objetos por classe
    python scripts/build_nebula_dataset.py --limit 20

    # 3. Rode completo (leva horas - e um servico publico, com pausa entre
    #    requisicoes por educacao e para nao ser bloqueado)
    python scripts/build_nebula_dataset.py

O resultado vai para ASTRO_DATA_ROOT/raw/nebulae/<classe>/*.jpg e um CSV
mestre em ASTRO_DATA_ROOT/catalogs/nebulae_index.csv.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from astro_classifier.data.catalogs import CATALOGS, inspect_catalog, query_catalog
from astro_classifier.data.cutouts import SURVEY_DSS2, download_cutouts
from astro_classifier.paths import get_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="so imprime as primeiras linhas de cada catalogo e sai (faca isto primeiro)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="maximo de objetos por classe (-1 = todos). Use um valor pequeno para testar.",
    )
    parser.add_argument("--size", type=int, default=256, help="lado do recorte em pixels")
    parser.add_argument(
        "--pause",
        type=float,
        default=0.5,
        help="segundos entre requisicoes ao servico publico (nao reduza muito)",
    )
    parser.add_argument("--survey", default=SURVEY_DSS2, help="HiPS de origem das imagens")
    args = parser.parse_args()

    if args.inspect:
        for spec in CATALOGS:
            inspect_catalog(spec)
        print(
            "\nConfira: cada catalogo trouxe coordenadas e um nome? "
            "Se alguma tabela veio vazia, o identificador VizieR mudou - "
            "ajuste CATALOGS em src/astro_classifier/data/catalogs.py."
        )
        return 0

    paths = get_paths().ensure()
    print(f"Dados vao para: {paths.root}")
    print(f"Levantamento (survey): {args.survey}\n")

    all_records: list[dict] = []

    for spec in CATALOGS:
        print(f"[{spec.label}] consultando {spec.vizier_id} ({spec.citation})")
        try:
            df = query_catalog(spec, limit=args.limit)
        except Exception as exc:  # noqa: BLE001 - um catalogo fora do ar nao pode matar os outros
            print(f"  [ERRO] falhou ao consultar {spec.vizier_id}: {exc}")
            print("  pulando esta classe. Rode --inspect para diagnosticar.")
            continue

        print(f"  {len(df)} objetos no catalogo, baixando recortes...")
        saved = download_cutouts(
            df.to_dict("records"),
            out_dir=paths.raw_nebulae,
            survey=args.survey,
            size_px=args.size,
            pause=args.pause,
            progress=print,
        )
        print(f"  [{spec.label}] {len(saved)}/{len(df)} imagens salvas\n")
        all_records.extend(saved)

    if not all_records:
        print("Nenhuma imagem baixada. Verifique a conexao e rode --inspect.")
        return 1

    index = pd.DataFrame(all_records)
    index_path = paths.catalogs / "nebulae_index.csv"
    index.to_csv(index_path, index=False)

    print(f"Indice salvo em {index_path}")
    print(f"Total: {len(index)} imagens")
    print(index["label"].value_counts().to_string())
    print(
        "\nProximo passo: inspecione VISUALMENTE uma amostra antes de treinar.\n"
        "Recortes vazios (objeto fraco demais para o DSS) ou com fov errado sao\n"
        "comuns e envenenam o treino. Um notebook em notebooks/ para folhear as\n"
        "imagens por classe vale muito mais do que confiar no catalogo."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
