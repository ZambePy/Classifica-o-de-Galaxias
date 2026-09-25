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

from astro_classifier.data.catalogs import (
    CATALOGS,
    deduplicate_by_position,
    inspect_catalog,
    query_catalog,
)
from astro_classifier.data.cutouts import SURVEY_DSS2, cutout_filename, download_cutouts
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
    parser.add_argument(
        "--prune",
        action="store_true",
        help="remove do disco imagens que nao estao mais no catalogo (orfas de "
        "uma coleta anterior, ou identificadas como duplicatas)",
    )
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

    # 1. Consulta TODOS os catalogos ANTES de baixar qualquer imagem. Sem
    #    isso nao da para deduplicar: os catalogos se sobrepoem (o LBN inclui
    #    dezenas de nebulosas de Sharpless) e a mesma nebulosa entraria duas
    #    vezes - podendo cair metade no treino e metade no teste.
    partes = []
    for spec in CATALOGS:
        print(f"[{spec.vizier_id}] {spec.citation}")
        try:
            partes.append(query_catalog(spec, limit=args.limit))
        except Exception as exc:  # noqa: BLE001 - um catalogo fora do ar nao mata os outros
            print(f"  [ERRO] falhou ao consultar {spec.vizier_id}: {exc}")
            print("  pulando. Rode --inspect para diagnosticar.")

    if not partes:
        print("Nenhum catalogo respondeu.")
        return 1

    todos = pd.concat(partes, ignore_index=True)
    print(f"\n{len(todos)} objetos somando os catalogos")
    todos = deduplicate_by_position(todos)
    print(f"{len(todos)} apos deduplicar: {todos['label'].value_counts().to_dict()}\n")

    # 2. Remove do disco o que nao esta mais no catalogo deduplicado. Sem
    #    isso, uma imagem baixada por uma versao anterior do catalogo fica
    #    orfa na pasta e entra no dataset pela varredura - inclusive as que
    #    a deduplicacao acabou de identificar como repetidas.
    if args.prune:
        esperados = {
            (paths.raw_nebulae / r["label"] / cutout_filename(r))
            for _, r in todos.iterrows()
        }
        removidos = 0
        for existente in paths.raw_nebulae.rglob("*.jpg"):
            if existente not in esperados:
                existente.unlink()
                removidos += 1
        print(f"  {removidos} imagens orfas removidas do disco\n")

    # 3. Agora sim, baixa.
    all_records = download_cutouts(
        todos.to_dict("records"),
        out_dir=paths.raw_nebulae,
        survey=args.survey,
        size_px=args.size,
        pause=args.pause,
        progress=print,
    )
    print(f"\n{len(all_records)}/{len(todos)} imagens salvas")

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
