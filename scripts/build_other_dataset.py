"""Monta a classe `other` - o que NAO e galaxia nem nebulosa.

Sem esta classe, o nivel 1 vira um classificador binario e o sistema e
obrigado a chamar tudo de galaxia ou nebulosa. Uma foto da Lua vira
"nebulosa planetaria, 97%". A classe `other` e o que permite recusar.

    # inspecione antes de baixar muito
    python scripts/build_other_dataset.py --limit 10

    # coleta completa
    python scripts/build_other_dataset.py

Quatro fontes, todas pela mesma pipeline de recorte das nebulosas:

    open_cluster      NGC2000.0 - aglomerados abertos. Objetos reais e
                      brilhantes que nao sao galaxia nem nebulosa.

    globular_cluster  NGC2000.0 e catalogo de Harris - aglomerados. Sao
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
from pathlib import Path

import pandas as pd

from astro_classifier.data.catalogs import (
    GLOBULAR_CLUSTERS,
    NGC2000,
    inspect_catalog,
    query_catalog,
)
from astro_classifier.data.cutouts import SURVEY_DSS2, cutout_filename, download_cutouts
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
    parser.add_argument(
        "--prune",
        action="store_true",
        help="remove do disco imagens que nao estao mais no catalogo (orfas de "
        "uma coleta anterior com outro esquema de nomes)",
    )
    args = parser.parse_args()

    if args.inspect:
        for spec in (NGC2000, GLOBULAR_CLUSTERS):
            inspect_catalog(spec)
        return 0

    paths = get_paths().ensure()
    out_dir = paths.raw_other
    print(f"Dados vao para: {out_dir}\n")

    n_empty = args.n_empty if args.limit < 0 else min(args.n_empty, args.limit)
    n_star = args.n_star if args.limit < 0 else min(args.n_star, args.limit)

    grupos: list[pd.DataFrame] = []

    # 1. Objetos reais que nao sao galaxia nem nebulosa. Sao muito mais
    #    informativos para a classe `other` do que campos de ceu sorteados:
    #    um aglomerado globular brilhante e justamente o caso dificil que o
    #    modelo precisa aprender a nao chamar de galaxia eliptica.
    for spec in (NGC2000, GLOBULAR_CLUSTERS):
        print(f"[{spec.vizier_id}] {spec.citation}")
        try:
            grupos.append(query_catalog(spec, limit=args.limit))
        except Exception as exc:  # noqa: BLE001 - um catalogo fora do ar nao mata o resto
            print(f"  [ERRO] {exc}")
            print("  pulando. Rode --inspect para diagnosticar.\n")

    # 2. e 3. Campos sorteados.
    #
    # O catalogo de nebulosas entra aqui como lista de EXCLUSAO, nao de
    # inclusao: nenhum campo sorteado pode conter um objeto catalogado. Sem
    # isso, 15% dos `star_field` continham uma nebulosa - e eles sao o grupo
    # de controle da analise de atalho, onde a premissa e justamente que nao
    # ha nebulosa nenhuma ali. Ver `drop_near_catalog`.
    #
    # A fonte e o CSV publicado em docs/dataset/, que ja esta no repositorio -
    # nao custa uma consulta de rede e e exatamente o conjunto que gerou as
    # imagens de nebulosa em disco.
    catalogo_neb = None
    csv_neb = Path(__file__).resolve().parents[1] / "docs" / "dataset" / "nebulae_catalog.csv"
    if csv_neb.exists():
        catalogo_neb = pd.read_csv(csv_neb)
        print(f"[exclusao] {len(catalogo_neb)} nebulosas de {csv_neb.name} serao evitadas")
    else:
        print(
            f"[aviso] {csv_neb} nao existe - os campos sorteados NAO serao\n"
            "        filtrados, e alguns vao conter nebulosas catalogadas.\n"
            "        Rode scripts/export_dataset.py antes para evitar isso."
        )

    print(f"[empty_field] sorteando {n_empty} direcoes em alta latitude galactica")
    grupos.append(
        sample_empty_fields(n_empty, seed=args.seed, catalogo=catalogo_neb, progress=print)
    )

    print(f"[star_field] sorteando {n_star} direcoes no plano galactico")
    grupos.append(
        sample_star_fields(n_star, seed=args.seed + 1, catalogo=catalogo_neb, progress=print)
    )

    from astro_classifier.data.catalogs import deduplicate_by_position

    todos = deduplicate_by_position(pd.concat(grupos, ignore_index=True))
    print(f"\n{len(todos)} recortes a baixar: {todos['label'].value_counts().to_dict()}\n")

    # Remove do disco o que nao esta mais no catalogo. Sem isto, mudar o
    # esquema de nomes deixa as imagens antigas orfas na pasta - e elas
    # entram no dataset pela varredura, duplicando cada objeto sob dois
    # nomes. Aconteceu: 735 imagens de `other` estavam duplicadas byte a
    # byte, e a mesma imagem podia cair no treino com um nome e no teste
    # com o outro.
    if args.prune:
        esperados = {
            out_dir / r["label"] / cutout_filename(r) for _, r in todos.iterrows()
        }
        removidos = 0
        for existente in out_dir.rglob("*.jpg"):
            # `non_astronomical` nao vem de catalogo: e preenchida a mao.
            if existente.parent.name == "non_astronomical":
                continue
            if existente not in esperados:
                existente.unlink()
                removidos += 1
        print(f"  {removidos} imagens orfas removidas do disco\n")

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
