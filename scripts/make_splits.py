"""Gera os CSVs de train/val/test, consistentes entre os tres niveis.

    # o normal: uma atribuicao global e os tres niveis derivados dela
    python scripts/make_splits.py --all --from-index galaxies_index.csv --cap-object 2000

    # regerar um nivel a partir da atribuicao que ja existe
    python scripts/make_splits.py --level galaxy

POR QUE UMA ATRIBUICAO GLOBAL

A versao anterior dividia cada nivel de forma independente. O resultado,
medido: das 300 galaxias do conjunto de TESTE do nivel 1, 218 estavam no
conjunto de TREINO do nivel 2. Avaliar a cascata assim mede um modelo
respondendo sobre imagens que ele ja viu.

Agora o destino de cada imagem e decidido UMA VEZ, estratificado pelo rotulo
mais fino, e gravado em splits/assignment.csv. Cada nivel filtra dali. Uma
imagem nunca troca de lado entre niveis.

DUAS FONTES DE IMAGENS, como antes:

  VARREDURA  subpastas por rotulo dentro de raw/ (nebulosas, other)
  INDICE     CSV com path,label,source - para dados numa pasta unica, como
             as 61 mil imagens do Galaxy Zoo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from astro_classifier.data.splits import (
    build_assignment,
    load_assignment,
    splits_from_assignment,
)
from astro_classifier.paths import get_paths
from astro_classifier.taxonomy import Level

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# Pastas de download bruto que nao seguem o padrao <classe>/<imagem>. Sao
# cobertas por um CSV de indice, entao a varredura as ignora em silencio.
SILENT_IGNORE = {"images_training_rev1", "images_test_rev1", "galaxy_zoo", "_cache"}

# Rotulo fino (o nome da subpasta ou o rotulo do indice) -> rotulo do nivel 1.
FINO_PARA_OBJETO = {
    "spiral": "galaxy",
    "elliptical": "galaxy",
    "irregular": "galaxy",
    "emission": "nebula",
    "reflection": "nebula",
    "planetary": "nebula",
    "supernova_remnant": "nebula",
    "globular_cluster": "other",
    "open_cluster": "other",
    "empty_field": "other",
    "star_field": "other",
    "non_astronomical": "other",
}

# Nos niveis 2 o rotulo do nivel E o proprio rotulo fino.
FINO_PARA_NIVEL: dict[Level, dict[str, str]] = {
    Level.OBJECT: FINO_PARA_OBJETO,
    Level.GALAXY: {c: c for c in Level.GALAXY.classes},
    Level.NEBULA: {c: c for c in Level.NEBULA.classes},
}


def scan_images(raw_root: Path) -> pd.DataFrame:
    """Varre raw/ e devolve (path relativo, label_fino, source)."""
    registros = []
    for caminho in raw_root.rglob("*"):
        if caminho.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rel = caminho.relative_to(raw_root)
        if len(rel.parts) < 2:
            continue  # imagem solta na raiz, sem pasta de rotulo
        registros.append(
            {
                "path": str(rel).replace("\\", "/"),
                "label_fino": rel.parts[-2],
                "source": rel.parts[0],
            }
        )
    return pd.DataFrame(registros)


def load_index(csv_path: Path, catalogs_dir: Path) -> pd.DataFrame:
    """Le um indice gerado por prepare_galaxy_zoo.py ou build_*_dataset.py."""
    resolvido = csv_path if csv_path.is_absolute() or csv_path.exists() else catalogs_dir / csv_path
    if not resolvido.exists():
        raise FileNotFoundError(f"indice nao encontrado: {resolvido}")

    df = pd.read_csv(resolvido)
    faltando = {"path", "label"} - set(df.columns)
    if faltando:
        raise ValueError(f"{resolvido} nao tem as colunas obrigatorias: {sorted(faltando)}")

    df = df.rename(columns={"label": "label_fino"})
    if "source" not in df.columns:
        df["source"] = resolvido.stem

    print(f"  indice {resolvido.name}: {len(df)} imagens  {df['label_fino'].value_counts().to_dict()}")

    # As colunas p_<classe> trazem a distribuicao de votos humanos, usada no
    # treino com soft labels. So o Galaxy Zoo as tem; para as nebulosas elas
    # ficam ausentes, o que e correto - nao ha votacao ali.
    probabilidades = [c for c in df.columns if c.startswith("p_")]
    if probabilidades:
        print(f"    com distribuicao de votos: {probabilidades}")
    return df[["path", "label_fino", "source"] + probabilidades]


def coletar_imagens(paths, indices: list[str], sem_varredura: bool) -> pd.DataFrame:
    """Une varredura e indices numa tabela unica de rotulos finos."""
    partes: list[pd.DataFrame] = []

    if not sem_varredura:
        varridas = scan_images(paths.raw)
        if not varridas.empty:
            print(f"{len(varridas)} imagens encontradas varrendo {paths.raw}")
            desconhecidas = [
                f
                for f in varridas["label_fino"].unique()
                if f not in FINO_PARA_OBJETO and f not in SILENT_IGNORE
            ]
            if desconhecidas:
                print(f"  [aviso] pastas sem mapeamento conhecido: {desconhecidas}")
            partes.append(varridas)

    for nome in indices:
        partes.append(load_index(Path(nome), paths.catalogs))

    if not partes:
        raise ValueError("nenhuma fonte de imagens: varredura vazia e nenhum indice informado")

    # O indice vence sobre a varredura: seu rotulo veio de um criterio
    # explicito, nao do nome de uma pasta.
    df = pd.concat(partes, ignore_index=True)
    duplicadas = int(df.duplicated(subset="path", keep="last").sum())
    if duplicadas:
        print(f"  {duplicadas} imagens em mais de uma fonte; mantido o rotulo do indice")
    return df.drop_duplicates(subset="path", keep="last").reset_index(drop=True)


def alertar_imagens_duplicadas(df: pd.DataFrame, raw_root: Path, amostra: int = 4) -> int:
    """Avisa quando duas imagens da tabela tem conteudo IDENTICO.

    ISTO EXISTE POR CAUSA DE UM BUG QUE ACONTECEU DUAS VEZES.

    A varredura de `raw/` entra por nome de pasta e aceita qualquer arquivo que
    esteja la. Se um coletor roda com um esquema de nomes e depois com outro, as
    imagens do primeiro ficam ORFAS no disco - e a varredura as soma ao dataset
    como se fossem objetos distintos. A mesma imagem entao pode cair no treino
    sob um nome e no teste sob o outro.

    Aconteceu com `other`: 848 orfas, 735 duplicadas byte a byte. E aconteceu
    com as nebulosas, por outra causa (deduplicacao incompleta por posicao).
    Nas duas vezes o log dizia que tudo estava bem, porque a gravacao FUNCIONOU -
    o que estava errado era o conjunto de arquivos existentes.

    A checagem custa um hash por imagem. Para 66 mil imagens sao alguns minutos,
    e roda uma vez por regeracao de splits - barato contra o preco de descobrir
    o vazamento depois de treinar.
    """
    import hashlib
    from collections import defaultdict

    por_hash: dict[str, list[str]] = defaultdict(list)
    ilegiveis = 0
    for rel in df["path"]:
        caminho = raw_root / rel
        try:
            por_hash[hashlib.md5(caminho.read_bytes()).hexdigest()].append(rel)
        except OSError:
            ilegiveis += 1

    grupos = {h: v for h, v in por_hash.items() if len(v) > 1}
    extras = sum(len(v) - 1 for v in grupos.values())

    if ilegiveis:
        print(f"  [aviso] {ilegiveis} imagens da tabela nao puderam ser lidas")

    if not grupos:
        print(f"  sem duplicatas de conteudo entre as {len(df)} imagens")
        return 0

    print(
        f"  [ATENCAO] {extras} imagens duplicadas byte a byte, em {len(grupos)} grupos.\n"
        "  A MESMA imagem pode cair em conjuntos diferentes - isso e vazamento."
    )
    for v in list(grupos.values())[:amostra]:
        print(f"      {v}")
    if len(grupos) > amostra:
        print(f"      ... e outros {len(grupos) - amostra} grupos")
    print(
        "  Rode o coletor da classe afetada com --prune para remover as orfas,\n"
        "  ou confira a deduplicacao por posicao se forem objetos de catalogo."
    )
    return extras


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="refaz a atribuicao global e os tres niveis")
    parser.add_argument("--level", choices=[lv.value for lv in Level], help="regera so este nivel")
    parser.add_argument("--from-index", action="append", default=[], metavar="CSV")
    parser.add_argument("--no-scan", action="store_true", help="usa apenas os indices")
    parser.add_argument(
        "--cap-object",
        type=int,
        default=2000,
        help="teto por classe no nivel 1 (0 = sem teto). Evita que dezenas de "
        "milhares de galaxias afoguem as nebulosas.",
    )
    parser.add_argument(
        "--skip-dup-check",
        action="store_true",
        help="pula a verificacao de imagens duplicadas por conteudo. Ela custa um "
        "hash por imagem (poucos minutos para 66 mil) e ja pegou dois vazamentos - "
        "pule so quando estiver iterando em splits e com pressa.",
    )
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.all and not args.level:
        parser.error("informe --all (recomendado) ou --level <nivel>")

    paths = get_paths().ensure()

    if args.all:
        print("=== 1. coletando todas as imagens ===")
        try:
            todas = coletar_imagens(paths, args.from_index, args.no_scan)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  [ERRO] {exc}")
            return 1

        print("\n=== 2. atribuicao global (estratificada pelo rotulo fino) ===")
        build_assignment(
            todas,
            paths.splits,
            val_size=args.val_size,
            test_size=args.test_size,
            seed=args.seed,
        )
        niveis = list(Level)
    else:
        niveis = [Level(args.level)]

    try:
        atribuicao = load_assignment(paths.splits)
    except FileNotFoundError as exc:
        print(f"[ERRO] {exc}")
        return 1

    print("\n=== 3. derivando os splits por nivel ===")
    for nivel in niveis:
        print(f"\n[{nivel.value}]")
        teto = args.cap_object if nivel is Level.OBJECT else 0
        try:
            splits_from_assignment(
                atribuicao,
                nivel,
                FINO_PARA_NIVEL[nivel],
                paths.splits,
                cap_per_class=teto,
                seed=args.seed,
            )
        except ValueError as exc:
            print(f"  [ERRO] {exc}")
            return 1

    print(f"\nSplits gravados em {paths.splits}")
    print("Consistentes entre niveis: uma imagem tem o mesmo destino em todos.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
