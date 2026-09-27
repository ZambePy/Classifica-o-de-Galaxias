"""Exporta o catalogo de nebulosas para que outra pessoa possa reconstrui-lo.

    python scripts/export_dataset.py

O QUE ISTO PRODUZ

As IMAGENS nao sao redistribuiveis: pertencem aos levantamentos (DSS2, via
CDS/hips2fits) e cada um tem seus termos. O que e nosso, e o que ninguem
mais tem, e a CURADORIA: quais objetos, de quais catalogos, com que rotulo,
com que enquadramento, depois de quais filtros e da deduplicacao.

Isso cabe num CSV de poucas centenas de kilobytes, entra no repositorio, e
permite que qualquer pessoa baixe exatamente o mesmo dataset rodando
`build_nebula_dataset.py`. E a forma usual de publicar dataset em astronomia
- catalogos de coordenadas, nao pixels.

POR QUE VALE PUBLICAR

Nao existe equivalente publico. Quem quiser treinar um classificador de
nebulosas hoje precisa refazer todo o trabalho: descobrir quais catalogos
servem, resolver que o VizieR muda esquemas, descobrir que o campo de visao
tem de ser proporcional ao diametro (sem isso as planetarias somem), e
deduplicar objetos que aparecem em varios catalogos com nomes diferentes.

Sai em docs/dataset/ com:

    nebulae_catalog.csv   os objetos, com coordenada, rotulo, fov e origem
    dataset_card.md       proveniencia, filtros, vieses conhecidos, licenca
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

CARTAO = """# Dataset de nebulosas — cartão de proveniência

Catálogo de **{n_total} nebulosas galácticas** com coordenadas, tipo e
enquadramento, montado a partir de {n_fontes} catálogos astronômicos publicados.

As imagens não acompanham este arquivo. Reconstrua-as com:

```powershell
python scripts\\build_nebula_dataset.py
```

O script consulta os mesmos catálogos no VizieR e baixa os recortes pelo
serviço `hips2fits` do CDS. O resultado é idêntico, porque tudo que varia
está fixado aqui: os identificadores, os filtros e a semente.

---

## Composição

| classe | objetos |
|---|---:|
{composicao}

## Catálogos de origem

| identificador | objetos | referência |
|---|---:|---|
{fontes}

## Como cada rótulo foi decidido

{regras}

## Enquadramento

O campo de visão é **proporcional ao diâmetro de cada objeto**, não fixo por
classe. Isso não é um detalhe: com campo fixo, **nenhuma** das 20 planetárias
que inspecionamos mostrava a nebulosa — as medianas de tamanho vão de 9
segundos de arco (planetárias) a 18 minutos (remanescentes), um fator de 120.

Objetos menores que o limite de resolução do levantamento são descartados, e
não ampliados: aumentar 9 arcsec do DSS2 produz um borrão.

| coluna | significado |
|---|---|
| `ra`, `dec` | coordenadas ICRS, em graus |
| `label` | classe |
| `fov_deg` | campo de visão usado no recorte, em graus |
| `diam_arcmin` | diâmetro do catálogo, quando disponível |
| `source` | identificador VizieR de origem |
| `name` | designação no catálogo de origem |

## Deduplicação

Catálogos se sobrepõem — o Lynds Bright Nebulae inclui dezenas de nebulosas
de Sharpless sob outra nomenclatura. Objetos a menos de **2 arcmin** um do
outro são tratados como o mesmo e mantém-se a primeira ocorrência, na ordem
declarada em `data/catalogs.py`. Foram removidos {n_dup} objetos.

A proximidade é tratada como **grafo**: todas as arestas abaixo da tolerância,
componentes conexas, um representante por componente. Isso importa — uma versão
anterior comparava cada objeto apenas com seu vizinho *mais próximo*, e deixava
43 objetos a menos de 2 arcmin no catálogo, incluindo pares com separação
exatamente zero que terminaram com a mesma imagem no treino e no teste.

**Alguns objetos duplicados têm classes diferentes nos catálogos de origem** —
Sharpless nº1 é `emission` e Magakian nº644, a 0,4 arcmin de distância, é
`reflection`. A deduplicação resolve pela ordem de prioridade, o que é uma
escolha e não uma verdade. Isso põe um piso na acurácia alcançável neste
dataset, e quem publicar um número deve mencioná-lo.

{nota_absorvidos}

---

## Vieses conhecidos

**Este dataset carrega um viés geográfico forte, e ele foi medido.**

Nebulosas galácticas ficam no plano da Via Láctea. A latitude galáctica
correlaciona com a classe:

{latitudes}

Um classificador treinado nestas imagens **pode aprender a geografia em vez
do objeto**. Medimos isso por ablação: desfocando o objeto e mantendo só o
fundo, um ResNet18 mantém **94% da acurácia**. Para comparação, o mesmo teste
num classificador de galáxias (Galaxy Zoo, alta latitude) derruba o
desempenho para abaixo do chute.

Quem usar este dataset deve reportar essa ablação junto com a acurácia.

**Outras limitações:**

- `supernova_remnant` tem sinal óptico fraco — a maioria foi descoberta em
  rádio, e o DSS2 mal os mostra.
- O rótulo `emission` vs `reflection` do catálogo Lynds vem da **cor
  dominante**, que é também a feature que um modelo usaria. Não é circular
  (a cor É o critério físico), mas merece menção.
- O DSS2 são placas fotográficas digitalizadas dos anos 1990. Para objetos
  fora do plano galáctico há levantamentos melhores.

---

## Licença e citação

Este **catálogo** (o CSV e a curadoria) é MIT, como o resto do repositório.

As **imagens** reconstruídas a partir dele pertencem aos levantamentos de
origem, distribuídos pelo CDS/Observatoire de Strasbourg. Verifique os termos
de cada um antes de redistribuir pixels.

Ao usar, cite os catálogos originais listados acima e o serviço hips2fits:

> Boch T., Fernique P., et al., *hips2fits*, Centre de Données
> astronomiques de Strasbourg.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None, help="pasta de saida (padrao: docs/dataset)")
    args = parser.parse_args()

    import numpy as np
    import pandas as pd
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    from astro_classifier.data.catalogs import CATALOGS, deduplicate_by_position, query_catalog

    print("consultando os catalogos...")
    partes = []
    for spec in CATALOGS:
        try:
            partes.append(query_catalog(spec))
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERRO] {spec.vizier_id}: {exc}")

    if not partes:
        print("nenhum catalogo respondeu")
        return 1

    bruto = pd.concat(partes, ignore_index=True)
    catalogo = deduplicate_by_position(bruto)
    n_dup = len(bruto) - len(catalogo)

    gal = SkyCoord(ra=catalogo["ra"].values * u.deg, dec=catalogo["dec"].values * u.deg).galactic
    catalogo = catalogo.assign(gal_b=np.round(gal.b.deg, 4))

    colunas = ["name", "label", "ra", "dec", "gal_b", "fov_deg", "diam_arcmin", "source"]
    catalogo = catalogo[[c for c in colunas if c in catalogo.columns]].sort_values(["label", "name"])

    destino = Path(args.out) if args.out else REPO / "docs" / "dataset"
    destino.mkdir(parents=True, exist_ok=True)

    csv = destino / "nebulae_catalog.csv"
    catalogo.to_csv(csv, index=False, float_format="%.6f")
    print(f"\n{csv}  ({len(catalogo)} objetos, {csv.stat().st_size // 1024} KB)")

    # --- cartao ---
    citacoes = {s.vizier_id: s.citation for s in CATALOGS}
    composicao = "\n".join(
        f"| `{c}` | {n} |" for c, n in catalogo["label"].value_counts().sort_index().items()
    )
    fontes = "\n".join(
        f"| `{src}` | {n} | {citacoes.get(src, '—')} |"
        for src, n in catalogo["source"].value_counts().items()
    )
    regras = "\n".join(
        f"- **`{s.label if not s.label_map else '/'.join(sorted(set(s.label_map.values())))}`** "
        f"— {s.vizier_id}: "
        + (
            f"rótulo derivado da coluna `{s.label_col}`; "
            if s.label_col
            else "todo o catálogo pertence à classe; "
        )
        + (
            f"objetos abaixo de {s.min_diam_arcmin:g} arcmin descartados"
            if s.min_diam_arcmin
            else "sem filtro de tamanho"
        )
        for s in CATALOGS
    )
    # Um catalogo pode acabar quase inteiro absorvido por outro que veio
    # antes. Sem explicar, o numero baixo parece erro de consulta.
    antes_dedup = bruto["source"].value_counts()
    depois_dedup = catalogo["source"].value_counts()
    absorvidos = [
        (src, int(antes_dedup[src]), int(depois_dedup.get(src, 0)))
        for src in antes_dedup.index
        if depois_dedup.get(src, 0) < 0.25 * antes_dedup[src]
    ]
    if absorvidos:
        linhas_nota = "\n".join(
            f"- `{src}` entrou com {a} objetos e ficou com **{d}**: os demais já "
            f"estavam num catálogo anterior da lista."
            for src, a, d in absorvidos
        )
        nota_absorvidos = (
            "**Catálogos majoritariamente absorvidos.** Um número baixo aqui não é\n"
            "erro de consulta — é a deduplicação funcionando:\n\n" + linhas_nota
        )
    else:
        nota_absorvidos = ""

    b = catalogo["gal_b"].abs()
    latitudes = "\n".join(
        ["| classe | \\|b\\| mediana | em \\|b\\| < 10° |", "|---|---:|---:|"]
        + [
            f"| `{c}` | {g.abs().median():.1f}° | {(g.abs() < 10).mean():.0%} |"
            for c, g in catalogo.groupby("label")["gal_b"]
        ]
    )
    del b

    cartao = destino / "dataset_card.md"
    cartao.write_text(
        CARTAO.format(
            n_total=len(catalogo),
            n_fontes=catalogo["source"].nunique(),
            composicao=composicao,
            fontes=fontes,
            regras=regras,
            n_dup=n_dup,
            nota_absorvidos=nota_absorvidos,
            latitudes=latitudes,
        ),
        encoding="utf-8",
    )
    print(f"{cartao}")
    print(f"\n{len(catalogo)} objetos | {n_dup} duplicatas removidas")
    print(catalogo["label"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
