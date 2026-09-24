"""Catalogos astronomicos -> tabela (nome, ra, dec, tipo).

Cada tipo de nebulosa vem de um catalogo classico diferente. Sao listas
compiladas por astronomos, publicadas e citaveis - o que da ao dataset uma
procedencia defensavel no texto academico. Todos sao consultados pelo
VizieR (CDS/Strasbourg) atraves do astroquery.

    emission           Sharpless (1959)       regioes HII
    reflection         van den Bergh (1966)   nebulosas de reflexao
    planetary          Acker+ (1992)          catalogo Strasbourg-ESO de PNe
    supernova_remnant  Green (2009+)          remanescentes galacticos

AVISO DE HONESTIDADE: os identificadores VizieR abaixo e os nomes de coluna
sao o ponto mais fragil do modulo - o VizieR muda esquemas entre edicoes.
`inspect_catalog()` existe para voce conferir o que voltou ANTES de baixar
milhares de imagens. Rode-o na primeira vez.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# Campo de visao tipico por tipo, em graus. E a diferenca entre baixar a
# nebulosa e baixar um pedaco vazio de ceu ao redor dela.
FOV_BY_LABEL: dict[str, float] = {
    "planetary": 0.05,  # ~3 arcmin: PNe sao compactas
    "reflection": 0.5,
    "emission": 0.8,  # regioes HII sao extensas
    "supernova_remnant": 0.7,
    "globular_cluster": 0.3,
    "empty_field": 0.5,
}


@dataclass(frozen=True)
class CatalogSpec:
    """Como consultar um catalogo e como traduzir suas colunas."""

    label: str
    vizier_id: str
    name_col: str
    citation: str


CATALOGS: list[CatalogSpec] = [
    CatalogSpec(
        label="emission",
        vizier_id="VII/20",
        name_col="Sh2",
        citation="Sharpless (1959), Catalogue of HII Regions, ApJS 4, 257",
    ),
    CatalogSpec(
        label="reflection",
        vizier_id="VII/21",
        name_col="vdB",
        citation="van den Bergh (1966), Catalogue of Reflection Nebulae, AJ 71, 990",
    ),
    CatalogSpec(
        label="planetary",
        vizier_id="V/84",
        name_col="PNG",
        citation="Acker+ (1992), Strasbourg-ESO Catalogue of Galactic Planetary Nebulae",
    ),
    CatalogSpec(
        label="supernova_remnant",
        vizier_id="VII/272",
        name_col="SNR",
        citation="Green (2009), A Catalogue of Galactic Supernova Remnants",
    ),
]

# Catalogo usado para parte da classe "other": aglomerados globulares sao
# objetos extensos e brilhantes que NAO sao galaxias nem nebulosas - sao
# exatamente o tipo de coisa que o modelo precisa aprender a recusar.
GLOBULAR_CLUSTERS = CatalogSpec(
    label="globular_cluster",
    vizier_id="VII/202",
    name_col="Name",
    citation="Harris (1996, 2010 ed.), Catalog of Parameters for Milky Way Globular Clusters",
)

RA_CANDIDATES = ["_RAJ2000", "RAJ2000", "RA_ICRS", "_RA"]
DEC_CANDIDATES = ["_DEJ2000", "DEJ2000", "DE_ICRS", "_DE"]


def query_catalog(spec: CatalogSpec, limit: int = -1) -> pd.DataFrame:
    """Consulta o VizieR e devolve colunas normalizadas.

    Colunas de saida: name, ra, dec, label, fov_deg, source.
    `limit=-1` traz o catalogo inteiro; use um limite pequeno para inspecionar.
    """
    from astroquery.vizier import Vizier

    vizier = Vizier(columns=["**", "_RAJ2000", "_DEJ2000"])
    vizier.ROW_LIMIT = limit

    tables = vizier.get_catalogs(spec.vizier_id)
    if len(tables) == 0:
        raise RuntimeError(f"VizieR nao devolveu tabela para {spec.vizier_id!r}")

    df = tables[0].to_pandas()

    ra_col = _first_present(df, RA_CANDIDATES)
    dec_col = _first_present(df, DEC_CANDIDATES)
    if ra_col is None or dec_col is None:
        raise RuntimeError(
            f"Nao encontrei colunas de coordenada em {spec.vizier_id!r}. "
            f"Colunas disponiveis: {list(df.columns)}. "
            "Ajuste RA_CANDIDATES/DEC_CANDIDATES ou o CatalogSpec."
        )

    name_col = spec.name_col if spec.name_col in df.columns else df.columns[0]

    out = pd.DataFrame(
        {
            "name": df[name_col].astype(str).str.strip(),
            "ra": pd.to_numeric(df[ra_col], errors="coerce"),
            "dec": pd.to_numeric(df[dec_col], errors="coerce"),
        }
    )
    out["label"] = spec.label
    out["fov_deg"] = FOV_BY_LABEL[spec.label]
    out["source"] = spec.vizier_id

    # Linhas sem coordenada valida sao inuteis para recorte.
    before = len(out)
    out = out.dropna(subset=["ra", "dec"]).reset_index(drop=True)
    if len(out) < before:
        print(f"  [{spec.label}] {before - len(out)} linhas descartadas por falta de coordenada")

    # Nomes vazios viram identificadores sinteticos, para nao colidir em disco.
    blank = out["name"].isin(["", "nan", "None"])
    out.loc[blank, "name"] = [f"{spec.label}_{i:05d}" for i in range(int(blank.sum()))]

    return out


def inspect_catalog(spec: CatalogSpec, n: int = 5) -> None:
    """Imprime as primeiras linhas cruas. RODE ISTO ANTES do download completo."""
    from astroquery.vizier import Vizier

    vizier = Vizier(columns=["**", "_RAJ2000", "_DEJ2000"])
    vizier.ROW_LIMIT = n
    tables = vizier.get_catalogs(spec.vizier_id)

    print(f"\n=== {spec.label} ({spec.vizier_id}) ===")
    print(f"Citacao: {spec.citation}")
    if len(tables) == 0:
        print("  NENHUMA TABELA RETORNADA - o identificador VizieR pode ter mudado.")
        return
    df = tables[0].to_pandas()
    print(f"  Colunas: {list(df.columns)}")
    print(df.head(n).to_string())


def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None
