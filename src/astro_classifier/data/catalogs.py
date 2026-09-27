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

import numpy as np
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
    "star_field": 0.5,
    "open_cluster": 0.25,  # mesmo campo do vazio, de proposito: a diferenca
    # entre os dois deve estar no conteudo (densidade de
    # estrelas), nao na escala da imagem
}


@dataclass(frozen=True)
class CatalogSpec:
    """Como consultar um catalogo, traduzir suas colunas e enquadrar o objeto.

    O enquadramento e a parte que decide se o dataset presta. Objetos
    astronomicos variam em tamanho aparente por ordens de grandeza - as
    planetarias deste catalogo tem mediana de 9 segundos de arco, os
    remanescentes de supernova tem mediana de 18 MINUTOS de arco: 120 vezes
    maiores. Um campo de visao fixo so pode servir bem a uma das duas.

    Por isso o campo e calculado por objeto:

        fov = diametro_do_objeto x fov_multiplier

    limitado a [fov_min_deg, fov_max_deg]. O multiplicador maior que 1 deixa
    contexto ao redor - um objeto colado nas bordas do quadro fica tao ruim
    de classificar quanto um objeto de tres pixels.

    Objetos menores que `min_diam_arcmin` sao DESCARTADOS. Nenhum
    enquadramento recupera algo que o levantamento nao resolveu: ampliar 9
    segundos de arco do DSS2 produz um borrao, nao uma nebulosa.
    """

    label: str
    vizier_id: str
    name_col: str
    citation: str

    # --- tamanho angular ---
    diam_col: str | None = None
    diam_table: str | None = None  # tabela separada; None = a mesma da consulta
    join_col: str | None = None  # chave para juntar diam_table a principal
    diam_to_arcmin: float = 1.0  # fator ate arcmin (arcsec -> 1/60; raio -> 2)
    diam_cols_max: tuple[str, ...] = ()  # usa o maior entre varias colunas
    min_diam_arcmin: float = 0.0  # abaixo disto, descarta

    # --- enquadramento ---
    fov_multiplier: float = 2.0
    fov_min_deg: float = 0.02
    fov_max_deg: float = 1.5

    # --- rotulo derivado de uma coluna ---
    # Alguns catalogos trazem varios tipos de objeto numa tabela so. O LBN,
    # por exemplo, tem 1125 nebulosas brilhantes e uma coluna `Color` que
    # diz a cor dominante - que e justamente o criterio fisico que separa
    # emissao de reflexao. Valores fora do mapa sao descartados.
    label_col: str | None = None
    label_map: dict | None = None

    # --- filtro de qualidade ---
    # (coluna, valor_maximo). No LBN, `Bright` vai de 1 (mais brilhante) a 6;
    # as mais fracas nao aparecem no DSS2 e so acrescentariam ruido.
    quality_filter: tuple[str, float] | None = None


CATALOGS: list[CatalogSpec] = [
    # A ORDEM IMPORTA. A deduplicacao por posicao mantem a PRIMEIRA
    # ocorrencia, entao catalogos mais completos ou com melhor medida de
    # tamanho vem antes. O Kohoutek (1989) e mais recente que o Acker (1992)
    # em cobertura de diametro, por isso abre a lista das planetarias.
    CatalogSpec(
        label="planetary",
        vizier_id="V/127A",
        name_col="PNG",
        citation="Kohoutek (2001), Catalogue of Galactic Planetary Nebulae",
        diam_col="MajDiam",  # arcsec
        diam_to_arcmin=1.0 / 60.0,
        min_diam_arcmin=0.5,
        fov_multiplier=4.0,
        fov_min_deg=0.02,
        fov_max_deg=0.3,
    ),
    # Catalogo consolidado de nebulosas de reflexao: reune vdB, DG, Ced e
    # outros numa lista so. E a maior fonte disponivel para a classe que
    # mais sofria com falta de dados (250 objetos ate aqui).
    # Nao traz tamanho angular, entao usa o fov de reserva da classe.
    CatalogSpec(
        label="reflection",
        vizier_id="J/A+A/399/141/table1",
        name_col="Seq",
        citation="Magakian (2003), Merged catalogue of reflection nebulae, A&A 399, 141",
        fov_min_deg=0.15,
        fov_max_deg=0.15,
    ),
    CatalogSpec(
        label="emission",
        vizier_id="VII/20",
        name_col="Sh2",
        citation="Sharpless (1959), Catalogue of HII Regions, ApJS 4, 257",
        diam_col="Diam",  # ja em arcmin
        min_diam_arcmin=1.0,
        fov_multiplier=2.0,
        fov_min_deg=0.05,
        fov_max_deg=1.5,
    ),
    CatalogSpec(
        label="reflection",
        vizier_id="VII/21",
        name_col="VdB",
        citation="van den Bergh (1966), Catalogue of Reflection Nebulae, AJ 71, 990",
        diam_col="BRadMax",  # RAIO em arcmin, dai o fator 2
        diam_to_arcmin=2.0,
        min_diam_arcmin=0.5,
        fov_multiplier=2.5,
        fov_min_deg=0.05,
        fov_max_deg=1.0,
    ),
    CatalogSpec(
        label="planetary",
        vizier_id="V/84/main",
        name_col="PNG",
        citation="Acker+ (1992), Strasbourg-ESO Catalogue of Galactic Planetary Nebulae",
        diam_col="oDiam",  # arcsec, numa tabela separada do catalogo
        diam_table="V/84/diam",
        join_col="PNG",
        diam_to_arcmin=1.0 / 60.0,
        # 30 segundos de arco. Corta ~75% do catalogo, e e o corte que faz a
        # classe existir: abaixo disso o DSS2 nao resolve a nebulosa e a
        # imagem vira um campo de estrelas indistinguivel de `star_field`.
        min_diam_arcmin=0.5,
        fov_multiplier=4.0,
        fov_min_deg=0.02,
        fov_max_deg=0.3,
    ),
    CatalogSpec(
        label="supernova_remnant",
        vizier_id="VII/272",
        name_col="SNR",
        citation="Green (2009), A Catalogue of Galactic Supernova Remnants",
        diam_col="Dmaj",  # ja em arcmin
        min_diam_arcmin=2.0,
        fov_multiplier=1.8,
        fov_min_deg=0.1,
        fov_max_deg=2.0,
    ),
    # --- ampliacao das duas classes mais carentes ---
    #
    # Ate aqui o nivel das nebulosas treinava com 722 imagens para quatro
    # classes, contra 26 mil galaxias. Com tao pouco dado o modelo se apoia
    # no contexto (o campo do plano galactico) em vez do objeto - foi o que
    # o Grad-CAM mostrou. Estes dois catalogos existem para atacar isso.
    CatalogSpec(
        label="emission",  # sobrescrito por label_map
        vizier_id="VII/9",
        name_col="Seq",
        citation="Lynds (1965), Catalogue of Bright Nebulae, ApJS 12, 163",
        diam_cols_max=("Diam1", "Diam2"),  # ja em arcmin
        min_diam_arcmin=2.0,
        fov_multiplier=2.0,
        fov_min_deg=0.05,
        fov_max_deg=1.5,
        # A coluna Color e o criterio fisico: azul dominante e luz estelar
        # ESPALHADA (reflexao); vermelho dominante e emissao em H-alfa. O
        # valor 2 (neutro) e ambiguo e fica de fora, como as galaxias sem
        # consenso do Galaxy Zoo.
        label_col="Color",
        label_map={1: "reflection", 3: "emission", 4: "emission"},
        # Bright vai de 1 (mais brilhante) a 6. Acima de 4 o objeto some no
        # DSS2 e a imagem so acrescenta ruido.
        quality_filter=("Bright", 4),
    ),
    CatalogSpec(
        label="emission",
        vizier_id="VII/216",
        name_col="RCW",
        citation="Rodgers, Campbell & Whiteoak (1960), Catalogue of Ha-emission regions",
        diam_col="MajAxis",
        min_diam_arcmin=2.0,
        fov_multiplier=2.0,
        fov_min_deg=0.05,
        fov_max_deg=1.5,
    ),
]

# NGC2000.0 reune 13 mil objetos NGC/IC com uma coluna `Type` que diz o que
# cada um e. Usamos os tipos que NAO sao galaxia nem nebulosa: aglomerados
# abertos, globulares e estrelas multiplas. Sao objetos reais, brilhantes e
# catalogados - bem mais informativos para a classe `other` do que campos de
# ceu sorteados, e exatamente o tipo de coisa que o modelo precisa recusar.
NGC2000 = CatalogSpec(
    label="open_cluster",  # sobrescrito por label_map
    vizier_id="VII/118",
    name_col="Name",
    citation="Sinnott (1988), NGC2000.0 - Complete NGC and IC catalogues",
    diam_col="size",  # arcmin
    min_diam_arcmin=1.0,
    fov_multiplier=2.5,
    fov_min_deg=0.05,
    fov_max_deg=1.0,
    label_col="Type",
    # Gx (galaxia) e Nb (nebulosa) ficam de fora de proposito: o primeiro ja
    # vem do Galaxy Zoo com morfologia, e o segundo e ambiguo demais - o
    # catalogo nao diz se a nebulosa e de emissao ou reflexao.
    label_map={"OC": "open_cluster", "Gb": "globular_cluster"},
)

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
    # Filtro de qualidade antes de tudo: nao adianta enquadrar bem um objeto
    # fraco demais para o levantamento.
    if spec.quality_filter is not None:
        coluna, maximo = spec.quality_filter
        if coluna in df.columns:
            valores = pd.to_numeric(df[coluna], errors="coerce")
            manter = (valores <= maximo) & valores.notna()
            antes = len(out)
            out, df = out[manter.values].reset_index(drop=True), df[manter].reset_index(drop=True)
            print(f"  [{spec.label}] {antes - len(out)} descartados por {coluna} > {maximo}")
        else:
            print(f"  [{spec.label}] coluna de qualidade '{coluna}' ausente; filtro ignorado")

    # O rotulo pode vir de uma coluna, e nao ser fixo para o catalogo inteiro.
    if spec.label_col is not None and spec.label_map is not None:
        if spec.label_col not in df.columns:
            raise RuntimeError(
                f"coluna de rotulo '{spec.label_col}' ausente em {spec.vizier_id}. "
                f"Colunas: {list(df.columns)}"
            )
        # O mapa pode ser indexado por numero (LBN Color) ou por texto
        # (NGC2000 Type). Tenta numerico e cai para texto.
        coluna = df[spec.label_col]
        primeira_chave = next(iter(spec.label_map))
        if isinstance(primeira_chave, str):
            rotulos = coluna.astype(str).str.strip().map(spec.label_map)
        else:
            rotulos = pd.to_numeric(coluna, errors="coerce").map(spec.label_map)
        conhecidos = rotulos.notna()
        antes = len(out)
        out = out[conhecidos.values].reset_index(drop=True)
        df = df[conhecidos].reset_index(drop=True)
        out["label"] = rotulos[conhecidos].values
        print(
            f"  [{spec.vizier_id}] {antes - len(out)} descartados por "
            f"{spec.label_col} fora do mapa; restaram {out['label'].value_counts().to_dict()}"
        )
    else:
        out["label"] = spec.label

    out["source"] = spec.vizier_id
    out["diam_arcmin"] = _diameters_arcmin(spec, df, out)

    # Linhas sem coordenada valida sao inuteis para recorte.
    before = len(out)
    out = out.dropna(subset=["ra", "dec"]).reset_index(drop=True)
    if len(out) < before:
        print(f"  [{spec.label}] {before - len(out)} linhas descartadas por falta de coordenada")

    # Nomes vazios viram identificadores sinteticos, para nao colidir em disco.
    blank = out["name"].isin(["", "nan", "None"])
    out.loc[blank, "name"] = [f"{spec.label}_{i:05d}" for i in range(int(blank.sum()))]

    # Objetos pequenos demais para o levantamento sao descartados: nenhum
    # enquadramento recupera o que o telescopio nao resolveu.
    if spec.min_diam_arcmin > 0 and out["diam_arcmin"].notna().any():
        antes = len(out)
        grandes = out["diam_arcmin"] >= spec.min_diam_arcmin
        # Sem medida de tamanho, damos o beneficio da duvida e mantemos.
        out = out[grandes | out["diam_arcmin"].isna()].reset_index(drop=True)
        if len(out) < antes:
            print(
                f"  [{spec.label}] {antes - len(out)} objetos descartados por serem "
                f"menores que {spec.min_diam_arcmin:.2f} arcmin ({len(out)} restantes)"
            )

    out["fov_deg"] = _fov_degrees(spec, out["diam_arcmin"])
    return out


def deduplicate_by_position(df: pd.DataFrame, tolerancia_arcmin: float = 2.0) -> pd.DataFrame:
    """Remove objetos repetidos entre catalogos, comparando POSICAO no ceu.

    Catalogos se sobrepoem: o LBN inclui varias nebulosas de Sharpless (os
    nomes comecam com 'S '). Sem isto, a mesma nebulosa entraria duas vezes
    no dataset, possivelmente com rotulos diferentes - e, pior, poderia cair
    metade no treino e metade no teste, vazando informacao.

    Comparar nome nao resolve (cada catalogo usa sua propria nomenclatura).
    Comparar coordenada resolve: dois objetos a menos de `tolerancia_arcmin`
    um do outro sao o mesmo objeto. Mantem-se a primeira ocorrencia, ou seja,
    a ordem de CATALOGS define a prioridade.

    POR QUE ISTO USA COMPONENTES CONEXAS, E NAO O VIZINHO MAIS PROXIMO

    A primeira versao pegava, para cada objeto, o VIZINHO MAIS PROXIMO
    (`match_to_catalog_sky(nthneighbor=2)`) e descartava o de indice maior do
    par. Parece equivalente e nao e: deixou 43 objetos a menos de 2 arcmin no
    catalogo, e 6 deles puseram a MESMA imagem no treino e no teste.

    Havia DOIS furos, e vale conhecer os dois porque nenhum quebra nada -
    ambos produzem um catalogo plausivel.

    FURO 1 - cadeias. "Estar a menos de 2 arcmin" e uma relacao entre PARES,
    mas o vizinho mais proximo e uma FUNCAO: cada objeto aponta para um so.
    Com A, B, C proximos entre si, se o mais proximo de A e B, o de B e C, e o
    de C e B:

        A -> B  (B vem depois de A, A e mantido)
        B -> C  (C vem depois de B, B e mantido)
        C -> B  (B vem antes de C, C e descartado)

    A e B ficam os dois, apesar de estarem a menos de 2 arcmin. Num campo
    denso - e o plano galactico e denso - isso e comum, nao excepcional. Medido
    num teste com 40 objetos aglomerados: sobravam 20, com 19 ainda proximos.

    FURO 2 - empates exatos. Quando dois objetos tem coordenada IDENTICA, a
    busca devolve o proprio objeto como "vizinho mais proximo" (`idx[i] == i`)
    para UM dos dois, e qual dos dois e arbitrario - depende da ordem interna
    da arvore de busca. Se o auto-casamento cai no de indice MAIOR, a condicao
    `idx[i] < i` e falsa para os dois e nenhum e removido. Foi o que aconteceu
    com LBN 770 e LBN 771, que tem separacao 0,000 arcmin e geraram dois
    arquivos byte a byte identicos, um no treino e um no teste.

    A correcao e tratar a proximidade como um GRAFO: toda aresta entre
    objetos a menos da tolerancia, depois uma componente conexa por objeto
    fisico, e um representante por componente - o de menor indice, que
    preserva a prioridade de CATALOGS.

    Efeito colateral honesto: a componente conexa e transitiva. Se A~B e B~C
    mas A esta a 3 arcmin de C, os tres colapsam num objeto so. Para
    tolerancia de 2 arcmin em catalogos de nebulosa isso e o comportamento
    desejado - sao erros de posicao do mesmo objeto, nao objetos distintos.
    """
    if len(df) < 2:
        return df

    from astropy import units as u
    from astropy.coordinates import SkyCoord

    coords = SkyCoord(ra=df["ra"].values * u.deg, dec=df["dec"].values * u.deg)

    # TODAS as arestas abaixo da tolerancia, nao so a do vizinho mais proximo.
    i1, i2, _, _ = coords.search_around_sky(coords, tolerancia_arcmin * u.arcmin)

    # Union-find: cada objeto comeca sozinho; cada aresta une duas componentes.
    pai = np.arange(len(df))

    def raiz(x: int) -> int:
        while pai[x] != x:
            pai[x] = pai[pai[x]]  # compressao de caminho
            x = pai[x]
        return x

    for a, b in zip(i1, i2, strict=True):
        if a == b:  # search_around_sky devolve o par (i, i)
            continue
        ra_, rb = raiz(int(a)), raiz(int(b))
        if ra_ != rb:
            # Une sempre sob o menor indice, para o representante da
            # componente ser o objeto do catalogo de maior prioridade.
            pai[max(ra_, rb)] = min(ra_, rb)

    # Mantem um objeto por componente: o de menor indice.
    manter = np.zeros(len(df), dtype=bool)
    vistas: set[int] = set()
    for i in range(len(df)):
        r = raiz(i)
        if r not in vistas:
            vistas.add(r)
            manter[r] = True

    removidos = int((~manter).sum())
    if removidos:
        print(f"  {removidos} objetos duplicados entre catalogos (mesma posicao no ceu)")
    return df[manter].reset_index(drop=True)


def _diameters_arcmin(spec: CatalogSpec, raw: pd.DataFrame, out: pd.DataFrame) -> pd.Series:
    """Diametro angular de cada objeto, em arcmin. NaN quando desconhecido."""
    # Varias colunas de tamanho: usa a maior (eixo principal do objeto).
    if spec.diam_cols_max:
        presentes = [c for c in spec.diam_cols_max if c in raw.columns]
        if not presentes:
            print(f"  [{spec.label}] colunas {spec.diam_cols_max} ausentes; usando fov fixo")
            return pd.Series([pd.NA] * len(out), dtype="Float64")
        valores = raw[presentes].apply(pd.to_numeric, errors="coerce").max(axis=1)
        return (valores * spec.diam_to_arcmin).astype("Float64").reset_index(drop=True)

    if spec.diam_col is None:
        return pd.Series([pd.NA] * len(out), dtype="Float64")

    if spec.diam_table is None:
        if spec.diam_col not in raw.columns:
            print(f"  [{spec.label}] coluna de tamanho '{spec.diam_col}' ausente; usando fov fixo")
            return pd.Series([pd.NA] * len(out), dtype="Float64")
        valores = pd.to_numeric(raw[spec.diam_col], errors="coerce")
        return (valores * spec.diam_to_arcmin).astype("Float64").reset_index(drop=True)

    # Tamanho mora em outra tabela do mesmo catalogo (caso do V/84).
    from astroquery.vizier import Vizier

    vizier = Vizier(columns=["**"])
    vizier.ROW_LIMIT = -1
    try:
        tabelas = vizier.get_catalogs(spec.diam_table)
        extra = tabelas[0].to_pandas()
    except Exception as exc:  # noqa: BLE001 - sem tamanho ainda da para usar fov fixo
        print(f"  [{spec.label}] nao consegui ler {spec.diam_table}: {exc}; usando fov fixo")
        return pd.Series([pd.NA] * len(out), dtype="Float64")

    if spec.join_col not in extra.columns or spec.join_col not in raw.columns:
        print(f"  [{spec.label}] chave de juncao '{spec.join_col}' ausente; usando fov fixo")
        return pd.Series([pd.NA] * len(out), dtype="Float64")

    chave = raw[spec.join_col].astype(str).str.strip()
    mapa = (
        extra.assign(_k=extra[spec.join_col].astype(str).str.strip())
        .drop_duplicates("_k")
        .set_index("_k")[spec.diam_col]
    )
    valores = pd.to_numeric(chave.map(mapa), errors="coerce")
    return (valores * spec.diam_to_arcmin).astype("Float64").reset_index(drop=True)


def _fov_degrees(spec: CatalogSpec, diam_arcmin: pd.Series) -> pd.Series:
    """Campo de visao por objeto. Sem diametro conhecido, cai no valor fixo."""
    fallback = FOV_BY_LABEL[spec.label]
    fov = (diam_arcmin.astype("Float64") / 60.0) * spec.fov_multiplier
    return (
        fov.clip(lower=spec.fov_min_deg, upper=spec.fov_max_deg)
        .fillna(fallback)
        .astype(float)
    )


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
