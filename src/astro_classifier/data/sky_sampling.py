"""Amostragem de coordenadas aleatorias no ceu, por regiao galactica.

Serve para montar a classe `other` sem catalogo: em vez de listar objetos,
sorteamos pedacos de ceu com uma propriedade conhecida.

A chave e a LATITUDE GALACTICA (b), que diz o quanto uma direcao se afasta
do plano da Via Lactea:

    |b| proximo de 0   ->  olhando ATRAVES do disco da galaxia.
                           Campo denso de estrelas, poeira, nebulosas.

    |b| grande         ->  olhando para FORA do disco, perpendicular a ele.
                           Poucas estrelas em primeiro plano, fundo quase
                           vazio (galaxias distantes e fracas demais para o
                           DSS aparecem raramente).

Por isso `empty_field` sorteia em alta latitude e `star_field` em baixa. As
duas classes usam o mesmo campo de visao de proposito: o que as distingue
deve ser o conteudo, nao a escala.

CUIDADO METODOLOGICO: "alta latitude" reduz muito a chance de cair em cima
de um objeto notavel, mas nao zera. Uma fracao pequena dos campos vazios vai
conter uma galaxia de verdade, e isso e ruido de rotulo dentro da classe
`other`. Inspecione uma amostra antes de treinar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from astro_classifier.data.catalogs import FOV_BY_LABEL

# Fronteiras em graus de latitude galactica.
HIGH_LATITUDE_MIN = 40.0  # |b| acima disto: longe do disco
LOW_LATITUDE_MAX = 5.0  # |b| abaixo disto: dentro do disco


def sample_galactic(
    n: int,
    b_min: float,
    b_max: float,
    seed: int = 42,
) -> pd.DataFrame:
    """Sorteia `n` direcoes com |b| entre b_min e b_max, uniformes na esfera.

    Detalhe que importa: sortear `b` uniformemente em graus concentraria os
    pontos perto dos polos galacticos. A amostragem uniforme em AREA exige
    sortear sin(b) uniformemente - e o que fazemos aqui.
    """
    if not 0.0 <= b_min < b_max <= 90.0:
        raise ValueError(f"faixa de latitude invalida: [{b_min}, {b_max}]")

    from astropy import units as u
    from astropy.coordinates import SkyCoord

    rng = np.random.default_rng(seed)

    lon = rng.uniform(0.0, 360.0, size=n)

    sin_lo, sin_hi = np.sin(np.radians([b_min, b_max]))
    sin_b = rng.uniform(sin_lo, sin_hi, size=n)
    lat = np.degrees(np.arcsin(sin_b))
    # Metade no hemisferio galactico sul.
    lat *= rng.choice([-1.0, 1.0], size=n)

    icrs = SkyCoord(l=lon * u.deg, b=lat * u.deg, frame="galactic").icrs

    return pd.DataFrame(
        {
            "ra": icrs.ra.deg,
            "dec": icrs.dec.deg,
            "gal_l": lon,
            "gal_b": lat,
        }
    )


def drop_near_catalog(
    campos: pd.DataFrame,
    catalogo: pd.DataFrame,
    margem_arcmin: float = 0.0,
    progress=None,
) -> pd.DataFrame:
    """Remove campos sorteados cujo recorte alcanca um objeto do catalogo.

    ISTO CORRIGE UM DEFEITO DE DESENHO QUE INVALIDAVA UM RESULTADO.

    `star_field` e usado como GRUPO DE CONTROLE na analise de atalho: sao
    pedacos do plano galactico "sem nebulosa alguma", e a fracao deles que o
    modelo chama de nebulosa mede quanto ele responde ao fundo em vez do
    objeto. O argumento so vale se realmente nao houver nebulosa ali.

    Nao havia essa garantia. Medido nos 300 campos coletados: o recorte usa
    30 arcmin de campo (meio-campo 15'), e 23 deles continham o CENTRO de uma
    nebulosa catalogada, 46 continham alguma parte. No conjunto de teste, 6 dos
    45 campos - e o relatorio dizia que 9 dos 45 eram "erros". Ou seja: ate dois
    tercos dos supostos erros podiam ser ACERTOS, e o numero do controle estava
    inflado.

    O criterio: o objeto entra no recorte se a separacao angular for menor que
    (meio-campo + raio do objeto + margem). Usa o `diam_arcmin` do catalogo
    quando existe; sem ele, trata o objeto como pontual, que e o caso
    conservador na direcao errada - por isso a `margem_arcmin`.
    """
    if campos.empty or catalogo.empty:
        return campos

    from astropy import units as u
    from astropy.coordinates import SkyCoord

    c = SkyCoord(ra=campos["ra"].values * u.deg, dec=campos["dec"].values * u.deg)
    cat = SkyCoord(ra=catalogo["ra"].values * u.deg, dec=catalogo["dec"].values * u.deg)
    idx, sep, _ = c.match_to_catalog_sky(cat)

    meio_campo = campos["fov_deg"].values * 60.0 / 2.0
    if "diam_arcmin" in catalogo.columns:
        raio = np.nan_to_num(
            pd.to_numeric(catalogo["diam_arcmin"], errors="coerce").values[idx] / 2.0,
            nan=0.0,
        )
    else:
        raio = np.zeros(len(campos))

    contaminado = sep.arcmin < (meio_campo + raio + margem_arcmin)
    n_fora = int(contaminado.sum())
    if n_fora and progress:
        progress(
            f"  {n_fora} de {len(campos)} campos sorteados alcancavam um objeto "
            f"do catalogo e foram descartados"
        )
    return campos[~contaminado].reset_index(drop=True)


def _sortear_limpo(
    n: int,
    b_min: float,
    b_max: float,
    seed: int,
    fov_deg: float,
    catalogo: pd.DataFrame | None,
    margem_arcmin: float,
    progress=None,
) -> pd.DataFrame:
    """Sorteia n campos livres do catalogo, sobre-amostrando quanto precisar.

    Sobre-amostra em vez de sortear exatamente n e filtrar depois: filtrar
    devolveria menos que n, e um conjunto menor que o pedido e o tipo de coisa
    que passa despercebida e desequilibra a classe.
    """
    if catalogo is None or catalogo.empty:
        df = sample_galactic(n, b_min, b_max, seed=seed)
        df["fov_deg"] = fov_deg
        return df

    limpos: list[pd.DataFrame] = []
    obtidos = 0
    for tentativa in range(6):
        # Pede o que falta com folga; a folga cresce se a regiao for suja.
        pedido = max(16, int((n - obtidos) * (1.6 + 0.4 * tentativa)))
        bruto = sample_galactic(pedido, b_min, b_max, seed=seed + 1000 * tentativa)
        bruto["fov_deg"] = fov_deg
        ok = drop_near_catalog(bruto, catalogo, margem_arcmin, progress=None)
        if not ok.empty:
            limpos.append(ok)
            obtidos += len(ok)
        if obtidos >= n:
            break

    if not limpos:
        raise RuntimeError(
            f"nao consegui sortear nenhum campo livre do catalogo em "
            f"|b| entre {b_min} e {b_max}"
        )

    df = pd.concat(limpos, ignore_index=True).head(n).reset_index(drop=True)
    if len(df) < n and progress:
        progress(
            f"  [aviso] pedi {n} campos limpos e consegui {len(df)} - a regiao "
            "esta densa de objetos catalogados"
        )
    return df


def sample_empty_fields(
    n: int,
    seed: int = 42,
    catalogo: pd.DataFrame | None = None,
    margem_arcmin: float = 1.0,
    progress=None,
) -> pd.DataFrame:
    """Campos de ceu em alta latitude galactica - fundo quase vazio.

    Passe `catalogo` (o DataFrame de nebulosas) para garantir que nenhum campo
    contenha um objeto catalogado. Em alta latitude isso quase nunca acontece -
    medido, 0 de 300 - mas o custo de checar e zero e a garantia deixa de
    depender de sorte.
    """
    df = _sortear_limpo(
        n, HIGH_LATITUDE_MIN, 90.0, seed,
        FOV_BY_LABEL["empty_field"], catalogo, margem_arcmin, progress,
    )
    df["label"] = "empty_field"
    df["source"] = f"aleatorio_|b|>{HIGH_LATITUDE_MIN:.0f}"
    df["name"] = [f"empty_{i:05d}" for i in range(len(df))]
    return df


def sample_star_fields(
    n: int,
    seed: int = 43,
    catalogo: pd.DataFrame | None = None,
    margem_arcmin: float = 1.0,
    progress=None,
) -> pd.DataFrame:
    """Campos no plano galactico - densos em estrelas.

    AQUI O `catalogo` IMPORTA DE VERDADE. Estes campos sao o grupo de controle
    da analise de atalho, e o plano galactico e justamente onde vivem as
    nebulosas: sem filtrar, 15% dos recortes continham uma delas. Ver
    `drop_near_catalog`.
    """
    df = _sortear_limpo(
        n, 0.0, LOW_LATITUDE_MAX, seed,
        FOV_BY_LABEL["star_field"], catalogo, margem_arcmin, progress,
    )
    df["label"] = "star_field"
    df["source"] = f"aleatorio_|b|<{LOW_LATITUDE_MAX:.0f}"
    df["name"] = [f"starfield_{i:05d}" for i in range(len(df))]
    return df
