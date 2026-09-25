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


def sample_empty_fields(n: int, seed: int = 42) -> pd.DataFrame:
    """Campos de ceu em alta latitude galactica - fundo quase vazio."""
    df = sample_galactic(n, HIGH_LATITUDE_MIN, 90.0, seed=seed)
    df["label"] = "empty_field"
    df["fov_deg"] = FOV_BY_LABEL["empty_field"]
    df["source"] = f"aleatorio_|b|>{HIGH_LATITUDE_MIN:.0f}"
    df["name"] = [f"empty_{i:05d}" for i in range(len(df))]
    return df


def sample_star_fields(n: int, seed: int = 43) -> pd.DataFrame:
    """Campos no plano galactico - densos em estrelas."""
    df = sample_galactic(n, 0.0, LOW_LATITUDE_MAX, seed=seed)
    df["label"] = "star_field"
    df["fov_deg"] = FOV_BY_LABEL["star_field"]
    df["source"] = f"aleatorio_|b|<{LOW_LATITUDE_MAX:.0f}"
    df["name"] = [f"starfield_{i:05d}" for i in range(len(df))]
    return df
