"""Baixa recortes do ceu por coordenada, via servico hips2fits do CDS.

Ideia central da coleta: nao existe um "Galaxy Zoo das nebulosas" com
imagens rotuladas. O que existe sao CATALOGOS - listas de objetos com
coordenadas e tipo, compiladas por astronomos ao longo de decadas. O
servico hips2fits permite pedir "me de a imagem do ceu nesta coordenada,
neste tamanho angular", em qualquer levantamento (survey) publicado.

Juntando os dois, o dataset se constroi sozinho:

    catalogo (ra, dec, tipo)  ->  hips2fits  ->  imagem rotulada

Servico: https://alasky.cds.unistra.fr/hips-image-services/hips2fits
Politica de uso: e um servico publico e gratuito. Baixe com pausa entre
requisicoes (o parametro `pause`) e nao paralelize agressivamente.
"""

from __future__ import annotations

import time
from pathlib import Path

import requests

HIPS2FITS_URL = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"

# Levantamentos disponiveis como HiPS.
# DSS2 color cobre o ceu inteiro - e o unico que serve para nebulosas
# galacticas, que ficam no plano da Via Lactea (onde o SDSS nao olhou).
# SDSS9 color tem qualidade melhor, mas cobre so ~1/3 do ceu (calotas
# galacticas) - e onde estao as galaxias do Galaxy Zoo.
SURVEY_DSS2 = "CDS/P/DSS2/color"
SURVEY_SDSS9 = "CDS/P/SDSS9/color"

DEFAULT_TIMEOUT = 60


class CutoutError(RuntimeError):
    """Falha ao obter um recorte (rede, servico fora do ar, coordenada invalida)."""


def fetch_cutout(
    ra: float,
    dec: float,
    fov_deg: float,
    *,
    survey: str = SURVEY_DSS2,
    size_px: int = 256,
    fmt: str = "jpg",
    timeout: int = DEFAULT_TIMEOUT,
) -> bytes:
    """Recorte centrado em (ra, dec) com campo de visao `fov_deg` graus.

    `fov_deg` importa muito: uma nebulosa planetaria tem segundos de arco e
    some num campo de 1 grau; uma nebulosa de emissao como a de Orion tem
    graus e nao cabe num campo de 0.05. Por isso o fov e por tipo de objeto
    (ver FOV_BY_LABEL em catalogs.py).
    """
    params = {
        "hips": survey,
        "ra": ra,
        "dec": dec,
        "fov": fov_deg,
        "width": size_px,
        "height": size_px,
        "projection": "TAN",
        "coordsys": "icrs",
        "format": fmt,
    }
    try:
        response = requests.get(HIPS2FITS_URL, params=params, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise CutoutError(f"falha ao baixar recorte ra={ra} dec={dec}: {exc}") from exc

    if not response.content:
        raise CutoutError(f"recorte vazio para ra={ra} dec={dec}")
    return response.content


def download_cutouts(
    rows: list[dict],
    out_dir: str | Path,
    *,
    survey: str = SURVEY_DSS2,
    size_px: int = 256,
    pause: float = 0.5,
    skip_existing: bool = True,
    progress=None,
) -> list[dict]:
    """Baixa um lote de recortes.

    Cada item de `rows` precisa de: name, ra, dec, fov_deg, label.
    Devolve os registros que foram salvos com sucesso, com a chave `path`
    adicionada (relativa a `out_dir.parent.parent`, pronta para o CSV).

    Falhas individuais nao abortam o lote - o ceu tem objetos mal
    catalogados e o servico as vezes recusa uma coordenada. O relatorio
    final diz quantos vieram.
    """
    out_dir = Path(out_dir)
    saved: list[dict] = []

    for i, row in enumerate(rows, start=1):
        label_dir = out_dir / row["label"]
        label_dir.mkdir(parents=True, exist_ok=True)
        dest = label_dir / cutout_filename(row)

        if skip_existing and dest.exists():
            saved.append({**row, "path": str(dest.relative_to(out_dir.parent)).replace("\\", "/")})
            continue

        try:
            content = fetch_cutout(
                ra=float(row["ra"]),
                dec=float(row["dec"]),
                fov_deg=float(row["fov_deg"]),
                survey=survey,
                size_px=size_px,
            )
        except CutoutError as exc:
            if progress:
                progress(f"  [falha] {row['name']}: {exc}")
            continue

        dest.write_bytes(content)
        saved.append({**row, "path": str(dest.relative_to(out_dir.parent)).replace("\\", "/")})

        if progress and i % 25 == 0:
            progress(f"  {i}/{len(rows)} processados, {len(saved)} salvos")
        time.sleep(pause)

    return saved


def cutout_filename(row: dict) -> str:
    """Nome de arquivo UNICO para um objeto, prefixado pelo catalogo.

    O prefixo nao e enfeite. Os catalogos usam numeracao propria e
    independente: a regiao numero 1 de Sharpless, a numero 1 de RCW e a
    numero 1 de Lynds sao tres objetos diferentes, em pontos distantes do
    ceu. Sem o prefixo, os tres viram `emission/1.jpg` e dois se perdem.

    Foi exatamente o que aconteceu: 301 dos 1667 objetos desapareceram,
    sobrescritos uns pelos outros, e o log ainda assim dizia "1667 salvos" -
    porque a gravacao funcionou, so que no mesmo arquivo. O bug so apareceu
    ao contar os arquivos em disco.
    """
    fonte = _slug(row.get("source", "")) if row.get("source") else ""
    nome = _slug(row["name"])
    return f"{fonte}_{nome}.jpg" if fonte else f"{nome}.jpg"


def _slug(name: str) -> str:
    """Texto -> pedaco de nome de arquivo seguro. 'Sh 2-155' -> 'Sh_2-155'."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name).strip())
    return safe.strip("_") or "unnamed"
