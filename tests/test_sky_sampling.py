"""Testes da amostragem de ceu e da exclusao de objetos catalogados.

O DEFEITO QUE ESTES TESTES IMPEDEM

`star_field` sao pedacos de ceu sorteados no plano galactico, e o projeto os usa
como GRUPO DE CONTROLE: a premissa e que nao ha nebulosa neles, entao chamar um
deles de nebulosa e prova de que o modelo decide pelo fundo.

A premissa nao era garantida. O recorte tem 30 arcmin de campo, o sorteio era
livre, e o plano galactico e exatamente onde as nebulosas moram. Medido nos 300
campos coletados:

    23 continham o CENTRO de uma nebulosa catalogada
    46 continham alguma parte dela

No conjunto de teste eram 6 de 45 - e o relatorio afirmava que 9 dos 45 eram
erros do modelo. Ate dois tercos dos "erros" podiam ser ACERTOS, e o numero que
sustentava a conclusao sobre o atalho estava inflado.

Nada disso quebrava: o codigo rodava, as imagens baixavam, o numero saia
plausivel. Era um erro de DESENHO EXPERIMENTAL, e o unico jeito de pegar e
checar a premissa contra o catalogo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from astro_classifier.data.sky_sampling import (
    HIGH_LATITUDE_MIN,
    LOW_LATITUDE_MAX,
    drop_near_catalog,
    sample_empty_fields,
    sample_star_fields,
)


def _campos(coords, fov_deg=0.5):
    return pd.DataFrame(
        {
            "ra": [c[0] for c in coords],
            "dec": [c[1] for c in coords],
            "fov_deg": [fov_deg] * len(coords),
        }
    )


def _catalogo(coords, diam_arcmin=None):
    d = pd.DataFrame({"ra": [c[0] for c in coords], "dec": [c[1] for c in coords]})
    if diam_arcmin is not None:
        d["diam_arcmin"] = diam_arcmin
    return d


def _contaminados(campos, catalogo) -> int:
    """Conta, de forma independente do codigo testado, quantos sobraram sujos."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    if campos.empty:
        return 0
    c = SkyCoord(ra=campos["ra"].values * u.deg, dec=campos["dec"].values * u.deg)
    cn = SkyCoord(ra=catalogo["ra"].values * u.deg, dec=catalogo["dec"].values * u.deg)
    idx, sep, _ = c.match_to_catalog_sky(cn)
    meio = campos["fov_deg"].values * 60.0 / 2.0
    raio = (
        np.nan_to_num(
            pd.to_numeric(catalogo["diam_arcmin"], errors="coerce").values[idx] / 2.0,
            nan=0.0,
        )
        if "diam_arcmin" in catalogo.columns
        else np.zeros(len(campos))
    )
    return int((sep.arcmin < (meio + raio)).sum())


class TestDropNearCatalog:
    def test_remove_objeto_dentro_do_recorte(self):
        """Campo de 30', objeto a 5' do centro: esta dentro, tem de sair."""
        m = 1.0 / 60.0
        campos = _campos([(10.0, 0.0)], fov_deg=0.5)  # 30 arcmin
        cat = _catalogo([(10.0 + 5.0 * m, 0.0)])
        assert len(drop_near_catalog(campos, cat)) == 0

    def test_mantem_objeto_fora_do_recorte(self):
        """Objeto a 40' do centro de um campo de 30': fora, pode ficar."""
        m = 1.0 / 60.0
        campos = _campos([(10.0, 0.0)], fov_deg=0.5)
        cat = _catalogo([(10.0 + 40.0 * m, 0.0)])
        assert len(drop_near_catalog(campos, cat)) == 1

    def test_considera_o_raio_do_objeto(self):
        """Objeto grande alcanca o recorte mesmo com centro fora dele.

        Centro a 20' (fora do meio-campo de 15'), mas a nebulosa tem 30' de
        diametro, logo 15' de raio: a borda dela entra no quadro.
        """
        m = 1.0 / 60.0
        campos = _campos([(10.0, 0.0)], fov_deg=0.5)
        longe = _catalogo([(10.0 + 20.0 * m, 0.0)], diam_arcmin=[0.1])
        grande = _catalogo([(10.0 + 20.0 * m, 0.0)], diam_arcmin=[30.0])

        assert len(drop_near_catalog(campos, longe)) == 1, "pontual nao deveria alcancar"
        assert len(drop_near_catalog(campos, grande)) == 0, "objeto de 30' alcanca"

    def test_margem_extra(self):
        m = 1.0 / 60.0
        campos = _campos([(10.0, 0.0)], fov_deg=0.5)
        cat = _catalogo([(10.0 + 20.0 * m, 0.0)])
        assert len(drop_near_catalog(campos, cat, margem_arcmin=0.0)) == 1
        assert len(drop_near_catalog(campos, cat, margem_arcmin=10.0)) == 0

    def test_entrada_vazia_passa_direto(self):
        vazio = pd.DataFrame(columns=["ra", "dec", "fov_deg"])
        cat = _catalogo([(10.0, 0.0)])
        assert drop_near_catalog(vazio, cat).empty
        assert len(drop_near_catalog(_campos([(10.0, 0.0)]), pd.DataFrame())) == 1

    def test_preserva_colunas_e_reinicia_indice(self):
        m = 1.0 / 60.0
        campos = _campos([(10.0, 0.0), (50.0, 0.0), (80.0, 0.0)])
        campos["label"] = "star_field"
        cat = _catalogo([(50.0 + 1.0 * m, 0.0)])
        r = drop_near_catalog(campos, cat)
        assert len(r) == 2
        assert "label" in r.columns
        assert list(r.index) == [0, 1]


class TestAmostragemComExclusao:
    """O caso real: sortear no plano galactico evitando o catalogo."""

    @staticmethod
    def _catalogo_denso(n=1500, seed=0):
        """Objetos no PLANO GALACTICO, com a densidade do catalogo real.

        Duas coisas que a primeira versao deste helper errou, e que valem ficar
        registradas porque sao faceis de repetir:

        1. Sorteava `dec` entre -5 e +5, ou seja, a banda EQUATORIAL. Mas os
           campos sao sorteados em |b| < 5, a banda GALACTICA - as duas estao
           inclinadas ~60 graus uma da outra e quase nao se cruzam. O teste
           media a sobreposicao de duas regioes diferentes do ceu, e por isso
           nunca achava contaminacao.

        2. Usava 400 objetos, densidade baixa demais. O catalogo real tem 2.737
           nebulosas concentradas no plano; com campo de 30 arcmin isso da ~15%
           de contaminacao, que e o numero medido. Com 400 espalhados dava
           quase zero, e um teste que nunca ve o problema nao protege de nada.
        """
        from astro_classifier.data.sky_sampling import sample_galactic

        rng = np.random.default_rng(seed)
        cat = sample_galactic(n, 0.0, LOW_LATITUDE_MAX, seed=seed)[["ra", "dec"]].copy()
        cat["diam_arcmin"] = rng.uniform(1, 30, len(cat))
        return cat

    def test_star_field_sem_catalogo_contamina(self):
        """Sem a exclusao, alguns campos contem objeto - e o bug original."""
        cat = self._catalogo_denso()
        campos = sample_star_fields(200, seed=7)
        assert _contaminados(campos, cat) > 0, (
            "com um catalogo denso no plano galactico, o sorteio livre tem de "
            "produzir contaminacao - se nao produz, o teste nao esta medindo nada"
        )

    def test_star_field_com_catalogo_fica_limpo(self):
        cat = self._catalogo_denso()
        campos = sample_star_fields(200, seed=7, catalogo=cat)
        assert _contaminados(campos, cat) == 0

    def test_entrega_a_quantidade_pedida(self):
        """Filtrar nao pode devolver menos campos que o pedido, em silencio."""
        cat = self._catalogo_denso()
        campos = sample_star_fields(200, seed=7, catalogo=cat)
        assert len(campos) == 200

    def test_empty_field_com_catalogo_fica_limpo(self):
        cat = self._catalogo_denso()
        campos = sample_empty_fields(100, seed=3, catalogo=cat)
        assert _contaminados(campos, cat) == 0
        assert len(campos) == 100

    def test_latitudes_continuam_corretas(self):
        """A exclusao nao pode empurrar os campos para fora da faixa pedida.

        Seria o jeito mais facil de "limpar" o conjunto e destruir o
        experimento: `star_field` FORA do plano galactico deixa de ser o
        controle que interessa.
        """
        from astropy import units as u
        from astropy.coordinates import SkyCoord

        cat = self._catalogo_denso()

        sf = sample_star_fields(120, seed=11, catalogo=cat)
        b = SkyCoord(ra=sf["ra"].values * u.deg, dec=sf["dec"].values * u.deg).galactic.b.deg
        assert np.abs(b).max() <= LOW_LATITUDE_MAX + 1e-6

        ef = sample_empty_fields(120, seed=12, catalogo=cat)
        b = SkyCoord(ra=ef["ra"].values * u.deg, dec=ef["dec"].values * u.deg).galactic.b.deg
        assert np.abs(b).min() >= HIGH_LATITUDE_MIN - 1e-6

    def test_colunas_esperadas(self):
        cat = self._catalogo_denso()
        for f in (sample_star_fields, sample_empty_fields):
            df = f(20, seed=5, catalogo=cat)
            assert {"ra", "dec", "label", "fov_deg", "source", "name"} <= set(df.columns)
            assert df["name"].nunique() == len(df), "nomes repetidos viram colisao de arquivo"
