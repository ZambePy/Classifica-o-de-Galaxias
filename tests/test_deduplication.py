"""Testes da deduplicacao por posicao no ceu.

O BUG QUE ESTES TESTES EXISTEM PARA IMPEDIR

A primeira versao de `deduplicate_by_position` comparava cada objeto com seu
VIZINHO MAIS PROXIMO e descartava o de indice maior do par. Passou por sete
meses parecendo correta, e a auditoria final encontrou 43 objetos a menos de
2 arcmin ainda no catalogo - incluindo pares com coordenada IDENTICA.

Consequencia medida: 10 dos 21 grupos de duplicatas caiam em conjuntos
diferentes, e 6 deles colocavam a MESMA imagem no treino e no teste. LBN 770 e
LBN 771 tem a mesma coordenada (separacao 0,000 arcmin) e produziram dois
arquivos byte-a-byte identicos, um em cada lado da divisao.

Havia duas raizes distintas, e os testes cobrem as duas:

  CADEIAS. "Estar a menos de 2 arcmin" e uma relacao entre PARES, mas o
  vizinho mais proximo e uma FUNCAO - cada objeto aponta para um unico outro.
  Em grupos de tres ou mais, dois objetos proximos podem ambos sobreviver
  porque nenhum dos dois e o vizinho MAIS proximo do outro. Medido: num campo
  de 40 objetos aglomerados, o algoritmo antigo deixava 20 objetos com 19
  ainda abaixo da tolerancia.

  EMPATES EXATOS. Com coordenada identica, a busca devolve o proprio objeto
  como vizinho mais proximo para UM dos dois, e qual e arbitrario. Caindo no
  de indice maior, a condicao `idx[i] < i` falha para ambos e nenhum sai. Foi
  o caso de LBN 770/771. Em par isolado o algoritmo antigo ate acertava - o
  que explica por que o bug sobreviveu a inspecao casual.

A correcao trata a proximidade como grafo e mantem um objeto por componente
conexa, o que elimina as duas causas de uma vez.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from astro_classifier.data.catalogs import deduplicate_by_position


def _catalogo(coords: list[tuple[float, float]], labels: list[str] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": [str(i) for i in range(len(coords))],
            "ra": [c[0] for c in coords],
            "dec": [c[1] for c in coords],
            "label": labels or ["emission"] * len(coords),
            "source": ["X"] * len(coords),
        }
    )


def _residuais(df: pd.DataFrame, tolerancia: float = 2.0) -> int:
    """Quantos objetos ainda tem vizinho abaixo da tolerancia."""
    if len(df) < 2:
        return 0
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    c = SkyCoord(ra=df["ra"].values * u.deg, dec=df["dec"].values * u.deg)
    _, sep, _ = c.match_to_catalog_sky(c, nthneighbor=2)
    return int((sep.arcmin < tolerancia).sum())


class TestCasosBasicos:
    def test_coordenada_identica_sobra_um(self):
        """O caso LBN 770/771: separacao exatamente zero.

        Em par ISOLADO o algoritmo antigo tambem passava neste teste - o
        auto-casamento caiu no indice menor. Quem pega aquele caso e
        `TestCatalogoRealPublicado`, com os 2.737 objetos reais.
        """
        df = _catalogo([(56.4939, 24.1551), (56.4939, 24.1551)])
        assert len(deduplicate_by_position(df)) == 1

    def test_par_de_separacao_zero_entre_vizinhos(self):
        """Par identico com vizinhos reais por perto, mas fora da tolerancia.

        Geometria tirada do catalogo publicado: o par em (56.4939, 24.1551) e
        seus vizinhos a ~13-15 arcmin, em direcoes diferentes de proposito -
        no mesmo azimute eles estariam a menos de 2 arcmin entre si e seriam
        duplicatas legitimas, mudando o numero esperado.
        """
        m = 1.0 / 60.0
        pontos = [
            (56.4939, 24.1551),  # LBN 770
            (56.4939, 24.1551),  # LBN 771 - separacao zero
            (56.4939 + 12.9 * m, 24.1551),
            (56.4939, 24.1551 + 13.3 * m),
            (56.4939, 24.1551 - 15.4 * m),
        ]
        resultado = deduplicate_by_position(_catalogo(pontos))
        assert len(resultado) == 4
        assert _residuais(resultado) == 0

    def test_objetos_distantes_sobrevivem(self):
        df = _catalogo([(10.0, 10.0), (200.0, -30.0), (100.0, 60.0)])
        assert len(deduplicate_by_position(df)) == 3

    def test_mantem_o_primeiro_da_lista(self):
        """A ordem de CATALOGS e a prioridade: o primeiro vence."""
        df = _catalogo([(10.0, 10.0), (10.0, 10.0)])
        df.loc[0, "source"] = "PRIORITARIO"
        df.loc[1, "source"] = "SECUNDARIO"
        resultado = deduplicate_by_position(df)
        assert len(resultado) == 1
        assert resultado["source"].iloc[0] == "PRIORITARIO"

    def test_dataframe_minusculo_passa_direto(self):
        for n in (0, 1):
            df = _catalogo([(10.0, 10.0)] * n)
            assert len(deduplicate_by_position(df)) == n


class TestOBugDoVizinhoMaisProximo:
    """O cenario exato que a versao antiga deixava passar."""

    def test_cadeia_de_tres_colapsa(self):
        """A-B-C todos proximos: tem de sobrar UM, nao dois.

        Com o algoritmo antigo, A e B sobreviviam: o vizinho mais proximo de A
        era B, o de B era C, e o de C era B. Nenhum dos dois primeiros tinha o
        outro como vizinho MAIS proximo, entao nenhum era descartado.

        As separacoes abaixo (0,5 e 0,9 arcmin) sao escolhidas para produzir
        exatamente essa assimetria.
        """
        # 1 arcmin = 1/60 grau. Em dec ~0, ra e dec tem a mesma escala.
        m = 1.0 / 60.0
        df = _catalogo([(10.0, 0.0), (10.0 + 0.9 * m, 0.0), (10.0 + 1.4 * m, 0.0)])
        resultado = deduplicate_by_position(df)

        assert len(resultado) == 1, (
            f"sobraram {len(resultado)} objetos de uma cadeia mutuamente proxima - "
            "e o bug do vizinho mais proximo de volta"
        )

    def test_nao_sobra_nenhum_residual_em_campo_denso(self):
        """Muitos objetos aglomerados: nada abaixo da tolerancia pode sobrar.

        E o teste mais importante do arquivo, porque e a propriedade que a
        funcao promete e que a versao antiga violava silenciosamente.
        """
        rng = np.random.default_rng(0)
        m = 1.0 / 60.0
        # 40 objetos espalhados em ~3 arcmin: garante muitos grupos ligados.
        pontos = [(10.0 + rng.normal(0, 1.5) * m, rng.normal(0, 1.5) * m) for _ in range(40)]
        resultado = deduplicate_by_position(_catalogo(pontos))

        assert _residuais(resultado) == 0, (
            f"{_residuais(resultado)} objetos ainda tem vizinho a menos de 2 arcmin"
        )

    def test_grupos_separados_nao_se_fundem(self):
        """Dois aglomerados distantes viram dois objetos, nao um."""
        m = 1.0 / 60.0
        df = _catalogo(
            [
                (10.0, 0.0),
                (10.0 + 0.5 * m, 0.0),  # grupo 1
                (50.0, 0.0),
                (50.0 + 0.5 * m, 0.0),  # grupo 2, longe
            ]
        )
        assert len(deduplicate_by_position(df)) == 2


class TestTolerancia:
    def test_respeita_a_tolerancia_dada(self):
        m = 1.0 / 60.0
        df = _catalogo([(10.0, 0.0), (10.0 + 5.0 * m, 0.0)])  # 5 arcmin

        assert len(deduplicate_by_position(df, tolerancia_arcmin=2.0)) == 2
        assert len(deduplicate_by_position(df, tolerancia_arcmin=10.0)) == 1

    def test_logo_acima_da_tolerancia_sobrevive(self):
        """Borda: 2,5 arcmin com tolerancia 2 tem de sobrar os dois."""
        m = 1.0 / 60.0
        df = _catalogo([(10.0, 0.0), (10.0 + 2.5 * m, 0.0)])
        assert len(deduplicate_by_position(df, tolerancia_arcmin=2.0)) == 2


class TestRotulosConflitantes:
    """Catalogos discordam sobre a classe do mesmo objeto.

    Achado da auditoria: varios pares duplicados tinham rotulos DIFERENTES -
    Sharpless nº1 e `emission` e Magakian nº644, a 0,4 arcmin, e `reflection`.
    A deduplicacao resolve pela prioridade do catalogo, o que e uma escolha,
    nao uma verdade. Este teste fixa o comportamento para que a escolha seja
    consciente.
    """

    def test_prioridade_decide_o_rotulo(self):
        df = _catalogo([(10.0, 0.0), (10.0, 0.0)], labels=["emission", "reflection"])
        resultado = deduplicate_by_position(df)
        assert len(resultado) == 1
        assert resultado["label"].iloc[0] == "emission", (
            "o rotulo mantido tem de ser o do primeiro catalogo da lista"
        )


class TestIndice:
    def test_indice_e_reiniciado(self):
        """Sem reset_index, codigo a jusante que usa .iloc/posicao quebra."""
        df = _catalogo([(10.0, 0.0), (10.0, 0.0), (50.0, 0.0)])
        resultado = deduplicate_by_position(df)
        assert list(resultado.index) == list(range(len(resultado)))

    def test_colunas_preservadas(self):
        df = _catalogo([(10.0, 0.0), (50.0, 0.0)])
        df["fov_deg"] = [1.5, 2.0]
        resultado = deduplicate_by_position(df)
        assert "fov_deg" in resultado.columns
        assert set(resultado.columns) == set(df.columns)


@pytest.mark.parametrize("n", [2, 5, 20])
def test_propriedade_nunca_sobra_residual(n):
    """Propriedade geral: a saida nunca tem par abaixo da tolerancia.

    Vale para qualquer entrada. E o invariante que o nome da funcao promete.
    """
    rng = np.random.default_rng(n)
    m = 1.0 / 60.0
    pontos = [(10.0 + rng.normal(0, 2) * m, rng.normal(0, 2) * m) for _ in range(n)]
    resultado = deduplicate_by_position(_catalogo(pontos))
    assert _residuais(resultado) == 0


class TestCatalogoRealPublicado:
    """Regressao contra os DADOS que expuseram o bug.

    Os testes sinteticos acima cobrem a logica, mas nenhum deles reproduz o
    empate exato que vazou para producao: qual elemento do par recebe o
    auto-casamento depende da arvore de busca construida sobre a distribuicao
    INTEIRA, e com cinco pontos ela nunca cai no caso ruim.

    Com os 2.737 objetos reais, cai. Este e o unico teste do arquivo que pega
    aquele caso, e ele funciona porque `docs/dataset/nebulae_catalog.csv` esta
    versionado no repositorio - o dado que revelou o problema viaja junto com
    a correcao.

    Medido no catalogo publicado: o algoritmo antigo removia 15 objetos e
    deixava 43 ainda a menos de 2 arcmin, entre eles LBN 770 e 771 nos indices
    2230 e 2232. O corrigido remove 22 e deixa zero.
    """

    @staticmethod
    def _catalogo_publicado():
        from pathlib import Path

        caminho = Path(__file__).resolve().parents[1] / "docs" / "dataset" / "nebulae_catalog.csv"
        if not caminho.exists():
            pytest.skip(f"{caminho} nao esta no repositorio")
        return pd.read_csv(caminho)

    def test_nenhum_residual_no_catalogo_real(self):
        """O invariante, no dado real. Falha com o algoritmo antigo."""
        resultado = deduplicate_by_position(self._catalogo_publicado())
        residuais = _residuais(resultado)
        assert residuais == 0, (
            f"{residuais} objetos do catalogo real ainda tem vizinho a menos de "
            "2 arcmin depois da deduplicacao"
        )

    def test_nao_remove_demais(self):
        """Sanidade na direcao oposta: nao pode dizimar o catalogo.

        Se um refactor trocasse a tolerancia de arcmin para grau, ou fundisse
        componentes por transitividade excessiva, o catalogo encolheria muito -
        e um teste que so olha residuais nao veria, porque menos objetos e
        sempre menos residuais. Este fixa o outro lado.
        """
        original = self._catalogo_publicado()
        resultado = deduplicate_by_position(original.copy())
        assert len(resultado) >= 0.95 * len(original), (
            f"a deduplicacao removeu {len(original) - len(resultado)} de "
            f"{len(original)} objetos - demais para duplicatas de catalogo"
        )

    def test_todas_as_classes_sobrevivem(self):
        original = self._catalogo_publicado()
        resultado = deduplicate_by_position(original.copy())
        assert set(resultado["label"]) == set(original["label"])


class TestGuardaDeDuplicatasEmDisco:
    """`make_splits.alertar_imagens_duplicadas` - a guarda que faltava.

    As duas vezes que duplicatas entraram no dataset, nada avisou: a coleta
    gravava com sucesso e a varredura de diretorio somava o que achava. Esta
    funcao roda na geracao dos splits e compara o CONTEUDO das imagens, que e a
    unica coisa que nao mente sobre duplicata.
    """

    @staticmethod
    def _arvore(tmp_path, arquivos: dict[str, bytes]):
        import pandas as pd

        raw = tmp_path / "raw"
        for rel, conteudo in arquivos.items():
            destino = raw / rel
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_bytes(conteudo)
        return raw, pd.DataFrame({"path": list(arquivos)})

    def test_detecta_duplicata_byte_a_byte(self, tmp_path, capsys):
        import sys

        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
        from make_splits import alertar_imagens_duplicadas

        raw, df = self._arvore(
            tmp_path,
            {
                "other/star_field/a.jpg": b"MESMO CONTEUDO",
                "other/star_field/VII_1_a.jpg": b"MESMO CONTEUDO",  # orfa do esquema antigo
                "other/empty_field/b.jpg": b"outro",
            },
        )
        extras = alertar_imagens_duplicadas(df, raw)
        assert extras == 1
        assert "ATENCAO" in capsys.readouterr().out

    def test_silencioso_quando_esta_tudo_certo(self, tmp_path, capsys):
        import sys

        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
        from make_splits import alertar_imagens_duplicadas

        raw, df = self._arvore(
            tmp_path,
            {
                "nebulae/emission/VII_20_1.jpg": b"um",
                "nebulae/emission/VII_9_1.jpg": b"dois",
                "nebulae/reflection/VII_21_1.jpg": b"tres",
            },
        )
        assert alertar_imagens_duplicadas(df, raw) == 0
        assert "sem duplicatas" in capsys.readouterr().out

    def test_arquivo_ausente_nao_derruba(self, tmp_path, capsys):
        """Um path na tabela sem arquivo em disco e aviso, nao excecao."""
        import sys

        import pandas as pd

        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
        from make_splits import alertar_imagens_duplicadas

        raw = tmp_path / "raw"
        raw.mkdir()
        df = pd.DataFrame({"path": ["nao/existe.jpg"]})
        assert alertar_imagens_duplicadas(df, raw) == 0
        assert "nao puderam ser lidas" in capsys.readouterr().out
