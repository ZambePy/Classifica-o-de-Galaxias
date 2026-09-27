"""Testes dos detectores de fora-de-dominio por energia e Mahalanobis.

O que estes testes protegem, concretamente:

  o SINAL da pontuacao. Os tres metodos tem convencoes naturais opostas - a
  energia e menor dentro do dominio, o MSP e maior. Se um deles entrar
  invertido na comparacao, a AUROC sai perto de 0 em vez de perto de 1, e o
  relatorio conclui que o metodo e pessimo quando ele e otimo. E um erro que
  nao quebra nada e produz um numero plausivel - a categoria de bug que este
  projeto ja teve sete vezes.

  a magnitude dos logits importar. E o ponto do metodo de energia. Um teste
  com dois vetores de logits que tem o MESMO softmax mas magnitudes
  diferentes garante que nao estamos calculando o MSP com outro nome.
"""

from __future__ import annotations

import numpy as np
import pytest

# O CI nao instala torch de proposito (precisa do indice CUDA certo), e os
# modulos de OOD dependem dele. Pular e o comportamento certo la; localmente
# roda normal. Mesma convencao de test_tta.py e test_numerical_sanity.py.
torch = pytest.importorskip("torch", reason="exige PyTorch")

from astro_classifier.ood.energy import (  # noqa: E402
    confidence_from_energy,
    energy_score,
)
from astro_classifier.ood.mahalanobis import (  # noqa: E402
    MAX_CONDICAO,
    MahalanobisDetector,
)


class TestEnergia:
    def test_logit_alto_da_energia_baixa(self):
        """Dentro do dominio = algum logit forte = energia baixa."""
        dentro = np.array([[10.0, 0.0, 0.0]])
        fora = np.array([[0.2, 0.1, 0.1]])
        assert energy_score(dentro)[0] < energy_score(fora)[0]

    def test_confianca_inverte_o_sinal(self):
        """`confidence_from_energy` tem de seguir a convencao do MSP."""
        dentro = np.array([[10.0, 0.0, 0.0]])
        fora = np.array([[0.2, 0.1, 0.1]])
        assert confidence_from_energy(dentro)[0] > confidence_from_energy(fora)[0]

    def test_nao_e_o_msp_disfarcado(self):
        """Mesmo softmax, magnitudes diferentes: a energia TEM de distinguir.

        Multiplicar logits por uma constante muda o softmax, entao para ter o
        mesmo softmax usamos deslocamento constante - que o softmax ignora por
        construcao, mas a energia nao.
        """
        a = np.array([[2.0, 1.0, 0.0]])
        b = a + 5.0  # softmax identico, magnitude bem maior

        msp_a = torch.softmax(torch.from_numpy(a), dim=1).max().item()
        msp_b = torch.softmax(torch.from_numpy(b), dim=1).max().item()
        assert msp_a == pytest.approx(msp_b, abs=1e-6), "o softmax deveria ser igual"

        assert energy_score(a)[0] != pytest.approx(energy_score(b)[0], abs=1e-3)
        # A de magnitude maior e a "mais dentro do dominio".
        assert energy_score(b)[0] < energy_score(a)[0]

    def test_temperatura_invalida(self):
        with pytest.raises(ValueError, match="temperature"):
            energy_score(np.zeros((2, 3)), temperature=0.0)

    def test_exige_matriz(self):
        with pytest.raises(ValueError, match="N, C"):
            energy_score(np.zeros(3))

    def test_aceita_tensor_e_array(self):
        x = np.array([[1.0, 2.0, 3.0]])
        assert energy_score(x)[0] == pytest.approx(energy_score(torch.from_numpy(x))[0])


class TestMahalanobis:
    @staticmethod
    def _duas_nuvens(n=200, d=8, seed=0):
        rng = np.random.default_rng(seed)
        f = np.vstack([rng.normal(0.0, 1.0, (n, d)), rng.normal(8.0, 1.0, (n, d))])
        y = np.array([0] * n + [1] * n)
        return f, y

    def test_ponto_de_treino_pontua_mais_que_ponto_distante(self):
        f, y = self._duas_nuvens()
        det = MahalanobisDetector.fit(f, y)
        assert det.score(f[:5]).mean() > det.score(np.full((5, f.shape[1]), 50.0)).mean()

    def test_distancia_e_minima_na_classe_certa(self):
        f, y = self._duas_nuvens()
        det = MahalanobisDetector.fit(f, y)
        d = det.distances(f)
        # A classe mais proxima de cada ponto tem de ser a classe dele.
        assert (d.argmin(axis=1) == y).mean() > 0.95

    def test_corrige_pela_escala_das_features(self):
        """O ponto da distancia de Mahalanobis: escala nao euclidiana.

        Uma nuvem esticada 100x numa direcao. Um deslocamento nessa direcao
        deve custar MENOS que o mesmo deslocamento na direcao estreita - a
        euclidiana daria os dois iguais.
        """
        rng = np.random.default_rng(1)
        n = 500
        f = np.stack([rng.normal(0, 100.0, n), rng.normal(0, 1.0, n)], axis=1)
        y = np.zeros(n, dtype=int)
        det = MahalanobisDetector.fit(f, y, shrinkage=1e-6)

        larga = det.distances(np.array([[10.0, 0.0]]))[0, 0]
        estreita = det.distances(np.array([[0.0, 10.0]]))[0, 0]
        assert estreita > larga * 10

    def test_encolhimento_sobe_quando_faltam_amostras(self):
        """Menos amostras que dimensoes: a covariancia e singular.

        ESTE TESTE PEGOU UM BUG REAL. A primeira versao do detector subia o
        encolhimento dentro de um `try/except` no Cholesky. Mas o Cholesky NAO
        falha aqui: somar 1e-9 na diagonal ja torna a matriz numericamente
        positiva-definida. O encolhimento ficava em 1e-9, nenhuma excecao era
        levantada, e as distancias saiam finitas - e sem significado, porque
        dominadas por autovalores que sao ruido de estimativa.

        Agora o criterio e o numero de condicao, que e o que realmente
        descreve o problema.
        """
        rng = np.random.default_rng(2)
        f = rng.normal(0, 1, (10, 40))  # 10 amostras, 40 features: rank <= 8
        y = np.array([0] * 5 + [1] * 5)
        det = MahalanobisDetector.fit(f, y, shrinkage=1e-9)

        assert det.shrinkage > 1e-9, "o encolhimento pedido era inviavel e nao subiu"
        assert det.condition_number <= MAX_CONDICAO
        assert np.isfinite(det.score(f)).all()

    def test_condicionamento_fica_dentro_do_limite(self):
        """Mesmo no caso bem-comportado, o condicionamento vai registrado."""
        f, y = self._duas_nuvens()
        det = MahalanobisDetector.fit(f, y)
        assert np.isfinite(det.condition_number)
        assert det.condition_number <= MAX_CONDICAO

    def test_classe_vazia_e_erro_claro(self):
        f, _ = self._duas_nuvens()
        y = np.zeros(len(f), dtype=int)
        y[0] = 2  # classe 1 fica vazia
        with pytest.raises(ValueError, match="classe 1"):
            MahalanobisDetector.fit(f, y)

    def test_dimensao_incompativel_e_erro_claro(self):
        f, y = self._duas_nuvens(d=8)
        det = MahalanobisDetector.fit(f, y)
        with pytest.raises(ValueError, match="dimensao"):
            det.score(np.zeros((1, 16)))

    def test_salva_e_carrega(self, tmp_path):
        f, y = self._duas_nuvens()
        det = MahalanobisDetector.fit(f, y, classes=["a", "b"])
        caminho = tmp_path / "maha.pt"
        det.save(caminho)

        volta = MahalanobisDetector.load(caminho)
        assert volta.classes == ["a", "b"]
        assert volta.shrinkage == det.shrinkage
        assert volta.n_fit == det.n_fit
        np.testing.assert_allclose(volta.score(f[:10]), det.score(f[:10]))

    def test_rotulos_e_features_de_tamanhos_diferentes(self):
        f, y = self._duas_nuvens()
        with pytest.raises(ValueError, match="contra"):
            MahalanobisDetector.fit(f, y[:-1])


class TestComparacaoEntreMetodos:
    """Os tres na mesma convencao, medidos do mesmo jeito.

    E o contrato que `scripts/compare_ood.py` depende: se as pontuacoes
    estiverem em orientacoes diferentes, a comparacao mente.
    """

    def test_os_tres_concordam_em_caso_obvio(self):
        rng = np.random.default_rng(3)
        d = 16
        # "dentro": features perto da media de treino, logits confiantes
        f_treino = rng.normal(0, 1, (300, d))
        y_treino = rng.integers(0, 3, 300)
        det = MahalanobisDetector.fit(f_treino, y_treino)

        f_dentro = rng.normal(0, 1, (50, d))
        f_fora = rng.normal(30, 1, (50, d))
        l_dentro = np.tile([12.0, 0.0, 0.0], (50, 1))
        l_fora = np.tile([0.3, 0.2, 0.1], (50, 1))

        msp_in = torch.softmax(torch.from_numpy(l_dentro), 1).numpy().max(axis=1)
        msp_out = torch.softmax(torch.from_numpy(l_fora), 1).numpy().max(axis=1)

        # Todos: dentro > fora.
        assert msp_in.mean() > msp_out.mean()
        assert confidence_from_energy(l_dentro).mean() > confidence_from_energy(l_fora).mean()
        assert det.score(f_dentro).mean() > det.score(f_fora).mean()


class TestCalibracaoDoPercentil:
    """O score cru do Mahalanobis nao e exibivel; o percentil e.

    A distancia de Mahalanobis vale -1500 numa imagem e -3 noutra. Um dashboard
    que ponha isso numa barra de progresso quebra, e o campo `domain.score` da
    API e justamente um numero que o dashboard exibe. O percentil resolve:
    fica em [0, 1], e interpretavel ("no pior 3% das legitimas") e faz o limiar
    de 95% de aceitacao ser exatamente 0,05.
    """

    @staticmethod
    def _ajustado(seed=0, d=16):
        rng = np.random.default_rng(seed)
        f = np.vstack([rng.normal(0, 1, (300, d)), rng.normal(6, 1, (300, d))])
        y = np.array([0] * 300 + [1] * 300)
        det = MahalanobisDetector.fit(f, y, classes=["a", "b"])
        val = np.vstack([rng.normal(0, 1, (150, d)), rng.normal(6, 1, (150, d))])
        det.calibrate(val)
        return det, rng, d

    def test_percentil_fica_no_intervalo_unitario(self):
        det, rng, d = self._ajustado()
        p = det.domain_percentile(rng.normal(0, 1, (40, d)))
        assert p.min() >= 0.0 and p.max() <= 1.0

    def test_dentro_do_dominio_passa_do_limiar(self):
        """~95% das legitimas devem ficar acima de 0,05, por construcao."""
        det, rng, d = self._ajustado()
        dentro = np.vstack([rng.normal(0, 1, (100, d)), rng.normal(6, 1, (100, d))])
        assert (det.domain_percentile(dentro) >= 0.05).mean() > 0.90

    def test_fora_do_dominio_fica_perto_de_zero(self):
        det, rng, d = self._ajustado()
        fora = rng.normal(40, 1, (50, d))
        assert det.domain_percentile(fora).max() < 0.05

    def test_sem_calibrar_e_erro_explicito(self):
        """Falhar alto, em vez de devolver um numero sem escala."""
        rng = np.random.default_rng(1)
        f = rng.normal(0, 1, (100, 8))
        det = MahalanobisDetector.fit(f, np.zeros(100, dtype=int))
        with pytest.raises(RuntimeError, match="nao calibrado"):
            det.domain_percentile(f[:1])

    def test_calibracao_sobrevive_ao_disco(self, tmp_path):
        det, rng, d = self._ajustado()
        amostra = rng.normal(0, 1, (20, d))
        esperado = det.domain_percentile(amostra)

        caminho = tmp_path / "maha.pt"
        det.save(caminho)
        volta = MahalanobisDetector.load(caminho)

        assert volta.calibration_scores is not None
        np.testing.assert_allclose(volta.domain_percentile(amostra), esperado)
