"""Testes da avaliacao de ponta a ponta.

As metricas da cascata sao as que vao para o trabalho academico. Um erro de
contagem aqui vira um numero errado no texto, e ninguem percebe - nao ha
nada com que comparar. Por isso os casos abaixo usam cenarios montados a
mao, onde o resultado certo e obvio.
"""

from __future__ import annotations

import pytest

from astro_classifier.evaluation.cascade import CascadeSample, evaluate_cascade


def amostra(obj_v, obj_p, sub_v=None, sub_p=None) -> CascadeSample:
    return CascadeSample(
        objeto_verdadeiro=obj_v,
        objeto_predito=obj_p,
        subtipo_verdadeiro=sub_v,
        subtipo_predito=sub_p,
    )


def test_tudo_certo_da_um():
    amostras = [
        amostra("galaxy", "galaxy", "spiral", "spiral"),
        amostra("nebula", "nebula", "planetary", "planetary"),
        amostra("other", "other"),
    ]
    r = evaluate_cascade(amostras)

    assert r.nivel1_acuracia == 1.0
    assert r.folha_acuracia == 1.0
    assert r.erro_cascata == 0.0
    assert r.subtipo_acuracia_condicional == 1.0


def test_erro_de_nivel_1_conta_como_erro_em_cascata():
    """Uma galaxia chamada de nebulosa recebe subtipo de nebulosa. O rotulo
    final esta errado mesmo que o modelo de nebulosa tenha 'funcionado'."""
    amostras = [
        amostra("galaxy", "nebula", "spiral", "planetary"),
        amostra("galaxy", "galaxy", "spiral", "spiral"),
    ]
    r = evaluate_cascade(amostras)

    assert r.nivel1_acuracia == 0.5
    assert r.erro_cascata == 0.5
    assert r.folha_acuracia == 0.5


def test_acuracia_condicional_ignora_quem_nao_teve_chance():
    """A condicional mede o nivel 2 SO onde o nivel 1 acertou. Sem isso, um
    subtipo bom pareceria ruim por culpa do nivel de cima."""
    amostras = [
        # nivel 1 errou: o subtipo nunca teve chance, nao deve pesar
        amostra("galaxy", "nebula", "spiral", "planetary"),
        # nivel 1 acertou e o subtipo tambem
        amostra("galaxy", "galaxy", "spiral", "spiral"),
        amostra("galaxy", "galaxy", "elliptical", "elliptical"),
    ]
    r = evaluate_cascade(amostras)

    assert r.subtipo_acuracia_condicional == 1.0, "as duas com chance acertaram"
    assert r.folha_acuracia == pytest.approx(2 / 3)


def test_other_nao_tem_subtipo_e_ainda_pode_estar_certo():
    amostras = [amostra("other", "other"), amostra("other", "other")]
    r = evaluate_cascade(amostras)

    assert r.folha_acuracia == 1.0
    assert r.por_ramo["other"]["subtipo_acuracia_condicional"] is None
    assert r.por_ramo["other"]["tem_submodelo"] is False


def test_folha_usa_subtipo_quando_existe():
    a = amostra("galaxy", "galaxy", "spiral", "elliptical")
    assert a.folha_verdadeira == "spiral"
    assert a.folha_predita == "elliptical"
    assert a.folha_correta is False
    assert a.nivel1_correto is True, "o nivel 1 acertou; quem errou foi o subtipo"


def test_subtipo_errado_nao_conta_como_erro_em_cascata():
    """Distincao central: errar o subtipo e diferente de nunca ter chegado
    ao submodelo certo. Confundir os dois esconde o custo da arquitetura."""
    amostras = [amostra("galaxy", "galaxy", "spiral", "irregular")]
    r = evaluate_cascade(amostras)

    assert r.erro_cascata == 0.0, "o nivel 1 acertou"
    assert r.folha_acuracia == 0.0
    assert r.subtipo_acuracia_condicional == 0.0


def test_matriz_da_folha_cruza_classes_de_ramos_diferentes():
    """A matriz por nivel nunca mostra uma espiral virando planetaria,
    porque nenhum nivel sozinho ve os dois rotulos. Esta mostra."""
    amostras = [
        amostra("galaxy", "nebula", "spiral", "planetary"),
        amostra("galaxy", "galaxy", "spiral", "spiral"),
    ]
    r = evaluate_cascade(amostras)

    assert "spiral" in r.classes_folha
    assert "planetary" in r.classes_folha

    i = r.classes_folha.index("spiral")
    j = r.classes_folha.index("planetary")
    assert r.matriz_folha[i][j] == 1, "uma espiral terminou como planetaria"
    assert r.matriz_folha[i][i] == 1


def test_metricas_por_ramo_somam_o_total():
    amostras = [
        amostra("galaxy", "galaxy", "spiral", "spiral"),
        amostra("galaxy", "nebula", "spiral", "planetary"),
        amostra("nebula", "nebula", "emission", "emission"),
        amostra("other", "galaxy"),
    ]
    r = evaluate_cascade(amostras)

    assert sum(m["n"] for m in r.por_ramo.values()) == r.n_amostras == 4
    assert r.erro_cascata == 0.5, "duas das quatro erraram no nivel 1"


def test_lista_vazia_falha_em_vez_de_dividir_por_zero():
    with pytest.raises(ValueError, match="nenhuma amostra"):
        evaluate_cascade([])
