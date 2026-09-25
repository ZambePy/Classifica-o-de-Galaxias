"""Testes das metricas de classificacao.

Sao os numeros que vao para o relatorio. Cada caso usa um cenario onde a
resposta correta da para conferir na mao.
"""

from __future__ import annotations

import numpy as np
import pytest

from astro_classifier.evaluation.metrics import (
    classification_metrics,
    metrics_to_markdown,
)

CLASSES = ["spiral", "elliptical", "irregular"]


def test_predicao_perfeita():
    y = np.array([0, 1, 2, 0, 1, 2])
    m = classification_metrics(y, y, CLASSES)

    assert m["accuracy"] == 1.0
    assert m["balanced_accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["cohen_kappa"] == 1.0
    assert m["matthews_corrcoef"] == 1.0
    for classe in CLASSES:
        assert m["per_class"][classe]["precision"] == 1.0
        assert m["per_class"][classe]["specificity"] == 1.0


def test_modelo_que_so_chuta_a_majoritaria():
    """O caso que motivou o projeto a nao usar acuracia: 60% de acuracia com
    macro-F1 de 0,25 e MCC zero. A acuracia sozinha esconderia isso."""
    y_true = np.array([0] * 6 + [1] * 2 + [2] * 2)
    y_pred = np.zeros(10, dtype=int)
    m = classification_metrics(y_true, y_pred, CLASSES)

    assert m["accuracy"] == pytest.approx(0.6)
    assert m["balanced_accuracy"] == pytest.approx(1 / 3)
    assert m["macro_f1"] < 0.30
    assert m["matthews_corrcoef"] == pytest.approx(0.0)
    assert m["cohen_kappa"] == pytest.approx(0.0)


def test_precisao_e_recall_medem_coisas_diferentes():
    """Modelo que grita 'spiral': acha todas as espirais (recall 1) mas erra
    muito ao dizer que algo e espiral (precisao baixa)."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 0, 0, 0, 0])
    m = classification_metrics(y_true, y_pred, CLASSES)["per_class"]["spiral"]

    assert m["recall"] == 1.0
    assert m["precision"] == pytest.approx(2 / 6)
    assert m["specificity"] == 0.0, "nenhum nao-espiral foi deixado de fora"


def test_especificidade_e_calculada_corretamente():
    # 'irregular': 1 verdadeiro positivo, 0 falsos positivos, 4 nao-irregulares
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 0, 1, 1, 2])
    m = classification_metrics(y_true, y_pred, CLASSES)["per_class"]["irregular"]

    assert m["specificity"] == 1.0
    assert m["support"] == 1


def test_metricas_de_ordenacao_aparecem_com_probabilidades():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    proba = np.array(
        [[0.9, 0.05, 0.05], [0.8, 0.1, 0.1], [0.1, 0.85, 0.05],
         [0.05, 0.9, 0.05], [0.05, 0.05, 0.9], [0.1, 0.1, 0.8]]
    )
    m = classification_metrics(y_true, y_pred, CLASSES, proba)

    assert "macro_roc_auc" in m
    assert m["macro_roc_auc"] == pytest.approx(1.0)
    assert m["ranking_per_class"]["spiral"]["roc_auc"] == pytest.approx(1.0)
    assert "average_precision" in m["ranking_per_class"]["spiral"]


def test_sem_probabilidades_as_demais_metricas_continuam():
    y = np.array([0, 1, 2])
    m = classification_metrics(y, y, CLASSES)

    assert "macro_roc_auc" not in m
    assert "ranking_per_class" not in m
    assert m["accuracy"] == 1.0


def test_classe_ausente_do_conjunto_nao_derruba():
    """Acontece de verdade em conjunto de teste pequeno: uma classe rara nao
    aparece. O relatorio precisa sair mesmo assim."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    proba = np.array([[0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.1, 0.9, 0.0], [0.2, 0.8, 0.0]])

    m = classification_metrics(y_true, y_pred, CLASSES, proba)
    assert m["per_class"]["irregular"]["support"] == 0
    assert "irregular" not in m.get("ranking_per_class", {})


def test_matriz_de_confusao_tem_o_formato_certo():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([1, 1, 2])
    m = classification_metrics(y_true, y_pred, CLASSES)
    cm = m["confusion_matrix"]

    assert len(cm) == 3 and len(cm[0]) == 3
    assert cm[0][1] == 1, "uma espiral foi predita como eliptica"
    assert sum(sum(linha) for linha in cm) == 3


def test_markdown_traz_todas_as_classes_e_as_medias():
    y = np.array([0, 1, 2, 0, 1, 2])
    texto = metrics_to_markdown(classification_metrics(y, y, CLASSES), "Teste")

    assert "### Teste" in texto
    assert "Macro-F1" in texto
    assert "Kappa de Cohen" in texto
    for classe in CLASSES:
        assert classe in texto
