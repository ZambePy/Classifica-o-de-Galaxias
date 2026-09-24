"""Testes do detector de dominio.

O detector e o que impede o dashboard de afirmar com 97% de confianca que
uma foto de cachorro e uma nebulosa planetaria. Vale ter teste.
"""

from __future__ import annotations

import numpy as np

from astro_classifier.ood.msp import OODDetector, calibrate_threshold, evaluate_detector


def test_confianca_alta_fica_dentro_do_dominio():
    detector = OODDetector(threshold=0.6)
    assert not detector.is_out_of_domain(np.array([0.95, 0.03, 0.02]))


def test_confianca_baixa_e_marcada_como_fora_do_dominio():
    detector = OODDetector(threshold=0.6)
    assert detector.is_out_of_domain(np.array([0.40, 0.35, 0.25]))


def test_score_e_a_probabilidade_maxima():
    detector = OODDetector()
    assert detector.score(np.array([0.1, 0.7, 0.2])) == 0.7


def test_calibracao_mantem_a_fracao_pedida_de_imagens_legitimas():
    rng = np.random.default_rng(42)
    # 1000 predicoes confiantes, como as de um conjunto de validacao.
    confidences = rng.uniform(0.55, 0.99, size=1000)
    probs = np.stack([confidences, 1 - confidences], axis=1)

    detector = calibrate_threshold(probs, target_tpr=0.95)
    aceitas = (probs.max(axis=1) >= detector.threshold).mean()

    assert aceitas >= 0.94, "o limiar rejeitou imagens legitimas demais"


def test_detector_separa_dominios_distintos():
    rng = np.random.default_rng(0)
    dentro = np.stack([c := rng.uniform(0.80, 0.99, 500), 1 - c], axis=1)
    fora = np.stack([c := rng.uniform(0.35, 0.60, 500), 1 - c], axis=1)

    detector = calibrate_threshold(dentro, target_tpr=0.95)
    resultado = evaluate_detector(detector, dentro, fora)

    assert resultado["auroc"] > 0.95
    assert resultado["fpr_out_domain"] < 0.10


def test_limiar_padrao_quando_o_arquivo_nao_existe(tmp_path):
    """A API precisa subir mesmo antes de qualquer calibracao."""
    detector = OODDetector.load(tmp_path / "nao_existe.json")
    assert 0.0 < detector.threshold < 1.0


def test_limiar_sobrevive_ao_disco(tmp_path):
    original = OODDetector(threshold=0.7123, target_tpr=0.9)
    destino = tmp_path / "ood.json"
    original.save(destino)

    recarregado = OODDetector.load(destino)
    assert recarregado.threshold == original.threshold
    assert recarregado.target_tpr == original.target_tpr
