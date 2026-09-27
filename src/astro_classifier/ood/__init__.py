"""Deteccao de imagens fora do dominio de treino.

Tres metodos, em ordem de quanto do modelo eles olham:

    msp          probabilidade maxima depois do softmax. Baseline da area.
    energy       log-sum-exp dos logits, antes de normalizar.
    mahalanobis  distancia no espaco de features do penultimo layer.

Nenhum exige retreino: todos leem um modelo ja treinado. Compare os tres com
`scripts/compare_ood.py`.
"""

from astro_classifier.ood.energy import confidence_from_energy, energy_score
from astro_classifier.ood.mahalanobis import MahalanobisDetector, extract_features
from astro_classifier.ood.msp import OODDetector, calibrate_threshold, evaluate_detector

__all__ = [
    "OODDetector",
    "calibrate_threshold",
    "evaluate_detector",
    "energy_score",
    "confidence_from_energy",
    "MahalanobisDetector",
    "extract_features",
]
