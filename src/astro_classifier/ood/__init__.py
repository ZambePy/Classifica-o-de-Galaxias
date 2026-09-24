"""Deteccao de imagens fora do dominio de treino."""

from astro_classifier.ood.msp import OODDetector, calibrate_threshold, evaluate_detector

__all__ = ["OODDetector", "calibrate_threshold", "evaluate_detector"]
