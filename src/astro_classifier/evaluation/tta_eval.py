"""Avaliacao com test-time augmentation.

Separado do loop normal porque o TTA processa UMA imagem por vez (cada uma
vira um batch de 8 vistas), enquanto o `evaluate` normal processa batches de
imagens. Misturar os dois complicaria o caminho rapido sem necessidade.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from astro_classifier.inference.tta import predict_proba_tta


def evaluate_with_tta(model, dataset, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devolve (y_true, y_pred, probabilidades) usando as 8 simetrias."""
    model.eval()
    verdadeiros, probabilidades = [], []

    for i in tqdm(range(len(dataset)), desc="TTA", leave=False):
        imagem, alvo = dataset[i]
        probs = predict_proba_tta(model, imagem.unsqueeze(0), device)
        probabilidades.append(probs[0].cpu().numpy())
        verdadeiros.append(int(alvo))

    proba = np.stack(probabilidades)
    return np.array(verdadeiros), proba.argmax(axis=1), proba
