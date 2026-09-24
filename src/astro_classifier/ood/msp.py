"""Deteccao de imagem fora do dominio (out-of-distribution).

O problema concreto do projeto: os modelos sao treinados em recortes de
levantamentos (DSS2/SDSS) - imagens fracas, de fundo escuro, com o objeto
pequeno no centro. O dashboard, porem, aceita upload livre: alguem vai
mandar uma foto processada do Hubble, uma selfie, ou o logo de um time.

Uma rede softmax NAO sabe dizer "nao sei". Ela sempre distribui 100% entre
as classes que conhece. Mandar uma foto da Lua produz "nebulosa planetaria,
97%" com toda a convicçao do mundo. Sem um detector, o dashboard mente.

Metodo implementado: MSP (Maximum Softmax Probability), de Hendrycks &
Gimpel (2017), "A Baseline for Detecting Misclassified and Out-of-
Distribution Examples in Neural Networks", ICLR. E o baseline da area:
imagens fora do dominio tendem a produzir confianca maxima mais baixa.
Simples, sem custo extra de inferencia, e - importante para o trabalho
academico - e o baseline contra o qual metodos melhores sao comparados.

Evolucao natural (Fase 2): energy-based (Liu+ 2020) ou Mahalanobis (Lee+
2018). Comparar os tres e um experimento pronto para o texto.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_THRESHOLD = 0.60


@dataclass
class OODDetector:
    """Decide se uma imagem esta fora do dominio, pelo MSP."""

    threshold: float = DEFAULT_THRESHOLD
    target_tpr: float = 0.95

    def is_out_of_domain(self, probabilities: np.ndarray) -> bool:
        return float(np.max(probabilities)) < self.threshold

    def score(self, probabilities: np.ndarray) -> float:
        """Confianca maxima. Quanto MAIOR, mais dentro do dominio."""
        return float(np.max(probabilities))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps({"threshold": self.threshold, "target_tpr": self.target_tpr}, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> OODDetector:
        """Carrega o limiar calibrado. Se o arquivo nao existe, usa o padrao -
        conservador, mas melhor do que nao avisar nada."""
        path = Path(path)
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(threshold=data["threshold"], target_tpr=data.get("target_tpr", 0.95))


def calibrate_threshold(
    in_domain_probs: np.ndarray, target_tpr: float = 0.95
) -> OODDetector:
    """Escolhe o limiar a partir do conjunto de VALIDACAO (dentro do dominio).

    Regra: manter `target_tpr` das imagens legitimas acima do limiar. Com
    target_tpr=0.95, aceitamos marcar 5% das imagens boas como suspeitas em
    troca de pegar as ruins. E um trade-off explicito - e essa escolha deve
    aparecer no texto, nao ficar escondida num numero magico.

    `in_domain_probs`: (N, num_classes), saidas softmax da validacao.
    """
    if in_domain_probs.ndim != 2:
        raise ValueError(f"esperado array (N, C), recebido {in_domain_probs.shape}")

    max_probs = in_domain_probs.max(axis=1)
    threshold = float(np.quantile(max_probs, 1.0 - target_tpr))
    return OODDetector(threshold=threshold, target_tpr=target_tpr)


def evaluate_detector(
    detector: OODDetector, in_domain_probs: np.ndarray, out_domain_probs: np.ndarray
) -> dict:
    """Mede o detector contra um conjunto realmente fora do dominio.

    Monte esse conjunto com o que o dashboard vai receber de verdade: fotos
    do Hubble baixadas da web, imagens do dia a dia, prints de tela. Sem essa
    medicao, o limiar e um chute.
    """
    from sklearn.metrics import roc_auc_score

    in_scores = in_domain_probs.max(axis=1)
    out_scores = out_domain_probs.max(axis=1)

    y_true = np.concatenate([np.ones_like(in_scores), np.zeros_like(out_scores)])
    y_score = np.concatenate([in_scores, out_scores])

    return {
        "threshold": detector.threshold,
        "auroc": float(roc_auc_score(y_true, y_score)),
        "tpr_in_domain": float((in_scores >= detector.threshold).mean()),
        "fpr_out_domain": float((out_scores >= detector.threshold).mean()),
        "n_in_domain": int(len(in_scores)),
        "n_out_domain": int(len(out_scores)),
    }
