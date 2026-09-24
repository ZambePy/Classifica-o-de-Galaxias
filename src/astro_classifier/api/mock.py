"""Gerador de respostas sinteticas que respeitam o contrato.

Existe para que o dashboard seja construido ANTES de qualquer modelo estar
treinado. A resposta e deterministica por imagem (hash do conteudo), entao
subir a mesma foto sempre devolve o mesmo resultado - o que torna possivel
escrever testes de frontend contra a API mock.
"""

from __future__ import annotations

import hashlib
import random

from astro_classifier.api.schemas import (
    ClassScore,
    DomainCheck,
    LevelPrediction,
    PredictionResponse,
)
from astro_classifier.taxonomy import (
    DISPLAY_PT,
    SUBMODEL_FOR_OBJECT,
    Level,
)

MOCK_VERSION = "mock-0.1"
MOCK_OOD_THRESHOLD = 0.60


def _rng_for(image_bytes: bytes) -> random.Random:
    """RNG estavel derivado do conteudo da imagem."""
    seed = int.from_bytes(hashlib.sha256(image_bytes).digest()[:8], "big")
    return random.Random(seed)


def _fake_scores(rng: random.Random, level: Level) -> list[ClassScore]:
    """Distribuicao plausivel: uma classe dominante, resto dividido."""
    raw = [rng.random() ** 2 for _ in level.classes]
    winner = rng.randrange(len(raw))
    raw[winner] += 1.5
    total = sum(raw)
    scores = [
        ClassScore(label=lbl, label_pt=DISPLAY_PT[lbl], probability=v / total)
        for lbl, v in zip(level.classes, raw, strict=True)
    ]
    return sorted(scores, key=lambda s: s.probability, reverse=True)


def _level_prediction(rng: random.Random, level: Level) -> LevelPrediction:
    scores = _fake_scores(rng, level)
    top = scores[0]
    return LevelPrediction(
        level=level.value,
        predicted=top.label,
        predicted_pt=top.label_pt,
        confidence=top.probability,
        scores=scores,
        model_version=MOCK_VERSION,
    )


def mock_prediction(image_bytes: bytes) -> PredictionResponse:
    """Resposta completa e coerente, sem modelo carregado."""
    rng = _rng_for(image_bytes)

    level1 = _level_prediction(rng, Level.OBJECT)
    levels = [level1]

    subtype = None
    subtype_confidence = None
    sub_level = SUBMODEL_FOR_OBJECT[level1.predicted]
    if sub_level is not None:
        level2 = _level_prediction(rng, sub_level)
        levels.append(level2)
        subtype = level2.predicted
        subtype_confidence = level2.confidence

    out_of_domain = level1.confidence < MOCK_OOD_THRESHOLD
    domain = DomainCheck(
        out_of_domain=out_of_domain,
        score=level1.confidence,
        threshold=MOCK_OOD_THRESHOLD,
        message_pt=(
            "Esta imagem parece estar fora do dominio de treino do modelo. "
            "Trate o resultado como pouco confiavel."
            if out_of_domain
            else "Imagem dentro do dominio de treino."
        ),
    )

    return PredictionResponse(
        mock=True,
        object=level1.predicted,
        confidence=level1.confidence,
        subtype=subtype,
        subtype_confidence=subtype_confidence,
        summary_pt=build_summary_pt(level1.predicted, subtype),
        levels=levels,
        domain=domain,
        inference_ms=round(rng.uniform(8.0, 40.0), 2),
    )


def build_summary_pt(obj: str, subtype: str | None) -> str:
    """'galaxy' + 'spiral' -> 'Galaxia espiral'. Usado tambem pela API real."""
    if subtype is None:
        return DISPLAY_PT[obj]
    if obj == "galaxy":
        return f"{DISPLAY_PT['galaxy']} {DISPLAY_PT[subtype].lower()}"
    return DISPLAY_PT[subtype]
