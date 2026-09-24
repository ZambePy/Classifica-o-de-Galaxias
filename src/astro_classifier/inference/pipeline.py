"""A cascata em producao: imagem -> nivel 1 -> nivel 2 -> resposta da API.

E aqui que a Abordagem A vira codigo. O nivel 1 decide o tipo de objeto; o
resultado dele ESCOLHE qual modelo de nivel 2 roda. Se o nivel 1 disser
"other", a cascata para - nao existe subtipo de "outro".

Consequencia conhecida e que deve ser reportada no trabalho: erro em
cascata. Uma galaxia classificada como nebulosa no nivel 1 vai receber um
subtipo de nebulosa, com confianca alta, e estara duplamente errada. Por
isso a resposta expoe `levels` com o detalhe de cada etapa, e nao so o
rotulo final - o dashboard consegue mostrar onde a decisao foi tomada.

Carregamento preguicoso (lazy): os modelos entram na memoria no primeiro
uso, nao no import. Isso mantem o modo mock leve e permite que a API suba
mesmo com apenas um dos tres checkpoints treinado.
"""

from __future__ import annotations

import io
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from astro_classifier.api.mock import build_summary_pt
from astro_classifier.api.schemas import (
    ClassScore,
    DomainCheck,
    LevelPrediction,
    PredictionResponse,
)
from astro_classifier.config import resolve_device
from astro_classifier.data.transforms import eval_transforms
from astro_classifier.models.classifier import AstroClassifier
from astro_classifier.ood.msp import OODDetector
from astro_classifier.paths import get_paths
from astro_classifier.taxonomy import DISPLAY_PT, SUBMODEL_FOR_OBJECT, Level


class ModelNotAvailable(RuntimeError):
    """Checkpoint ausente. A API traduz isto em HTTP 503."""


class HierarchicalPipeline:
    def __init__(self, checkpoints_dir: str | Path | None = None, device: str | None = None) -> None:
        paths = get_paths()
        self.checkpoints_dir = Path(checkpoints_dir or paths.checkpoints)
        self.device = device or resolve_device()
        self.transform = eval_transforms()
        self._models: dict[Level, AstroClassifier] = {}
        self._versions: dict[Level, str] = {}
        self.ood = OODDetector.load(self.checkpoints_dir / "ood_threshold.json")

    def checkpoint_path(self, level: Level) -> Path:
        return self.checkpoints_dir / f"{level.value}_best.pt"

    def loaded_levels(self) -> list[str]:
        """Quais niveis tem checkpoint disponivel em disco."""
        return [lvl.value for lvl in Level if self.checkpoint_path(lvl).exists()]

    def _get_model(self, level: Level) -> AstroClassifier:
        if level not in self._models:
            path = self.checkpoint_path(level)
            if not path.exists():
                raise ModelNotAvailable(
                    f"Checkpoint do nivel '{level.value}' nao encontrado em {path}. "
                    f"Treine com: python scripts/train.py --config configs/{_config_name(level)}"
                )
            model, ckpt = AstroClassifier.load(path, device=self.device)
            self._models[level] = model
            self._versions[level] = f"{level.value}-e{ckpt.get('extra', {}).get('epoch', '?')}"
        return self._models[level]

    def _run_level(self, tensor: torch.Tensor, level: Level) -> tuple[LevelPrediction, np.ndarray]:
        model = self._get_model(level)
        probs = model.predict_proba(tensor.to(self.device))[0].cpu().numpy()

        scores = sorted(
            (
                ClassScore(label=lbl, label_pt=DISPLAY_PT[lbl], probability=float(p))
                for lbl, p in zip(level.classes, probs, strict=True)
            ),
            key=lambda s: s.probability,
            reverse=True,
        )
        prediction = LevelPrediction(
            level=level.value,
            predicted=scores[0].label,
            predicted_pt=scores[0].label_pt,
            confidence=scores[0].probability,
            scores=scores,
            model_version=self._versions[level],
        )
        return prediction, probs

    def predict_bytes(self, image_bytes: bytes) -> PredictionResponse:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.predict_image(image)

    def predict_image(self, image: Image.Image) -> PredictionResponse:
        started = time.perf_counter()
        tensor = self.transform(image).unsqueeze(0)

        level1, probs1 = self._run_level(tensor, Level.OBJECT)
        levels = [level1]

        subtype = None
        subtype_confidence = None
        sub_level = SUBMODEL_FOR_OBJECT[level1.predicted]
        if sub_level is not None:
            try:
                level2, _ = self._run_level(tensor, sub_level)
                levels.append(level2)
                subtype = level2.predicted
                subtype_confidence = level2.confidence
            except ModelNotAvailable:
                # O modelo de topo existe mas o de refinamento ainda nao foi
                # treinado. Melhor devolver o nivel 1 do que falhar inteiro -
                # e exatamente o estado do projeto entre uma fase e outra.
                pass

        # O aviso de dominio usa o nivel 1: e ele que ve a imagem inteira e
        # cuja incerteza indica "isto nao se parece com nada que eu conheco".
        out_of_domain = self.ood.is_out_of_domain(probs1)
        domain = DomainCheck(
            out_of_domain=out_of_domain,
            score=self.ood.score(probs1),
            threshold=self.ood.threshold,
            message_pt=(
                "Esta imagem parece estar fora do dominio de treino do modelo "
                "(recortes de levantamentos DSS2/SDSS). Trate o resultado como "
                "pouco confiavel."
                if out_of_domain
                else "Imagem dentro do dominio de treino."
            ),
        )

        return PredictionResponse(
            mock=False,
            object=level1.predicted,
            confidence=level1.confidence,
            subtype=subtype,
            subtype_confidence=subtype_confidence,
            summary_pt=build_summary_pt(level1.predicted, subtype),
            levels=levels,
            domain=domain,
            inference_ms=round((time.perf_counter() - started) * 1000, 2),
        )


def _config_name(level: Level) -> str:
    return {
        Level.OBJECT: "level1_object.yaml",
        Level.GALAXY: "level2_galaxy.yaml",
        Level.NEBULA: "level3_nebula.yaml",
    }[level]


@lru_cache(maxsize=1)
def get_pipeline() -> HierarchicalPipeline:
    """Instancia unica compartilhada pela API (evita recarregar por requisicao)."""
    return HierarchicalPipeline()
