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
import os
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
    def __init__(
        self,
        checkpoints_dir: str | Path | None = None,
        device: str | None = None,
        use_tta: bool = False,
        ood_method: str | None = None,
    ) -> None:
        # TTA: classifica as 8 simetrias da imagem e usa a media. No ceu nao
        # existe orientacao privilegiada, entao as 8 vistas sao igualmente
        # validas - ver inference/tta.py. Custa 8x mais inferencias (~60ms
        # em vez de ~20ms), o que continua instantaneo para um dashboard.
        self.use_tta = use_tta
        paths = get_paths()
        self.checkpoints_dir = Path(checkpoints_dir or paths.checkpoints)
        self.device = device or resolve_device()
        self.transform = eval_transforms()
        self._models: dict[Level, AstroClassifier] = {}
        self._versions: dict[Level, str] = {}
        self.ood = OODDetector.load(self.checkpoints_dir / "ood_threshold.json")

        # --- qual metodo decide "fora do dominio" ---
        #
        # O padrao continua o MSP, e NAO por ele ser melhor: medido em tres
        # levantamentos, o MSP tem AUROC media 0,727 e aceita 2 de cada 3
        # imagens de fora, contra 0,976 e 1 em 7 do Mahalanobis (ver
        # docs/results.md). O padrao e o MSP porque trocar o metodo muda o
        # SIGNIFICADO do campo `domain.score` na resposta da API, e esse campo
        # e parte do contrato v1.0 com o dashboard. Quebrar isso sem combinar
        # nao se faz.
        #
        # Para ligar o metodo melhor: ood_method="mahalanobis", ou a variavel
        # de ambiente ASTRO_OOD_METHOD=mahalanobis. O score continua em [0, 1]
        # (e o percentil entre as imagens legitimas, nao a max softmax), entao
        # uma barra de progresso no dashboard continua funcionando - o que muda
        # e a interpretacao do numero.
        self.ood_method = (ood_method or os.environ.get("ASTRO_OOD_METHOD") or "msp").lower()
        if self.ood_method not in {"msp", "mahalanobis"}:
            raise ValueError(
                f"ood_method '{self.ood_method}' invalido. Use: msp | mahalanobis"
            )

        self.maha = None
        if self.ood_method == "mahalanobis":
            caminho = self.checkpoints_dir / "ood_mahalanobis.pt"
            if not caminho.exists():
                raise ModelNotAvailable(
                    f"ood_method='mahalanobis' pedido, mas {caminho} nao existe. "
                    "Gere e calibre o detector com: python scripts/compare_ood.py"
                )
            from astro_classifier.ood.mahalanobis import MahalanobisDetector

            self.maha = MahalanobisDetector.load(caminho)
            if self.maha.calibration_scores is None:
                raise ModelNotAvailable(
                    f"{caminho} existe mas nao esta calibrado - sem isso o score "
                    "nao tem escala para exibir. Rode scripts/compare_ood.py de novo."
                )

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
        if self.use_tta:
            from astro_classifier.inference.tta import predict_proba_tta

            probs = predict_proba_tta(model, tensor, self.device)[0].cpu().numpy()
        else:
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
        if self.maha is not None:
            # Features do penultimo layer do MESMO modelo de nivel 1. Nao ha
            # custo de inferencia extra relevante: uma passada a mais no
            # backbone, ~20 ms, contra a chance de aceitar uma imagem que nao
            # deveria ser aceita.
            modelo1 = self._get_model(Level.OBJECT)
            with torch.inference_mode():
                feats = modelo1.backbone(tensor.to(self.device)).flatten(1).cpu().numpy()
            # Limiar 0,05 = aceitar 95% das imagens legitimas, por construcao
            # do percentil. Nao e numero magico: e o mesmo alvo de TPR que
            # calibra o MSP.
            limiar = round(1.0 - self.ood.target_tpr, 4)
            pontuacao = float(self.maha.domain_percentile(feats)[0])
            out_of_domain = pontuacao < limiar
        else:
            out_of_domain = self.ood.is_out_of_domain(probs1)
            pontuacao = self.ood.score(probs1)
            limiar = self.ood.threshold

        domain = DomainCheck(
            out_of_domain=out_of_domain,
            score=pontuacao,
            threshold=limiar,
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
