"""O classificador de um nivel da cascata.

Deliberadamente simples: backbone + dropout + linear. Toda a complexidade
do projeto esta nos dados e na hierarquia, nao na arquitetura - e um modelo
simples e um baseline honesto, que e o que um trabalho academico precisa
antes de comparar qualquer coisa.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from astro_classifier.config import ExperimentConfig
from astro_classifier.models.backbone import build_backbone
from astro_classifier.taxonomy import Level


class AstroClassifier(nn.Module):
    def __init__(
        self,
        level: Level,
        backbone: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.level = level
        self.backbone_name = backbone
        self.classes = list(level.classes)

        self.backbone, n_features = build_backbone(backbone, pretrained)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_features, level.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Devolve LOGITS (nao probabilidades) - a loss espera logits."""
        return self.head(self.backbone(x))

    @torch.inference_mode()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=1)

    def save(self, path: str | Path, config: ExperimentConfig, extra: dict[str, Any] | None = None) -> None:
        """Checkpoint auto-descritivo: pesos + config + taxonomia.

        Guardar as classes junto dos pesos evita o pior bug possivel do
        projeto - carregar um checkpoint antigo depois de mudar a ordem das
        classes e receber predicoes silenciosamente trocadas.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "level": self.level.value,
                "classes": self.classes,
                "backbone": self.backbone_name,
                "config": config.to_dict(),
                "extra": extra or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> tuple[AstroClassifier, dict[str, Any]]:
        """Reconstroi o modelo a partir do checkpoint. Devolve (modelo, metadados)."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        level = Level(ckpt["level"])

        if ckpt["classes"] != level.classes:
            raise RuntimeError(
                f"Checkpoint '{path}' foi treinado com as classes {ckpt['classes']}, "
                f"mas a taxonomia atual do nivel '{level.value}' e {level.classes}. "
                "Retreine o modelo ou reverta a mudanca em taxonomy.py."
            )

        model = cls(
            level=level,
            backbone=ckpt["backbone"],
            pretrained=False,  # os pesos vem do checkpoint
            dropout=ckpt["config"]["model"]["dropout"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()
        return model, ckpt


def build_model(config: ExperimentConfig) -> AstroClassifier:
    return AstroClassifier(
        level=config.level,
        backbone=config.model.backbone,
        pretrained=config.model.pretrained,
        dropout=config.model.dropout,
    )
