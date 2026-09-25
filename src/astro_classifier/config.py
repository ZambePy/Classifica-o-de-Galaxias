"""Configuracao de experimentos, carregada de YAML.

Um YAML por modelo da cascata (configs/level1_object.yaml, etc). Todo
treino grava o YAML usado junto do checkpoint - sem isso o resultado nao e
reproduzivel, e reprodutibilidade e o ponto do projeto.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from astro_classifier.taxonomy import Level


@dataclass
class DataConfig:
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    # Balanceamento da loss. Ver AstroImageDataset.class_weights():
    #   balanced       peso cheio, inverso da frequencia
    #   sqrt_balanced  raiz do peso cheio - use quando a razao entre a maior
    #                  e a menor classe passa de ~5x, para nao trocar recall
    #                  alto por precisao baixa
    #   none           sem peso
    class_weights: str = "sqrt_balanced"
    augment: bool = True
    # Treinar com a distribuicao de votos humanos em vez do rotulo vencedor.
    # Exige colunas p_<classe> no CSV (ver prepare_galaxy_zoo.py). A
    # VALIDACAO continua usando o rotulo duro - e contra ele que as metricas
    # sao reportadas.
    soft_labels: bool = False
    soft_label_temperature: float = 1.0


@dataclass
class ModelConfig:
    backbone: str = "resnet18"  # resnet18 | resnet50 | efficientnet_b0
    pretrained: bool = True
    dropout: float = 0.2
    # Congela o backbone nas primeiras epocas (util com datasets pequenos).
    freeze_backbone_epochs: int = 0


@dataclass
class OptimConfig:
    epochs: int = 15
    lr: float = 3e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"  # cosine | none
    label_smoothing: float = 0.0
    early_stopping_patience: int = 5
    # AMP: essencial nos 6 GB da GTX 1660 Ti.
    mixed_precision: bool = True


@dataclass
class ExperimentConfig:
    name: str
    level: Level
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=raw["name"],
            level=Level(raw["level"]),
            seed=raw.get("seed", 42),
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            optim=OptimConfig(**raw.get("optim", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["level"] = self.level.value
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )


def resolve_device(requested: str | None = None) -> str:
    """'auto' vira cuda se houver GPU, senao cpu. Importa torch so aqui."""
    requested = requested or os.getenv("ASTRO_DEVICE", "auto")
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
