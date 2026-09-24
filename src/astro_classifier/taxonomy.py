"""Taxonomia do classificador hierarquico.

Fonte unica de verdade para nomes e ordem das classes. Treino, avaliacao,
API e dashboard leem daqui - se uma classe mudar, muda em um lugar so.

A ordem das listas DEFINE o indice de saida de cada modelo. Nao reordene
sem retreinar: o checkpoint antigo passa a apontar para a classe errada.
"""

from __future__ import annotations

from enum import Enum

# Nivel 1 - que tipo de objeto e este?
OBJECT_CLASSES: list[str] = ["galaxy", "nebula", "other"]

# Nivel 2a - morfologia de galaxia (Hubble simplificado)
GALAXY_CLASSES: list[str] = ["spiral", "elliptical", "irregular"]

# Nivel 2b - tipo de nebulosa
NEBULA_CLASSES: list[str] = ["emission", "reflection", "planetary", "supernova_remnant"]


class Level(str, Enum):
    """Os tres modelos independentes da cascata."""

    OBJECT = "object"
    GALAXY = "galaxy"
    NEBULA = "nebula"

    @property
    def classes(self) -> list[str]:
        return LEVEL_CLASSES[self]

    @property
    def num_classes(self) -> int:
        return len(self.classes)


LEVEL_CLASSES: dict[Level, list[str]] = {
    Level.OBJECT: OBJECT_CLASSES,
    Level.GALAXY: GALAXY_CLASSES,
    Level.NEBULA: NEBULA_CLASSES,
}

# Qual modelo de nivel 2 roda depois de cada predicao do nivel 1.
# "other" nao tem refinamento - a cascata para ali.
SUBMODEL_FOR_OBJECT: dict[str, Level | None] = {
    "galaxy": Level.GALAXY,
    "nebula": Level.NEBULA,
    "other": None,
}

# Rotulos legiveis para o dashboard (pt-BR).
DISPLAY_PT: dict[str, str] = {
    "galaxy": "Galaxia",
    "nebula": "Nebulosa",
    "other": "Outro objeto",
    "spiral": "Espiral",
    "elliptical": "Eliptica",
    "irregular": "Irregular",
    "emission": "Nebulosa de emissao",
    "reflection": "Nebulosa de reflexao",
    "planetary": "Nebulosa planetaria",
    "supernova_remnant": "Remanescente de supernova",
}


def label_to_index(level: Level, label: str) -> int:
    """Indice de saida do modelo para um rotulo. Levanta ValueError se invalido."""
    try:
        return level.classes.index(label)
    except ValueError as exc:
        raise ValueError(
            f"'{label}' nao e uma classe valida do nivel {level.value}. "
            f"Validas: {level.classes}"
        ) from exc


def index_to_label(level: Level, index: int) -> str:
    """Rotulo correspondente a um indice de saida do modelo."""
    classes = level.classes
    if not 0 <= index < len(classes):
        raise ValueError(f"indice {index} fora do intervalo do nivel {level.value} (0..{len(classes)-1})")
    return classes[index]
