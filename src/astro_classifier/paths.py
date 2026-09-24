"""Onde cada coisa mora no disco.

Regra do projeto: CODIGO no repositorio (sincronizado/versionado),
DADOS e CHECKPOINTS em ASTRO_DATA_ROOT, fora do OneDrive.

Motivo: o Galaxy Zoo sozinho tem ~61 mil imagens. Dentro de uma pasta
sincronizada, o cliente do OneDrive tenta subir cada arquivo e degrada a
maquina inteira. Alem disso, checkpoints de 45 MB versionados a cada
epoca estouram qualquer cota.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(REPO_ROOT / ".env")

DEFAULT_DATA_ROOT = Path("C:/astro-data") if os.name == "nt" else Path.home() / "astro-data"


@dataclass(frozen=True)
class Paths:
    """Arvore de diretorios de dados, derivada de uma unica raiz."""

    root: Path

    @property
    def raw(self) -> Path:
        """Downloads intactos, exatamente como vieram da fonte."""
        return self.root / "raw"

    @property
    def raw_galaxy_zoo(self) -> Path:
        return self.raw / "galaxy_zoo"

    @property
    def raw_nebulae(self) -> Path:
        return self.raw / "nebulae"

    @property
    def raw_other(self) -> Path:
        return self.raw / "other"

    @property
    def processed(self) -> Path:
        """Imagens ja recortadas/redimensionadas, prontas para o DataLoader."""
        return self.root / "processed"

    @property
    def catalogs(self) -> Path:
        """CSVs de catalogos astronomicos (coordenadas + tipo)."""
        return self.root / "catalogs"

    @property
    def splits(self) -> Path:
        """CSVs train/val/test. Versionavel, pequeno - copie para docs se quiser citar."""
        return self.root / "splits"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def runs(self) -> Path:
        """Logs, metricas e figuras de cada execucao de treino."""
        return self.root / "runs"

    def ensure(self) -> Paths:
        """Cria toda a arvore. Idempotente."""
        for p in (
            self.raw_galaxy_zoo,
            self.raw_nebulae,
            self.raw_other,
            self.processed,
            self.catalogs,
            self.splits,
            self.checkpoints,
            self.runs,
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths() -> Paths:
    """Le ASTRO_DATA_ROOT do ambiente (.env ou variavel de sistema)."""
    root = os.getenv("ASTRO_DATA_ROOT")
    return Paths(Path(root).expanduser() if root else DEFAULT_DATA_ROOT)
