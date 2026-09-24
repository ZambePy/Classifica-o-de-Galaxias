"""Dataset do PyTorch alimentado por CSV de split.

Formato do CSV (uma linha por imagem):

    path,label,source
    nebulae/planetary/HASH_0421.jpg,planetary,HASH
    galaxy_zoo/100008.jpg,spiral,GalaxyZoo2

`path` e relativo a raiz de imagens passada ao Dataset. Manter os splits em
CSV (e nao em pastas por classe) permite que a mesma imagem participe de
niveis diferentes da cascata sem duplicar arquivo em disco.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from astro_classifier.taxonomy import Level, label_to_index

REQUIRED_COLUMNS = {"path", "label"}


class AstroImageDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        images_root: str | Path,
        level: Level,
        transform=None,
    ) -> None:
        self.df = pd.read_csv(csv_path)
        missing = REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            raise ValueError(f"{csv_path} nao tem as colunas obrigatorias: {sorted(missing)}")

        unknown = set(self.df["label"]) - set(level.classes)
        if unknown:
            raise ValueError(
                f"{csv_path} contem rotulos que nao pertencem ao nivel "
                f"'{level.value}': {sorted(unknown)}. Esperadas: {level.classes}"
            )

        self.images_root = Path(images_root)
        self.level = level
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        image = Image.open(self.images_root / row["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label_to_index(self.level, row["label"])

    def class_counts(self) -> dict[str, int]:
        counts = self.df["label"].value_counts().to_dict()
        return {c: int(counts.get(c, 0)) for c in self.level.classes}

    def class_weights(self) -> torch.Tensor:
        """Pesos inversamente proporcionais a frequencia.

        Sem isto, com ~2500 nebulosas planetarias contra ~150 de reflexao, o
        modelo aprende a chutar 'planetary' e acerta 80% - uma acuracia alta e
        completamente inutil.
        """
        counts = torch.tensor(
            [max(self.class_counts()[c], 1) for c in self.level.classes], dtype=torch.float
        )
        weights = counts.sum() / (len(counts) * counts)
        return weights


def build_dataloaders(
    splits_dir: str | Path,
    images_root: str | Path,
    level: Level,
    batch_size: int,
    num_workers: int,
    train_tf,
    eval_tf,
) -> tuple[DataLoader, DataLoader, AstroImageDataset]:
    """Monta os loaders de treino e validacao. Devolve tambem o dataset de
    treino, porque quem chama precisa dele para os pesos de classe."""
    splits_dir = Path(splits_dir)
    train_ds = AstroImageDataset(splits_dir / f"{level.value}_train.csv", images_root, level, train_tf)
    val_ds = AstroImageDataset(splits_dir / f"{level.value}_val.csv", images_root, level, eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_ds
