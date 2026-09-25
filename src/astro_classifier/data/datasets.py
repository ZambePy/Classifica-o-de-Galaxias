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
        soft_labels: bool = False,
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

        # Soft labels: o alvo deixa de ser um indice e passa a ser a
        # distribuicao de votos humanos. Exige colunas p_<classe> no CSV,
        # gravadas por prepare_galaxy_zoo.py.
        self.soft_labels = soft_labels
        self.prob_cols = [f"p_{c}" for c in level.classes]
        if soft_labels:
            faltando = [c for c in self.prob_cols if c not in self.df.columns]
            if faltando:
                raise ValueError(
                    f"{csv_path} nao tem as colunas de probabilidade {faltando}. "
                    "Soft labels exigem um indice gerado por prepare_galaxy_zoo.py "
                    "e splits regerados depois dele."
                )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int | torch.Tensor]:
        row = self.df.iloc[idx]
        image = Image.open(self.images_root / row["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if self.soft_labels:
            probs = torch.tensor(
                [float(row[c]) for c in self.prob_cols], dtype=torch.float32
            )
            soma = probs.sum()
            # Linha degenerada vira uniforme em vez de NaN na loss.
            probs = probs / soma if soma > 0 else torch.full_like(probs, 1.0 / len(probs))
            return image, probs

        return image, label_to_index(self.level, row["label"])

    def class_counts(self) -> dict[str, int]:
        counts = self.df["label"].value_counts().to_dict()
        return {c: int(counts.get(c, 0)) for c in self.level.classes}

    def class_weights(self, mode: str = "balanced") -> torch.Tensor:
        """Pesos por classe, para compensar o desbalanceamento na loss.

        Sem peso nenhum, com 15 mil espirais contra 1,6 mil irregulares, o
        modelo aprende a ignorar a classe rara.

        Com peso cheio (`balanced`), o efeito contrario aparece: o peso de
        5,5x que o `irregular` recebia levou o modelo a prever essa classe
        em excesso - recall 0,886 mas precisao 0,579, com 184 espirais
        classificadas como irregulares. Compensar demais tambem e erro.

        `sqrt_balanced` usa a raiz do peso cheio (5,5 -> 2,3). Continua
        favorecendo a classe rara, sem transformar o modelo num gerador de
        falsos positivos. E o meio-termo usual quando a razao entre classes
        passa de ~5x.

            balanced       n_total / (n_classes * n_da_classe)
            sqrt_balanced  raiz do anterior
            none           todos iguais a 1
        """
        contagem = torch.tensor(
            [max(self.class_counts()[c], 1) for c in self.level.classes], dtype=torch.float
        )
        cheio = contagem.sum() / (len(contagem) * contagem)

        if mode == "balanced":
            return cheio
        if mode == "sqrt_balanced":
            return cheio.sqrt()
        if mode == "none":
            return torch.ones_like(cheio)
        raise ValueError(
            f"class_weights '{mode}' invalido. Use: balanced | sqrt_balanced | none"
        )


def build_dataloaders(
    splits_dir: str | Path,
    images_root: str | Path,
    level: Level,
    batch_size: int,
    num_workers: int,
    train_tf,
    eval_tf,
    soft_labels: bool = False,
) -> tuple[DataLoader, DataLoader, AstroImageDataset]:
    """Monta os loaders de treino e validacao. Devolve tambem o dataset de
    treino, porque quem chama precisa dele para os pesos de classe."""
    splits_dir = Path(splits_dir)
    # So o TREINO usa soft labels. A validacao permanece com o rotulo duro,
    # porque e contra ele que acuracia, F1 e matriz de confusao fazem
    # sentido - e e ele que sera comparado com a rodada sem soft labels.
    train_ds = AstroImageDataset(
        splits_dir / f"{level.value}_train.csv", images_root, level, train_tf,
        soft_labels=soft_labels,
    )
    val_ds = AstroImageDataset(splits_dir / f"{level.value}_val.csv", images_root, level, eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        # Sem isto, os workers sao destruidos e recriados a cada epoca. No
        # Windows, criar um worker significa um processo novo reimportando
        # torch inteiro - caro, e um pico de memoria a cada epoca.
        persistent_workers=num_workers > 0,
    )

    # A validacao usa METADE dos workers, e no minimo 1. Os dois loaders
    # coexistem durante o treino, entao o custo e a soma dos dois. No Windows
    # cada worker e um processo que reimporta torch, numpy e sklearn: ~600 MB
    # de memoria comprometida. Com 8+8 workers isso estoura o arquivo de
    # paginacao e os workers morrem no meio da epoca, com uma mensagem que
    # nao menciona memoria nenhuma ("DataLoader worker exited unexpectedly").
    val_workers = max(1, num_workers // 2) if num_workers > 0 else 0
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=True,
        persistent_workers=val_workers > 0,
    )
    return train_loader, val_loader, train_ds
