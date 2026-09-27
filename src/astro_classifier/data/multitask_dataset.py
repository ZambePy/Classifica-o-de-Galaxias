"""Dataset para o modelo de backbone compartilhado.

Devolve uma imagem e TRES rotulos - um por cabeca - usando -1 onde a imagem
nao tem rotulo naquele nivel. Uma espiral tem rotulo de objeto (`galaxy`) e
de galaxia (`spiral`), e -1 no nivel de nebulosa.

A DECISAO QUE FAZ A COMPARACAO COM A CASCATA SER JUSTA

Este dataset e a UNIAO dos tres splits da cascata, e cada imagem so recebe
rotulo nos niveis em que ela realmente aparecia:

    esta em object_train  -> ganha rotulo de objeto
    esta em galaxy_train  -> ganha rotulo de galaxia
    esta em nebula_train  -> ganha rotulo de nebulosa

Ou seja: o modelo multi-tarefa ve exatamente a mesma supervisao que os tres
modelos da cascata viram somados, nem mais nem menos. Se em vez disso
derivassemos o rotulo de objeto de toda imagem de `galaxy_train`, a cabeca de
objeto receberia 26401 galaxias contra 1389 `other` - e perderia do nivel 1
da cascata por desbalanceamento de amostragem, nao por arquitetura. A
conclusao sairia errada, e parecendo certa.

Como os splits vem da atribuicao GLOBAL (`splits.py`), uma imagem tem um
unico destino - treino, validacao ou teste - em todos os niveis. Sem isso
esta uniao vazaria teste para dentro do treino.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from astro_classifier.models.multitask import LEVELS
from astro_classifier.taxonomy import Level

# Nome do arquivo de split para cada nivel: object_train.csv, galaxy_val.csv...
_PREFIXO = {Level.OBJECT: "object", Level.GALAXY: "galaxy", Level.NEBULA: "nebula"}


class MultiTaskDataset(Dataset):
    """Imagem + um rotulo por nivel, com -1 onde nao ha supervisao."""

    def __init__(
        self,
        splits_dir: str | Path,
        images_root: str | Path,
        split: str,
        transform=None,
    ) -> None:
        self.images_root = Path(images_root)
        self.transform = transform
        self.split = split

        # indice por caminho -> {nivel: indice da classe}
        rotulos: dict[str, dict[str, int]] = {}
        self.contagem: dict[str, int] = {}

        for nivel in LEVELS:
            csv = Path(splits_dir) / f"{_PREFIXO[nivel]}_{split}.csv"
            if not csv.exists():
                raise FileNotFoundError(
                    f"{csv} nao existe - rode scripts/make_splits.py --all antes"
                )
            df = pd.read_csv(csv)
            ordem = {c: i for i, c in enumerate(nivel.classes)}

            desconhecidas = set(df["label"]) - set(ordem)
            if desconhecidas:
                raise ValueError(
                    f"{csv} tem rotulos fora da taxonomia do nivel "
                    f"'{nivel.value}': {sorted(desconhecidas)}"
                )

            for caminho, rot in zip(df["path"], df["label"], strict=True):
                rotulos.setdefault(caminho, {})[nivel.value] = ordem[rot]
            self.contagem[nivel.value] = len(df)

        self.paths = sorted(rotulos)
        self.rotulos = rotulos

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        caminho = self.paths[i]
        imagem = Image.open(self.images_root / caminho).convert("RGB")
        if self.transform is not None:
            imagem = self.transform(imagem)

        tem = self.rotulos[caminho]
        alvos = {lv.value: torch.tensor(tem.get(lv.value, -1), dtype=torch.long) for lv in LEVELS}
        return imagem, alvos

    def resumo(self) -> str:
        linhas = [f"{len(self)} imagens distintas em '{self.split}'"]
        for nivel in LEVELS:
            n = self.contagem[nivel.value]
            linhas.append(f"  {nivel.value:<8} {n:>6} com rotulo ({n / len(self):.0%} do total)")
        return "\n".join(linhas)
