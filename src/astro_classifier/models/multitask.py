"""Alternativa a cascata: UM backbone, tres cabecas.

A cascata (Abordagem A) treina tres redes independentes. Funciona, e o
projeto mediu o preco: o erro em cascata, amostras que o nivel 1 manda para
o submodelo errado e que nao tem mais como ser salvas.

A alternativa e compartilhar o backbone e pendurar uma cabeca linear por
nivel:

                          /-- cabeca objeto   (3 classes)
    imagem -> backbone --+--- cabeca galaxia  (3 classes)
                          \-- cabeca nebulosa (4 classes)

O QUE ISSO MUDA, E O QUE NAO MUDA

Muda o custo: um backbone em vez de tres. Na inferencia e ~3x menos memoria
e uma passada so em vez de duas. Para um dashboard que roda numa maquina
qualquer, isso e o argumento principal.

Muda o aprendizado: as cabecas dividem features. Reconhecer uma espiral e
reconhecer uma nebulosa de emissao podem exigir coisas parecidas nas
camadas iniciais, e treinar junto e uma forma de regularizacao - cada tarefa
vira dado extra para as outras. Pode ajudar o nivel 3, que tem so 1916
imagens de treino e e o mais fraco da cascata.

NAO muda o erro em cascata. Isto e importante e e facil errar: na inferencia
ainda e a cabeca de objeto que decide qual cabeca de subtipo ler. Se ela
disser "galaxia" para uma nebulosa, lemos a cabeca de galaxia e a resposta
final continua errada. O ganho, se houver, vem de features melhores - nao da
topologia.

A ARMADILHA DO TREINO CONJUNTO

Cada imagem tem rotulo para ALGUMAS cabecas, nao todas. Uma espiral nao tem
rotulo de nebulosa. Se a loss somar as tres cabecas sempre, a cabeca de
nebulosa recebe gradiente de imagens que nao sao nebulosa - e aprende lixo.
A loss aqui e MASCARADA: cada cabeca so ve as amostras que tem rotulo para
ela, e a media e sobre essas, nao sobre o lote.

Consequencia pratica: um lote pode nao ter nenhuma nebulosa. A cabeca de
nebulosa entao contribui zero naquele passo - nao NaN, que e o que acontece
se alguem dividir por um contador vazio. O codigo trata isso explicitamente.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from astro_classifier.config import ExperimentConfig
from astro_classifier.models.backbone import build_backbone
from astro_classifier.taxonomy import Level

# A ordem e fixa e vira a ordem dos tensores de rotulo. Nao reordene sem
# migrar os checkpoints: seria o mesmo bug de classes trocadas que o
# `AstroClassifier.load` protege.
LEVELS: tuple[Level, ...] = (Level.OBJECT, Level.GALAXY, Level.NEBULA)


class MultiTaskClassifier(nn.Module):
    """Backbone compartilhado + uma cabeca linear por nivel da hierarquia."""

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.levels = LEVELS

        self.backbone, n_features = build_backbone(backbone, pretrained)
        # ModuleDict com a chave sendo o valor do enum, para o state_dict
        # ficar legivel e o checkpoint auto-descritivo.
        self.heads = nn.ModuleDict(
            {
                lv.value: nn.Sequential(nn.Dropout(dropout), nn.Linear(n_features, lv.num_classes))
                for lv in LEVELS
            }
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Devolve LOGITS por nivel: {'object': (N,3), 'galaxy': (N,3), ...}

        Roda o backbone UMA vez e distribui para as cabecas - e o ponto
        inteiro da arquitetura.
        """
        f = self.backbone(x).flatten(1)
        return {nome: cabeca(f) for nome, cabeca in self.heads.items()}

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Vetor do penultimo layer, para Grad-CAM e Mahalanobis."""
        return self.backbone(x).flatten(1)

    def save(self, path: str | Path, config: ExperimentConfig, extra: dict[str, Any] | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "multitask": True,
                "levels": [lv.value for lv in self.levels],
                "classes": {lv.value: list(lv.classes) for lv in self.levels},
                "backbone": self.backbone_name,
                "config": config.to_dict(),
                "extra": extra or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> tuple[MultiTaskClassifier, dict[str, Any]]:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if not ckpt.get("multitask"):
            raise RuntimeError(
                f"'{path}' nao e um checkpoint multi-tarefa. Use AstroClassifier.load."
            )

        # Mesma protecao do modelo em cascata: pesos e taxonomia tem de casar.
        for nome, classes in ckpt["classes"].items():
            atual = Level(nome).classes
            if classes != atual:
                raise RuntimeError(
                    f"Checkpoint '{path}' foi treinado com {classes} no nivel '{nome}', "
                    f"mas a taxonomia atual e {atual}. Retreine ou reverta taxonomy.py."
                )

        model = cls(
            backbone=ckpt["backbone"],
            pretrained=False,
            dropout=ckpt["config"]["model"]["dropout"],
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device).eval()
        return model, ckpt


def masked_multitask_loss(
    logits: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Cross-entropy por cabeca, so nas amostras que tem rotulo para ela.

    `targets[nivel]` usa -1 para "esta amostra nao tem rotulo neste nivel".
    E a convencao do `ignore_index` do PyTorch, mas nao basta passar
    ignore_index: com TODAS as amostras marcadas -1 o PyTorch devolve NaN, e
    um lote sem nenhuma nebulosa e comum aqui (1916 nebulosas contra 26401
    galaxias). Por isso a contagem e explicita.

    Devolve (loss total, {nivel: loss daquela cabeca}) - as parciais vao para
    o log, porque a media escondida de tres tarefas nao diz qual esta ruim.
    """
    pesos = weights or {}
    total = None
    partes: dict[str, float] = {}

    for nome, logit in logits.items():
        alvo = targets[nome]
        valido = alvo >= 0
        n = int(valido.sum())
        if n == 0:
            # Nenhuma amostra deste nivel no lote. Contribuir zero e correto;
            # somar um tensor NaN destruiria o passo inteiro.
            partes[nome] = 0.0
            continue

        perda = nn.functional.cross_entropy(logit[valido], alvo[valido])
        partes[nome] = float(perda.detach())
        termo = pesos.get(nome, 1.0) * perda
        total = termo if total is None else total + termo

    if total is None:
        raise ValueError(
            "nenhuma cabeca recebeu amostra valida neste lote - o dataset "
            "provavelmente esta devolvendo -1 em todos os niveis"
        )
    return total, partes
