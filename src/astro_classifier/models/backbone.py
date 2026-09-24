"""Backbones pre-treinados no ImageNet.

Por que transfer learning e nao uma CNN do zero: com 3 mil imagens de
nebulosa, uma rede treinada do zero decora o conjunto de treino. As features
de baixo nivel do ImageNet (bordas, texturas, gradientes) transferem bem
mesmo para um dominio tao diferente quanto imagens de telescopio.

Na GTX 1660 Ti (6 GB), resnet18 e efficientnet_b0 treinam com batch 32 em
224x224 confortavelmente. resnet50 exige batch 16 + precisao mista.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

SUPPORTED = ("resnet18", "resnet50", "efficientnet_b0")


def build_backbone(name: str, pretrained: bool = True) -> tuple[nn.Module, int]:
    """Devolve (backbone sem a camada final, numero de features de saida)."""
    if name not in SUPPORTED:
        raise ValueError(f"backbone '{name}' nao suportado. Use um de: {SUPPORTED}")

    if name == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        n_features = net.fc.in_features
        net.fc = nn.Identity()
    elif name == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        n_features = net.fc.in_features
        net.fc = nn.Identity()
    else:  # efficientnet_b0
        net = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        n_features = net.classifier[1].in_features
        net.classifier = nn.Identity()

    return net, n_features


def set_backbone_trainable(backbone: nn.Module, trainable: bool) -> None:
    """Congela/descongela o backbone (usado por freeze_backbone_epochs)."""
    for param in backbone.parameters():
        param.requires_grad = trainable
