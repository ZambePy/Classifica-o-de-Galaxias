"""Pre-processamento e augmentation.

Uma decisao importante e especifica de astronomia: usamos flips e rotacoes
LIVRES (0-360 graus). Em fotos do dia a dia isso seria errado - um cachorro
de cabeca para baixo e raro. No ceu nao existe "em pe": a orientacao de uma
galaxia na imagem e arbitraria, definida pela posicao do telescopio. Entao a
rotacao e uma simetria real do problema, nao um ruido artificial.

O que NAO fazemos: alterar cor agressivamente. Em nebulosas a cor carrega
sinal fisico (H-alfa avermelhado em nebulosas de emissao, azul espalhado em
nebulosas de reflexao). Distorcer cor apagaria a feature mais discriminativa.
"""

from __future__ import annotations

from torchvision import transforms

# Estatisticas do ImageNet - os backbones pre-treinados as esperam.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transforms(image_size: int = 224, augment: bool = True) -> transforms.Compose:
    steps = [transforms.Resize((image_size, image_size))]
    if augment:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            # Brilho/contraste leves: simulam diferencas de exposicao entre
            # telescopios. Saturacao e matiz ficam de fora de proposito.
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(steps)


def eval_transforms(image_size: int = 224) -> transforms.Compose:
    """Sem aleatoriedade - validacao, teste e inferencia usam isto."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
