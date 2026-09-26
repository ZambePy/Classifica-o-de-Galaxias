"""Test-time augmentation: classificar a mesma imagem varias vezes.

A ideia: em vez de uma predicao, faz oito - a imagem original, suas rotacoes
de 90/180/270 graus e os espelhamentos de cada uma - e devolve a media das
probabilidades.

POR QUE ISSO FUNCIONA PARTICULARMENTE BEM AQUI

Em fotos do dia a dia, girar a imagem seria trapaca: um cachorro de cabeca
para baixo e um caso raro que o modelo nao deveria ter que resolver. No ceu
e o contrario. Nao existe "em pe": a orientacao de uma galaxia na imagem e
consequencia de onde o telescopio estava apontado, nao do objeto.

As oito transformacoes sao, portanto, OITO VISTAS IGUALMENTE VALIDAS do
mesmo objeto - o grupo de simetria do quadrado (D4). O modelo deveria dar a
mesma resposta para todas; onde ele nao da, a media corrige.

E o mesmo motivo pelo qual o treino ja usa rotacao livre como augmentation
(ver data/transforms.py). Aqui a simetria e explorada na outra ponta.

CUSTO: 8x mais inferencias. Na GTX 1660 Ti isso leva a predicao de ~20ms
para ~60ms (as oito vistas vao num batch so), o que continua instantaneo
para um dashboard.
"""

from __future__ import annotations

import torch


def d4_transforms(x: torch.Tensor) -> torch.Tensor:
    """As 8 simetrias do quadrado, empilhadas num batch.

    Entrada (1, C, H, W) -> saida (8, C, H, W). Exige imagem quadrada, que e
    o caso do projeto inteiro (224x224).
    """
    if x.dim() != 4 or x.size(0) != 1:
        raise ValueError(f"esperado tensor (1, C, H, W), recebido {tuple(x.shape)}")
    if x.size(-1) != x.size(-2):
        raise ValueError(
            f"as rotacoes de 90 graus exigem imagem quadrada; recebido "
            f"{x.size(-2)}x{x.size(-1)}"
        )

    vistas = []
    for k in range(4):  # 0, 90, 180, 270 graus
        girada = torch.rot90(x, k, dims=(-2, -1))
        vistas.append(girada)
        vistas.append(torch.flip(girada, dims=(-1,)))  # + espelhamento
    return torch.cat(vistas, dim=0)


@torch.inference_mode()
def predict_proba_tta(model, x: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """Probabilidades medias sobre as 8 vistas. Devolve (1, num_classes).

    A media e feita sobre as PROBABILIDADES, nao sobre os logits. Media de
    logits equivale a media geometrica das probabilidades, que e dominada
    pelas vistas mais confiantes - justamente o oposto do que queremos, ja
    que o ganho do TTA vem de suavizar as vistas discordantes.
    """
    vistas = d4_transforms(x).to(device)
    probs = torch.softmax(model(vistas), dim=1)
    return probs.mean(dim=0, keepdim=True)
