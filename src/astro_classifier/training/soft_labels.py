"""Treino com soft labels: usar a incerteza humana em vez de apaga-la.

O PROBLEMA QUE ISTO ATACA

O Galaxy Zoo nao diz "esta galaxia e espiral". Ele diz "73% dos voluntarios
viram um disco com bracos, 21% viram algo liso, 6% acharam estranho". Ao
reduzir isso a um rotulo unico, jogamos fora a informacao mais interessante
do dataset: O QUANTO os humanos concordaram.

E o custo aparece na classe mais fragil. O `irregular` sai com precisao
0,664 e recall 0,825 - o modelo o preve demais. Boa parte disso e rotulo
ruim: o Galaxy Zoo 2 nao tem pergunta direta para "irregular", e a
aproximacao por `Class6.1` ("tem algo estranho") mistura galaxias
genuinamente irregulares com fusoes, aneis e artefatos. Forcar 100% de
certeza num rotulo desses ensina o modelo a ser confiante onde nem os
humanos foram.

COMO FUNCIONA

Em vez de um alvo [0, 0, 1], o alvo e a propria distribuicao de votos:

    voto majoritario   spiral=1.00  elliptical=0.00  irregular=0.00
    soft label         spiral=0.73  elliptical=0.21  irregular=0.06

E a loss deixa de ser cross-entropy contra um indice e passa a ser
divergencia de Kullback-Leibler entre a distribuicao prevista e a humana.
O modelo e penalizado por estar CERTO DEMAIS numa galaxia ambigua - que e
exatamente o comportamento que queremos.

A PERGUNTA DE PESQUISA

    Treinar com a distribuicao completa de votos dos voluntarios melhora o
    desempenho de um classificador morfologico, em relacao a treinar com o
    rotulo de voto majoritario?

Nao e retorica: pode nao melhorar. O ganho esperado esta na calibracao e nas
classes ambiguas, nao necessariamente na acuracia geral. Medir os dois e o
experimento - e o resultado, seja qual for, vale um capitulo.

Referencia: Hinton et al. (2015), Distilling the Knowledge in a Neural
Network - a mesma ideia, com o professor sendo aqui um coletivo de humanos.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftLabelLoss(nn.Module):
    """KL-divergencia contra a distribuicao de votos, com peso por classe.

    `temperature` > 1 suaviza ainda mais a distribuicao alvo, aumentando o
    peso relativo das classes minoritarias dentro de cada exemplo. Comece
    em 1.0 (a distribuicao como veio) e so mexa se houver motivo.

    `class_weights` continua valendo: soft labels tratam a ambiguidade DE
    CADA EXEMPLO, nao o desbalanceamento ENTRE classes. Sao problemas
    diferentes e as duas correcoes se somam.
    """

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        temperature: float = 1.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature precisa ser > 0, recebido {temperature}")
        self.register_buffer(
            "class_weights", class_weights if class_weights is not None else None
        )
        self.temperature = temperature
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """`targets` e (N, C) com as probabilidades, nao (N,) com indices."""
        if targets.dim() != 2:
            raise ValueError(
                f"soft labels esperam alvo (N, C); recebido {tuple(targets.shape)}. "
                "Use AstroImageDataset com soft_labels=True."
            )

        alvo = targets.float()
        if self.temperature != 1.0:
            alvo = torch.softmax(torch.log(alvo.clamp_min(1e-8)) / self.temperature, dim=1)
        if self.label_smoothing > 0:
            n = alvo.size(1)
            alvo = (1 - self.label_smoothing) * alvo + self.label_smoothing / n

        log_pred = F.log_softmax(logits, dim=1)
        # KL por exemplo, sem reduzir ainda - precisamos ponderar antes.
        kl = (alvo * (torch.log(alvo.clamp_min(1e-8)) - log_pred)).sum(dim=1)

        if self.class_weights is not None:
            # Peso do exemplo = peso da classe dominante nele.
            peso = self.class_weights[alvo.argmax(dim=1)]
            return (kl * peso).sum() / peso.sum()
        return kl.mean()


def votes_to_soft_labels(
    votos: torch.Tensor, min_vote: float = 0.0
) -> torch.Tensor:
    """Normaliza votos brutos numa distribuicao de probabilidade.

    `min_vote` zera classes com apoio irrisorio antes de renormalizar - sem
    isso, ruido de 1% em classes que ninguem votou vira sinal de treino.
    """
    v = votos.float().clamp_min(0.0)
    if min_vote > 0:
        v = torch.where(v >= min_vote, v, torch.zeros_like(v))

    soma = v.sum(dim=1, keepdim=True)
    # Linha inteira zerada (ninguem votou em nada): distribuicao uniforme, que
    # e o alvo honesto para "nao sabemos".
    uniforme = torch.full_like(v, 1.0 / v.size(1))
    return torch.where(soma > 0, v / soma.clamp_min(1e-8), uniforme)
