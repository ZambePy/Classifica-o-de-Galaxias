"""Deteccao de fora-de-dominio por ENERGIA.

O MSP (o metodo que o projeto usa hoje) olha a probabilidade maxima depois do
softmax. O problema e que o softmax NORMALIZA: ele divide pela soma, entao
perde a magnitude absoluta dos logits. Uma imagem que o modelo acha
fracamente parecida com tudo e outra que ele acha fortemente parecida com uma
classe podem sair com a mesma confianca.

A energia preserva essa magnitude:

    E(x) = -T * log( sum_i exp( logit_i / T ) )

E o log-sum-exp dos logits, com sinal trocado. Vale a intuicao fisica: a
energia e BAIXA onde o modelo esta "em casa" - onde algum logit e grande - e
ALTA onde nenhuma classe responde forte.

A relacao com o MSP e exata e vale entender: o softmax e
`exp(logit_i) / sum_j exp(logit_j)`. O denominador e justamente o que a
energia mede, e e exatamente o que o softmax joga fora ao normalizar.

Referencia: Liu et al. (2020), Energy-based Out-of-distribution Detection,
NeurIPS. O artigo mostra que a energia supera o MSP sem exigir retreino,
mudanca de arquitetura ou dados de fora-de-dominio para calibrar - que e
precisamente o caso deste projeto.
"""

from __future__ import annotations

import numpy as np
import torch


def energy_score(logits: torch.Tensor | np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Energia de cada amostra. MENOR = mais dentro do dominio.

    Recebe LOGITS, nao probabilidades - a magnitude e o ponto do metodo, e o
    softmax ja a descartou.
    """
    if temperature <= 0:
        raise ValueError(f"temperature precisa ser > 0, recebido {temperature}")

    t = torch.as_tensor(logits, dtype=torch.float32)
    if t.dim() != 2:
        raise ValueError(f"esperado (N, C) de logits; recebido {tuple(t.shape)}")

    return (-temperature * torch.logsumexp(t / temperature, dim=1)).cpu().numpy()


def confidence_from_energy(
    logits: torch.Tensor | np.ndarray, temperature: float = 1.0
) -> np.ndarray:
    """Energia com sinal trocado, para ficar na mesma orientacao do MSP.

    MAIOR = mais dentro do dominio. So existe para que as comparacoes com o
    MSP usem a mesma convencao e a AUROC saia no mesmo sentido.
    """
    return -energy_score(logits, temperature)
