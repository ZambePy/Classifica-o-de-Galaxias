"""Testes do treino com soft labels.

Uma loss errada aqui nao explode: ela treina, converge para algo e produz
metricas plausiveis. Por isso os casos abaixo verificam o COMPORTAMENTO
esperado - que a loss seja minima quando a predicao bate com a distribuicao
humana, e que penalize confianca excessiva em exemplos ambiguos.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="exige PyTorch")

from astro_classifier.training.soft_labels import (  # noqa: E402
    SoftLabelLoss,
    votes_to_soft_labels,
)


def logits_para(probs: torch.Tensor) -> torch.Tensor:
    """Logits que produzem exatamente `probs` apos softmax."""
    return torch.log(probs.clamp_min(1e-8))


def test_loss_e_praticamente_zero_quando_a_predicao_bate_com_os_votos():
    alvo = torch.tensor([[0.73, 0.21, 0.06]])
    perda = SoftLabelLoss()(logits_para(alvo), alvo)
    assert perda.item() == pytest.approx(0.0, abs=1e-5)


def test_confianca_excessiva_em_exemplo_ambiguo_e_punida():
    """O ponto central do metodo: numa galaxia onde os humanos se dividiram,
    responder 99% de certeza deve custar mais do que responder 73%."""
    alvo = torch.tensor([[0.50, 0.30, 0.20]])
    loss = SoftLabelLoss()

    honesto = loss(logits_para(alvo), alvo)
    confiante = loss(logits_para(torch.tensor([[0.99, 0.005, 0.005]])), alvo)

    assert confiante > honesto
    assert confiante > 0.5, "a punicao precisa ser substancial, nao simbolica"


def test_acertar_a_classe_dominante_com_distribuicao_errada_ainda_custa():
    """argmax certo nao basta - o formato da distribuicao tambem conta."""
    alvo = torch.tensor([[0.60, 0.35, 0.05]])
    loss = SoftLabelLoss()
    perda = loss(logits_para(torch.tensor([[0.95, 0.03, 0.02]])), alvo)
    assert perda.item() > 0.1


def test_alvo_com_indices_e_recusado_com_mensagem_util():
    """Passar (N,) em vez de (N,C) e o erro mais provavel de quem liga isto
    ao loop de treino sem trocar o dataset."""
    with pytest.raises(ValueError, match="soft labels esperam alvo"):
        SoftLabelLoss()(torch.randn(4, 3), torch.tensor([0, 1, 2, 0]))


def test_pesos_de_classe_sao_aplicados():
    """Soft labels tratam ambiguidade DE CADA exemplo; pesos tratam o
    desbalanceamento ENTRE classes. As duas correcoes devem coexistir."""
    alvo = torch.tensor([[0.9, 0.05, 0.05], [0.05, 0.05, 0.9]])
    pred = logits_para(torch.tensor([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]]))

    sem_peso = SoftLabelLoss()(pred, alvo)
    # peso alto na classe 2, que e a dominante do segundo exemplo
    com_peso = SoftLabelLoss(class_weights=torch.tensor([1.0, 1.0, 10.0]))(pred, alvo)
    assert com_peso != sem_peso


def test_temperatura_suaviza_o_alvo():
    alvo = torch.tensor([[0.90, 0.08, 0.02]])
    pred = logits_para(torch.tensor([[0.5, 0.3, 0.2]]))

    t1 = SoftLabelLoss(temperature=1.0)(pred, alvo)
    t3 = SoftLabelLoss(temperature=3.0)(pred, alvo)
    assert t3 < t1, "alvo mais suave deve ficar mais perto de uma predicao incerta"


def test_temperatura_invalida_falha_cedo():
    with pytest.raises(ValueError, match="temperature"):
        SoftLabelLoss(temperature=0.0)


def test_votos_viram_distribuicao_que_soma_um():
    votos = torch.tensor([[7.0, 2.0, 1.0], [1.0, 1.0, 1.0]])
    probs = votes_to_soft_labels(votos)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2))
    assert probs[0, 0] == pytest.approx(0.7)


def test_linha_sem_voto_nenhum_vira_uniforme():
    """'Ninguem votou' e diferente de 'todos votaram na classe 0'. Sem este
    tratamento a normalizacao daria NaN e contaminaria o batch inteiro."""
    probs = votes_to_soft_labels(torch.tensor([[0.0, 0.0, 0.0]]))
    assert torch.allclose(probs, torch.full((1, 3), 1 / 3))
    assert not torch.isnan(probs).any()


def test_min_vote_descarta_apoio_irrisorio():
    votos = torch.tensor([[0.80, 0.19, 0.01]])
    probs = votes_to_soft_labels(votos, min_vote=0.05)
    assert probs[0, 2] == 0.0
    assert probs.sum().item() == pytest.approx(1.0)


def test_gradiente_flui():
    alvo = torch.tensor([[0.6, 0.3, 0.1]])
    logits = torch.randn(1, 3, requires_grad=True)
    SoftLabelLoss()(logits, alvo).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
