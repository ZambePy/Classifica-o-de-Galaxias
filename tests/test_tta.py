"""Testes do test-time augmentation.

O TTA e facil de implementar errado de um jeito que nao quebra: se as
transformacoes estiverem repetidas ou se a media for feita sobre logits em
vez de probabilidades, o resultado continua plausivel e o ganho some.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="exige PyTorch")

import torch.nn as nn  # noqa: E402

from astro_classifier.inference.tta import d4_transforms, predict_proba_tta  # noqa: E402


def test_produz_oito_vistas():
    saida = d4_transforms(torch.randn(1, 3, 32, 32))
    assert saida.shape == (8, 3, 32, 32)


def test_as_oito_vistas_sao_distintas():
    """Uma imagem assimetrica precisa gerar 8 resultados diferentes. Se duas
    coincidirem, ha transformacao repetida e o TTA perde eficacia."""
    x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    vistas = d4_transforms(x)
    achatadas = {tuple(v.flatten().tolist()) for v in vistas}
    assert len(achatadas) == 8, "ha transformacoes repetidas entre as 8 vistas"


def test_a_primeira_vista_e_a_original():
    x = torch.randn(1, 3, 16, 16)
    assert torch.allclose(d4_transforms(x)[0], x[0])


def test_todas_as_vistas_preservam_o_conteudo():
    """Rotacao e espelhamento reordenam pixels, nunca alteram valores."""
    x = torch.randn(1, 1, 8, 8)
    original = sorted(x.flatten().tolist())
    for vista in d4_transforms(x):
        assert sorted(vista.flatten().tolist()) == pytest.approx(original)


def test_imagem_nao_quadrada_falha_com_mensagem_clara():
    with pytest.raises(ValueError, match="quadrada"):
        d4_transforms(torch.randn(1, 3, 32, 64))


def test_batch_maior_que_um_e_recusado():
    with pytest.raises(ValueError, match=r"\(1, C, H, W\)"):
        d4_transforms(torch.randn(4, 3, 32, 32))


class ModeloInvariante(nn.Module):
    """Responde o mesmo para qualquer entrada - o TTA nao pode mudar nada."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[2.0, 1.0, 0.0]]).repeat(x.size(0), 1)


class ModeloSensivelARotacao(nn.Module):
    """Decide pela soma da metade superior: muda conforme a orientacao."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        metade = x[:, :, : x.size(2) // 2, :].sum(dim=(1, 2, 3))
        return torch.stack([metade, -metade, torch.zeros_like(metade)], dim=1)


def test_modelo_invariante_nao_e_afetado():
    x = torch.randn(1, 3, 16, 16)
    com_tta = predict_proba_tta(ModeloInvariante(), x)
    sem_tta = torch.softmax(ModeloInvariante()(x), dim=1)
    assert torch.allclose(com_tta, sem_tta, atol=1e-6)


def test_probabilidades_somam_um():
    p = predict_proba_tta(ModeloSensivelARotacao(), torch.randn(1, 3, 16, 16))
    assert p.shape == (1, 3)
    assert p.sum().item() == pytest.approx(1.0, abs=1e-5)


class ModeloDiscordante(nn.Module):
    """Logits MODERADOS e diferentes por vista, para que a distincao entre
    media aritmetica e geometrica seja visivel.

    Metade das vistas vota com folga na classe 0; a outra metade vota, com a
    mesma folga, na classe 1. E o caso que o TTA existe para resolver: o
    modelo nao e invariante a rotacao e as vistas discordam.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        saida = torch.zeros(n, 3)
        metade = n // 2
        saida[:metade, 0] = 2.0
        saida[metade:, 1] = 2.0
        return saida


def test_media_e_sobre_probabilidades_e_nao_sobre_logits():
    """A distincao importa: media de logits equivale a media GEOMETRICA das
    probabilidades, dominada pelas vistas mais confiantes - o oposto do que
    o TTA deveria fazer, que e suavizar a discordancia entre vistas.

    Com metade das vistas votando em cada classe, a media aritmetica devolve
    as duas empatadas; a geometrica achata tudo em direcao ao uniforme.
    """
    modelo = ModeloDiscordante()
    x = torch.randn(1, 3, 16, 16)

    vistas = d4_transforms(x)
    logits = modelo(vistas)
    aritmetica = torch.softmax(logits, dim=1).mean(dim=0)
    geometrica = torch.softmax(logits.mean(dim=0, keepdim=True), dim=1)[0]

    resultado = predict_proba_tta(modelo, x)[0]

    assert torch.allclose(resultado, aritmetica, atol=1e-6), "deveria ser a media aritmetica"
    assert not torch.allclose(resultado, geometrica, atol=1e-3), (
        "esta usando media de logits (geometrica) em vez de probabilidades"
    )
    # As duas classes disputadas empatam, e ficam acima da terceira.
    assert resultado[0].item() == pytest.approx(resultado[1].item(), abs=1e-6)
    assert resultado[0] > resultado[2]


def test_e_deterministico():
    modelo = ModeloSensivelARotacao()
    x = torch.randn(1, 3, 16, 16)
    assert torch.allclose(predict_proba_tta(modelo, x), predict_proba_tta(modelo, x))
