"""Testes da verificacao de sanidade numerica.

Nascem de um bug real: na GTX 1660 Ti o cuDNN produzia NaN no forward sob
precisao mista a partir de batch 64. O treino rodava, gravava checkpoint e
reportava acuracia plausivel - mas o modelo previa sempre a mesma classe,
porque argmax sobre NaN devolve o indice 0.

O caro nao foi o bug, foi ele ser SILENCIOSO. Estes testes garantem que o
proximo dessa familia grite antes da primeira epoca.

Rodam na CPU, sem GPU - o que testamos e a deteccao, nao o bug da placa.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="testes de treino exigem PyTorch")

import torch.nn as nn  # noqa: E402

from astro_classifier.training.loops import (  # noqa: E402
    NumericalSanityError,
    check_numerical_sanity,
)


class ModeloSao(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(1)[:, :3]


class ModeloComNaN(nn.Module):
    """Imita o sintoma: forward valido na forma, NaN no conteudo."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        saida = x.flatten(1)[:, :3].clone()
        saida[:] = float("nan")
        return saida


class ModeloComInf(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        saida = x.flatten(1)[:, :3].clone()
        saida[:] = float("inf")
        return saida


@pytest.fixture
def amostra() -> torch.Tensor:
    return torch.randn(4, 3, 32, 32)


def test_modelo_sao_passa(amostra):
    check_numerical_sanity(ModeloSao(), amostra, device="cpu", use_amp=False)


def test_nan_no_forward_e_detectado(amostra):
    with pytest.raises(NumericalSanityError, match="NaN/inf"):
        check_numerical_sanity(ModeloComNaN(), amostra, device="cpu", use_amp=False)


def test_inf_no_forward_e_detectado(amostra):
    with pytest.raises(NumericalSanityError, match="NaN/inf"):
        check_numerical_sanity(ModeloComInf(), amostra, device="cpu", use_amp=False)


def test_mensagem_diz_quantos_e_o_tamanho_do_batch(amostra):
    """O batch aparece na mensagem porque o bug original SO acontecia a
    partir de batch 64 - sem esse numero, o diagnostico fica pela metade."""
    with pytest.raises(NumericalSanityError) as exc:
        check_numerical_sanity(ModeloComNaN(), amostra, device="cpu", use_amp=False)

    texto = str(exc.value)
    assert "12" in texto, "deveria contar os valores ruins (4 amostras x 3 classes)"
    assert "batch de 4" in texto


def test_modelo_volta_para_train_mode(amostra):
    """A verificacao usa eval() internamente. Se esquecer de restaurar, o
    BatchNorm treina com estatisticas congeladas e o treino inteiro degrada
    de um jeito dificil de perceber."""
    model = ModeloSao()
    model.train()
    check_numerical_sanity(model, amostra, device="cpu", use_amp=False)
    assert model.training is True


def test_modelo_volta_para_train_mode_mesmo_falhando(amostra):
    model = ModeloComNaN()
    model.train()
    with pytest.raises(NumericalSanityError):
        check_numerical_sanity(model, amostra, device="cpu", use_amp=False)
    assert model.training is True


def test_configs_do_repositorio_nao_ligam_amp_nesta_maquina():
    """Guarda a decisao tomada apos medir: nesta GPU o AMP produz NaN e
    ainda deixa o treino 4,6x mais lento. Se alguem reativar sem querer,
    este teste avisa - e quem reativar de proposito atualiza o teste junto."""
    from pathlib import Path

    from astro_classifier.config import ExperimentConfig

    configs = Path(__file__).resolve().parents[1] / "configs"
    for yaml_file in sorted(configs.glob("*.yaml")):
        config = ExperimentConfig.from_yaml(yaml_file)
        assert config.optim.mixed_precision is False, (
            f"{yaml_file.name} religou mixed_precision. Nesta GPU (GTX 1660 Ti, "
            "sem tensor cores) isso produz NaN a partir de batch 64. Em placa com "
            "tensor cores, pode religar - e atualize este teste."
        )
