"""Testes da regra de nomeacao de checkpoint.

Nascem de um bug flagrado em execucao: a bateria de experimentos treinou a
variante EfficientNet-B0 do nivel 1 e gravou por cima de `object_best.pt` -
o checkpoint que a cascata e a API carregam. O ResNet18 em producao foi
substituido em silencio, sem erro, sem aviso.

A regra: `<nivel>_best.pt` pertence ao config CANONICO do nivel. Qualquer
variante (outro backbone, outra semente, soft labels) grava com o proprio
nome.

Os testes abaixo verificam a regra na forma de funcao pura, para nao
depender de treinar nada.
"""

from __future__ import annotations

import pytest

from astro_classifier.taxonomy import Level

CANONICO = {
    Level.OBJECT: "level1_object",
    Level.GALAXY: "level2_galaxy",
    Level.NEBULA: "level3_nebula",
}


def nome_do_checkpoint(config_name: str, level: Level, name_suffix: str = "") -> str:
    """Replica a regra de scripts/train.py."""
    e_producao = config_name == CANONICO[level] and not name_suffix
    return f"{level.value}_best.pt" if e_producao else f"{config_name}.pt"


@pytest.mark.parametrize("level", list(Level))
def test_config_canonico_escreve_o_checkpoint_de_producao(level):
    assert nome_do_checkpoint(CANONICO[level], level) == f"{level.value}_best.pt"


@pytest.mark.parametrize(
    "variante",
    ["level1_object_efficientnet_b0", "level1_object_resnet50", "level2_galaxy_soft"],
)
def test_variante_nao_toca_o_checkpoint_de_producao(variante):
    """A REGRESSAO PRINCIPAL. Sem isto, treinar uma variante troca o modelo
    que esta em producao e o sintoma aparece muito depois."""
    nivel = Level.GALAXY if "galaxy" in variante else Level.OBJECT
    nome = nome_do_checkpoint(variante, nivel)
    assert nome == f"{variante}.pt"
    assert nome != f"{nivel.value}_best.pt"


def test_sufixo_de_nome_tambem_protege_producao():
    """Usado pelas sementes multiplas: mesmo config, sementes diferentes."""
    nome = nome_do_checkpoint("level1_object_seed7", Level.OBJECT, name_suffix="_seed7")
    assert nome == "level1_object_seed7.pt"


def test_config_canonico_com_sufixo_nao_e_producao():
    """`--name-suffix` sinaliza um treino experimental mesmo partindo do
    config canonico - tipico de um teste rapido com poucas epocas."""
    nome = nome_do_checkpoint("level1_object_smoke", Level.OBJECT, name_suffix="_smoke")
    assert nome != "object_best.pt"


def test_cada_nivel_tem_seu_proprio_arquivo_de_producao():
    nomes = {nome_do_checkpoint(CANONICO[lv], lv) for lv in Level}
    assert len(nomes) == len(list(Level))


def test_os_configs_do_repositorio_respeitam_a_regra():
    """Percorre os YAMLs de verdade: exatamente tres podem gravar em
    producao, um por nivel. Se alguem criar uma variante com o nome canonico,
    este teste quebra."""
    from pathlib import Path

    from astro_classifier.config import ExperimentConfig

    configs = Path(__file__).resolve().parents[1] / "configs"
    producao = []
    for yaml_file in sorted(configs.glob("*.yaml")):
        cfg = ExperimentConfig.from_yaml(yaml_file)
        if nome_do_checkpoint(cfg.name, cfg.level) == f"{cfg.level.value}_best.pt":
            producao.append((yaml_file.name, cfg.level.value))

    niveis = [lv for _, lv in producao]
    assert len(niveis) == len(set(niveis)), (
        f"mais de um config grava no mesmo checkpoint de producao: {producao}"
    )
    assert len(producao) == 3, f"esperados 3 configs canonicos, achados {producao}"
