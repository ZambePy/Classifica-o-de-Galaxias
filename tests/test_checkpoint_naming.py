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


class TestAvisoBackboneDivergente:
    """`evaluate.py` avisa quando a config e o checkpoint discordam de backbone.

    O BUG QUE ISTO IMPEDE (ja aconteceu, bug nº 9 do projeto)

    Sem `--checkpoint-name`, `evaluate.py` carrega `<nivel>_best.pt` - o de
    producao - qualquer que seja o backbone da config. Rodar a config do
    EfficientNet avaliava o ResNet50 de producao e gravava o relatorio com o nome
    do EfficientNet. A comparacao de backbones saiu com tres numeros identicos e
    a conclusao foi "o backbone nao importa"; o ganho real era 4,8 pontos de
    macro-F1 no nivel 3.

    Nada falhava: pesos carregavam, metricas saiam, arquivo tinha o nome
    esperado. So o rotulo mentia.
    """

    @staticmethod
    def _aviso(backbone_config: str, backbone_ckpt: str):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from astro_classifier.config import ExperimentConfig
        from astro_classifier.taxonomy import Level
        from evaluate import aviso_backbone_divergente

        config = ExperimentConfig(name="teste_config", level=Level.NEBULA)
        config.model.backbone = backbone_config
        return aviso_backbone_divergente(config, {"backbone": backbone_ckpt}, "nebula_best.pt")

    def test_avisa_quando_diverge(self):
        aviso = self._aviso("efficientnet_b0", "resnet50")
        assert aviso is not None
        assert "efficientnet_b0" in aviso and "resnet50" in aviso
        assert "nebula_best.pt" in aviso
        assert "--checkpoint-name" in aviso, "o aviso tem de dizer como consertar"

    def test_silencioso_quando_coincide(self):
        assert self._aviso("resnet50", "resnet50") is None

    def test_silencioso_quando_o_checkpoint_nao_declara(self):
        """Checkpoint antigo sem o campo `backbone` nao deve gerar ruido."""
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from astro_classifier.config import ExperimentConfig
        from astro_classifier.taxonomy import Level
        from evaluate import aviso_backbone_divergente

        config = ExperimentConfig(name="t", level=Level.NEBULA)
        assert aviso_backbone_divergente(config, {}, "x.pt") is None
