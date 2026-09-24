"""Testes de configuracao e divisao de dados.

Nao dependem de torch, GPU nem de imagens - rodam em segundos no CI.
"""

from __future__ import annotations

import pandas as pd
import pytest

from astro_classifier.config import ExperimentConfig
from astro_classifier.data.splits import make_splits
from astro_classifier.paths import Paths
from astro_classifier.taxonomy import Level


def test_configs_do_repositorio_carregam(tmp_path):
    """Os tres YAMLs versionados precisam ser validos. Um typo aqui so
    apareceria depois de horas de download de dados."""
    from pathlib import Path

    repo_configs = Path(__file__).resolve().parents[1] / "configs"
    for yaml_file in sorted(repo_configs.glob("*.yaml")):
        config = ExperimentConfig.from_yaml(yaml_file)
        assert config.name
        assert isinstance(config.level, Level)
        assert config.optim.epochs > 0
        assert config.data.batch_size > 0


def test_config_sobrevive_ao_disco(tmp_path):
    """Todo treino grava sua config junto dos resultados. Se o ciclo
    salvar/carregar perder informacao, o experimento nao e reprodutivel."""
    original = ExperimentConfig.from_yaml(
        _repo_config("level3_nebula.yaml")
    )
    destino = tmp_path / "config.yaml"
    original.save(destino)

    recarregada = ExperimentConfig.from_yaml(destino)
    assert recarregada.to_dict() == original.to_dict()


def test_paths_derivam_todos_de_uma_raiz(tmp_path):
    paths = Paths(root=tmp_path).ensure()

    assert paths.raw.exists()
    assert paths.checkpoints.exists()
    assert paths.splits.exists()
    # Tudo precisa estar sob a raiz - e essa propriedade que mantem os dados
    # fora do OneDrive.
    for p in (paths.raw, paths.processed, paths.checkpoints, paths.runs, paths.splits):
        assert tmp_path in p.parents or p.parent == tmp_path


def test_ensure_e_idempotente(tmp_path):
    paths = Paths(root=tmp_path)
    paths.ensure()
    paths.ensure()  # nao pode explodir na segunda chamada
    assert paths.raw.exists()


def test_split_e_estratificado(tmp_path):
    df = pd.DataFrame(
        {
            "path": [f"img_{i}.jpg" for i in range(300)],
            "label": (["spiral"] * 150) + (["elliptical"] * 100) + (["irregular"] * 50),
        }
    )
    splits = make_splits(df, Level.GALAXY, tmp_path, seed=42)

    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == 300
    # Toda classe precisa aparecer nos tres conjuntos, senao a metrica dela
    # no teste nao existe.
    for part in splits.values():
        assert set(part["label"]) == {"spiral", "elliptical", "irregular"}


def test_split_nao_vaza_imagem_entre_conjuntos(tmp_path):
    """O erro mais caro do projeto: a mesma imagem em treino e teste faz a
    acuracia de teste parecer otima e ser mentira."""
    df = pd.DataFrame(
        {
            "path": [f"img_{i}.jpg" for i in range(200)],
            "label": (["spiral"] * 100) + (["elliptical"] * 100),
        }
    )
    splits = make_splits(df, Level.GALAXY, tmp_path, seed=7)

    treino = set(splits["train"]["path"])
    validacao = set(splits["val"]["path"])
    teste = set(splits["test"]["path"])

    assert treino & validacao == set()
    assert treino & teste == set()
    assert validacao & teste == set()


def test_split_e_reprodutivel_com_a_mesma_seed(tmp_path):
    df = pd.DataFrame(
        {
            "path": [f"img_{i}.jpg" for i in range(100)],
            "label": (["spiral"] * 50) + (["elliptical"] * 50),
        }
    )
    a = make_splits(df, Level.GALAXY, tmp_path / "a", seed=13)
    b = make_splits(df, Level.GALAXY, tmp_path / "b", seed=13)

    assert list(a["test"]["path"]) == list(b["test"]["path"])


def test_classe_rara_demais_e_descartada_com_aviso(tmp_path, capsys):
    df = pd.DataFrame(
        {
            "path": [f"img_{i}.jpg" for i in range(103)],
            "label": (["spiral"] * 50) + (["elliptical"] * 50) + (["irregular"] * 3),
        }
    )
    splits = make_splits(df, Level.GALAXY, tmp_path, seed=42, min_per_class=10)

    assert "irregular" not in set(splits["train"]["label"])
    assert "removidas" in capsys.readouterr().out


def test_dataframe_sem_colunas_obrigatorias_falha(tmp_path):
    df = pd.DataFrame({"arquivo": ["a.jpg"], "classe": ["spiral"]})
    with pytest.raises(ValueError, match="colunas obrigatorias"):
        make_splits(df, Level.GALAXY, tmp_path)


def _repo_config(name: str):
    from pathlib import Path

    return Path(__file__).resolve().parents[1] / "configs" / name
