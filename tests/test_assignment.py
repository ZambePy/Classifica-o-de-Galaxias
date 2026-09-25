"""Testes da atribuicao global de splits.

Existem por causa de um bug que passou despercebido ate a avaliacao da
cascata dar um numero impossivel: com os niveis divididos de forma
independente, 218 das 300 galaxias do TESTE do nivel 1 estavam no TREINO do
nivel 2. Avaliar a cascata assim mede um modelo respondendo sobre imagens
que ele ja viu.

O primeiro teste abaixo e o que trava essa regressao.
"""

from __future__ import annotations

import pandas as pd
import pytest

from astro_classifier.data.splits import (
    build_assignment,
    load_assignment,
    splits_from_assignment,
)
from astro_classifier.taxonomy import Level

FINO_PARA_OBJETO = {
    "spiral": "galaxy",
    "elliptical": "galaxy",
    "irregular": "galaxy",
    "emission": "nebula",
    "planetary": "nebula",
    "star_field": "other",
}
IDENTIDADE_GALAXIA = {c: c for c in Level.GALAXY.classes}


@pytest.fixture
def catalogo() -> pd.DataFrame:
    linhas = []
    for rotulo, n in [
        ("spiral", 120), ("elliptical", 90), ("irregular", 60),
        ("emission", 50), ("planetary", 40), ("star_field", 45),
    ]:
        linhas += [{"path": f"{rotulo}/{i:04d}.jpg", "label_fino": rotulo} for i in range(n)]
    return pd.DataFrame(linhas)


def test_uma_imagem_tem_o_mesmo_destino_em_todos_os_niveis(tmp_path, catalogo):
    """A REGRESSAO PRINCIPAL. Se este teste falhar, a metrica da cascata
    volta a medir um modelo respondendo sobre o proprio treino."""
    atribuicao = build_assignment(catalogo, tmp_path, seed=1)

    splits_from_assignment(atribuicao, Level.OBJECT, FINO_PARA_OBJETO, tmp_path)
    splits_from_assignment(atribuicao, Level.GALAXY, IDENTIDADE_GALAXIA, tmp_path)

    teste_objeto = set(pd.read_csv(tmp_path / "object_test.csv")["path"])
    treino_galaxia = set(pd.read_csv(tmp_path / "galaxy_train.csv")["path"])
    val_galaxia = set(pd.read_csv(tmp_path / "galaxy_val.csv")["path"])
    teste_galaxia = set(pd.read_csv(tmp_path / "galaxy_test.csv")["path"])

    assert teste_objeto & treino_galaxia == set(), "imagem do teste do nivel 1 no TREINO do nivel 2"
    assert teste_objeto & val_galaxia == set(), "imagem do teste do nivel 1 na VALIDACAO do nivel 2"

    # E o outro lado: toda galaxia do teste do nivel 1 precisa estar no teste
    # do nivel 2, senao a cascata nao tem subtipo verdadeiro para comparar.
    galaxias_no_teste = {p for p in teste_objeto if p.split("/")[0] in IDENTIDADE_GALAXIA}
    assert galaxias_no_teste <= teste_galaxia


def test_splits_nao_se_sobrepoem(tmp_path, catalogo):
    atribuicao = build_assignment(catalogo, tmp_path, seed=1)
    partes = splits_from_assignment(atribuicao, Level.GALAXY, IDENTIDADE_GALAXIA, tmp_path)

    treino = set(partes["train"]["path"])
    validacao = set(partes["val"]["path"])
    teste = set(partes["test"]["path"])
    assert treino & validacao == set()
    assert treino & teste == set()
    assert validacao & teste == set()


def test_atribuicao_e_estratificada_pelo_rotulo_fino(tmp_path, catalogo):
    """Estratificar pelo fino tambem acerta o grosso, porque o grosso e
    funcao do fino. O contrario nao vale."""
    atribuicao = build_assignment(catalogo, tmp_path, seed=1)

    for rotulo in catalogo["label_fino"].unique():
        do_rotulo = atribuicao[atribuicao["label_fino"] == rotulo]
        assert set(do_rotulo["split"]) == {"train", "val", "test"}, f"{rotulo} faltou em algum split"


def test_atribuicao_cobre_todas_as_imagens_uma_vez(tmp_path, catalogo):
    atribuicao = build_assignment(catalogo, tmp_path, seed=1)
    assert len(atribuicao) == len(catalogo)
    assert atribuicao["path"].is_unique
    assert set(atribuicao["split"]) == {"train", "val", "test"}


def test_e_reprodutivel_com_a_mesma_semente(tmp_path, catalogo):
    a = build_assignment(catalogo, tmp_path / "a", seed=7)
    b = build_assignment(catalogo, tmp_path / "b", seed=7)
    assert a.sort_values("path")["split"].tolist() == b.sort_values("path")["split"].tolist()


def test_rotulo_fino_vira_rotulo_grosso_no_nivel_1(tmp_path, catalogo):
    atribuicao = build_assignment(catalogo, tmp_path, seed=1)
    partes = splits_from_assignment(atribuicao, Level.OBJECT, FINO_PARA_OBJETO, tmp_path)

    treino = partes["train"]
    assert set(treino["label"]) <= set(Level.OBJECT.classes)
    # uma espiral precisa aparecer como 'galaxy', mantendo o fino na coluna
    espirais = treino[treino["label_fino"] == "spiral"]
    assert (espirais["label"] == "galaxy").all()


def test_o_teto_por_classe_respeita_os_splits(tmp_path, catalogo):
    """O teto e aplicado DENTRO de cada split. Aplicado antes, desfaria a
    proporcao que a atribuicao global acabou de garantir."""
    atribuicao = build_assignment(catalogo, tmp_path, seed=1)
    partes = splits_from_assignment(
        atribuicao, Level.OBJECT, FINO_PARA_OBJETO, tmp_path, cap_per_class=60
    )

    for nome, parte in partes.items():
        for _, n in parte["label"].value_counts().items():
            assert n <= 60, f"teto estourado em {nome}"
    # e continua havendo as tres classes em todos os splits
    for parte in partes.values():
        assert len(set(parte["label"])) == 3


def test_imagem_sem_traducao_para_o_nivel_sai(tmp_path, catalogo):
    """'star_field' nao pertence ao nivel 2 de galaxias; precisa sumir dali."""
    atribuicao = build_assignment(catalogo, tmp_path, seed=1)
    partes = splits_from_assignment(atribuicao, Level.GALAXY, IDENTIDADE_GALAXIA, tmp_path)

    for parte in partes.values():
        assert set(parte["label"]) <= set(Level.GALAXY.classes)
        assert "star_field" not in set(parte["label_fino"])


def test_assignment_sobrevive_ao_disco(tmp_path, catalogo):
    build_assignment(catalogo, tmp_path, seed=1)
    recarregado = load_assignment(tmp_path)
    assert len(recarregado) == len(catalogo)
    assert {"path", "label_fino", "split"} <= set(recarregado.columns)


def test_erro_claro_quando_o_assignment_nao_existe(tmp_path):
    with pytest.raises(FileNotFoundError, match="make_splits.py --all"):
        load_assignment(tmp_path)


def test_coluna_obrigatoria_ausente_falha(tmp_path):
    df = pd.DataFrame({"path": ["a.jpg"], "classe": ["spiral"]})
    with pytest.raises(ValueError, match="colunas obrigatorias"):
        build_assignment(df, tmp_path)
