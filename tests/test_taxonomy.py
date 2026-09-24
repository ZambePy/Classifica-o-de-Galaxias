"""A taxonomia e a fonte de verdade de todo o resto - se ela quebrar em
silencio, os modelos passam a prever a classe errada sem erro nenhum."""

from __future__ import annotations

import pytest

from astro_classifier.taxonomy import (
    DISPLAY_PT,
    GALAXY_CLASSES,
    NEBULA_CLASSES,
    OBJECT_CLASSES,
    SUBMODEL_FOR_OBJECT,
    Level,
    index_to_label,
    label_to_index,
)


def test_cada_nivel_conhece_suas_classes():
    assert Level.OBJECT.classes == OBJECT_CLASSES
    assert Level.GALAXY.classes == GALAXY_CLASSES
    assert Level.NEBULA.classes == NEBULA_CLASSES


def test_num_classes_bate_com_a_lista():
    for level in Level:
        assert level.num_classes == len(level.classes)


def test_label_e_index_sao_inversos():
    for level in Level:
        for i, label in enumerate(level.classes):
            assert label_to_index(level, label) == i
            assert index_to_label(level, i) == label


def test_label_invalido_falha_explicitamente():
    with pytest.raises(ValueError, match="nao e uma classe valida"):
        label_to_index(Level.GALAXY, "planetary")


def test_index_fora_do_intervalo_falha():
    with pytest.raises(ValueError, match="fora do intervalo"):
        index_to_label(Level.OBJECT, 99)


def test_toda_classe_tem_rotulo_em_portugues():
    """O dashboard exibe DISPLAY_PT. Uma classe sem traducao viraria KeyError
    em producao, no meio de uma demonstracao."""
    for level in Level:
        for label in level.classes:
            assert label in DISPLAY_PT, f"'{label}' nao tem rotulo em DISPLAY_PT"


def test_cascata_cobre_todas_as_classes_do_nivel_1():
    assert set(SUBMODEL_FOR_OBJECT) == set(OBJECT_CLASSES)


def test_other_encerra_a_cascata():
    assert SUBMODEL_FOR_OBJECT["other"] is None


def test_classes_nao_se_repetem_dentro_do_nivel():
    for level in Level:
        assert len(level.classes) == len(set(level.classes))
