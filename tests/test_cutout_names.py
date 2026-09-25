"""Testes do nome de arquivo dos recortes.

Nascem de um bug que apagou 18% do dataset em silencio: os catalogos usam
numeracao propria e independente, entao a regiao 1 de Sharpless, a 1 de RCW
e a 1 de Lynds - tres objetos em pontos distantes do ceu - viravam todas
`emission/1.jpg`. 301 dos 1667 objetos foram sobrescritos.

O pior: o log dizia "1667/1667 imagens salvas". A gravacao funcionou; so
que no mesmo arquivo. So contar os arquivos em disco revelou.
"""

from __future__ import annotations

from astro_classifier.data.cutouts import cutout_filename


def test_mesmo_numero_em_catalogos_diferentes_nao_colide():
    """A REGRESSAO PRINCIPAL."""
    sharpless = {"name": "1", "source": "VII/20", "label": "emission"}
    rcw = {"name": "1", "source": "VII/216", "label": "emission"}
    lynds = {"name": "1", "source": "VII/9", "label": "emission"}

    nomes = {cutout_filename(sharpless), cutout_filename(rcw), cutout_filename(lynds)}
    assert len(nomes) == 3, f"objetos de catalogos diferentes colidiram: {nomes}"


def test_o_catalogo_aparece_no_nome():
    nome = cutout_filename({"name": "42", "source": "VII/20"})
    assert "VII" in nome and "20" in nome and "42" in nome
    assert nome.endswith(".jpg")


def test_barra_do_identificador_nao_vira_subpasta():
    """'VII/20' tem uma barra. Se ela sobrevivesse, o arquivo iria parar numa
    subpasta inexistente e a gravacao falharia."""
    nome = cutout_filename({"name": "1", "source": "VII/20"})
    assert "/" not in nome and "\\" not in nome


def test_sem_fonte_ainda_gera_nome_valido():
    """Campos de ceu sorteados nao vem de catalogo nenhum."""
    nome = cutout_filename({"name": "empty_00042"})
    assert nome == "empty_00042.jpg"


def test_caracteres_problematicos_sao_neutralizados():
    nome = cutout_filename({"name": "Sh 2-155", "source": "VII/20"})
    assert " " not in nome
    assert nome.endswith(".jpg")
    assert "2-155" in nome, "a parte distintiva do nome precisa sobreviver"


def test_nomes_diferentes_no_mesmo_catalogo_continuam_diferentes():
    a = cutout_filename({"name": "PNG 000.1+02.3", "source": "V/84/main"})
    b = cutout_filename({"name": "PNG 000.1+02.4", "source": "V/84/main"})
    assert a != b


def test_nome_vazio_nao_gera_arquivo_sem_nome():
    nome = cutout_filename({"name": "", "source": "VII/20"})
    assert nome.endswith(".jpg")
    assert len(nome) > len(".jpg")


def test_catalogo_inteiro_produz_nomes_unicos():
    """Simula o caso real: tres catalogos com numeracao sobreposta."""
    linhas = [
        {"name": str(i), "source": fonte, "label": "emission"}
        for fonte in ("VII/20", "VII/216", "VII/9")
        for i in range(1, 51)
    ]
    nomes = [cutout_filename(r) for r in linhas]
    assert len(set(nomes)) == len(linhas) == 150
