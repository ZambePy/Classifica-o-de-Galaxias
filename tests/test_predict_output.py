"""Testes do desenho da saida de `scripts/predict.py`.

O BUG QUE ESTES TESTES IMPEDEM

A saida do `predict.py` usa blocos ('█', '░') e uma seta ('►') para desenhar as
barras de probabilidade. O terminal padrao do Windows usa cp1252, que nao tem
nenhum desses caracteres, e o comando morria com

    UnicodeEncodeError: 'charmap' codec can't encode character '\\u25ba'

no MEIO da saida - depois de imprimir o nome do arquivo e a resposta. A pessoa
via o resultado comecar e o comando explodir. E `predict.py` e o caminho que o
README documenta para classificar uma imagem, ou seja, provavelmente a primeira
coisa que alguem roda no projeto.

Nenhum teste pegava isso porque nenhum teste chamava a formatacao. A logica de
inferencia estava certa; o que quebrava era imprimir.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture
def pred():
    """O modulo recarregado, para `UNICODE_OK` comecar no padrao."""
    import predict

    return importlib.reload(predict)


class TestDesenhoUnicode:
    def test_barra_usa_blocos_quando_da(self, pred):
        pred.UNICODE_OK = True
        b = pred.barra(0.5, largura=10)
        assert b == "█" * 5 + "░" * 5

    def test_barra_cai_para_ascii(self, pred):
        pred.UNICODE_OK = False
        b = pred.barra(0.5, largura=10)
        assert b == "#" * 5 + "." * 5
        assert b.isascii(), "no fallback nada pode estar fora do ASCII"

    def test_marcador_nos_dois_modos(self, pred):
        pred.UNICODE_OK = True
        assert pred.marcador(True) == "►"
        pred.UNICODE_OK = False
        assert pred.marcador(True) == ">"
        assert pred.marcador(False) == " "

    def test_fallback_inteiro_e_codificavel_em_cp1252(self, pred):
        """A prova direta: no modo ASCII, tudo passa pelo cp1252.

        É o teste que teria pegado o bug. Se alguem acrescentar um caractere
        bonito ao desenho sem pensar no console do Windows, isto falha.
        """
        pred.UNICODE_OK = False
        pedacos = [pred.barra(f / 10, largura=12) for f in range(11)]
        pedacos += [pred.marcador(True), pred.marcador(False)]
        for p in pedacos:
            p.encode("cp1252")  # levanta UnicodeEncodeError se houver algo fora

    def test_barra_nos_extremos(self, pred):
        pred.UNICODE_OK = True
        assert pred.barra(0.0, largura=8) == "░" * 8
        assert pred.barra(1.0, largura=8) == "█" * 8

    def test_barra_tem_sempre_a_largura_pedida(self, pred):
        """Barra de largura variavel desalinha a tabela inteira."""
        for modo in (True, False):
            pred.UNICODE_OK = modo
            for f in range(0, 101):
                assert len(pred.barra(f / 100, largura=24)) == 24


class TestPrepararSaida:
    def test_aceita_stdout_utf8(self, pred, monkeypatch):
        class Fake:
            encoding = "utf-8"

        monkeypatch.setattr(pred.sys, "stdout", Fake())
        assert pred._preparar_saida() is True

    def test_reconfigura_quando_possivel(self, pred, monkeypatch):
        """cp1252 + `reconfigure` disponivel: troca para UTF-8 e mantem o visual."""

        class Fake:
            encoding = "cp1252"

            def __init__(self):
                self.chamou = None

            def reconfigure(self, encoding=None):
                self.chamou = encoding

        f = Fake()
        monkeypatch.setattr(pred.sys, "stdout", f)
        assert pred._preparar_saida() is True
        assert f.chamou == "utf-8"

    def test_cai_para_ascii_quando_nao_da(self, pred, monkeypatch):
        """cp1252 e `reconfigure` falhando: degradar, nao explodir."""

        class Fake:
            encoding = "cp1252"

            def reconfigure(self, encoding=None):
                raise OSError("fluxo nao suporta")

        monkeypatch.setattr(pred.sys, "stdout", Fake())
        assert pred._preparar_saida() is False

    def test_sem_reconfigure(self, pred, monkeypatch):
        class Fake:
            encoding = "cp1252"

        monkeypatch.setattr(pred.sys, "stdout", Fake())
        assert pred._preparar_saida() is False

    def test_encoding_ausente(self, pred, monkeypatch):
        """Alguns fluxos nao expoem `encoding`; nao pode estourar."""

        class Fake:
            encoding = None

        monkeypatch.setattr(pred.sys, "stdout", Fake())
        assert pred._preparar_saida() is False
