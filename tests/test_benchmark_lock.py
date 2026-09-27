"""Testes do lock que impede dois treinos no mesmo checkpoint.

O INCIDENTE QUE ORIGINOU ISTO

Uma cadeia de scripts rodava `benchmark_galaxy10.py --task 3class` enquanto o
mesmo comando era disparado a mao. Os dois gravam `galaxy10_3class.pt` quando a
validacao melhora, e duas chamadas de `torch.save` no mesmo caminho nao dao
erro: uma trunca a outra. O arquivo resultante pode nao carregar - ou carregar
uma mistura das duas, o que e pior porque nao falha.

O projeto ja tinha sofrido de um checkpoint sobrescrito por outro caminho (um
experimento gravando sobre o de producao), e a correcao de la foi uma regra de
nomes. Faltava a protecao contra CONCORRENCIA.

Um destes testes existe por um bug DO PROPRIO LOCK: a primeira versao imprimia
"lock antigo ignorado (0s > 900s)" quando `--force` encontrava um lock fresco -
uma frase que se contradiz, e que faria alguem concluir que o lock estava velho
quando havia um treino vivo do outro lado.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from benchmark_galaxy10 import (  # noqa: E402
    JANELA_LOCK_S,
    checar_lock,
    checar_loss_finita,
    liberar_lock,
)


@pytest.fixture
def ckpt(tmp_path):
    return tmp_path / "galaxy10_10class.pt"


def _envelhecer(lock: Path, segundos: float) -> None:
    quando = time.time() - segundos
    os.utime(lock, (quando, quando))


class TestLock:
    def test_cria_lock_com_o_pid(self, ckpt):
        lock = checar_lock(ckpt, force=False)
        assert lock.exists()
        assert lock.read_text(encoding="utf-8").strip() == str(os.getpid())
        assert lock.suffix == ".lock"

    def test_lock_ativo_aborta(self, ckpt):
        checar_lock(ckpt, force=False)
        with pytest.raises(SystemExit) as exc:
            checar_lock(ckpt, force=False)

        msg = str(exc.value)
        # A mensagem tem de dizer COMO sair da situacao, nao so que deu errado.
        assert "--tag" in msg
        assert "--force" in msg
        assert ckpt.name in msg

    def test_force_passa_por_cima(self, ckpt, capsys):
        checar_lock(ckpt, force=False)
        checar_lock(ckpt, force=True)  # nao levanta

        saida = capsys.readouterr().out
        assert "--force" in saida
        assert "ATIVO" in saida, "o aviso tem de dizer que o lock estava ATIVO"
        # O BUG QUE ISTO PEGA: nao pode chamar de "abandonado" um lock fresco.
        assert "abandonado" not in saida

    def test_lock_abandonado_nao_bloqueia(self, ckpt, capsys):
        lock = checar_lock(ckpt, force=False)
        _envelhecer(lock, JANELA_LOCK_S + 60)

        checar_lock(ckpt, force=False)  # nao levanta
        saida = capsys.readouterr().out
        assert "abandonado" in saida
        assert "ATIVO" not in saida

    def test_fronteira_da_janela(self, ckpt):
        """Logo abaixo da janela bloqueia; logo acima, nao."""
        lock = checar_lock(ckpt, force=False)

        _envelhecer(lock, JANELA_LOCK_S - 30)
        with pytest.raises(SystemExit):
            checar_lock(ckpt, force=False)

        _envelhecer(lock, JANELA_LOCK_S + 30)
        checar_lock(ckpt, force=False)

    def test_tags_diferentes_nao_colidem(self, tmp_path):
        """E a saida recomendada pela mensagem de erro: tem de funcionar."""
        a = tmp_path / "galaxy10_3class.pt"
        b = tmp_path / "galaxy10_3class_v2.pt"

        checar_lock(a, force=False)
        checar_lock(b, force=False)  # nao levanta: outro arquivo, outro lock

        assert a.with_suffix(".lock").exists()
        assert b.with_suffix(".lock").exists()

    def test_lock_ilegivel_nao_derruba(self, ckpt):
        """Lock corrompido continua bloqueando, sem estourar na leitura."""
        lock = ckpt.with_suffix(".lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_bytes(b"\xff\xfe\x00nao-e-texto")

        with pytest.raises(SystemExit):
            checar_lock(ckpt, force=False)

    def test_cria_a_pasta_se_faltar(self, tmp_path):
        fundo = tmp_path / "a" / "b" / "galaxy10_10class.pt"
        lock = checar_lock(fundo, force=False)
        assert lock.exists()


class TestGuardaDeNaN:
    """`checar_loss_finita` - aborta no primeiro NaN em vez de treinar no vazio.

    O INCIDENTE

    Uma rodada de `--task 3class` divergiu para NaN na epoca 3 e seguiu treinando
    ate a 7, com a acuracia de validacao travada em 0,5525 - exatamente a fracao
    da classe majoritaria. O modelo passou a prever sempre `spiral`, e nada
    avisou: a loss virou `nan`, o laco continuou, o early stopping "funcionou", e
    o JSON final saiu com aparencia normal.

    Pior: aquele run gravava no mesmo checkpoint de um run sadio, e a avaliacao
    final dele carregou os pesos do OUTRO. O relatorio saiu com 0,9208 de
    acuracia para um modelo que era lixo.

    E o mesmo sintoma do primeiro bug silencioso do projeto (acuracia 0,567
    prevendo sempre a mesma classe). O projeto ja tinha guarda para isso em
    `training.loops.check_numerical_sanity`; este script nao usava porque tem
    laco de treino proprio.
    """

    def test_loss_normal_passa(self):
        for v in (0.0, 0.5, 1e-9, 12.5, 1e30):
            checar_loss_finita(v, epoca=1, lote=1)  # nao levanta

    def test_nan_aborta(self):
        with pytest.raises(SystemExit) as exc:
            checar_loss_finita(float("nan"), epoca=3, lote=17)
        msg = str(exc.value)
        assert "epoca 3" in msg and "lote 17" in msg
        assert "classe" in msg, "a mensagem tem de explicar o sintoma"

    def test_infinitos_abortam(self):
        for v in (float("inf"), -float("inf")):
            with pytest.raises(SystemExit):
                checar_loss_finita(v, epoca=1, lote=1)

    def test_mensagem_sugere_o_que_fazer(self):
        with pytest.raises(SystemExit) as exc:
            checar_loss_finita(float("nan"), epoca=1, lote=1)
        msg = str(exc.value)
        assert "--lr" in msg and "--batch-size" in msg


class TestLiberarLock:
    """`liberar_lock` - a limpeza, que NAO EXISTIA e ninguem viu.

    A primeira versao removia o lock na ultima linha de `main()`. Dois furos:

      1. Nao rodava quando o treino abortava, entao um NaN ou um Ctrl-C deixava o
         lock para tras e bloqueava a proxima tentativa por 15 minutos -
         exatamente quando a pessoa quer tentar de novo.

      2. Aquela remocao nunca chegou a existir no arquivo. O patch que a
         adicionou falhou em silencio (um escape que nao casou), e nenhum teste
         cobria a limpeza - so a criacao. O lock orfao foi achado depois de uma
         rodada que terminou BEM, com o PID de um processo ja morto.

    Hoje a limpeza roda via `atexit`, e estes testes cobrem os dois lados.
    """

    def test_remove_o_lock(self, ckpt):
        checar_lock(ckpt, force=False)
        assert liberar_lock(ckpt) is True
        assert not ckpt.with_suffix(".lock").exists()

    def test_sem_lock_nao_reclama(self, ckpt):
        assert liberar_lock(ckpt) is False

    def test_chamar_duas_vezes_e_seguro(self, ckpt):
        """`atexit` pode coexistir com uma remocao explicita."""
        checar_lock(ckpt, force=False)
        assert liberar_lock(ckpt) is True
        assert liberar_lock(ckpt) is False

    def test_depois_de_liberar_pode_treinar_de_novo(self, ckpt):
        """O ponto de tudo: uma segunda rodada nao deve ser recusada."""
        checar_lock(ckpt, force=False)
        liberar_lock(ckpt)
        checar_lock(ckpt, force=False)  # nao levanta
