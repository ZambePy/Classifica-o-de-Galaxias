"""Testes do modelo de backbone compartilhado.

O bug que estes testes existem para impedir:

  UM LOTE SEM NEBULOSA. O nivel 3 tem 1916 imagens de treino contra 26401 de
  galaxia. Com lote de 32, a maioria dos lotes nao tem nenhuma nebulosa. Se a
  loss dividir por um contador zerado, o passo inteiro vira NaN e o treino
  morre em silencio - exatamente o que aconteceu no projeto com precisao
  mista, quando a acuracia ficou em 0,567 predizendo sempre a mesma classe.

  SUPERVISAO VAZANDO ENTRE CABECAS. Se a cabeca de nebulosa receber gradiente
  de imagens de galaxia, ela aprende lixo e o resultado da comparacao com a
  cascata fica errado - parecendo certo.
"""

from __future__ import annotations

import pandas as pd
import pytest

# Ver nota em test_tta.py: o CI nao instala torch.
torch = pytest.importorskip("torch", reason="exige PyTorch")

from astro_classifier.data.multitask_dataset import MultiTaskDataset  # noqa: E402
from astro_classifier.models.multitask import (  # noqa: E402
    LEVELS,
    masked_multitask_loss,
)
from astro_classifier.taxonomy import Level  # noqa: E402


def _logits(n=4):
    return {lv.value: torch.randn(n, lv.num_classes, requires_grad=True) for lv in LEVELS}


class TestLossMascarada:
    def test_cabeca_sem_amostra_contribui_zero_e_nao_nan(self):
        """O caso comum no treino real: lote sem nenhuma nebulosa."""
        alvos = {
            "object": torch.tensor([0, 1, 2, 0]),
            "galaxy": torch.tensor([0, 1, -1, -1]),
            "nebula": torch.tensor([-1, -1, -1, -1]),
        }
        total, partes = masked_multitask_loss(_logits(), alvos)
        assert torch.isfinite(total), "loss virou NaN com uma cabeca vazia"
        assert partes["nebula"] == 0.0

    def test_nenhuma_cabeca_valida_e_erro_explicito(self):
        """Tudo -1 e bug de dataset, nao um lote legitimo. Tem de falhar alto."""
        alvos = {lv.value: torch.full((4,), -1) for lv in LEVELS}
        with pytest.raises(ValueError, match="nenhuma cabeca"):
            masked_multitask_loss(_logits(), alvos)

    def test_gradiente_nao_chega_na_cabeca_sem_rotulo(self):
        """A prova de que nao ha vazamento de supervisao entre tarefas."""
        logits = _logits()
        alvos = {
            "object": torch.tensor([0, 1, 2, 0]),
            "galaxy": torch.tensor([-1, -1, -1, -1]),
            "nebula": torch.tensor([-1, -1, -1, -1]),
        }
        total, _ = masked_multitask_loss(logits, alvos)
        total.backward()

        assert logits["object"].grad is not None
        assert logits["object"].grad.abs().sum() > 0
        # Sem rotulo => sem gradiente. `None` ou exatamente zero, ambos ok.
        for nome in ("galaxy", "nebula"):
            g = logits[nome].grad
            assert g is None or g.abs().sum() == 0, f"gradiente vazou para '{nome}'"

    def test_media_e_sobre_as_amostras_validas(self):
        """A loss de uma cabeca nao pode ser diluida pelas amostras sem rotulo.

        Duas amostras com rotulo entre quatro devem dar a MESMA loss que as
        mesmas duas amostras sozinhas. Se o codigo dividisse pelo tamanho do
        lote, o valor cairia pela metade e a cabeca aprenderia mais devagar
        quanto mais raro fosse o rotulo - justamente ao contrario do desejado.
        """
        torch.manual_seed(0)
        base = torch.randn(2, Level.OBJECT.num_classes)
        alvo2 = torch.tensor([0, 1])

        pequeno = {lv.value: torch.zeros(2, lv.num_classes) for lv in LEVELS}
        pequeno["object"] = base
        a_peq = {lv.value: torch.full((2,), -1) for lv in LEVELS}
        a_peq["object"] = alvo2
        loss_peq, _ = masked_multitask_loss(pequeno, a_peq)

        grande = {lv.value: torch.zeros(4, lv.num_classes) for lv in LEVELS}
        grande["object"] = torch.cat([base, torch.randn(2, Level.OBJECT.num_classes)])
        a_gra = {lv.value: torch.full((4,), -1) for lv in LEVELS}
        a_gra["object"] = torch.tensor([0, 1, -1, -1])
        loss_gra, _ = masked_multitask_loss(grande, a_gra)

        assert float(loss_peq) == pytest.approx(float(loss_gra), abs=1e-6)

    def test_pesos_por_cabeca(self):
        alvos = {
            "object": torch.tensor([0, 1, 2, 0]),
            "galaxy": torch.tensor([0, 1, 2, 0]),
            "nebula": torch.tensor([-1, -1, -1, -1]),
        }
        logits = _logits()
        sem_peso, _ = masked_multitask_loss(logits, alvos)
        com_peso, _ = masked_multitask_loss(logits, alvos, weights={"galaxy": 0.0})
        assert float(com_peso.detach()) < float(sem_peso.detach())


class TestModelo:
    def test_uma_passada_do_backbone_alimenta_tres_cabecas(self):
        from astro_classifier.models.multitask import MultiTaskClassifier

        modelo = MultiTaskClassifier(backbone="resnet18", pretrained=False)
        saida = modelo(torch.randn(2, 3, 224, 224))

        assert set(saida) == {lv.value for lv in LEVELS}
        for lv in LEVELS:
            assert saida[lv.value].shape == (2, lv.num_classes)

    def test_checkpoint_recusa_taxonomia_diferente(self, tmp_path, monkeypatch):
        """Mesma protecao do AstroClassifier: pesos e classes tem de casar.

        E o bug mais perigoso possivel aqui - carregar pesos antigos depois de
        mudar a ordem das classes devolve predicoes trocadas em silencio.
        """
        from astro_classifier.config import ExperimentConfig
        from astro_classifier.models.multitask import MultiTaskClassifier

        modelo = MultiTaskClassifier(backbone="resnet18", pretrained=False)
        config = ExperimentConfig(name="t", level=Level.OBJECT)
        caminho = tmp_path / "mt.pt"
        modelo.save(caminho, config)

        # Falsifica as classes gravadas, como se a taxonomia tivesse mudado.
        ck = torch.load(caminho, map_location="cpu", weights_only=False)
        ck["classes"]["object"] = ["outra", "coisa", "aqui"]
        torch.save(ck, caminho)

        with pytest.raises(RuntimeError, match="taxonomia atual"):
            MultiTaskClassifier.load(caminho)

    def test_recusa_checkpoint_da_cascata(self, tmp_path):
        from astro_classifier.config import ExperimentConfig
        from astro_classifier.models.classifier import AstroClassifier
        from astro_classifier.models.multitask import MultiTaskClassifier

        simples = AstroClassifier(level=Level.OBJECT, backbone="resnet18", pretrained=False)
        caminho = tmp_path / "simples.pt"
        simples.save(caminho, ExperimentConfig(name="t", level=Level.OBJECT))

        with pytest.raises(RuntimeError, match="nao e um checkpoint multi-tarefa"):
            MultiTaskClassifier.load(caminho)


class TestDataset:
    @staticmethod
    def _montar(tmp_path):
        """Tres splits minimos, com uma imagem em dois niveis."""
        splits = tmp_path / "splits"
        splits.mkdir()
        cols = ["path", "label", "label_fino", "source"]

        pd.DataFrame(
            [
                ["a.jpg", "galaxy", "spiral", "gz"],
                ["b.jpg", "nebula", "emission", "neb"],
                ["c.jpg", "other", "star_field", "other"],
            ],
            columns=cols,
        ).to_csv(splits / "object_train.csv", index=False)

        pd.DataFrame([["a.jpg", "spiral", "spiral", "gz"]], columns=cols).to_csv(
            splits / "galaxy_train.csv", index=False
        )
        pd.DataFrame([["b.jpg", "emission", "emission", "neb"]], columns=cols).to_csv(
            splits / "nebula_train.csv", index=False
        )
        return splits

    def test_uniao_com_rotulo_so_onde_havia_supervisao(self, tmp_path):
        splits = self._montar(tmp_path)
        ds = MultiTaskDataset(splits, tmp_path, "train", transform=None)

        assert len(ds) == 3  # a, b, c - uniao sem repetir
        assert ds.rotulos["a.jpg"] == {
            "object": Level.OBJECT.classes.index("galaxy"),
            "galaxy": Level.GALAXY.classes.index("spiral"),
        }
        # `c` e `other`: nao tem subtipo em nivel nenhum.
        assert set(ds.rotulos["c.jpg"]) == {"object"}
        # `b` nao pode ter rotulo de galaxia.
        assert "galaxy" not in ds.rotulos["b.jpg"]

    def test_split_faltando_da_erro_util(self, tmp_path):
        splits = tmp_path / "splits"
        splits.mkdir()
        with pytest.raises(FileNotFoundError, match="make_splits"):
            MultiTaskDataset(splits, tmp_path, "train")

    def test_rotulo_fora_da_taxonomia(self, tmp_path):
        splits = self._montar(tmp_path)
        pd.DataFrame(
            [["z.jpg", "quasar", "quasar", "x"]],
            columns=["path", "label", "label_fino", "source"],
        ).to_csv(splits / "object_train.csv", index=False)

        with pytest.raises(ValueError, match="quasar"):
            MultiTaskDataset(splits, tmp_path, "train")
