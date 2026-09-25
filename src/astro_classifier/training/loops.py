"""Loop de treino de um nivel da cascata.

Enxuto de proposito: sem framework de treino, sem abstracao de callback. Sao
~150 linhas que voce consegue ler inteiras e explicar numa banca - e e isso
que um baseline reprodutivel precisa ser.

Notas de hardware (GTX 1660 Ti, 6 GB), todas medidas nesta maquina:
  - precisao mista DESLIGADA. Ao contrario do que o senso comum sugere,
    nesta placa o AMP deixa o treino 4,6x mais lento (sem tensor cores, o
    fp16 nao acelera nada) e o cuDNN produz NaN a partir de batch 64.
    Ver check_numerical_sanity() abaixo.
  - resnet18 + batch 64 + 224x224 usa ~1,6 GB de VRAM: sobra muito
  - o gargalo e o DataLoader (CPU decodificando JPG), nao a GPU
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from astro_classifier.config import ExperimentConfig
from astro_classifier.models.backbone import set_backbone_trainable
from astro_classifier.models.classifier import AstroClassifier


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_macro_f1: float
    seconds: float


@dataclass
class TrainingHistory:
    epochs: list[EpochResult] = field(default_factory=list)
    best_epoch: int = -1
    best_macro_f1: float = -1.0

    def to_dict(self) -> dict:
        return {
            "best_epoch": self.best_epoch,
            "best_macro_f1": self.best_macro_f1,
            "epochs": [vars(e) for e in self.epochs],
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


class NumericalSanityError(RuntimeError):
    """O modelo produziu NaN/inf antes mesmo de comecar a treinar."""


def check_numerical_sanity(
    model: nn.Module, sample: torch.Tensor, device: str, use_amp: bool
) -> None:
    """Um forward de teste antes do treino. Falha em segundos, nao de manha.

    Existe por causa de um bug real encontrado neste projeto: na GTX 1660 Ti
    (Turing TU116, sem tensor cores), o cuDNN produz NaN no forward sob
    precisao mista a partir de batch 64. Os pesos nunca se corrompem - o
    GradScaler rejeita todo passo - entao o treino roda a noite inteira,
    grava checkpoints, e de manha o modelo preve sempre a mesma classe.
    A loss aparece como `nan` no log, mas a acuracia parece plausivel.

    Verificar leva um segundo. Nao verificar custa uma noite.
    """
    model.eval()
    try:
        with torch.inference_mode():
            with torch.amp.autocast("cuda", enabled=use_amp):
                saida = model(sample.to(device))
        ruim = int(torch.isnan(saida.float()).sum()) + int(torch.isinf(saida.float()).sum())
    finally:
        model.train()

    if ruim == 0:
        return

    detalhe = ""
    if use_amp:
        with torch.inference_mode():
            limpo = model(sample.to(device))
        if int(torch.isnan(limpo.float()).sum()) == 0:
            detalhe = (
                "\n\nO mesmo forward em fp32 funciona - o problema e a PRECISAO MISTA.\n"
                "Esta GPU provavelmente nao lida bem com fp16 (a serie GTX 16xx nao\n"
                "tem tensor cores). Ponha `mixed_precision: false` no YAML: nessas\n"
                "placas o AMP nao acelera nada e ainda quebra o treino."
            )

    raise NumericalSanityError(
        f"O modelo produziu {ruim} valores NaN/inf num forward de teste, ANTES do "
        f"treino comecar (batch de {sample.size(0)}, amp={use_amp})." + detalhe
    )


def set_seed(seed: int) -> None:
    """Reprodutibilidade. cudnn.deterministic custa velocidade, mas o ponto
    do projeto e poder repetir o numero publicado."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(
    model: AstroClassifier,
    config: ExperimentConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    checkpoint_path: str | Path,
    class_weights: torch.Tensor | None = None,
) -> TrainingHistory:
    """Treina, valida a cada epoca e salva o melhor checkpoint por macro-F1.

    Por que macro-F1 e nao acuracia: as classes sao desbalanceadas. Um modelo
    que so chuta a classe majoritaria tem acuracia alta e macro-F1 baixo -
    escolher pelo macro-F1 impede que esse modelo inutil seja salvo.
    """
    from astro_classifier.evaluation.metrics import classification_metrics

    model.to(device)

    # Duas losses, conforme o alvo. A validacao SEMPRE usa cross-entropy
    # contra o rotulo duro - e contra ele que as metricas sao reportadas e
    # comparadas entre rodadas.
    if config.data.soft_labels:
        from astro_classifier.training.soft_labels import SoftLabelLoss

        criterion = SoftLabelLoss(
            class_weights=class_weights.to(device) if class_weights is not None else None,
            temperature=config.data.soft_label_temperature,
            label_smoothing=config.optim.label_smoothing,
        ).to(device)
        print(
            f"treinando com SOFT LABELS (temperatura {config.data.soft_label_temperature}): "
            "o alvo e a distribuicao de votos humanos, nao o rotulo vencedor"
        )
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None,
            label_smoothing=config.optim.label_smoothing,
        )

    val_criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.optim.lr, weight_decay=config.optim.weight_decay
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.optim.epochs)
        if config.optim.scheduler == "cosine"
        else None
    )

    use_amp = config.optim.mixed_precision and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Antes de qualquer epoca: um forward de teste com o batch real. Se esta
    # combinacao de GPU, precisao e tamanho de batch produzir NaN, e melhor
    # saber agora do que amanha de manha.
    amostra, _ = next(iter(train_loader))
    check_numerical_sanity(model, amostra, device, use_amp)

    history = TrainingHistory()
    epochs_without_improvement = 0

    for epoch in range(1, config.optim.epochs + 1):
        started = time.perf_counter()

        # Warm-up com backbone congelado: com datasets pequenos, deixar o
        # backbone solto desde a epoca 1 destroi as features do ImageNet
        # antes da cabeca aprender qualquer coisa.
        frozen = epoch <= config.model.freeze_backbone_epochs
        set_backbone_trainable(model.backbone, trainable=not frozen)

        train_loss = _train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp, epoch, frozen
        )
        val_loss, y_true, y_pred, y_proba = evaluate(model, val_loader, val_criterion, device)
        metrics = classification_metrics(y_true, y_pred, model.classes, y_proba)

        if scheduler is not None:
            scheduler.step()

        result = EpochResult(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_accuracy=metrics["accuracy"],
            val_macro_f1=metrics["macro_f1"],
            seconds=time.perf_counter() - started,
        )
        history.epochs.append(result)
        print(
            f"epoca {epoch:>3}/{config.optim.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"acc={metrics['accuracy']:.4f}  macroF1={metrics['macro_f1']:.4f}  "
            f"({result.seconds:.0f}s)"
        )

        if metrics["macro_f1"] > history.best_macro_f1:
            history.best_macro_f1 = metrics["macro_f1"]
            history.best_epoch = epoch
            model.save(checkpoint_path, config, extra={"val_metrics": metrics, "epoch": epoch})
            print(f"  -> novo melhor modelo salvo em {checkpoint_path}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.optim.early_stopping_patience:
                print(
                    f"  -> early stopping: {epochs_without_improvement} epocas sem melhora"
                )
                break

    return history


def _train_one_epoch(
    model, loader, criterion, optimizer, scaler, device, use_amp, epoch, frozen
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    tag = f"epoca {epoch}" + (" [backbone congelado]" if frozen else "")
    nao_finitos = 0

    for images, targets in tqdm(loader, desc=tag, leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = criterion(model(images), targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Uma unica loss NaN contamina a media da epoca inteira. Contamos
        # separado para que o numero no log continue significando algo e o
        # problema apareca como aviso, nao como um "nan" mudo.
        if torch.isfinite(loss):
            total_loss += loss.item()
            n_batches += 1
        else:
            nao_finitos += 1

    if nao_finitos:
        total = nao_finitos + n_batches
        print(
            f"  [AVISO] {nao_finitos}/{total} batches produziram loss nao finita. "
            "O treino nao esta aprendendo nesses passos - verifique precisao "
            "mista, taxa de aprendizado e pesos de classe."
        )

    return total_loss / max(n_batches, 1) if n_batches else float("nan")


@torch.inference_mode()
def evaluate(
    model, loader: DataLoader, criterion, device: str
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Devolve (loss media, y_true, y_pred, probabilidades)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_true, all_pred, all_probs = [], [], []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        total_loss += criterion(logits, targets).item()
        n_batches += 1

        probs = torch.softmax(logits, dim=1)
        all_true.append(targets.cpu().numpy())
        all_pred.append(probs.argmax(dim=1).cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    return (
        total_loss / max(n_batches, 1),
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_probs),
    )
