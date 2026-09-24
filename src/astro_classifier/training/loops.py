"""Loop de treino de um nivel da cascata.

Enxuto de proposito: sem framework de treino, sem abstracao de callback. Sao
~150 linhas que voce consegue ler inteiras e explicar numa banca - e e isso
que um baseline reprodutivel precisa ser.

Notas de hardware (GTX 1660 Ti, 6 GB):
  - precisao mista (AMP) e praticamente obrigatoria; corta a memoria pela
    metade e acelera o treino
  - resnet18 + batch 32 + 224x224 cabe folgado
  - resnet50 exige batch 16
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

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None,
        label_smoothing=config.optim.label_smoothing,
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
        val_loss, y_true, y_pred, _ = evaluate(model, val_loader, criterion, device)
        metrics = classification_metrics(y_true, y_pred, model.classes)

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
    for images, targets in tqdm(loader, desc=tag, leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = criterion(model(images), targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


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
