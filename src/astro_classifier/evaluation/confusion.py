"""Figuras de avaliacao: matriz de confusao e curvas de treino.

Sao as duas figuras que entram em qualquer apresentacao do projeto. A matriz
normalizada por linha e a que interessa: ela responde "das nebulosas de
reflexao reais, quantas o modelo acertou?", que e a pergunta certa quando as
classes tem tamanhos diferentes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sem janela: funciona em servidor e em CI
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    matrix: list[list[int]] | np.ndarray,
    classes: list[str],
    out_path: str | Path,
    *,
    normalize: bool = True,
    title: str = "Matriz de confusao",
) -> Path:
    cm = np.asarray(matrix, dtype=float)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        # Classe sem nenhum exemplo no conjunto: evita divisao por zero.
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(1.6 * len(classes) + 2, 1.4 * len(classes) + 2))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)

    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")
    ax.set_title(title + (" (normalizada por linha)" if normalize else ""))

    threshold = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=9,
            )

    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_training_curves(history: dict, out_path: str | Path) -> Path:
    """Loss e macro-F1 por epoca. A figura que revela overfitting num olhar:
    train_loss caindo enquanto val_loss sobe."""
    epochs = [e["epoch"] for e in history["epochs"]]

    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, [e["train_loss"] for e in history["epochs"]], label="treino")
    ax_loss.plot(epochs, [e["val_loss"] for e in history["epochs"]], label="validacao")
    ax_loss.set_xlabel("epoca")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_f1.plot(epochs, [e["val_macro_f1"] for e in history["epochs"]], color="tab:green")
    ax_f1.axvline(history["best_epoch"], ls="--", color="gray", label="melhor epoca")
    ax_f1.set_xlabel("epoca")
    ax_f1.set_ylabel("macro-F1")
    ax_f1.set_title("Macro-F1 (validacao)")
    ax_f1.legend()
    ax_f1.grid(alpha=0.3)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
