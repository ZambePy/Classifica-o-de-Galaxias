"""Metricas de classificacao.

Acuracia sozinha mente em dataset desbalanceado. O projeto reporta sempre o
conjunto completo - acuracia, precisao/recall/F1 por classe, macro-F1 e a
matriz de confusao - porque e a por-classe que mostra qual tipo de nebulosa
o modelo simplesmente nao aprendeu.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]
) -> dict:
    """Pacote completo de metricas, pronto para gravar em JSON."""
    labels = list(range(len(classes)))

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {
            c: {
                "precision": float(report[c]["precision"]),
                "recall": float(report[c]["recall"]),
                "f1": float(report[c]["f1-score"]),
                "support": int(report[c]["support"]),
            }
            for c in classes
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classes": classes,
    }


def save_metrics(metrics: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def print_metrics(metrics: dict) -> None:
    """Resumo legivel no terminal."""
    print(f"\nacuracia          {metrics['accuracy']:.4f}")
    print(f"acuracia balanc.  {metrics['balanced_accuracy']:.4f}")
    print(f"macro-F1          {metrics['macro_f1']:.4f}")
    print(f"\n{'classe':<22}{'prec':>8}{'recall':>8}{'F1':>8}{'n':>8}")
    print("-" * 54)
    for name, m in metrics["per_class"].items():
        print(
            f"{name:<22}{m['precision']:>8.3f}{m['recall']:>8.3f}"
            f"{m['f1']:>8.3f}{m['support']:>8d}"
        )
