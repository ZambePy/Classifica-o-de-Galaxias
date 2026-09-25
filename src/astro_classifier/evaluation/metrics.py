"""Metricas de classificacao.

Acuracia sozinha mente em dataset desbalanceado: um modelo que so chuta a
classe majoritaria pode ter 57% de acuracia e nao ter aprendido nada. Por
isso o projeto reporta sempre o conjunto completo.

O QUE CADA METRICA RESPONDE

  acuracia            de tudo que o modelo viu, quanto ele acertou?
                      Enganosa quando as classes tem tamanhos diferentes.

  acuracia balanceada media do recall por classe. Responde "o modelo vai bem
                      em TODAS as classes?" - chutar a majoritaria derruba
                      esta metrica na hora.

  precisao (classe)   dos que o modelo CHAMOU de espiral, quantos eram?
                      Baixa = o modelo grita essa classe demais.

  recall (classe)     das espirais que EXISTEM, quantas ele achou?
                      Baixo = o modelo deixa passar.

  F1 (classe)         media harmonica de precisao e recall. Punitiva: so e
                      alta quando as duas sao.

  macro-F1            media simples dos F1. Cada classe pesa igual, mesmo a
                      minoritaria. E a metrica que escolhe o checkpoint.

  weighted-F1         media dos F1 ponderada pelo tamanho da classe. Mais
                      otimista que a macro; mostrar as duas juntas revela o
                      quanto o desempenho depende das classes grandes.

  especificidade      dos que NAO sao espiral, quantos ele deixou de fora?

  ROC AUC             o modelo ORDENA bem? Independe do limiar escolhido.
                      0,5 = chute; 1,0 = separacao perfeita.

  average precision   area sob a curva precisao-recall. Mais informativa que
                      a ROC AUC quando a classe e rara.

  kappa de Cohen      concordancia corrigida pelo acaso. Compara o modelo
                      com um chutador que respeita a distribuicao das classes.

  MCC                 correlacao entre predito e verdadeiro, de -1 a +1.
                      A metrica mais conservadora para dados desbalanceados:
                      so fica alta quando as quatro celulas da matriz de
                      confusao vao bem.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    y_proba: np.ndarray | None = None,
) -> dict:
    """Pacote completo de metricas, pronto para gravar em JSON.

    `y_proba` (N, C) e opcional. Sem ele, as metricas que dependem de
    ordenacao (ROC AUC, average precision) sao omitidas - as demais saem
    normalmente.
    """
    labels = list(range(len(classes)))

    report = classification_report(
        y_true, y_pred, labels=labels, target_names=classes,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    metrics = {
        "n_amostras": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {
            nome: {
                "precision": float(report[nome]["precision"]),
                "recall": float(report[nome]["recall"]),
                "f1": float(report[nome]["f1-score"]),
                "specificity": _specificity(cm, i),
                "support": int(report[nome]["support"]),
            }
            for i, nome in enumerate(classes)
        },
        "confusion_matrix": cm.tolist(),
        "classes": classes,
    }

    if y_proba is not None:
        metrics.update(_ranking_metrics(y_true, np.asarray(y_proba), classes, labels))

    return metrics


def _specificity(cm: np.ndarray, indice: int) -> float:
    """Verdadeiros negativos sobre todos os negativos reais, para uma classe."""
    verdadeiros_positivos = cm[indice, indice]
    falsos_positivos = cm[:, indice].sum() - verdadeiros_positivos
    falsos_negativos = cm[indice, :].sum() - verdadeiros_positivos
    verdadeiros_negativos = cm.sum() - verdadeiros_positivos - falsos_positivos - falsos_negativos
    denominador = verdadeiros_negativos + falsos_positivos
    return float(verdadeiros_negativos / denominador) if denominador else 0.0


def _ranking_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, classes: list[str], labels: list[int]
) -> dict:
    """ROC AUC e average precision, no esquema um-contra-o-resto."""
    saida: dict = {}

    # Uma classe ausente do conjunto torna sua AUC indefinida. Acontece de
    # verdade em conjuntos de teste pequenos e nao pode derrubar o relatorio.
    presentes = set(np.unique(y_true).tolist())
    if len(presentes) < 2:
        return saida

    binario = np.zeros((len(y_true), len(classes)), dtype=int)
    binario[np.arange(len(y_true)), y_true] = 1

    try:
        saida["macro_roc_auc"] = float(
            roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=labels)
        )
        saida["weighted_roc_auc"] = float(
            roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted", labels=labels)
        )
    except ValueError:
        pass

    por_classe: dict[str, dict] = {}
    for i, nome in enumerate(classes):
        if i not in presentes:
            continue
        entrada: dict[str, float] = {}
        try:
            entrada["roc_auc"] = float(roc_auc_score(binario[:, i], y_proba[:, i]))
        except ValueError:
            pass
        try:
            entrada["average_precision"] = float(
                average_precision_score(binario[:, i], y_proba[:, i])
            )
        except ValueError:
            pass
        if entrada:
            por_classe[nome] = entrada

    if por_classe:
        saida["ranking_per_class"] = por_classe
    return saida


def save_metrics(metrics: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def print_metrics(metrics: dict, titulo: str = "") -> None:
    """Resumo legivel no terminal."""
    if titulo:
        print(f"\n{'=' * 72}\n{titulo}\n{'=' * 72}")

    print(f"\namostras              {metrics['n_amostras']}")
    print(f"acuracia              {metrics['accuracy']:.4f}")
    print(f"acuracia balanceada   {metrics['balanced_accuracy']:.4f}")
    print(f"kappa de Cohen        {metrics['cohen_kappa']:.4f}")
    print(f"MCC                   {metrics['matthews_corrcoef']:.4f}")
    if "macro_roc_auc" in metrics:
        print(f"ROC AUC (macro)       {metrics['macro_roc_auc']:.4f}")

    print(f"\n{'medias':<14}{'precisao':>10}{'recall':>10}{'F1':>10}")
    print("-" * 44)
    print(
        f"{'macro':<14}{metrics['macro_precision']:>10.4f}"
        f"{metrics['macro_recall']:>10.4f}{metrics['macro_f1']:>10.4f}"
    )
    print(
        f"{'ponderada':<14}{metrics['weighted_precision']:>10.4f}"
        f"{metrics['weighted_recall']:>10.4f}{metrics['weighted_f1']:>10.4f}"
    )

    ranking = metrics.get("ranking_per_class", {})
    tem_auc = bool(ranking)
    cabecalho = f"\n{'classe':<22}{'precisao':>9}{'recall':>9}{'F1':>8}{'especif.':>10}{'n':>7}"
    if tem_auc:
        cabecalho += f"{'ROC AUC':>9}{'AP':>8}"
    print(cabecalho)
    print("-" * (len(cabecalho) - 1))

    for nome, m in metrics["per_class"].items():
        linha = (
            f"{nome:<22}{m['precision']:>9.3f}{m['recall']:>9.3f}"
            f"{m['f1']:>8.3f}{m['specificity']:>10.3f}{m['support']:>7d}"
        )
        if tem_auc:
            r = ranking.get(nome, {})
            linha += f"{r.get('roc_auc', float('nan')):>9.3f}{r.get('average_precision', float('nan')):>8.3f}"
        print(linha)


def metrics_to_markdown(metrics: dict, titulo: str = "Resultados") -> str:
    """Tabela em Markdown, para colar direto no relatorio ou no README."""
    ranking = metrics.get("ranking_per_class", {})
    linhas = [
        f"### {titulo}",
        "",
        f"- Amostras: **{metrics['n_amostras']}**",
        f"- Acurácia: **{metrics['accuracy']:.4f}** · "
        f"Acurácia balanceada: **{metrics['balanced_accuracy']:.4f}**",
        f"- Macro-F1: **{metrics['macro_f1']:.4f}** · "
        f"F1 ponderado: **{metrics['weighted_f1']:.4f}**",
        f"- Kappa de Cohen: **{metrics['cohen_kappa']:.4f}** · "
        f"MCC: **{metrics['matthews_corrcoef']:.4f}**",
    ]
    if "macro_roc_auc" in metrics:
        linhas.append(f"- ROC AUC (macro): **{metrics['macro_roc_auc']:.4f}**")

    cabecalho = "| Classe | Precisão | Recall | F1 | Especificidade | N |"
    separador = "|---|---:|---:|---:|---:|---:|"
    if ranking:
        cabecalho += " ROC AUC | AP |"
        separador += "---:|---:|"
    linhas += ["", cabecalho, separador]

    for nome, m in metrics["per_class"].items():
        linha = (
            f"| {nome} | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{m['f1']:.3f} | {m['specificity']:.3f} | {m['support']} |"
        )
        if ranking:
            r = ranking.get(nome, {})
            auc = f"{r['roc_auc']:.3f}" if "roc_auc" in r else "-"
            ap = f"{r['average_precision']:.3f}" if "average_precision" in r else "-"
            linha += f" {auc} | {ap} |"
        linhas.append(linha)

    return "\n".join(linhas) + "\n"


def save_markdown(metrics: dict, path: str | Path, titulo: str = "Resultados") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(metrics_to_markdown(metrics, titulo), encoding="utf-8")
