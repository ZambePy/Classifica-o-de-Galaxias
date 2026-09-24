"""Divisao train / val / test.

Duas regras que o projeto nao negocia:

1. ESTRATIFICADO por classe. Com ~150 nebulosas de reflexao contra ~2500
   planetarias, um split aleatorio pode deixar o conjunto de teste quase sem
   reflexao - e a metrica dessa classe vira ruido.

2. SEMENTE FIXA e splits salvos em CSV. Reprodutibilidade nao e so "usar a
   mesma seed": e poder apontar exatamente quais imagens estavam em cada
   conjunto quando aquele numero foi medido. Copie os CSVs para docs/ ao
   publicar resultados.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from astro_classifier.taxonomy import Level


def make_splits(
    df: pd.DataFrame,
    level: Level,
    out_dir: str | Path,
    *,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    min_per_class: int = 10,
) -> dict[str, pd.DataFrame]:
    """Divide `df` (precisa das colunas path e label) e grava os tres CSVs.

    Classes com menos de `min_per_class` exemplos sao removidas com aviso:
    manter uma classe com 4 imagens produz metricas sem significado.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = {"path", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame sem as colunas obrigatorias: {sorted(missing)}")

    df = df[df["label"].isin(level.classes)].copy()

    counts = df["label"].value_counts()
    too_few = counts[counts < min_per_class]
    if len(too_few) > 0:
        print(
            f"  [aviso] classes removidas por terem menos de {min_per_class} "
            f"exemplos: {too_few.to_dict()}"
        )
        df = df[~df["label"].isin(too_few.index)]

    if df.empty:
        raise ValueError(f"Nenhum dado restante para o nivel {level.value!r}.")

    train_df, temp_df = train_test_split(
        df, test_size=val_size + test_size, stratify=df["label"], random_state=seed
    )
    rel_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=rel_test, stratify=temp_df["label"], random_state=seed
    )

    splits = {"train": train_df, "val": val_df, "test": test_df}
    for name, part in splits.items():
        dest = out_dir / f"{level.value}_{name}.csv"
        part.sort_values("path").to_csv(dest, index=False)
        print(f"  {dest.name}: {len(part)} imagens  {part['label'].value_counts().to_dict()}")

    return splits
