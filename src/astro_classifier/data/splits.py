"""Divisao train / val / test.

Tres regras que o projeto nao negocia:

1. ESTRATIFICADO por classe. Com ~150 nebulosas de reflexao contra ~2500
   planetarias, um split aleatorio pode deixar o conjunto de teste quase sem
   reflexao - e a metrica dessa classe vira ruido.

2. SEMENTE FIXA e splits salvos em CSV. Reprodutibilidade nao e so "usar a
   mesma seed": e poder apontar exatamente quais imagens estavam em cada
   conjunto quando aquele numero foi medido. Copie os CSVs para docs/ ao
   publicar resultados.

3. CONSISTENTE ENTRE NIVEIS. Uma imagem tem UM destino, valido para a
   cascata inteira.

A regra 3 foi aprendida do jeito caro. A primeira versao dividia cada nivel
de forma independente, e o resultado foi este:

    das 300 galaxias do conjunto de TESTE do nivel 1,
    218 estavam no conjunto de TREINO do nivel 2

Ou seja: ao avaliar a cascata de ponta a ponta, o modelo de morfologia ja
tinha visto a maioria daquelas imagens. A metrica ficou inutilizavel, e o
pior e que ela parecia funcionar - ninguem desconfia de um numero que nao
tem com o que ser comparado.

A correcao e decidir o destino de cada imagem UMA VEZ (build_assignment),
estratificando pelo rotulo mais FINO disponivel, e cada nivel filtra desse
conjunto (splits_from_assignment). Assim uma imagem nunca troca de lado.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from astro_classifier.taxonomy import Level

ASSIGNMENT_FILE = "assignment.csv"


def build_assignment(
    df: pd.DataFrame,
    out_dir: str | Path,
    *,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    min_per_class: int = 10,
) -> pd.DataFrame:
    """Decide train/val/test de cada imagem, uma vez para todos os niveis.

    `df` precisa de `path` e `label_fino` - o rotulo mais especifico que a
    imagem tem (spiral, planetary, star_field...), nunca o rotulo grosso.
    Estratificar pelo fino garante proporcao correta tambem nos niveis
    superiores, porque o rotulo grosso e funcao do fino.

    Devolve o DataFrame com a coluna `split`, e grava assignment.csv.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    faltando = {"path", "label_fino"} - set(df.columns)
    if faltando:
        raise ValueError(f"DataFrame sem as colunas obrigatorias: {sorted(faltando)}")

    df = df.drop_duplicates(subset="path").reset_index(drop=True)

    # Classes com pouquissimos exemplos quebram a estratificacao (o
    # train_test_split exige ao menos 2 por classe em cada corte).
    contagem = df["label_fino"].value_counts()
    raras = contagem[contagem < min_per_class]
    if len(raras) > 0:
        print(f"  [aviso] rotulos finos com menos de {min_per_class} exemplos: {raras.to_dict()}")
        df = df[~df["label_fino"].isin(raras.index)].reset_index(drop=True)

    treino, resto = train_test_split(
        df, test_size=val_size + test_size, stratify=df["label_fino"], random_state=seed
    )
    proporcao_teste = test_size / (val_size + test_size)
    validacao, teste = train_test_split(
        resto, test_size=proporcao_teste, stratify=resto["label_fino"], random_state=seed
    )

    df = df.copy()
    df["split"] = "train"
    df.loc[df["path"].isin(validacao["path"]), "split"] = "val"
    df.loc[df["path"].isin(teste["path"]), "split"] = "test"

    destino = out_dir / ASSIGNMENT_FILE
    df.sort_values("path").to_csv(destino, index=False)

    print(f"  {destino.name}: {len(df)} imagens")
    resumo = df.groupby(["split", "label_fino"]).size().unstack(fill_value=0)
    print(resumo.to_string().replace("\n", "\n  "))
    return df


def load_assignment(splits_dir: str | Path) -> pd.DataFrame:
    caminho = Path(splits_dir) / ASSIGNMENT_FILE
    if not caminho.exists():
        raise FileNotFoundError(
            f"{caminho} nao existe. Gere-o antes com:\n"
            "  python scripts/make_splits.py --all"
        )
    return pd.read_csv(caminho)


def splits_from_assignment(
    assignment: pd.DataFrame,
    level: Level,
    fino_para_nivel: dict[str, str],
    out_dir: str | Path,
    *,
    cap_per_class: int = 0,
    seed: int = 42,
    min_per_class: int = 10,
) -> dict[str, pd.DataFrame]:
    """Deriva os tres CSVs de um nivel a partir da atribuicao global.

    `fino_para_nivel` traduz o rotulo fino no rotulo deste nivel (no nivel 1,
    'spiral' -> 'galaxy'; nos niveis 2, a identidade). Imagens sem traducao
    simplesmente nao pertencem a este nivel e saem.

    `cap_per_class` e aplicado DENTRO de cada split, proporcionalmente - se
    fosse aplicado antes, mudaria as proporcoes que a atribuicao global
    acabou de garantir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = assignment.copy()
    df["label"] = df["label_fino"].map(fino_para_nivel)
    df = df.dropna(subset=["label"])
    df = df[df["label"].isin(level.classes)]

    if df.empty:
        raise ValueError(f"Nenhuma imagem pertence as classes do nivel '{level.value}'.")

    contagem = df["label"].value_counts()
    raras = contagem[contagem < min_per_class]
    if len(raras) > 0:
        print(f"  [aviso] classes removidas (menos de {min_per_class} exemplos): {raras.to_dict()}")
        df = df[~df["label"].isin(raras.index)]

    # As colunas p_<classe> carregam a distribuicao de votos humanos e sao o
    # insumo do treino com soft labels. Se forem perdidas aqui, o erro so
    # aparece no inicio do treino, depois de regerar tudo.
    colunas = [
        c
        for c in ("path", "label", "label_fino", "source")
        if c in df.columns
    ] + [c for c in df.columns if c.startswith("p_")]
    resultado: dict[str, pd.DataFrame] = {}

    for nome, fracao in (("train", 1 - 0.30), ("val", 0.15), ("test", 0.15)):
        parte = df[df["split"] == nome]

        if cap_per_class > 0:
            teto = max(1, round(cap_per_class * fracao))
            parte = (
                parte.sample(frac=1, random_state=seed)
                .groupby("label", sort=False)
                .head(teto)
            )

        parte = parte[colunas].sort_values("path")
        destino = out_dir / f"{level.value}_{nome}.csv"
        parte.to_csv(destino, index=False)
        resultado[nome] = parte
        print(f"  {destino.name}: {len(parte)} imagens  {parte['label'].value_counts().to_dict()}")

    return resultado


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
