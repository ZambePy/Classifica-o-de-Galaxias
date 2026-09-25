"""Avalia a CASCATA INTEIRA no conjunto de teste, como ela roda em producao.

    python scripts/evaluate_cascade.py

Diferente de evaluate.py, que mede um nivel isolado, este script passa cada
imagem pelo sistema completo - nivel 1 decide, o resultado escolhe o modelo
de nivel 2 - e mede o rotulo FINAL. E o numero que responde "o sistema
funciona?", e o unico que expoe o erro em cascata.

Exige os checkpoints dos tres niveis. Rode depois de scripts/train_all.ps1.

Gera em runs/_cascade/:
    metrics.json        todas as metricas, incluindo por ramo
    metrics.md          tabela pronta para colar no relatorio
    confusion.png       matriz de confusao sobre os rotulos finais
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from astro_classifier.config import resolve_device
from astro_classifier.data.transforms import eval_transforms
from astro_classifier.evaluation.cascade import (
    CascadeSample,
    evaluate_cascade,
    print_cascade_report,
)
from astro_classifier.evaluation.confusion import plot_confusion_matrix
from astro_classifier.inference.pipeline import HierarchicalPipeline, ModelNotAvailable
from astro_classifier.paths import get_paths
from astro_classifier.taxonomy import GALAXY_CLASSES, NEBULA_CLASSES, Level

# Rotulo fino (spiral, planetary...) -> rotulo do nivel 1.
FINO_PARA_OBJETO = (
    {c: "galaxy" for c in GALAXY_CLASSES}
    | {c: "nebula" for c in NEBULA_CLASSES}
)


def carregar_verdade(paths) -> pd.DataFrame:
    """Monta a tabela de teste da cascata a partir do split do nivel 1.

    O rotulo fino vem da coluna `label_fino`, gravada pela atribuicao global.
    Uma imagem de `other` nao tem subtipo - correto, a cascata para nela.

    ISTO JA ESTEVE ERRADO. A versao anterior cruzava o teste do nivel 1 com
    os testes dos niveis 2. Como os splits eram independentes, a maioria das
    imagens nao tinha correspondencia, ficava sem subtipo, e o codigo
    comparava o rotulo grosso ('galaxy') contra o predito fino ('spiral') -
    contando como erro. A acuracia de folha saiu 0,38 com niveis de 0,97 e
    0,93. Com a atribuicao global isso desaparece por construcao: se a
    imagem esta no teste do nivel 1, seu rotulo fino esta na mesma linha.
    """
    arquivo = paths.splits / "object_test.csv"
    objeto = pd.read_csv(arquivo)

    if "label_fino" not in objeto.columns:
        raise ValueError(
            f"{arquivo} nao tem a coluna 'label_fino'. Ele foi gerado por uma "
            "versao antiga do make_splits.py, que dividia cada nivel de forma "
            "independente e vazava imagens de treino para o teste da cascata.\n"
            "Regere os splits com:\n"
            "  python scripts/make_splits.py --all --from-index galaxies_index.csv"
        )

    objeto = objeto.rename(columns={"label": "objeto_verdadeiro"})

    # O subtipo so existe para galaxias e nebulosas. Em `other`, o rotulo
    # fino (star_field, empty_field...) nao e um subtipo da taxonomia - a
    # cascata nao tenta refinar ali.
    subtipos = set(GALAXY_CLASSES) | set(NEBULA_CLASSES)
    objeto["subtipo_verdadeiro"] = objeto["label_fino"].where(
        objeto["label_fino"].isin(subtipos)
    )
    return objeto


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=0, help="avalia so N imagens (teste rapido)")
    args = parser.parse_args()

    paths = get_paths()
    device = resolve_device(args.device)

    if not (paths.splits / "object_test.csv").exists():
        print("Split de teste do nivel 1 nao encontrado.")
        print("Rode: python scripts/make_splits.py --level object ...")
        return 1

    pipeline = HierarchicalPipeline(device=device)
    disponiveis = pipeline.loaded_levels()
    print(f"checkpoints disponiveis: {disponiveis or 'nenhum'}")
    if "object" not in disponiveis:
        print("\nO checkpoint do nivel 1 e obrigatorio. Treine antes:")
        print("  python scripts/train.py --config configs/level1_object.yaml")
        return 1
    faltando = {"galaxy", "nebula"} - set(disponiveis)
    if faltando:
        print(f"[aviso] sem checkpoint de {sorted(faltando)} - esses ramos ficarao sem subtipo")

    verdade = carregar_verdade(paths)
    if args.limit:
        verdade = verdade.head(args.limit)
    print(f"avaliando {len(verdade)} imagens de teste em {device}\n")

    transformacao = eval_transforms()
    amostras: list[CascadeSample] = []
    ilegiveis = 0

    for linha in tqdm(verdade.itertuples(index=False), total=len(verdade), desc="cascata"):
        caminho = paths.raw / linha.path
        try:
            imagem = Image.open(caminho).convert("RGB")
        except OSError:
            ilegiveis += 1
            continue

        tensor = transformacao(imagem).unsqueeze(0)
        with torch.inference_mode():
            nivel1, _ = pipeline._run_level(tensor, Level.OBJECT)

            subtipo = None
            from astro_classifier.taxonomy import SUBMODEL_FOR_OBJECT

            ramo = SUBMODEL_FOR_OBJECT[nivel1.predicted]
            if ramo is not None:
                try:
                    nivel2, _ = pipeline._run_level(tensor, ramo)
                    subtipo = nivel2.predicted
                except ModelNotAvailable:
                    pass

        verdadeiro_fino = linha.subtipo_verdadeiro
        if isinstance(verdadeiro_fino, float):  # NaN do merge
            verdadeiro_fino = None

        amostras.append(
            CascadeSample(
                objeto_verdadeiro=linha.objeto_verdadeiro,
                objeto_predito=nivel1.predicted,
                subtipo_verdadeiro=verdadeiro_fino,
                subtipo_predito=subtipo,
            )
        )

    if ilegiveis:
        print(f"[aviso] {ilegiveis} imagens ilegiveis foram puladas")
    if not amostras:
        print("Nenhuma imagem avaliada.")
        return 1

    relatorio = evaluate_cascade(amostras)
    print_cascade_report(relatorio)

    destino = paths.runs / "_cascade"
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "metrics.json").write_text(
        json.dumps(relatorio.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_confusion_matrix(
        relatorio.matriz_folha,
        relatorio.classes_folha,
        destino / "confusion.png",
        title="Cascata completa - rotulo final",
    )
    _escrever_markdown(relatorio, destino / "metrics.md")

    print(f"\nResultados em {destino}")
    return 0


def _escrever_markdown(relatorio, caminho: Path) -> None:
    linhas = [
        "### Cascata completa (ponta a ponta)",
        "",
        f"- Amostras: **{relatorio.n_amostras}**",
        f"- Acurácia do nível 1: **{relatorio.nivel1_acuracia:.4f}**",
        f"- Acurácia da folha (rótulo final): **{relatorio.folha_acuracia:.4f}**",
        f"- Erro em cascata: **{relatorio.erro_cascata:.4f}**",
        f"- Subtipo dado nível 1 correto: **{relatorio.subtipo_acuracia_condicional:.4f}**",
        "",
        "| Ramo | N | Nível 1 | Folha | Subtipo \\| nível 1 ok |",
        "|---|---:|---:|---:|---:|",
    ]
    for nome, m in relatorio.por_ramo.items():
        cond = (
            f"{m['subtipo_acuracia_condicional']:.3f}"
            if m["subtipo_acuracia_condicional"] is not None
            else "—"
        )
        linhas.append(
            f"| {nome} | {m['n']} | {m['nivel1_acuracia']:.3f} | {m['folha_acuracia']:.3f} | {cond} |"
        )
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
