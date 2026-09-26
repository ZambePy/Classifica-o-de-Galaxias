"""Roda a bateria de experimentos prevista no roteiro e agrega os resultados.

    # tudo (varias horas)
    python scripts/run_experiments.py

    # so as sementes multiplas do nivel 1
    python scripts/run_experiments.py --only seeds --levels object

O QUE ELE RESOLVE

Comparar arquiteturas ou medir variabilidade a mao significa rodar dezenas de
comandos, anotar numeros em algum lugar e torcer para nao trocar uma linha.
Este script roda, coleta os JSON de metricas e escreve uma tabela unica.

EXPERIMENTOS

  seeds      o mesmo modelo com sementes diferentes. Sem isso nao ha como
             saber se uma diferenca de 1 ponto entre dois modelos e real ou
             ruido - e a resposta costuma ser ruido.

  backbones  ResNet18 x EfficientNet-B0 x ResNet50, so nos niveis baratos.
             O nivel 2 leva ~70 min por treino e ficaria caro demais.

  tta        reavalia os checkpoints ja treinados com test-time augmentation.
             Nao treina nada; so mede se a media sobre as 8 simetrias rende.

Cada treino grava seus proprios artefatos em runs/<nome>/, entao nada se
perde se a bateria for interrompida no meio.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

BACKBONES = ("efficientnet_b0", "resnet50")
CONFIG_BASE = {
    "object": "configs/level1_object.yaml",
    "galaxy": "configs/level2_galaxy.yaml",
    "nebula": "configs/level3_nebula.yaml",
}


def rodar(cmd: list[str], descricao: str) -> bool:
    """Executa e informa. Falha de um experimento nao aborta a bateria."""
    print(f"\n{'=' * 74}\n{descricao}\n{'=' * 74}", flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=REPO)
    dt = (time.perf_counter() - t0) / 60
    if r.returncode == 0:
        print(f"-> concluido em {dt:.1f} min", flush=True)
        return True
    print(f"-> FALHOU (codigo {r.returncode}) apos {dt:.1f} min", flush=True)
    return False


def ler_metricas(runs: Path, nome: str) -> dict | None:
    arquivo = runs / nome / "metrics_test.json"
    if not arquivo.exists():
        return None
    return json.loads(arquivo.read_text(encoding="utf-8"))


def alertar_resultados_identicos(linhas: list[dict]) -> None:
    """Avisa quando variantes diferentes produzem numeros identicos.

    Dois modelos com arquiteturas diferentes nao dao a mesma acuracia ate a
    quarta casa decimal. Quando isso aparece, a explicacao quase sempre e que
    a avaliacao carregou o MESMO checkpoint nas duas - e o relatorio sai com
    uma conclusao errada ("o backbone nao faz diferenca") sem nenhum sinal de
    erro.

    Foi o que aconteceu nesta bateria: uma correcao no proprio script chegou
    depois que o processo ja estava em memoria, as avaliacoes rodaram sem
    apontar o checkpoint certo, e os tres backbones mediram o ResNet18.
    Reavaliados a mao, diferiam em ate 4,8 pontos de macro-F1.
    """
    from collections import defaultdict

    por_grupo: dict[tuple, list[str]] = defaultdict(list)
    for linha in linhas:
        chave = (linha["experimento"], linha["nivel"], linha["acuracia"], linha["macro_f1"])
        por_grupo[chave].append(linha["variante"])

    for (exp, nivel, acc, f1), variantes in por_grupo.items():
        if len(variantes) > 1 and exp != "seeds":
            print(
                f"\n[ALERTA] {exp}/{nivel}: as variantes {variantes} deram numeros\n"
                f"         IDENTICOS (acuracia {acc}, macro-F1 {f1}). Arquiteturas\n"
                f"         diferentes nao fazem isso - provavelmente a avaliacao\n"
                f"         carregou o mesmo checkpoint. Confira --checkpoint-name.",
                flush=True,
            )


def experimento_seeds(niveis: list[str], sementes: list[int], runs: Path) -> list[dict]:
    """O mesmo config com sementes diferentes, para medir a variabilidade."""
    linhas = []
    for nivel in niveis:
        config = CONFIG_BASE[nivel]
        for semente in sementes:
            nome = f"{Path(config).stem}_seed{semente}"
            ok = rodar(
                [str(PYTHON), "scripts/train.py", "--config", config,
                 "--seed", str(semente), "--name-suffix", f"_seed{semente}"],
                f"SEMENTE {semente} | nivel {nivel}",
            )
            if not ok:
                continue
            rodar(
                [str(PYTHON), "scripts/evaluate.py", "--config", config,
                 "--no-gradcam", "--checkpoint-name", f"{nome}.pt",
                 "--run-name", nome],
                f"avaliando {nome}",
            )
            m = ler_metricas(runs, nome)
            if m:
                linhas.append({"experimento": "seeds", "nivel": nivel,
                               "variante": f"seed {semente}", **resumo(m, f"{nome}.pt")})
    return linhas


def experimento_backbones(niveis: list[str], runs: Path) -> list[dict]:
    linhas = []
    for nivel in niveis:
        if nivel == "galaxy":
            print("[pulando] backbones no nivel 2 custam ~70 min cada", flush=True)
            continue
        base = Path(CONFIG_BASE[nivel]).stem
        for bb in BACKBONES:
            config = f"configs/{base}_{bb}.yaml"
            if not (REPO / config).exists():
                print(f"[pulando] {config} nao existe", flush=True)
                continue
            nome = f"{base}_{bb}"
            if not rodar([str(PYTHON), "scripts/train.py", "--config", config],
                         f"BACKBONE {bb} | nivel {nivel}"):
                continue
            rodar([str(PYTHON), "scripts/evaluate.py", "--config", config,
                   "--no-gradcam", "--checkpoint-name", f"{nome}.pt"],
                  f"avaliando {nome}")
            m = ler_metricas(runs, nome)
            if m:
                linhas.append({"experimento": "backbones", "nivel": nivel,
                               "variante": bb, **resumo(m, f"{nome}.pt")})
    return linhas


def experimento_tta(niveis: list[str], runs: Path) -> list[dict]:
    """Reavalia os checkpoints ja treinados com as 8 simetrias."""
    linhas = []
    for nivel in niveis:
        config = CONFIG_BASE[nivel]
        nome = Path(config).stem
        antes = ler_metricas(runs, nome)
        if not rodar([str(PYTHON), "scripts/evaluate.py", "--config", config,
                      "--tta", "--no-gradcam", "--run-name", f"{nome}_tta"],
                     f"TTA | nivel {nivel}"):
            continue
        depois = ler_metricas(runs, f"{nome}_tta")
        if antes:
            linhas.append({"experimento": "tta", "nivel": nivel,
                           "variante": "sem TTA", **resumo(antes)})
        if depois:
            linhas.append({"experimento": "tta", "nivel": nivel,
                           "variante": "com TTA", **resumo(depois)})
    return linhas


def resumo(m: dict, checkpoint: str = "") -> dict:
    return {
        "checkpoint": checkpoint,
        "n": m["n_amostras"],
        "acuracia": round(m["accuracy"], 4),
        "macro_f1": round(m["macro_f1"], 4),
        "mcc": round(m["matthews_corrcoef"], 4),
        "roc_auc": round(m.get("macro_roc_auc", float("nan")), 4),
    }


def escrever_tabela(linhas: list[dict], destino: Path) -> None:
    import pandas as pd

    if not linhas:
        print("nenhum resultado para agregar")
        return

    df = pd.DataFrame(linhas)
    destino.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destino.with_suffix(".csv"), index=False)

    partes = ["# Bateria de experimentos\n"]
    for exp, g in df.groupby("experimento", sort=False):
        partes.append(f"\n## {exp}\n")
        for nivel, gn in g.groupby("nivel", sort=False):
            partes.append(f"\n### nível `{nivel}`\n")
            partes.append("| variante | n | acurácia | macro-F1 | MCC | ROC AUC |")
            partes.append("|---|---:|---:|---:|---:|---:|")
            for _, r in gn.iterrows():
                partes.append(
                    f"| {r['variante']} | {r['n']} | {r['acuracia']:.4f} | "
                    f"{r['macro_f1']:.4f} | {r['mcc']:.4f} | {r['roc_auc']:.4f} |"
                )
            # Variabilidade entre sementes: o desvio e a barra de erro que
            # falta em quase todo relatorio de projeto de curso.
            if exp == "seeds" and len(gn) > 1:
                partes.append(
                    f"\n**média ± desvio** — acurácia {gn['acuracia'].mean():.4f} ± "
                    f"{gn['acuracia'].std():.4f} · macro-F1 {gn['macro_f1'].mean():.4f} ± "
                    f"{gn['macro_f1'].std():.4f}\n"
                )
    destino.write_text("\n".join(partes) + "\n", encoding="utf-8")
    print(f"\ntabela em {destino}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", default=["seeds", "backbones", "tta"],
                        choices=["seeds", "backbones", "tta"])
    parser.add_argument("--levels", nargs="*", default=["object", "nebula"],
                        choices=["object", "galaxy", "nebula"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[1, 2])
    args = parser.parse_args()

    from astro_classifier.paths import get_paths

    runs = get_paths().runs
    linhas: list[dict] = []
    t0 = time.perf_counter()

    if "tta" in args.only:
        linhas += experimento_tta(args.levels, runs)
    if "backbones" in args.only:
        linhas += experimento_backbones(args.levels, runs)
    if "seeds" in args.only:
        linhas += experimento_seeds(args.levels, args.seeds, runs)

    alertar_resultados_identicos(linhas)
    escrever_tabela(linhas, runs / "_experiments" / "resultados.md")
    print(f"\nbateria concluida em {(time.perf_counter() - t0) / 60:.0f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
