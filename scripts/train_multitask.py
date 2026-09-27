"""Treina o modelo de backbone compartilhado e compara com a cascata.

    python scripts/train_multitask.py
    python scripts/train_multitask.py --eval-only   # so compara, sem treinar

Fecha o item do roteiro "Cascata x multi-tarefa". A pergunta e concreta:
compartilhar o backbone entre os tres niveis ganha ou perde da cascata de
tres redes independentes?

COMO A COMPARACAO E MANTIDA JUSTA

1. MESMA supervisao. O dataset e a uniao dos tres splits de treino, e cada
   imagem so recebe rotulo nos niveis em que ela aparecia (ver
   multitask_dataset.py). O multi-tarefa nao ve um rotulo a mais que a
   cascata viu somada.

2. MESMO conjunto de teste e MESMO codigo de metrica. A avaliacao final
   monta `CascadeSample` e chama `evaluate_cascade` - a mesmissima funcao que
   mede a cascata em evaluate_cascade.py. Se ela tiver um vies, os dois
   numeros sofrem o mesmo vies.

3. MESMO backbone do nivel 1 (resnet50, o mais forte dos tres). Perder com um
   backbone menor nao diria nada sobre a arquitetura.

4. MESMA travessia na inferencia. A cabeca de objeto decide qual cabeca de
   subtipo ler. O erro em cascata NAO desaparece por compartilhar o
   backbone - e um efeito da hierarquia, nao da topologia da rede.

O QUE O RESULTADO SIGNIFICA, NOS DOIS SENTIDOS

Se o multi-tarefa empatar, ele ganha na pratica: um backbone em vez de tres
e ~1/3 da memoria de inferencia, o que importa para o dashboard.

Se perder, tambem e resultado publicavel: mostra que as tres tarefas
disputam capacidade em vez de se ajudarem - e o `other` do nivel 1, que
inclui campos de estrelas e fotos do dia a dia, e bem diferente de
morfologia de galaxia.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/multitask.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="nao treina; carrega o checkpoint existente e so compara com a cascata",
    )
    args = parser.parse_args()

    # Imports pesados aqui dentro: `spawn` do Windows reexecuta este arquivo
    # em cada worker do DataLoader.
    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from astro_classifier.config import ExperimentConfig, resolve_device
    from astro_classifier.data.multitask_dataset import MultiTaskDataset
    from astro_classifier.data.transforms import eval_transforms, train_transforms
    from astro_classifier.evaluation.cascade import (
        CascadeSample,
        evaluate_cascade,
        print_cascade_report,
    )
    from astro_classifier.models.multitask import (
        LEVELS,
        MultiTaskClassifier,
        masked_multitask_loss,
    )
    from astro_classifier.paths import get_paths
    from astro_classifier.taxonomy import (
        GALAXY_CLASSES,
        NEBULA_CLASSES,
        SUBMODEL_FOR_OBJECT,
        Level,
    )

    config = ExperimentConfig.from_yaml(args.config)
    device = resolve_device(args.device)
    paths = get_paths().ensure()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    destino = paths.checkpoints / "multitask_best.pt"
    saida = paths.runs / "_multitask"
    saida.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ treino
    if not args.eval_only:
        treino = MultiTaskDataset(
            paths.splits, paths.raw, "train", train_transforms(config.data.image_size, config.data.augment)
        )
        validacao = MultiTaskDataset(
            paths.splits, paths.raw, "val", eval_transforms(config.data.image_size)
        )
        print(treino.resumo())
        print(validacao.resumo())
        print()

        loader_treino = DataLoader(
            treino,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            persistent_workers=config.data.num_workers > 0,
            drop_last=True,
        )
        loader_val = DataLoader(
            validacao,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=max(1, config.data.num_workers // 2),
            persistent_workers=config.data.num_workers > 0,
        )

        modelo = MultiTaskClassifier(
            backbone=config.model.backbone,
            pretrained=config.model.pretrained,
            dropout=config.model.dropout,
        ).to(device)

        otimizador = torch.optim.AdamW(
            modelo.parameters(), lr=config.optim.lr, weight_decay=config.optim.weight_decay
        )
        agendador = (
            torch.optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=config.optim.epochs)
            if config.optim.scheduler == "cosine"
            else None
        )

        def validar() -> tuple[dict[str, float], float]:
            """Acuracia por cabeca + a media macro entre elas.

            A media e sobre as CABECAS, nao sobre as amostras. Media por
            amostra seria dominada pelas 5659 galaxias da validacao e a
            cabeca de nebulosa (410 imagens) quase nao contaria - justamente
            a que mais interessa melhorar.
            """
            modelo.eval()
            certos = {lv.value: 0 for lv in LEVELS}
            totais = {lv.value: 0 for lv in LEVELS}
            with torch.inference_mode():
                for imagens, alvos in loader_val:
                    logits = modelo(imagens.to(device))
                    for nome, logit in logits.items():
                        alvo = alvos[nome].to(device)
                        m = alvo >= 0
                        if not m.any():
                            continue
                        certos[nome] += int((logit[m].argmax(1) == alvo[m]).sum())
                        totais[nome] += int(m.sum())
            accs = {n: (certos[n] / totais[n] if totais[n] else 0.0) for n in certos}
            return accs, float(np.mean(list(accs.values())))

        melhor = -1.0
        melhor_epoca = 0
        sem_melhora = 0
        historico = []

        print(f"treinando em {device} | {sum(p.numel() for p in modelo.parameters()):,} parametros\n")
        for epoca in range(1, config.optim.epochs + 1):
            modelo.train()
            t0 = time.time()
            soma, n_lotes = 0.0, 0
            somas_cabeca = {lv.value: 0.0 for lv in LEVELS}

            for imagens, alvos in loader_treino:
                imagens = imagens.to(device, non_blocking=True)
                alvos = {k: v.to(device, non_blocking=True) for k, v in alvos.items()}

                otimizador.zero_grad(set_to_none=True)
                perda, partes = masked_multitask_loss(modelo(imagens), alvos)
                perda.backward()
                otimizador.step()

                soma += float(perda.detach())
                for k, v in partes.items():
                    somas_cabeca[k] += v
                n_lotes += 1

            if agendador is not None:
                agendador.step()

            accs, macro = validar()
            dt = time.time() - t0
            historico.append(
                {"epoca": epoca, "loss": soma / n_lotes, "val_acc": accs, "val_macro": macro}
            )
            detalhe = "  ".join(f"{k[:3]} {v:.4f}" for k, v in accs.items())
            print(
                f"epoca {epoca:>2}/{config.optim.epochs}  loss {soma / n_lotes:.4f}  "
                f"[{detalhe}]  macro {macro:.4f}  {dt:.0f}s"
            )

            if macro > melhor:
                melhor, melhor_epoca, sem_melhora = macro, epoca, 0
                modelo.save(destino, config, extra={"val_macro": macro, "epoca": epoca})
                print(f"  -> novo melhor salvo em {destino}")
            else:
                sem_melhora += 1
                if sem_melhora >= config.optim.early_stopping_patience:
                    print(f"  -> early stopping: {sem_melhora} epocas sem melhora")
                    break

        (saida / "history.json").write_text(
            json.dumps(historico, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nmelhor macro de validacao: {melhor:.4f} (epoca {melhor_epoca})")

    # ------------------------------------------------- avaliacao tipo cascata
    if not destino.exists():
        print(f"\nCheckpoint nao encontrado: {destino}")
        return 1

    print("\n" + "=" * 72)
    print("AVALIACAO PONTA A PONTA - mesmo teste e mesmo codigo da cascata")
    print("=" * 72)

    modelo, ckpt = MultiTaskClassifier.load(destino, device=device)

    # Mesma construcao de verdade de evaluate_cascade.py: o subtipo so existe
    # para galaxias e nebulosas; em `other` o rotulo fino (star_field,
    # empty_field...) nao e subtipo da taxonomia.
    verdade = pd.read_csv(paths.splits / "object_test.csv")
    subtipos = set(GALAXY_CLASSES) | set(NEBULA_CLASSES)
    verdade["subtipo_verdadeiro"] = verdade["label_fino"].where(
        verdade["label_fino"].isin(subtipos)
    )

    from PIL import Image
    from tqdm import tqdm

    transformacao = eval_transforms(config.data.image_size)
    amostras: list[CascadeSample] = []
    ilegiveis = 0

    modelo.eval()
    for linha in tqdm(verdade.itertuples(index=False), total=len(verdade), desc="multi-tarefa"):
        try:
            imagem = Image.open(paths.raw / linha.path).convert("RGB")
        except OSError:
            ilegiveis += 1
            continue

        with torch.inference_mode():
            # UMA passada pelo backbone serve as tres cabecas - o ganho de
            # custo da arquitetura aparece exatamente aqui.
            logits = modelo(transformacao(imagem).unsqueeze(0).to(device))

        objeto = Level.OBJECT.classes[int(logits[Level.OBJECT.value].argmax())]
        ramo = SUBMODEL_FOR_OBJECT[objeto]
        subtipo = (
            ramo.classes[int(logits[ramo.value].argmax())] if ramo is not None else None
        )

        fino = linha.subtipo_verdadeiro
        amostras.append(
            CascadeSample(
                objeto_verdadeiro=linha.label,
                objeto_predito=objeto,
                subtipo_verdadeiro=None if pd.isna(fino) else fino,
                subtipo_predito=subtipo,
            )
        )

    if ilegiveis:
        print(f"[aviso] {ilegiveis} imagens ilegiveis foram puladas")

    relatorio = evaluate_cascade(amostras)
    print_cascade_report(relatorio)

    resultado = {
        "backbone": ckpt["backbone"],
        "parametros": sum(p.numel() for p in modelo.parameters()),
        "val_macro": ckpt.get("extra", {}).get("val_macro"),
        "multitarefa": relatorio.to_dict(),
    }

    # --- comparacao direta com a cascata, se ela ja foi medida ---
    cascata_json = paths.runs / "_cascade" / "metrics.json"
    if cascata_json.exists():
        casc = json.loads(cascata_json.read_text(encoding="utf-8"))
        resultado["cascata"] = {
            k: casc.get(k)
            for k in ("n_amostras", "nivel1_acuracia", "folha_acuracia", "erro_cascata",
                      "subtipo_acuracia_condicional")
        }

        print("\n" + "=" * 72)
        print("CASCATA x MULTI-TAREFA")
        print("=" * 72)
        if casc.get("n_amostras") != relatorio.n_amostras:
            print(
                f"[AVISO] a cascata foi medida em {casc.get('n_amostras')} amostras e este\n"
                f"modelo em {relatorio.n_amostras}. Os numeros NAO sao comparaveis - rode\n"
                "scripts/evaluate_cascade.py de novo para alinhar os conjuntos."
            )
        print(f"\n  {'metrica':<32}{'cascata':>10}{'multi':>10}{'dif':>9}")
        print("  " + "-" * 61)
        for chave, rotulo in [
            ("nivel1_acuracia", "acuracia do nivel 1"),
            ("folha_acuracia", "acuracia da folha"),
            ("subtipo_acuracia_condicional", "subtipo dado nivel 1 correto"),
            ("erro_cascata", "erro em cascata"),
        ]:
            a = casc.get(chave)
            b = getattr(relatorio, chave)
            if a is None:
                continue
            print(f"  {rotulo:<32}{a:>10.4f}{b:>10.4f}{b - a:>+9.4f}")

        dif = relatorio.folha_acuracia - casc.get("folha_acuracia", 0.0)
        print(
            f"\n  Na acuracia da folha o multi-tarefa fica {dif:+.4f}.\n"
            "  Lembre do outro lado da balanca: um backbone em vez de tres."
        )
    else:
        print(
            f"\n[aviso] {cascata_json} nao existe - sem comparacao direta.\n"
            "Rode: python scripts/evaluate_cascade.py"
        )

    destino_json = saida / "metrics.json"
    destino_json.write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nResultados em {destino_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
