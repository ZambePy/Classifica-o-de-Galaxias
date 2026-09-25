"""Mede o detector de fora-de-dominio (MSP) do nivel 1.

    # o que da para medir sem dados extras: deteccao de ERRO
    python scripts/evaluate_ood.py

    # com um conjunto realmente fora do dominio
    python scripts/evaluate_ood.py --ood-dir "C:/astrofotos_hubble"

DUAS COISAS DIFERENTES, E O PROJETO PRECISA DAS DUAS

O artigo que originou este metodo (Hendrycks & Gimpel, ICLR 2017) se chama
"A Baseline for Detecting Misclassified AND Out-of-Distribution Examples".
Sao dois usos do mesmo sinal:

  DETECCAO DE ERRO       a confianca baixa avisa que ESTA predicao
                         provavelmente esta errada? Mede-se com o proprio
                         conjunto de teste, separando acertos de erros.
                         E o que roda por padrao aqui.

  DETECCAO DE DOMINIO    a confianca baixa avisa que a imagem nao se parece
                         com nada do treino? Exige imagens de OUTRO dominio,
                         que o modelo nunca viu.

CUIDADO QUE CUSTOU UMA MEDICAO ERRADA AQUI

As 600 fotos comuns de `non_astronomical` NAO servem como conjunto fora do
dominio: elas foram treinadas, como parte da classe `other`. O modelo as
conhece. Media-las daria uma AUROC alta e vazia.

Para a deteccao de dominio de verdade, aponte --ood-dir para imagens de
outro dominio que o modelo nunca viu - astrofotos processadas do Hubble ou
do JWST sao o caso de uso real do dashboard, ja que e isso que as pessoas
sobem.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from astro_classifier.config import ExperimentConfig, resolve_device
from astro_classifier.data.datasets import AstroImageDataset
from astro_classifier.data.transforms import eval_transforms
from astro_classifier.models.classifier import AstroClassifier
from astro_classifier.ood.msp import OODDetector, evaluate_detector
from astro_classifier.paths import get_paths
from astro_classifier.training.loops import evaluate

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def deteccao_de_erro(probs: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, det: OODDetector) -> dict:
    """A confianca separa predicoes certas das erradas?"""
    from sklearn.metrics import roc_auc_score

    confianca = probs.max(axis=1)
    acertou = (y_true == y_pred).astype(int)

    if acertou.min() == acertou.max():
        return {"erro": "todas as predicoes acertaram ou todas erraram"}

    return {
        "auroc": float(roc_auc_score(acertou, confianca)),
        "confianca_media_acertos": float(confianca[acertou == 1].mean()),
        "confianca_media_erros": float(confianca[acertou == 0].mean()),
        "limiar": det.threshold,
        "acertos_acima_do_limiar": float((confianca[acertou == 1] >= det.threshold).mean()),
        "erros_abaixo_do_limiar": float((confianca[acertou == 0] < det.threshold).mean()),
        "n_acertos": int((acertou == 1).sum()),
        "n_erros": int((acertou == 0).sum()),
    }


def probs_de_pasta(model, pasta: Path, device: str, image_size: int) -> np.ndarray:
    """Softmax do nivel 1 para toda imagem de uma pasta."""
    arquivos = [p for p in pasta.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES]
    if not arquivos:
        raise ValueError(f"nenhuma imagem em {pasta}")

    transformacao = eval_transforms(image_size)
    saidas = []
    for caminho in tqdm(arquivos, desc="fora do dominio"):
        try:
            imagem = Image.open(caminho).convert("RGB")
        except OSError:
            continue
        with torch.inference_mode():
            t = transformacao(imagem).unsqueeze(0).to(device)
            saidas.append(torch.softmax(model(t), dim=1)[0].cpu().numpy())
    return np.stack(saidas)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/level1_object.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--ood-dir",
        default=None,
        help="pasta com imagens de OUTRO dominio, nunca vistas no treino",
    )
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    device = resolve_device(args.device)
    paths = get_paths()

    checkpoint = paths.checkpoints / "object_best.pt"
    if not checkpoint.exists():
        print(f"Checkpoint nao encontrado: {checkpoint}")
        return 1

    model, _ = AstroClassifier.load(checkpoint, device=device)
    detector = OODDetector.load(paths.checkpoints / "ood_threshold.json")
    print(f"limiar calibrado: {detector.threshold:.4f} (alvo: {detector.target_tpr:.0%})\n")

    test_ds = AstroImageDataset(
        paths.splits / "object_test.csv", paths.raw, config.level, eval_transforms(config.data.image_size)
    )
    loader = DataLoader(test_ds, batch_size=config.data.batch_size, shuffle=False, num_workers=0)
    _, y_true, y_pred, probs = evaluate(model, loader, torch.nn.CrossEntropyLoss(), device)

    resultado = {"limiar": detector.threshold, "target_tpr": detector.target_tpr}

    print("=" * 68)
    print("DETECCAO DE ERRO - a confianca avisa quando a predicao esta errada?")
    print("=" * 68)
    erro = deteccao_de_erro(probs, y_true, y_pred, detector)
    resultado["deteccao_de_erro"] = erro
    if "erro" in erro:
        print(f"  {erro['erro']}")
    else:
        print(f"  AUROC                        {erro['auroc']:.4f}")
        print(f"  confianca media nos acertos  {erro['confianca_media_acertos']:.4f}  (n={erro['n_acertos']})")
        print(f"  confianca media nos erros    {erro['confianca_media_erros']:.4f}  (n={erro['n_erros']})")
        print(f"  acertos acima do limiar      {erro['acertos_acima_do_limiar']:.1%}")
        print(f"  erros pegos pelo limiar      {erro['erros_abaixo_do_limiar']:.1%}")
        print(
            "\n  AUROC acima de ~0,80 significa que vale exibir o aviso de baixa\n"
            "  confianca no dashboard: ele acerta mais do que erra."
        )

    if args.ood_dir:
        print("\n" + "=" * 68)
        print("DETECCAO DE DOMINIO - contra imagens de outro dominio")
        print("=" * 68)
        try:
            ood_probs = probs_de_pasta(model, Path(args.ood_dir), device, config.data.image_size)
        except ValueError as exc:
            print(f"  [ERRO] {exc}")
            return 1
        medida = evaluate_detector(detector, probs, ood_probs)
        resultado["deteccao_de_dominio"] = medida
        print(f"  AUROC                     {medida['auroc']:.4f}")
        print(f"  dentro do dominio aceito  {medida['tpr_in_domain']:.1%}  (n={medida['n_in_domain']})")
        print(f"  fora do dominio ACEITO    {medida['fpr_out_domain']:.1%}  (n={medida['n_out_domain']})  <- quanto menor, melhor")
    else:
        print("\n" + "=" * 68)
        print("DETECCAO DE DOMINIO - nao medida")
        print("=" * 68)
        print(
            "  Falta um conjunto de outro dominio. NAO use raw/other/non_astronomical:\n"
            "  aquelas fotos foram treinadas como classe `other`, o modelo as conhece,\n"
            "  e a medicao sairia alta e sem significado.\n\n"
            "  Junte algumas dezenas de astrofotos processadas (Hubble, JWST,\n"
            "  astrofotografia amadora) numa pasta e rode:\n"
            "    python scripts/evaluate_ood.py --ood-dir <pasta>"
        )

    destino = paths.runs / "_ood" / "metrics.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultados em {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
