"""Avalia um checkpoint no conjunto de TESTE e calibra o detector de dominio.

    python scripts/evaluate.py --config configs/level1_object.yaml

O conjunto de teste so deve ser tocado no fim. Se voce ajustar
hiperparametros olhando o teste, ele deixa de medir generalizacao e vira
mais um conjunto de validacao - e o numero que voce publicar estara inflado.

Para o nivel 1, o script tambem calibra o limiar de deteccao de dominio
(OOD) e o grava em checkpoints/ood_threshold.json, de onde a API o le.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from astro_classifier.config import ExperimentConfig, resolve_device
from astro_classifier.data.datasets import AstroImageDataset
from astro_classifier.data.transforms import eval_transforms
from astro_classifier.evaluation.confusion import plot_confusion_matrix
from astro_classifier.evaluation.gradcam import save_gradcam_examples
from astro_classifier.evaluation.metrics import (
    classification_metrics,
    print_metrics,
    save_markdown,
    save_metrics,
)
from astro_classifier.models.classifier import AstroClassifier
from astro_classifier.ood.msp import calibrate_threshold
from astro_classifier.paths import get_paths
from astro_classifier.taxonomy import Level
from astro_classifier.training.loops import evaluate


def aviso_backbone_divergente(config, ckpt: dict, nome_arquivo: str) -> str | None:
    """Avisa quando a config declara um backbone e o checkpoint traz outro.

    ISTO JA CUSTOU UMA CONCLUSAO ERRADA AO PROJETO.

    Sem `--checkpoint-name`, este script carrega `<nivel>_best.pt` - o de
    PRODUCAO - qualquer que seja o backbone declarado na config. Passar
    `level3_nebula_efficientnet_b0.yaml` avalia entao o checkpoint em producao
    (que pode ser resnet50) e grava o relatorio sob o nome da config do
    efficientnet.

    Foi assim que a comparacao de backbones saiu com tres numeros IDENTICOS e a
    conclusao "o backbone nao importa" - quando o ganho real era de 4,8 pontos de
    macro-F1 no nivel 3.

    Nada falha nesse caminho: os pesos carregam, as metricas saem, o arquivo tem
    o nome esperado. So o rotulo esta errado. Por isso o aviso.

    Devolve o texto do aviso, ou None quando esta tudo coerente.
    """
    do_checkpoint = ckpt.get("backbone")
    da_config = getattr(getattr(config, "model", None), "backbone", None)
    if not do_checkpoint or not da_config or do_checkpoint == da_config:
        return None

    return (
        f"\n[ATENCAO] a config '{config.name}' declara backbone '{da_config}', "
        f"mas {nome_arquivo} foi treinado com '{do_checkpoint}'.\n"
        f"          O que sera avaliado e o '{do_checkpoint}' do checkpoint, e o "
        f"relatorio sairia\n"
        f"          rotulado como '{config.name}'. Se a intencao era avaliar o "
        f"'{da_config}',\n"
        f"          passe --checkpoint-name com o arquivo certo "
        f"(ex.: {config.name}.pt).\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--images-root", default=None)
    parser.add_argument(
        "--tta",
        action="store_true",
        help="test-time augmentation: media sobre as 8 simetrias da imagem. "
        "Custa 8x mais inferencias e costuma render 1-2 pontos.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=None,
        help="nome do arquivo de checkpoint (padrao: <nivel>_best.pt). Use para "
        "avaliar uma variante gravada com --name-suffix.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="nome da pasta de resultados (padrao: o nome do experimento)",
    )
    parser.add_argument("--no-gradcam", action="store_true", help="pula as figuras de Grad-CAM")
    parser.add_argument(
        "--gradcam-per-class", type=int, default=4, help="exemplos por classe no Grad-CAM"
    )
    parser.add_argument(
        "--target-tpr",
        type=float,
        default=0.95,
        help="fracao de imagens legitimas que o detector de dominio deve aceitar",
    )
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    device = resolve_device(args.device)
    paths = get_paths()
    images_root = Path(args.images_root) if args.images_root else paths.raw

    checkpoint = paths.checkpoints / (args.checkpoint_name or f"{config.level.value}_best.pt")
    if not checkpoint.exists():
        print(f"Checkpoint nao encontrado: {checkpoint}")
        print(f"Treine primeiro: python scripts/train.py --config {args.config}")
        return 1

    model, ckpt = AstroClassifier.load(checkpoint, device=device)
    aviso = aviso_backbone_divergente(config, ckpt, checkpoint.name)
    if aviso:
        print(aviso)

    test_ds = AstroImageDataset(
        paths.splits / f"{config.level.value}_test.csv",
        images_root,
        config.level,
        eval_transforms(config.data.image_size),
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.data.batch_size, shuffle=False, num_workers=config.data.num_workers
    )

    sufixo_tta = " + TTA" if args.tta else ""
    print(f"Avaliando {config.name} em {len(test_ds)} imagens de TESTE ({device}{sufixo_tta})")
    if args.tta:
        from astro_classifier.evaluation.tta_eval import evaluate_with_tta

        y_true, y_pred, y_proba = evaluate_with_tta(model, test_ds, device)
    else:
        _, y_true, y_pred, y_proba = evaluate(
            model, test_loader, torch.nn.CrossEntropyLoss(), device
        )
    metrics = classification_metrics(y_true, y_pred, config.level.classes, y_proba)
    print_metrics(metrics, titulo=f"{config.name} - TESTE{sufixo_tta}")

    run_dir = paths.runs / (args.run_name or config.name)
    save_metrics(metrics, run_dir / "metrics_test.json")
    save_markdown(metrics, run_dir / "metrics_test.md", f"{config.name} — teste")
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        config.level.classes,
        run_dir / "confusion_test.png",
        title=f"{config.name} - teste",
    )
    # Grad-CAM de acertos e erros. Nao e enfeite: e como se descobre que o
    # modelo acertou pelo motivo errado - olhando o fundo em vez do objeto.
    # Importa especialmente nas classes de F1 baixo.
    if not args.no_gradcam:
        try:
            geradas = save_gradcam_examples(
                model=model,
                dataset=test_ds,
                y_true=y_true,
                y_pred=y_pred,
                classes=config.level.classes,
                out_dir=run_dir / "gradcam",
                device=device,
                per_class=args.gradcam_per_class,
            )
            print(f"Grad-CAM: {geradas} figuras em {run_dir / 'gradcam'}")
        except Exception as exc:  # noqa: BLE001 - visualizacao nunca derruba a avaliacao
            print(f"[aviso] Grad-CAM falhou: {exc}")

    print(f"\nResultados em {run_dir}")

    # O limiar de OOD e calibrado na VALIDACAO, nunca no teste - calibrar no
    # teste seria vazamento de informacao.
    if config.level is Level.OBJECT:
        val_ds = AstroImageDataset(
            paths.splits / "object_val.csv",
            images_root,
            config.level,
            eval_transforms(config.data.image_size),
        )
        val_loader = DataLoader(val_ds, batch_size=config.data.batch_size, shuffle=False)
        _, _, _, val_probs = evaluate(model, val_loader, torch.nn.CrossEntropyLoss(), device)

        detector = calibrate_threshold(val_probs, target_tpr=args.target_tpr)
        detector.save(paths.checkpoints / "ood_threshold.json")
        print(
            f"\nDetector de dominio calibrado: limiar={detector.threshold:.4f} "
            f"(aceita {args.target_tpr:.0%} das imagens de validacao)"
        )
        print(
            "Proximo passo: monte um conjunto realmente fora do dominio (fotos do\n"
            "Hubble, imagens do dia a dia) e meca com ood.evaluate_detector(). Sem\n"
            "essa medicao, o limiar e so um chute plausivel."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
