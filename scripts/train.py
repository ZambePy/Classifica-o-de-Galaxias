"""Treina um nivel da cascata.

    python scripts/train.py --config configs/level1_object.yaml
    python scripts/train.py --config configs/level2_galaxy.yaml
    python scripts/train.py --config configs/level3_nebula.yaml

Cada execucao grava em ASTRO_DATA_ROOT/runs/<nome>/:
    config.yaml       a configuracao exata usada (reprodutibilidade)
    history.json      metricas por epoca
    metrics_val.json  metricas finais da melhor epoca
    curves.png        loss e macro-F1 ao longo do treino
    confusion.png     matriz de confusao da validacao

e o melhor checkpoint em ASTRO_DATA_ROOT/checkpoints/<nivel>_best.pt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from astro_classifier.config import ExperimentConfig, resolve_device
from astro_classifier.data.datasets import build_dataloaders
from astro_classifier.data.transforms import eval_transforms, train_transforms
from astro_classifier.evaluation.confusion import plot_confusion_matrix, plot_training_curves
from astro_classifier.evaluation.metrics import classification_metrics, print_metrics, save_metrics
from astro_classifier.models.classifier import build_model
from astro_classifier.paths import get_paths
from astro_classifier.training.loops import evaluate, set_seed, train_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="caminho do YAML de experimento")
    parser.add_argument("--device", default=None, help="cuda | cpu | auto (padrao: do .env)")
    parser.add_argument(
        "--images-root",
        default=None,
        help="raiz das imagens (padrao: ASTRO_DATA_ROOT/raw)",
    )
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    device = resolve_device(args.device)
    paths = get_paths().ensure()
    images_root = Path(args.images_root) if args.images_root else paths.raw

    set_seed(config.seed)

    print(f"experimento : {config.name}")
    print(f"nivel       : {config.level.value}  ->  {config.level.classes}")
    print(f"device      : {device}")
    if device.startswith("cuda"):
        print(f"gpu         : {torch.cuda.get_device_name(0)}")
    print(f"imagens     : {images_root}")
    print(f"splits      : {paths.splits}\n")

    train_loader, val_loader, train_ds = build_dataloaders(
        splits_dir=paths.splits,
        images_root=images_root,
        level=config.level,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        train_tf=train_transforms(config.data.image_size, config.data.augment),
        eval_tf=eval_transforms(config.data.image_size),
    )

    print(f"treino: {len(train_ds)} imagens  {train_ds.class_counts()}")
    class_weights = train_ds.class_weights() if config.data.class_weights == "balanced" else None
    if class_weights is not None:
        print(f"pesos de classe: {dict(zip(config.level.classes, class_weights.tolist(), strict=True))}\n")

    run_dir = paths.runs / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    config.save(run_dir / "config.yaml")

    model = build_model(config)
    checkpoint_path = paths.checkpoints / f"{config.level.value}_best.pt"

    history = train_model(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_path=checkpoint_path,
        class_weights=class_weights,
    )
    history.save(run_dir / "history.json")

    # Avaliacao final com o MELHOR checkpoint, nao com o estado da ultima
    # epoca - que pode ser pior por causa do early stopping.
    print(f"\nRecarregando melhor checkpoint (epoca {history.best_epoch}) para avaliacao final")
    from astro_classifier.models.classifier import AstroClassifier

    best_model, _ = AstroClassifier.load(checkpoint_path, device=device)
    _, y_true, y_pred, _ = evaluate(best_model, val_loader, torch.nn.CrossEntropyLoss(), device)
    metrics = classification_metrics(y_true, y_pred, config.level.classes)

    print_metrics(metrics)
    save_metrics(metrics, run_dir / "metrics_val.json")
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        config.level.classes,
        run_dir / "confusion.png",
        title=f"{config.name} - validacao",
    )
    plot_training_curves(history.to_dict(), run_dir / "curves.png")

    print(f"\nArtefatos em: {run_dir}")
    print(f"Checkpoint  : {checkpoint_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
