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

# ATENCAO AO QUE E IMPORTADO NO TOPO DESTE ARQUIVO.
#
# No Windows o DataLoader cria cada worker com `spawn`, e spawn RE-EXECUTA
# este modulo dentro do processo filho. Tudo que estiver aqui no topo e
# carregado de novo por worker - inclusive o que o worker nunca usa.
#
# Medido nesta maquina: com sklearn e matplotlib no topo, cada um dos 8
# processos ocupava ~750 MB, somando 6 GB dos 16 GB de RAM. A memoria
# comprometida livre caiu para 0,5 GB e o sistema comecou a derrubar
# processos no meio do treino.
#
# Os workers so precisam ler imagem e aplicar transformacoes: torch, PIL,
# pandas, torchvision.transforms. Metricas, graficos e o modelo sao usados
# apenas pelo processo principal - por isso sao importados DENTRO de main().
from astro_classifier.config import ExperimentConfig, resolve_device
from astro_classifier.data.datasets import build_dataloaders
from astro_classifier.data.transforms import eval_transforms, train_transforms
from astro_classifier.paths import get_paths


def main() -> int:
    # Importados aqui, e nao no topo, para nao serem carregados por cada
    # worker do DataLoader. Ver o comentario acima.
    import torch

    from astro_classifier.evaluation.confusion import (
        plot_confusion_matrix,
        plot_training_curves,
    )
    from astro_classifier.evaluation.metrics import (
        classification_metrics,
        print_metrics,
        save_markdown,
        save_metrics,
    )
    from astro_classifier.models.classifier import AstroClassifier, build_model
    from astro_classifier.training.loops import evaluate, set_seed, train_model

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="caminho do YAML de experimento")
    parser.add_argument("--device", default=None, help="cuda | cpu | auto (padrao: do .env)")
    parser.add_argument(
        "--images-root",
        default=None,
        help="raiz das imagens (padrao: ASTRO_DATA_ROOT/raw)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="sobrescreve as epocas do YAML. Util para validar o pipeline "
        "com --epochs 1 antes de deixar um treino longo rodando.",
    )
    parser.add_argument(
        "--name-suffix",
        default="",
        help="sufixo no nome do experimento, para nao sobrescrever um run anterior",
    )
    args = parser.parse_args()

    config = ExperimentConfig.from_yaml(args.config)
    if args.epochs is not None:
        config.optim.epochs = args.epochs
    if args.name_suffix:
        config.name = f"{config.name}{args.name_suffix}"
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
        soft_labels=config.data.soft_labels,
    )

    print(f"treino: {len(train_ds)} imagens  {train_ds.class_counts()}")
    class_weights = (
        None
        if config.data.class_weights == "none"
        else train_ds.class_weights(config.data.class_weights)
    )
    if class_weights is not None:
        print(f"pesos de classe: {dict(zip(config.level.classes, class_weights.tolist(), strict=True))}\n")

    run_dir = paths.runs / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    config.save(run_dir / "config.yaml")

    model = build_model(config)
    # Sem sufixo, o checkpoint usa o nome canonico do nivel - e o que a API
    # procura. Com sufixo, grava separado para nao sobrescrever um treino bom
    # com um teste rapido.
    checkpoint_path = paths.checkpoints / (
        f"{config.name}.pt" if args.name_suffix else f"{config.level.value}_best.pt"
    )

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

    best_model, _ = AstroClassifier.load(checkpoint_path, device=device)
    _, y_true, y_pred, y_proba = evaluate(
        best_model, val_loader, torch.nn.CrossEntropyLoss(), device
    )
    metrics = classification_metrics(y_true, y_pred, config.level.classes, y_proba)

    print_metrics(metrics, titulo=f"{config.name} - VALIDACAO")
    save_metrics(metrics, run_dir / "metrics_val.json")
    save_markdown(metrics, run_dir / "metrics_val.md", f"{config.name} — validação")
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
