"""Compara os tres detectores de fora-de-dominio no mesmo conjunto.

    python scripts/compare_ood.py

Fecha o item do roteiro "Metodos melhores de OOD: energy-based, Mahalanobis".
Nenhum dos tres exige retreino - todos leem o checkpoint que ja existe. E
por isso que a comparacao e honesta: mesmo modelo, mesmas imagens, mesma
particao. So muda a funcao que transforma a saida da rede num numero.

    MSP          max softmax.            Hendrycks & Gimpel, ICLR 2017.
    ENERGIA      -logsumexp(logits).     Liu et al., NeurIPS 2020.
    MAHALANOBIS  distancia nas features. Lee et al., NeurIPS 2018.

AS TRES METRICAS, E POR QUE A TERCEIRA E A QUE IMPORTA AQUI

    AUROC       probabilidade de uma imagem de dentro receber pontuacao
                maior que uma de fora. Independente de limiar. E o numero
                que os artigos reportam.

    AUPR        area sob precisao-recall, tratando FORA como a classe
                positiva. Mais informativo que a AUROC quando os conjuntos
                tem tamanhos muito diferentes - que e o nosso caso.

    FPR@95TPR   fixando o limiar para aceitar 95% das imagens legitimas,
                que fracao das imagens de fora passa? E a metrica que
                descreve o dashboard de verdade: o usuario nao ve a AUROC,
                ve um aviso que aparece ou nao. MENOR e melhor.

CUIDADO HERDADO DE `evaluate_ood.py`, QUE VALE REPETIR

`raw/other/non_astronomical` NAO serve como conjunto de fora: aquelas fotos
foram TREINADAS como parte da classe `other`. O modelo as conhece. Usa-se
`C:/astro-data/ood/*`, que sao levantamentos que o modelo nunca viu.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/level1_object.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--ood-root",
        default="C:/astro-data/ood",
        help="pasta com uma subpasta por conjunto de fora-de-dominio",
    )
    parser.add_argument(
        "--shrinkage",
        type=float,
        default=0.01,
        help="encolhimento inicial da covariancia do Mahalanobis",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    # Imports pesados aqui dentro: no Windows o DataLoader usa `spawn`, que
    # reexecuta este arquivo em cada worker. Importar sklearn/torch no topo
    # do modulo multiplicaria o consumo de memoria por worker.
    import numpy as np
    import torch
    from PIL import Image
    from sklearn.metrics import average_precision_score, roc_auc_score
    from torch.utils.data import DataLoader

    from astro_classifier.config import ExperimentConfig, resolve_device
    from astro_classifier.data.datasets import AstroImageDataset
    from astro_classifier.data.transforms import eval_transforms
    from astro_classifier.models.classifier import AstroClassifier
    from astro_classifier.ood.energy import confidence_from_energy
    from astro_classifier.ood.mahalanobis import MahalanobisDetector
    from astro_classifier.paths import get_paths

    config = ExperimentConfig.from_yaml(args.config)
    device = resolve_device(args.device)
    paths = get_paths()

    checkpoint = paths.checkpoints / "object_best.pt"
    if not checkpoint.exists():
        print(f"Checkpoint nao encontrado: {checkpoint}")
        return 1

    model, _ = AstroClassifier.load(checkpoint, device=device)
    transformacao = eval_transforms(config.data.image_size)

    @torch.inference_mode()
    def logits_e_features(loader):
        """Uma passada so, devolvendo as duas coisas.

        Os tres metodos precisam de saidas diferentes da MESMA rede: o MSP e
        a energia querem logits, o Mahalanobis quer as features. Rodar duas
        vezes desperdicaria metade do tempo e abriria a porta para compara-los
        em imagens em ordens diferentes.
        """
        model.eval()
        ls, fs, ys = [], [], []
        for lote in loader:
            x = lote[0].to(device)
            f = model.backbone(x).flatten(1)
            ls.append(model.head(f).cpu().numpy())
            fs.append(f.cpu().numpy())
            if len(lote) > 1:
                y = lote[1].cpu()
                ys.append((y.argmax(dim=1) if y.dim() > 1 else y).numpy())
        return (
            np.concatenate(ls),
            np.concatenate(fs),
            np.concatenate(ys) if ys else None,
        )

    class PastaSimples(torch.utils.data.Dataset):
        """Imagens soltas numa pasta, sem rotulo. Os conjuntos de fora nao tem."""

        def __init__(self, pasta: Path):
            self.arquivos = sorted(
                p for p in pasta.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
            )
            if not self.arquivos:
                raise ValueError(f"nenhuma imagem em {pasta}")

        def __len__(self):
            return len(self.arquivos)

        def __getitem__(self, i):
            return (transformacao(Image.open(self.arquivos[i]).convert("RGB")),)

    def carregar(ds):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # --- dentro do dominio ---
    print("extraindo do conjunto de TREINO (para ajustar o Mahalanobis)...")
    treino_ds = AstroImageDataset(
        paths.splits / "object_train.csv", paths.raw, config.level, transformacao
    )
    _, f_treino, y_treino = logits_e_features(carregar(treino_ds))

    print(f"ajustando gaussianas em {f_treino.shape[0]} imagens x {f_treino.shape[1]} features")
    maha = MahalanobisDetector.fit(
        f_treino, y_treino, classes=config.level.classes, shrinkage=args.shrinkage
    )
    print(f"  encolhimento efetivo: {maha.shrinkage:g}")
    if maha.shrinkage > args.shrinkage:
        print(
            f"  [AVISO] o encolhimento pedido ({args.shrinkage:g}) nao bastou para a\n"
            f"  covariancia ficar positiva-definida. Com {f_treino.shape[1]} features e\n"
            f"  {f_treino.shape[0]} imagens isso e esperado, mas o numero vai no relatorio."
        )
    # Calibracao do percentil na VALIDACAO, nao no treino nem no teste.
    #
    # No treino as gaussianas foram ajustadas, entao as distancias sairiam
    # otimistas e o percentil deslocado. No teste seria pior ainda: o limiar
    # passaria a depender do conjunto onde a gente mede, e o numero final
    # ficaria contaminado. A validacao existe para exatamente esta decisao.
    print("\nextraindo da VALIDACAO (para calibrar o percentil)...")
    val_ds = AstroImageDataset(
        paths.splits / "object_val.csv", paths.raw, config.level, transformacao
    )
    _, f_val, _ = logits_e_features(carregar(val_ds))
    maha.calibrate(f_val)
    print(f"  calibrado em {len(f_val)} imagens de validacao")
    maha.save(paths.checkpoints / "ood_mahalanobis.pt")

    print("\nextraindo do conjunto de TESTE (dentro do dominio)...")
    teste_ds = AstroImageDataset(
        paths.splits / "object_test.csv", paths.raw, config.level, transformacao
    )
    l_teste, f_teste, _ = logits_e_features(carregar(teste_ds))

    def pontuacoes(logits, features) -> dict:
        """As tres pontuacoes, todas na convencao MAIOR = mais dentro."""
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        return {
            "MSP": probs.max(axis=1),
            "energia": confidence_from_energy(logits),
            "Mahalanobis": maha.score(features),
        }

    dentro = pontuacoes(l_teste, f_teste)
    metodos = list(dentro.keys())

    # --- conjuntos de fora ---
    raiz = Path(args.ood_root)
    conjuntos = sorted(p for p in raiz.iterdir() if p.is_dir()) if raiz.exists() else []
    if not conjuntos:
        print(f"\nNenhum conjunto de fora-de-dominio em {raiz}")
        print("Rode primeiro: python scripts/build_ood_set.py")
        return 1

    resultado: dict = {
        "n_dentro": int(len(l_teste)),
        "mahalanobis_shrinkage": maha.shrinkage,
        "mahalanobis_dim": maha.dim,
        "n_ajuste": maha.n_fit,
        "conjuntos": {},
    }

    for pasta in conjuntos:
        print(f"\nextraindo de {pasta.name}...")
        try:
            l_fora, f_fora, _ = logits_e_features(carregar(PastaSimples(pasta)))
        except ValueError as exc:
            print(f"  [pulando] {exc}")
            continue
        fora = pontuacoes(l_fora, f_fora)

        linhas = {}
        for m in metodos:
            s_in, s_out = dentro[m], fora[m]
            y = np.concatenate([np.ones_like(s_in), np.zeros_like(s_out)])
            s = np.concatenate([s_in, s_out])
            # Limiar que aceita 95% das legitimas; quantas de fora passam?
            limiar = float(np.quantile(s_in, 0.05))
            linhas[m] = {
                "auroc": float(roc_auc_score(y, s)),
                # FORA como classe positiva -> pontuacao invertida
                "aupr_fora": float(average_precision_score(1 - y, -s)),
                "fpr_em_95tpr": float((s_out >= limiar).mean()),
                "limiar_95tpr": limiar,
            }
        resultado["conjuntos"][pasta.name] = {"n": int(len(l_fora)), "metodos": linhas}

        print(f"  {'metodo':<14}{'AUROC':>8}{'AUPR':>8}{'FPR@95':>9}")
        for m in metodos:
            d = linhas[m]
            print(
                f"  {m:<14}{d['auroc']:>8.4f}{d['aupr_fora']:>8.4f}"
                f"{d['fpr_em_95tpr']:>8.1%}"
            )

    # --- media entre conjuntos, que e o que resume a comparacao ---
    print("\n" + "=" * 64)
    print("MEDIA ENTRE OS CONJUNTOS")
    print("=" * 64)
    print(f"  {'metodo':<14}{'AUROC':>8}{'AUPR':>8}{'FPR@95':>9}")
    medias = {}
    for m in metodos:
        vals = [c["metodos"][m] for c in resultado["conjuntos"].values()]
        medias[m] = {
            k: float(np.mean([v[k] for v in vals]))
            for k in ("auroc", "aupr_fora", "fpr_em_95tpr")
        }
        d = medias[m]
        print(
            f"  {m:<14}{d['auroc']:>8.4f}{d['aupr_fora']:>8.4f}{d['fpr_em_95tpr']:>8.1%}"
        )
    resultado["media"] = medias

    melhor = max(medias, key=lambda m: medias[m]["auroc"])
    print(f"\n  melhor AUROC media: {melhor}")
    if melhor != "MSP":
        ganho = medias[melhor]["auroc"] - medias["MSP"]["auroc"]
        reducao = medias["MSP"]["fpr_em_95tpr"] - medias[melhor]["fpr_em_95tpr"]
        print(
            f"  contra o MSP: {ganho:+.4f} de AUROC e {reducao:+.1%} de imagens\n"
            f"  de fora barradas no mesmo ponto de operacao."
        )
    else:
        print(
            "  o baseline ganhou. E um resultado legitimo e vale reportar: os\n"
            "  metodos mais novos nao ganham em todo dominio, e um artigo que\n"
            "  mostra onde eles NAO ganham tem valor."
        )

    destino = paths.runs / "_ood" / "comparacao.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nResultados em {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
