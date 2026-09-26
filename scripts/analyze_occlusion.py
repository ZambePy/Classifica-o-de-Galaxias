"""Ablacao por oclusao: o modelo precisa do OBJETO, ou o fundo ja basta?

    python scripts/analyze_occlusion.py --config configs/level3_nebula.yaml

O PROBLEMA

As evidencias anteriores de atalho - Grad-CAM, latitude galactica, grupo de
controle - sao indiretas. Todas sugerem que o modelo se apoia no contexto,
nenhuma isola a contribuicao dele. Este script isola.

O DESENHO

A mesma imagem de teste, em tres condicoes:

    COMPLETA   como o modelo sempre viu
    SO OBJETO  o centro preservado, o resto apagado
    SO FUNDO   o centro apagado, o resto preservado

O objeto esta centralizado por construcao: os recortes vem de hips2fits
centrados na coordenada do catalogo, e o campo de visao e proporcional ao
diametro (ver data/catalogs.py). Entao o centro da imagem E o objeto.

COMO LER O RESULTADO

    SO OBJETO alto, SO FUNDO baixo    o modelo usa o objeto. E o desejado.

    SO FUNDO alto                     o contexto sozinho ja decide. O modelo
                                      aprendeu geografia, nao morfologia -
                                      e a acuracia da imagem completa esta
                                      inflada por isso.

    os dois medianos                  o modelo usa as duas coisas; a
                                      diferenca para a imagem completa mede
                                      o quanto cada parte contribui.

POR QUE DESFOQUE E NAO RECORTE

A primeira versao deste script apagava a regiao com um circulo de borda
dura. O resultado foi inutilizavel, e vale registrar por que:

    completa    0,797
    so objeto   0,259   <- ABAIXO de chutar a majoritaria (0,362)
    so fundo    0,377   <- no nivel do chute

As DUAS condicoes desabaram. Isso nao diz que o modelo usa o objeto nem que
usa o fundo: diz que a borda circular abrupta e, ela mesma, uma imagem que a
rede nunca viu. O teste estava medindo o artefato da mascara.

A versao atual DESFOCA a regiao em vez de apaga-la. O desfoque preserva
brilho medio, cor e estatistica local - destroi a estrutura fina (bracos
espirais, aneis, filamentos) sem introduzir uma fronteira artificial. A
imagem continua parecendo uma imagem de ceu.

Referencia: Zeiler & Fergus (2014), Visualizing and Understanding
Convolutional Networks - a oclusao sistematica como diagnostico.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def peso_suave(tamanho: int, raio_fracao: float, dentro: bool, suavidade: float = 0.15):
    """Peso continuo (H, W) em [0,1], com transicao suave na borda.

    `dentro=True` da peso 1 ao centro e 0 as bordas. A transicao em rampa,
    e nao em degrau, evita criar uma fronteira dura - que foi o que
    inutilizou a primeira versao deste teste.
    """
    import torch

    eixo = np.linspace(-1.0, 1.0, tamanho)
    xx, yy = np.meshgrid(eixo, eixo)
    distancia = np.sqrt(xx**2 + yy**2)

    # rampa linear de raio-suavidade ate raio+suavidade
    peso = np.clip((raio_fracao + suavidade - distancia) / (2 * suavidade), 0.0, 1.0)
    if not dentro:
        peso = 1.0 - peso
    return torch.from_numpy(peso.astype(np.float32))


def desfocar(imagens, sigma: float = 6.0):
    """Desfoque gaussiano forte: apaga a estrutura, preserva a estatistica."""
    from torchvision.transforms import functional as TF

    tamanho = int(sigma * 4) | 1  # kernel impar
    return TF.gaussian_blur(imagens, kernel_size=[tamanho, tamanho], sigma=[sigma, sigma])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/level3_nebula.yaml")
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--sigma",
        type=float,
        default=6.0,
        help="intensidade do desfoque aplicado a regiao suprimida",
    )
    parser.add_argument(
        "--raio",
        type=float,
        default=0.45,
        help="fracao do quadro ocupada pelo circulo central (0,45 cobre o "
        "objeto na maioria dos recortes, dado o fov proporcional ao diametro)",
    )
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    from astro_classifier.config import ExperimentConfig, resolve_device
    from astro_classifier.data.datasets import AstroImageDataset
    from astro_classifier.data.transforms import eval_transforms
    from astro_classifier.evaluation.metrics import classification_metrics
    from astro_classifier.models.classifier import AstroClassifier
    from astro_classifier.paths import get_paths

    config = ExperimentConfig.from_yaml(args.config)
    device = resolve_device(args.device)
    paths = get_paths()

    checkpoint = paths.checkpoints / (
        args.checkpoint_name or f"{config.level.value}_best.pt"
    )
    if not checkpoint.exists():
        print(f"Checkpoint nao encontrado: {checkpoint}")
        return 1

    model, _ = AstroClassifier.load(checkpoint, device=device)
    model.eval()

    dataset = AstroImageDataset(
        paths.splits / f"{config.level.value}_test.csv",
        paths.raw,
        config.level,
        eval_transforms(config.data.image_size),
    )
    loader = DataLoader(dataset, batch_size=config.data.batch_size, shuffle=False, num_workers=0)

    tamanho = config.data.image_size
    # peso 1 = preserva o original ali; peso 0 = usa a versao desfocada
    condicoes = {
        "completa": None,
        "so_objeto": peso_suave(tamanho, args.raio, dentro=True),
        "so_fundo": peso_suave(tamanho, args.raio, dentro=False),
    }

    print(f"modelo: {checkpoint.name} | nivel {config.level.value} | {len(dataset)} imagens de teste")
    print(f"raio do circulo central: {args.raio:.2f} do quadro\n")

    resultados: dict[str, dict] = {}
    for nome, mascara in condicoes.items():
        verdadeiros, preditos, probabilidades = [], [], []

        for imagens, alvos in loader:
            imagens = imagens.to(device)
            if mascara is not None:
                # Combina a imagem nitida com a desfocada segundo o peso:
                # onde peso=1 fica nitido, onde peso=0 fica desfocado.
                peso = mascara.to(device).view(1, 1, tamanho, tamanho)
                imagens = imagens * peso + desfocar(imagens) * (1.0 - peso)

            with torch.inference_mode():
                probs = torch.softmax(model(imagens), dim=1)

            probabilidades.append(probs.cpu().numpy())
            preditos.append(probs.argmax(dim=1).cpu().numpy())
            verdadeiros.append(alvos.numpy())

        y_true = np.concatenate(verdadeiros)
        y_pred = np.concatenate(preditos)
        proba = np.concatenate(probabilidades)
        m = classification_metrics(y_true, y_pred, config.level.classes, proba)
        resultados[nome] = m

        print(f"{nome:<12} acuracia {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}  "
              f"MCC {m['matthews_corrcoef']:.4f}")

    # --- leitura ---
    completa = resultados["completa"]["accuracy"]
    objeto = resultados["so_objeto"]["accuracy"]
    fundo = resultados["so_fundo"]["accuracy"]
    chute = max(
        v["support"] for v in resultados["completa"]["per_class"].values()
    ) / resultados["completa"]["n_amostras"]

    print(f"\n{'=' * 70}")
    print(f"chutar a classe majoritaria daria {chute:.4f}")
    print(f"{'=' * 70}")
    print(f"  so o objeto nitido preserva  {objeto / completa:.1%} da acuracia original")
    print(f"  so o fundo nitido preserva   {fundo / completa:.1%} da acuracia original")

    acima_do_chute = (fundo - chute) / (completa - chute) if completa > chute else 0.0
    print(f"\n  Descontando o chute, o FUNDO sozinho entrega {acima_do_chute:.1%}")
    print("  do que a imagem completa entrega.")

    if acima_do_chute > 0.5:
        print(
            "\n  >>> Mais da metade do desempenho vem do CONTEXTO, nao do objeto.\n"
            "      A acuracia da imagem completa esta inflada por essa via."
        )
    elif acima_do_chute > 0.25:
        print(
            "\n  >>> O contexto contribui de forma relevante, mas o objeto ainda\n"
            "      carrega a maior parte da decisao."
        )
    else:
        print(
            "\n  >>> O modelo depende do objeto. O contexto sozinho pouco explica -\n"
            "      e a evidencia mais forte possivel contra o atalho."
        )

    # Por classe: o atalho nao atinge todas igualmente.
    print(f"\n{'classe':<22}{'completa':>10}{'so objeto':>11}{'so fundo':>10}")
    print("-" * 53)
    for classe in config.level.classes:
        linha = f"{classe:<22}"
        for nome in ("completa", "so_objeto", "so_fundo"):
            linha += f"{resultados[nome]['per_class'][classe]['f1']:>10.3f} "
        print(linha)

    destino = paths.runs / "_occlusion" / f"{config.level.value}.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(
        json.dumps(
            {
                "raio": args.raio,
                "checkpoint": checkpoint.name,
                "chute_majoritaria": chute,
                "fundo_sobre_completa_descontado_chute": acima_do_chute,
                "condicoes": {k: v for k, v in resultados.items()},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nResultados em {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
