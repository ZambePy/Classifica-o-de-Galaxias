"""Classifica imagens pela linha de comando, sem precisar subir a API.

    # uma imagem
    python scripts/predict.py foto.jpg

    # uma pasta inteira
    python scripts/predict.py C:/minhas_fotos

    # com mapa de calor mostrando onde o modelo olhou
    python scripts/predict.py foto.jpg --gradcam

    # saida JSON, para encadear com outros comandos
    python scripts/predict.py foto.jpg --json

Usa a mesma cascata da API (`inference/pipeline.py`), entao o resultado e
identico ao que o dashboard receberia. A diferenca e so o transporte.

O AVISO DE DOMINIO E A PARTE MAIS IMPORTANTE DA SAIDA

Os modelos foram treinados em recortes de levantamentos astronomicos - DSS2
e SDSS. Uma astrofoto processada do Hubble e outro dominio: o modelo vai
responder alguma coisa, e pode responder com confianca. Quando a linha
`FORA DO DOMINIO` aparecer, o resultado acima dela nao vale.

Medido: contra astrofotografia amadora processada, o detector acerta 2 de
cada 3 casos (AUROC 0,887).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

VERDE = "\033[32m"
AMARELO = "\033[33m"
VERMELHO = "\033[31m"
CINZA = "\033[90m"
NEGRITO = "\033[1m"
FIM = "\033[0m"


def _preparar_saida() -> bool:
    """Garante que o console aguenta os caracteres de bloco. Devolve se aguenta.

    O CONSOLE DO WINDOWS QUEBRAVA ESTE SCRIPT.

    O terminal padrao do Windows usa cp1252, que nao tem '►' nem os blocos
    '█'/'░'. Imprimir a tabela de probabilidades levantava

        UnicodeEncodeError: 'charmap' codec can't encode character '\\u25ba'

    no meio da saida - depois de ja ter impresso o titulo, o que e pior: a
    pessoa via o resultado comecar e o comando morrer. E este e o caminho que o
    README documenta para classificar uma imagem.

    A correcao tem duas camadas. Primeiro tenta trocar a codificacao do stdout
    para UTF-8, que resolve em consoles modernos. Se nao der, avisa o chamador
    para usar o desenho em ASCII - degradar o visual e melhor que falhar.
    """
    fluxo = sys.stdout
    codificacao = (getattr(fluxo, "encoding", "") or "").lower()
    if "utf" in codificacao:
        return True

    reconfigurar = getattr(fluxo, "reconfigure", None)
    if reconfigurar is not None:
        try:
            reconfigurar(encoding="utf-8")
            return True
        except (ValueError, OSError):
            pass
    return False


# Definido em main(): False forca o desenho em ASCII.
UNICODE_OK = True


def barra(fracao: float, largura: int = 24) -> str:
    cheio = round(fracao * largura)
    if UNICODE_OK:
        return "█" * cheio + "░" * (largura - cheio)
    return "#" * cheio + "." * (largura - cheio)


def marcador(selecionado: bool) -> str:
    if not selecionado:
        return " "
    return "►" if UNICODE_OK else ">"


def cor_da_confianca(valor: float) -> str:
    if valor >= 0.85:
        return VERDE
    if valor >= 0.60:
        return AMARELO
    return VERMELHO


def imprimir(caminho: Path, resposta, colorido: bool = True) -> None:
    c = (lambda s, cor: f"{cor}{s}{FIM}") if colorido else (lambda s, _: s)

    print(f"\n{c(caminho.name, NEGRITO)}")

    fora = resposta.domain.out_of_domain
    if fora:
        aviso = "⚠" if UNICODE_OK else "!"
        print(c(f"  {aviso}  FORA DO DOMINIO DE TREINO - resultado pouco confiavel", AMARELO))
        print(c(f"     confianca {resposta.domain.score:.3f} < limiar {resposta.domain.threshold:.3f}", CINZA))

    rotulo = resposta.summary_pt
    conf = resposta.confidence
    print(f"  {c(rotulo, NEGRITO if not fora else CINZA)}  "
          f"{c(f'{conf:.1%}', cor_da_confianca(conf) if not fora else CINZA)}")

    for nivel in resposta.levels:
        print(f"\n  {c(nivel.level, CINZA)}  (modelo {c(nivel.model_version, CINZA)})")
        for score in nivel.scores:
            marca = marcador(score.label == nivel.predicted)
            print(
                f"    {marca} {score.label_pt:<26} {barra(score.probability)} "
                f"{score.probability:>6.1%}"
            )

    print(c(f"\n  inferencia {resposta.inference_ms:.1f} ms", CINZA))


def salvar_gradcam(caminho: Path, pipeline, destino: Path) -> Path | None:
    """Gera o mapa de calor do nivel 1 sobreposto a imagem."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from astro_classifier.evaluation.gradcam import GradCAM, denormalize, overlay_heatmap
    from astro_classifier.taxonomy import Level

    modelo = pipeline._get_model(Level.OBJECT)
    tensor = pipeline.transform(Image.open(caminho).convert("RGB")).unsqueeze(0)

    with GradCAM(modelo) as cam:
        mapa = cam(tensor.to(pipeline.device))

    original = denormalize(tensor[0])
    fig, (a, b) = plt.subplots(1, 2, figsize=(8, 4.2))
    a.imshow(original)
    a.set_title("imagem", fontsize=10)
    b.imshow(overlay_heatmap(original, mapa))
    b.set_title("onde o modelo olhou", fontsize=10)
    for eixo in (a, b):
        eixo.axis("off")
    fig.suptitle(caminho.name, fontsize=11)
    fig.tight_layout()

    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=130)
    plt.close(fig)
    return destino


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("entrada", help="imagem ou pasta de imagens")
    parser.add_argument("--json", action="store_true", help="saida JSON em vez de texto")
    parser.add_argument("--gradcam", action="store_true", help="gera mapa de calor por imagem")
    parser.add_argument("--out", default=None, help="pasta para os Grad-CAM (padrao: runs/predict)")
    parser.add_argument("--device", default=None, help="cuda | cpu | auto")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args()

    # Antes de qualquer print: o console do Windows usa cp1252 e nao aguenta
    # os blocos nem as setas. Ver `_preparar_saida`.
    global UNICODE_OK
    UNICODE_OK = _preparar_saida()

    entrada = Path(args.entrada)
    if not entrada.exists():
        print(f"nao encontrado: {entrada}", file=sys.stderr)
        return 1

    imagens = (
        sorted(p for p in entrada.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if entrada.is_dir()
        else [entrada]
    )
    if not imagens:
        print(f"nenhuma imagem em {entrada}", file=sys.stderr)
        return 1

    from PIL import Image, UnidentifiedImageError

    from astro_classifier.inference.pipeline import HierarchicalPipeline, ModelNotAvailable
    from astro_classifier.paths import get_paths

    try:
        pipeline = HierarchicalPipeline(device=args.device)
        disponiveis = pipeline.loaded_levels()
    except Exception as exc:  # noqa: BLE001
        print(f"falha ao preparar a cascata: {exc}", file=sys.stderr)
        return 1

    if "object" not in disponiveis:
        print(
            "Nenhum modelo treinado encontrado.\n"
            "Treine antes:  python scripts/train.py --config configs/level1_object.yaml",
            file=sys.stderr,
        )
        return 1

    if not args.json:
        cabecalho = f"modelos: {', '.join(disponiveis)} | {pipeline.device}"
        print(cabecalho if args.no_color else f"{CINZA}{cabecalho}{FIM}", file=sys.stderr)

    destino_cam = Path(args.out) if args.out else get_paths().runs / "predict"
    resultados = []
    falhas = 0
    fora_do_dominio = 0

    for caminho in imagens:
        try:
            with Image.open(caminho) as img:
                resposta = pipeline.predict_image(img.convert("RGB"))
        except (UnidentifiedImageError, OSError):
            print(f"  [ilegivel] {caminho.name}", file=sys.stderr)
            falhas += 1
            continue
        except ModelNotAvailable as exc:
            print(f"  [modelo ausente] {exc}", file=sys.stderr)
            falhas += 1
            continue

        fora_do_dominio += int(resposta.domain.out_of_domain)

        if args.json:
            resultados.append({"arquivo": str(caminho), **resposta.model_dump()})
        else:
            imprimir(caminho, resposta, colorido=not args.no_color)

        if args.gradcam:
            try:
                figura = salvar_gradcam(caminho, pipeline, destino_cam / f"{caminho.stem}_gradcam.png")
                if not args.json:
                    print(f"  {CINZA}grad-cam: {figura}{FIM}")
            except Exception as exc:  # noqa: BLE001 - visualizacao nao derruba a predicao
                print(f"  [grad-cam falhou] {exc}", file=sys.stderr)

    if args.json:
        print(json.dumps(resultados, indent=2, ensure_ascii=False))
    elif len(imagens) > 1:
        classificadas = len(imagens) - falhas
        resumo = f"\n{classificadas} imagens classificadas"
        if falhas:
            resumo += f", {falhas} ilegiveis"
        if fora_do_dominio:
            resumo += (
                f"\n{fora_do_dominio} marcadas como FORA DO DOMINIO "
                f"({fora_do_dominio / max(classificadas, 1):.0%}) - nessas, "
                "o rotulo nao e confiavel"
            )
        print(resumo)

    return 0


if __name__ == "__main__":
    sys.exit(main())
