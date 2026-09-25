"""Grad-CAM: onde o modelo olhou para decidir.

Implementacao direta (sem dependencia externa), ~80 linhas. O metodo:
capturamos as ativacoes e os gradientes da ultima camada convolucional,
pesamos cada canal pelo gradiente medio e somamos - o resultado e um mapa de
calor do que sustentou a predicao.

Por que isso importa AQUI e nao e so enfeite de dashboard: e como voce
descobre que o modelo aprendeu a coisa errada. Se o mapa de calor de uma
"galaxia espiral" acende no ruido do fundo e nao nos bracos espirais, sua
acuracia alta e um artefato do dataset. Isso ja aconteceu em trabalhos
publicados de classificacao de galaxias - vale um paragrafo no seu texto.

Referencia: Selvaraju et al. (2017), Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization, ICCV.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from astro_classifier.models.classifier import AstroClassifier


class GradCAM:
    """Uso:

        cam = GradCAM(model)
        heatmap = cam(input_tensor)   # (H, W) em [0, 1]
        cam.close()
    """

    def __init__(self, model: AstroClassifier, target_layer: nn.Module | None = None) -> None:
        self.model = model
        self.model.eval()
        self.layer = target_layer or self._last_conv_layer(model.backbone)

        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles = [
            self.layer.register_forward_hook(self._save_activations),
            self.layer.register_full_backward_hook(self._save_gradients),
        ]

    @staticmethod
    def _last_conv_layer(module: nn.Module) -> nn.Module:
        convs = [m for m in module.modules() if isinstance(m, nn.Conv2d)]
        if not convs:
            raise ValueError("Nenhuma camada Conv2d encontrada no backbone.")
        return convs[-1]

    def _save_activations(self, _module, _inp, output) -> None:
        self._activations = output.detach()

    def _save_gradients(self, _module, _grad_in, grad_out) -> None:
        self._gradients = grad_out[0].detach()

    def __call__(self, image: torch.Tensor, class_index: int | None = None) -> np.ndarray:
        """`image`: tensor (1, 3, H, W) ja normalizado. Devolve mapa (H, W) em [0,1]."""
        if image.dim() != 4 or image.size(0) != 1:
            raise ValueError(f"esperado tensor (1, 3, H, W), recebido {tuple(image.shape)}")

        # Grad-CAM precisa de gradientes: inference_mode nao serve aqui.
        with torch.enable_grad():
            logits = self.model(image)
            if class_index is None:
                class_index = int(logits.argmax(dim=1).item())
            self.model.zero_grad(set_to_none=True)
            logits[0, class_index].backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("hooks nao capturaram ativacoes/gradientes")

        # Peso de cada canal = gradiente medio espacial (Global Average Pooling).
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)

        cam = cam[0, 0].cpu().numpy()
        span = cam.max() - cam.min()
        # Mapa uniforme (span ~ 0) significa que nada se destacou: devolve zeros
        # em vez de amplificar ruido numerico.
        return (cam - cam.min()) / span if span > 1e-8 else np.zeros_like(cam)

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def overlay_heatmap(
    image_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    """Sobrepoe o mapa de calor a imagem original. Ambos (H, W, 3) / (H, W).

    Devolve uint8 (H, W, 3), pronto para PIL.Image.fromarray.
    """
    import matplotlib

    colored = matplotlib.colormaps["jet"](heatmap)[..., :3]
    base = image_rgb.astype(np.float32) / 255.0 if image_rgb.dtype == np.uint8 else image_rgb
    blended = (1 - alpha) * base + alpha * colored
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Desfaz a normalizacao do ImageNet. (3,H,W) -> (H,W,3) em [0,1]."""
    from astro_classifier.data.transforms import IMAGENET_MEAN, IMAGENET_STD

    media = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    desvio = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    imagem = (tensor.cpu() * desvio + media).clamp(0, 1)
    return imagem.permute(1, 2, 0).numpy()


def save_gradcam_examples(
    model: AstroClassifier,
    dataset,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    out_dir,
    device: str = "cpu",
    per_class: int = 4,
) -> int:
    """Uma figura por classe, com acertos e erros lado a lado.

    Escolhemos acertos E erros de proposito. Um acerto mostra em que o modelo
    se apoia quando da certo; um erro mostra o que ele confundiu - e e no erro
    que aparece o atalho. Se o mapa de uma classe acende no fundo, no ruido ou
    na borda da placa fotografica em vez de no objeto, a acuracia daquela
    classe e um artefato do dataset, nao conhecimento.

    Devolve quantas figuras foram gravadas.
    """
    import matplotlib

    matplotlib.use("Agg")
    from pathlib import Path

    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    geradas = 0

    with GradCAM(model) as cam:
        for indice, nome in enumerate(classes):
            acertos = np.flatnonzero((y_true == indice) & (y_pred == indice))[:per_class]
            erros = np.flatnonzero((y_true == indice) & (y_pred != indice))[:per_class]
            escolhidos = list(acertos) + list(erros)
            if not escolhidos:
                continue

            fig, axes = plt.subplots(2, len(escolhidos), figsize=(2.1 * len(escolhidos), 4.6))
            axes = np.atleast_2d(axes)
            if axes.shape[0] == 1:  # uma coluna so
                axes = axes.reshape(2, -1)

            for coluna, idx in enumerate(escolhidos):
                tensor, _ = dataset[int(idx)]
                entrada = tensor.unsqueeze(0).to(device)
                mapa = cam(entrada, class_index=int(y_pred[idx]))
                original = denormalize(tensor)

                axes[0, coluna].imshow(original)
                axes[1, coluna].imshow(overlay_heatmap(original, mapa))

                acertou = y_pred[idx] == indice
                axes[0, coluna].set_title(
                    ("OK " if acertou else "ERRO -> ") + ("" if acertou else classes[y_pred[idx]]),
                    fontsize=8,
                    color="green" if acertou else "crimson",
                )
                for linha in (0, 1):
                    axes[linha, coluna].axis("off")

            axes[0, 0].set_ylabel("original", fontsize=8)
            axes[1, 0].set_ylabel("Grad-CAM", fontsize=8)
            fig.suptitle(f"Grad-CAM - classe '{nome}'", fontsize=11)
            fig.tight_layout()
            fig.savefig(out_dir / f"{nome}.png", dpi=120)
            plt.close(fig)
            geradas += 1

    return geradas
