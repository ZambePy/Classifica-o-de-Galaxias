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
    import matplotlib.cm as cm

    colored = cm.get_cmap("jet")(heatmap)[..., :3]
    base = image_rgb.astype(np.float32) / 255.0 if image_rgb.dtype == np.uint8 else image_rgb
    blended = (1 - alpha) * base + alpha * colored
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)
