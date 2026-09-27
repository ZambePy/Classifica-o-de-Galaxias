"""Deteccao de fora-de-dominio pela distancia de MAHALANOBIS.

O MSP e a energia olham a SAIDA da rede - os logits. Este metodo olha um
nivel antes: o vetor de features do penultimo layer, antes da camada linear.

A ideia e modelar onde as imagens de treino VIVEM nesse espaco. Para cada
classe ajusta-se uma gaussiana, todas compartilhando a mesma matriz de
covariancia, e a pontuacao de uma imagem nova e a distancia ate a gaussiana
mais proxima:

    score(x) = - min_c  (f(x) - mu_c)^T  Sigma^-1  (f(x) - mu_c)

A distancia de Mahalanobis nao e a euclidiana: o `Sigma^-1` corrige pela
forma da nuvem de pontos. Uma direcao em que as features de treino variam
muito conta pouco; uma direcao em que elas quase nao variam conta muito. Sem
essa correcao, as poucas dimensoes de alta variancia dominariam tudo.

POR QUE ISTO PODE GANHAR DO MSP NESTE PROJETO

Uma imagem fora do dominio pode produzir logits confiantes por acidente -
basta que ela ative as features erradas com forca. Mas e bem mais dificil
que ela caia DENTRO da nuvem de features do treino. O sinal e mais rico:
512 ou 2048 dimensoes contra 3 ou 4 logits.

A ARMADILHA NUMERICA, QUE E REAL AQUI

A covariancia tem d x d entradas - 2048 x 2048 = 4,2 milhoes de parametros
para o resnet50 - estimados a partir de alguns milhares de imagens. A matriz
sai SINGULAR, e inverter uma matriz singular produz numeros sem sentido que
parecem numeros. Por isso usamos encolhimento (shrinkage):

    Sigma' = (1 - a) * Sigma  +  a * (tr(Sigma)/d) * I

que empurra a matriz na direcao de uma esfera. E o mesmo remedio que o
artigo original usa via `sklearn.EmpiricalCovariance`, so explicito.

O `a` nao e fixo: ele sobe ate a matriz ficar bem CONDICIONADA, e o objeto
registra tanto o `a` usado quanto o numero de condicao resultante. Um `a`
alto e sinal de que ha poucas imagens para a dimensao das features, e isso
tem de aparecer no relatorio em vez de ficar escondido.

E o criterio e o condicionamento, nao "o Cholesky falhou". A diferenca nao e
teorica: uma covariancia matematicamente singular ACEITA Cholesky depois de
somar 1e-9 na diagonal. Nao ha excecao, as distancias saem finitas, enormes e
sem significado - dominadas por autodirecoes que sao puro ruido de
estimativa. Um teste deste modulo cobre exatamente esse caso.

Referencia: Lee et al. (2018), A Simple Unified Framework for Detecting
Out-of-Distribution Samples and Adversarial Attacks, NeurIPS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

# Condicionamento maximo aceito para a covariancia encolhida.
#
# O numero de condicao e a razao entre o maior e o menor autovalor, e diz
# quanto a direcao mais estreita da nuvem de features e AMPLIFICADA no calculo
# da distancia. Com 1e6, uma direcao estimada com variancia um milhao de vezes
# menor que a maior ainda contribui, mas nao domina.
#
# Por que nao um valor mais folgado: float64 tem ~16 digitos, entao 1e10
# "cabe" numericamente. Mas caber nao e o critério. Um autovalor 1e10 vezes
# menor que o maior, estimado de alguns milhares de amostras, NAO e uma
# direcao estreita de verdade - e ruido de estimativa, e a distancia de
# Mahalanobis divide por ele. A primeira versao deste arquivo usava 1e10 e um
# caso de 10 amostras x 40 features passou com condicionamento 8,5e9 e
# encolhimento intacto: exatamente o cenario que o limite deveria barrar.
#
# O preco de apertar: mais encolhimento aproxima a distancia da euclidiana e
# perde parte do branqueamento que faz o metodo melhor que a euclidiana. E um
# meio-termo, e por isso o valor usado vai SEMPRE no relatorio - quem le
# precisa poder julgar se foi apertado demais para o dataset dele.
MAX_CONDICAO = 1e6


@dataclass
class MahalanobisDetector:
    """Gaussianas por classe com covariancia compartilhada, no espaco de features.

    Nao construa direto: use `fit()`.
    """

    means: torch.Tensor  # (C, d) - centroide de cada classe
    cholesky: torch.Tensor  # (d, d) - fator L de Sigma, triangular inferior
    shrinkage: float  # o `a` efetivamente usado
    condition_number: float  # condicionamento de Sigma' apos o encolhimento
    n_fit: int  # imagens usadas no ajuste
    classes: list[str] = field(default_factory=list)
    # Scores ordenados de um conjunto dentro do dominio, para `domain_percentile`.
    # Preenchido por `calibrate()`; None significa nao calibrado.
    calibration_scores: torch.Tensor | None = None

    @property
    def dim(self) -> int:
        return int(self.means.shape[1])

    def distances(self, features: torch.Tensor | np.ndarray) -> np.ndarray:
        """Distancia de Mahalanobis ao quadrado ate cada classe. (N, C)

        Resolvida por substituicao no fator de Cholesky, nao invertendo
        Sigma: `L z = (f - mu)` da `z` com `|z|^2 = d_M^2`. Inverter a matriz
        explicitamente perde precisao sem necessidade.
        """
        f = torch.as_tensor(features, dtype=torch.float64)
        if f.dim() == 1:
            f = f.unsqueeze(0)
        if f.shape[1] != self.dim:
            raise ValueError(
                f"features tem dimensao {f.shape[1]}, o detector foi ajustado em "
                f"{self.dim}. Provavelmente e outro backbone."
            )

        saida = torch.empty((f.shape[0], self.means.shape[0]), dtype=torch.float64)
        for c in range(self.means.shape[0]):
            delta = (f - self.means[c]).T  # (d, N)
            z = torch.linalg.solve_triangular(self.cholesky, delta, upper=False)
            saida[:, c] = (z * z).sum(dim=0)
        return saida.numpy()

    def score(self, features: torch.Tensor | np.ndarray) -> np.ndarray:
        """MAIOR = mais dentro do dominio, para casar com a convencao do MSP.

        E a distancia ate a classe mais proxima, com sinal trocado. A escala e
        arbitraria - pode valer -1500 - entao NAO exiba este numero numa
        interface. Use `domain_percentile`.
        """
        return -self.distances(features).min(axis=1)

    def calibrate(self, features: torch.Tensor | np.ndarray) -> None:
        """Guarda os scores de um conjunto DENTRO do dominio, para o percentil.

        Passe a VALIDACAO, nao o treino: as gaussianas foram ajustadas no
        treino, e pontuar o proprio treino daria distancias otimistas e um
        percentil deslocado.
        """
        s = np.sort(self.score(features))
        self.calibration_scores = torch.as_tensor(s, dtype=torch.float64)

    def domain_percentile(self, features: torch.Tensor | np.ndarray) -> np.ndarray:
        """Score em [0, 1]: fracao das imagens legitimas que pontuam PIOR.

        Existe por um motivo pratico. A distancia de Mahalanobis crua e um
        numero sem escala fixa - um dashboard que a mostre numa barra de
        progresso quebra. O percentil e interpretavel ("esta imagem esta no
        pior 3% das legitimas") e, de quebra, torna o limiar trivial: aceitar
        95% das imagens boas e exatamente `percentil >= 0,05`.

        Requer `calibrate()` antes.
        """
        if self.calibration_scores is None:
            raise RuntimeError(
                "detector nao calibrado - chame calibrate() com as features da "
                "validacao antes de usar domain_percentile()"
            )
        ref = self.calibration_scores.numpy()
        # searchsorted devolve quantos valores de referencia sao menores.
        return np.searchsorted(ref, self.score(features), side="left") / len(ref)

    @classmethod
    def fit(
        cls,
        features: torch.Tensor | np.ndarray,
        labels: torch.Tensor | np.ndarray,
        classes: list[str] | None = None,
        shrinkage: float = 0.01,
    ) -> MahalanobisDetector:
        """Ajusta nas features do TREINO.

        Use o conjunto de treino, nao o de validacao: queremos saber onde o
        modelo aprendeu a colocar as coisas. E use as features sem
        aumentacao - `eval_transforms`.

        A covariancia e AGRUPADA (tied): cada classe entra centrada na sua
        propria media, e todas somam numa unica matriz. E o que o artigo faz,
        e com poucas imagens por classe e a unica escolha estavel - uma
        covariancia por classe multiplicaria por C o numero de parametros a
        estimar.
        """
        f = torch.as_tensor(features, dtype=torch.float64)
        y = torch.as_tensor(labels).long().flatten()

        if f.dim() != 2:
            raise ValueError(f"esperado (N, d) de features; recebido {tuple(f.shape)}")
        if len(f) != len(y):
            raise ValueError(f"{len(f)} features contra {len(y)} rotulos")

        n, d = f.shape
        n_classes = int(y.max().item()) + 1

        medias = torch.zeros((n_classes, d), dtype=torch.float64)
        centrado = torch.empty_like(f)
        for c in range(n_classes):
            mascara = y == c
            if not mascara.any():
                raise ValueError(
                    f"classe {c} nao tem nenhuma imagem no conjunto de ajuste - "
                    "nao da para estimar o centroide dela"
                )
            medias[c] = f[mascara].mean(dim=0)
            centrado[mascara] = f[mascara] - medias[c]

        # Covariancia agrupada. Divide por n, nao por n-C: e o estimador de
        # maxima verossimilhanca, que e o do artigo.
        sigma = (centrado.T @ centrado) / n

        # --- encolhimento guiado pelo NUMERO DE CONDICAO ---
        #
        # A versao anterior deste codigo tentava o Cholesky e aumentava o
        # encolhimento quando ele falhava. ISSO NAO PROTEGE NADA, e um teste
        # pegou: com 10 amostras e 40 features a covariancia tem rank <= 8 e e
        # matematicamente singular, mas somar 1e-9 * I ja a torna
        # NUMERICAMENTE positiva-definida. O Cholesky passa, nao ha excecao, e
        # as distancias saem dominadas pelas autodirecoes de autovalor
        # ~1e-9 - numeros enormes, finitos, e sem significado.
        #
        # O criterio certo e o condicionamento. Como
        #     Sigma'(a) = (1-a) Sigma + a s I
        # tem autovalores (1-a) lambda_i + a s, basta UM autodecomposicao de
        # Sigma para saber o condicionamento de qualquer `a` - sem refatorar a
        # matriz a cada tentativa.
        escala = float(torch.diagonal(sigma).mean())
        autovalores = torch.linalg.eigvalsh(sigma).clamp(min=0.0)
        lmin, lmax = float(autovalores[0]), float(autovalores[-1])

        def condicao(a: float) -> float:
            num = (1.0 - a) * lmax + a * escala
            den = (1.0 - a) * lmin + a * escala
            return num / den if den > 0 else float("inf")

        a = float(shrinkage)
        if condicao(a) > MAX_CONDICAO:
            # Menor `a` na grade que atinge o condicionamento alvo.
            for candidato in (10 ** (e / 4) for e in range(-32, 1)):
                if candidato >= a and condicao(candidato) <= MAX_CONDICAO:
                    a = candidato
                    break
            else:
                raise RuntimeError(
                    f"as features nao ficam bem condicionadas nem com encolhimento 1,0: "
                    f"d={d}, n={n}, autovalores entre {lmin:.3e} e {lmax:.3e}. "
                    "Ha features demais para imagens de menos - use um backbone "
                    "com menos features ou ajuste em mais imagens."
                )

        identidade = torch.eye(d, dtype=torch.float64)
        L = torch.linalg.cholesky((1.0 - a) * sigma + a * escala * identidade)

        if classes is not None and len(classes) != n_classes:
            raise ValueError(
                f"{len(classes)} nomes de classe para {n_classes} classes nos rotulos"
            )

        return cls(
            means=medias,
            cholesky=L,
            shrinkage=a,
            condition_number=condicao(a),
            n_fit=int(n),
            classes=list(classes) if classes else [str(i) for i in range(n_classes)],
        )

    def save(self, path: str | Path) -> None:
        """Matrizes nao cabem em JSON de forma honesta - vai como .pt."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "means": self.means,
                "cholesky": self.cholesky,
                "shrinkage": self.shrinkage,
                "condition_number": self.condition_number,
                "n_fit": self.n_fit,
                "classes": self.classes,
                "calibration_scores": self.calibration_scores,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> MahalanobisDetector:
        d = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            means=d["means"],
            cholesky=d["cholesky"],
            shrinkage=d["shrinkage"],
            condition_number=d.get("condition_number", float("nan")),
            n_fit=d["n_fit"],
            classes=d.get("classes", []),
            calibration_scores=d.get("calibration_scores"),
        )


@torch.inference_mode()
def extract_features(model, loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Features do penultimo layer + rotulos, para um DataLoader inteiro.

    `model.backbone(x)` ja devolve o vetor achatado depois do pooling - e
    exatamente o que entra na camada linear. Nao precisa de hook.
    """
    model.eval()
    feats: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for lote in loader:
        x, y = lote[0], lote[1]
        f = model.backbone(x.to(device))
        feats.append(f.flatten(1).cpu().numpy())
        # Com rotulos suaves o `y` vem como distribuicao - a classe e o argmax.
        y = y.cpu()
        ys.append((y.argmax(dim=1) if y.dim() > 1 else y).numpy())
    return np.concatenate(feats), np.concatenate(ys)
