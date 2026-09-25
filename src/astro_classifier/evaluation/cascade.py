"""Avaliacao da cascata inteira, de ponta a ponta.

Medir cada nivel isoladamente nao diz como o SISTEMA se comporta. O nivel 2
de galaxias e sempre medido em galaxias de verdade - mas em producao ele
recebe o que o nivel 1 mandar, inclusive nebulosas classificadas errado.

O que so aparece aqui:

  acuracia da folha    de ponta a ponta, o rotulo final saiu certo?
                       Sempre menor que o produto das acuracias por nivel.

  erro em cascata      quantas amostras o nivel 1 errou e, por isso,
                       nem chegaram ao modelo certo? E o preco explicito
                       da Abordagem A, e precisa estar no trabalho.

  acuracia condicional entre as que o nivel 1 acertou, quanto o nivel 2
                       acertou? Separa "o subtipo esta ruim" de "o subtipo
                       nunca teve chance".

Exemplo do que isso revela: nivel 1 com 92% e nivel 2 com 88% NAO dao 88%
no fim. Dao ~81%, e os 11 pontos perdidos sao amostras que o modelo de
subtipo nunca viu - ele recebeu nebulosas e respondeu com morfologia de
galaxia.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from astro_classifier.taxonomy import SUBMODEL_FOR_OBJECT


@dataclass
class CascadeSample:
    """Uma amostra atravessando a cascata."""

    objeto_verdadeiro: str
    objeto_predito: str
    subtipo_verdadeiro: str | None = None
    subtipo_predito: str | None = None

    @property
    def nivel1_correto(self) -> bool:
        return self.objeto_verdadeiro == self.objeto_predito

    @property
    def folha_verdadeira(self) -> str:
        """Rotulo final esperado: o subtipo quando existe, senao o objeto."""
        return self.subtipo_verdadeiro or self.objeto_verdadeiro

    @property
    def folha_predita(self) -> str:
        return self.subtipo_predito or self.objeto_predito

    @property
    def folha_correta(self) -> bool:
        return self.folha_verdadeira == self.folha_predita


@dataclass
class CascadeReport:
    n_amostras: int = 0
    nivel1_acuracia: float = 0.0
    folha_acuracia: float = 0.0
    erro_cascata: float = 0.0
    subtipo_acuracia_condicional: float = 0.0
    por_ramo: dict = field(default_factory=dict)
    matriz_folha: list = field(default_factory=list)
    classes_folha: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_amostras": self.n_amostras,
            "nivel1_acuracia": self.nivel1_acuracia,
            "folha_acuracia": self.folha_acuracia,
            "erro_cascata": self.erro_cascata,
            "subtipo_acuracia_condicional": self.subtipo_acuracia_condicional,
            "por_ramo": self.por_ramo,
            "matriz_folha": self.matriz_folha,
            "classes_folha": self.classes_folha,
        }


def evaluate_cascade(amostras: list[CascadeSample]) -> CascadeReport:
    """Calcula as metricas de ponta a ponta a partir das travessias."""
    if not amostras:
        raise ValueError("nenhuma amostra para avaliar")

    n = len(amostras)
    nivel1_ok = sum(a.nivel1_correto for a in amostras)
    folha_ok = sum(a.folha_correta for a in amostras)

    # Amostras que tinham subtipo esperado e cujo nivel 1 acertou: sao as
    # unicas em que o modelo de subtipo teve chance real.
    com_chance = [a for a in amostras if a.nivel1_correto and a.subtipo_verdadeiro is not None]
    subtipo_ok = sum(a.subtipo_predito == a.subtipo_verdadeiro for a in com_chance)

    # Perdidas exclusivamente pelo erro do nivel 1.
    perdidas = sum(1 for a in amostras if not a.nivel1_correto)

    relatorio = CascadeReport(
        n_amostras=n,
        nivel1_acuracia=nivel1_ok / n,
        folha_acuracia=folha_ok / n,
        erro_cascata=perdidas / n,
        subtipo_acuracia_condicional=(subtipo_ok / len(com_chance)) if com_chance else 0.0,
    )

    # Por ramo: cada tipo de objeto real tem um destino diferente na cascata.
    for objeto in sorted({a.objeto_verdadeiro for a in amostras}):
        do_ramo = [a for a in amostras if a.objeto_verdadeiro == objeto]
        acertou_nivel1 = [a for a in do_ramo if a.nivel1_correto]
        tem_subtipo = [a for a in acertou_nivel1 if a.subtipo_verdadeiro is not None]

        relatorio.por_ramo[objeto] = {
            "n": len(do_ramo),
            "nivel1_acuracia": len(acertou_nivel1) / len(do_ramo),
            "folha_acuracia": sum(a.folha_correta for a in do_ramo) / len(do_ramo),
            "subtipo_acuracia_condicional": (
                sum(a.subtipo_predito == a.subtipo_verdadeiro for a in tem_subtipo) / len(tem_subtipo)
                if tem_subtipo
                else None
            ),
            "tem_submodelo": SUBMODEL_FOR_OBJECT.get(objeto) is not None,
        }

    matriz, classes = _matriz_folha(amostras)
    relatorio.matriz_folha = matriz
    relatorio.classes_folha = classes
    return relatorio


def _matriz_folha(amostras: list[CascadeSample]) -> tuple[list, list]:
    """Matriz de confusao sobre os rotulos FINAIS da cascata.

    E a figura mais honesta do sistema: mostra, por exemplo, espirais
    terminando como `planetary` - erro que nenhuma matriz por nivel revela,
    porque nenhum nivel sozinho ve os dois rotulos.
    """
    classes = sorted({a.folha_verdadeira for a in amostras} | {a.folha_predita for a in amostras})
    indice = {c: i for i, c in enumerate(classes)}

    matriz = np.zeros((len(classes), len(classes)), dtype=int)
    for a in amostras:
        matriz[indice[a.folha_verdadeira], indice[a.folha_predita]] += 1
    return matriz.tolist(), classes


def print_cascade_report(relatorio: CascadeReport) -> None:
    print(f"\n{'=' * 72}\nCASCATA COMPLETA (ponta a ponta)\n{'=' * 72}")
    print(f"\namostras                          {relatorio.n_amostras}")
    print(f"acuracia do nivel 1               {relatorio.nivel1_acuracia:.4f}")
    print(f"acuracia da folha (rotulo final)  {relatorio.folha_acuracia:.4f}")
    print(f"erro em cascata                   {relatorio.erro_cascata:.4f}")
    print(f"subtipo dado nivel 1 correto      {relatorio.subtipo_acuracia_condicional:.4f}")

    print(f"\n{'ramo':<14}{'n':>7}{'nivel 1':>10}{'folha':>10}{'subtipo|ok':>12}")
    print("-" * 53)
    for nome, m in relatorio.por_ramo.items():
        cond = f"{m['subtipo_acuracia_condicional']:.3f}" if m["subtipo_acuracia_condicional"] is not None else "-"
        print(f"{nome:<14}{m['n']:>7}{m['nivel1_acuracia']:>10.3f}{m['folha_acuracia']:>10.3f}{cond:>12}")

    perdido = relatorio.nivel1_acuracia - relatorio.folha_acuracia
    print(
        f"\nDo nivel 1 ate a folha perdem-se {perdido:.3f} de acuracia. Parte e erro\n"
        "do modelo de subtipo; parte sao amostras que chegaram ao submodelo errado.\n"
        "Essa diferenca e o custo da arquitetura em cascata - reporte-a."
    )
