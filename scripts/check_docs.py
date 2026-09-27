"""Confere se os numeros da documentacao batem com os dados e as metricas.

    python scripts/check_docs.py

POR QUE ISTO EXISTE

A auditoria de 26/09 encontrou, so na documentacao:

  - o total do dataset citado como 66.409 quando a soma das partes que o
    proprio paragrafo lista da 42.437 - e o numero em disco e 66.299;
  - "seis catalogos" em quatro lugares, quando sao oito;
  - a contagem de testes desatualizada em tres lugares;
  - `supernova_remnant` com F1 0,575 na secao de limitacoes e 0,673 na tabela
    do mesmo arquivo;
  - a dependencia de contexto do nivel 3 como 89% num lugar e 82,8% noutro;
  - `star_field -> nebulosa` como 7,8% quando o JSON dizia 4 de 90.

Nenhum desses erros quebra codigo. Todos quebram a confianca de quem le, que e
o unico produto de um trabalho academico. E todos sao detectaveis por
comparacao mecanica.

O QUE ISTO NAO FAZ

Nao confere prosa, nao confere raciocinio, e nao adivinha numeros que so
existem em texto. Confere o que da para ancorar: contagens de arquivos,
metricas gravadas em JSON, e a soma de tabelas. Falhar aqui e erro; passar aqui
nao e garantia de que o texto esta certo.

Saida: uma linha por verificacao, e codigo 1 se alguma falhar - da para rodar
no CI.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

_OK, _FALHA, _PULADO = "ok", "FALHA", "pulado"


class Verificador:
    def __init__(self) -> None:
        self.linhas: list[tuple[str, str, str]] = []

    def checar(self, nome: str, esperado, encontrado, detalhe: str = "") -> None:
        estado = _OK if esperado == encontrado else _FALHA
        msg = detalhe or f"doc diz {encontrado}, dados dizem {esperado}"
        self.linhas.append((estado, nome, "" if estado == _OK else msg))

    def pular(self, nome: str, motivo: str) -> None:
        self.linhas.append((_PULADO, nome, motivo))

    def relatorio(self) -> int:
        largura = max(len(n) for _, n, _ in self.linhas) + 2
        falhas = 0
        for estado, nome, msg in self.linhas:
            marca = {"ok": "  ok  ", "FALHA": " FALHA", "pulado": "pulado"}[estado]
            print(f"[{marca}] {nome:<{largura}}{msg}")
            falhas += estado == _FALHA
        print()
        if falhas:
            print(f"{falhas} verificacao(oes) falharam - a documentacao esta desatualizada.")
        else:
            print("todos os numeros ancoraveis da documentacao batem com os dados.")
        return 1 if falhas else 0


def _texto(rel: str) -> str | None:
    caminho = REPO / rel
    return caminho.read_text(encoding="utf-8") if caminho.exists() else None


def _numeros_com_ponto(texto: str, padrao: str) -> list[int]:
    """Acha numeros escritos no estilo pt-BR (66.299) apos um padrao."""
    achados = re.findall(padrao, texto)
    return [int(a.replace(".", "").replace(" ", "")) for a in achados]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="raiz dos dados (padrao: ASTRO_DATA_ROOT do .env)",
    )
    args = parser.parse_args()

    from astro_classifier.paths import get_paths

    paths = get_paths()
    raiz = Path(args.data_root) if args.data_root else paths.raw.parent
    raw = raiz / "raw"

    v = Verificador()
    readme = _texto("README.md") or ""
    results = _texto("docs/results.md") or ""
    datasets = _texto("docs/datasets.md") or ""

    # ---------------------------------------------------- 1. contagem de testes
    try:
        import subprocess

        saida = subprocess.run(
            [
                sys.executable, "-m", "pytest", "--collect-only", "-q",
                "--no-header", "-p", "no:warnings",
            ],
            capture_output=True,
            text=True,
            cwd=REPO,
            timeout=300,
        ).stdout
        # Com `-q` o pytest imprime uma linha por arquivo: "tests/x.py: 12".
        # Somar essas linhas e mais estavel que casar a frase de resumo, que
        # muda de formato entre versoes e desaparece com -p no:warnings.
        por_arquivo = [int(n) for n in re.findall(r"^tests[/\\].+?: (\d+)$", saida, re.M)]
        if por_arquivo:
            real = sum(por_arquivo)
            citados = {int(x) for x in re.findall(r"(\d+) testes", readme)}
            if citados:
                v.checar(
                    "contagem de testes no README",
                    {real},
                    citados,
                    f"README cita {sorted(citados)}, pytest coleta {real}",
                )
            else:
                v.pular("contagem de testes no README", "nenhuma mencao encontrada")
        else:
            v.pular("contagem de testes", "nao consegui ler a saida do pytest")
    except Exception as exc:  # noqa: BLE001
        v.pular("contagem de testes", f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------- 2. imagens em disco
    if raw.exists():
        em_disco = sum(
            1 for p in raw.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
        )
        citados = set(_numeros_com_ponto(readme, r"[Tt]otal em disco: ([\d.]+)"))
        if citados:
            v.checar("total de imagens em disco", {em_disco}, citados)
        else:
            v.pular("total de imagens em disco", f"{em_disco} em disco; README nao cita")
    else:
        v.pular("total de imagens em disco", f"{raw} nao existe")

    # ------------------------------------------------- 3. imagens rotuladas
    usadas: set[str] = set()
    import pandas as pd

    if paths.splits.exists():
        for f in paths.splits.glob("*_*.csv"):
            if f.stem == "assignment":
                continue
            try:
                usadas |= set(pd.read_csv(f)["path"])
            except Exception:  # noqa: BLE001
                pass
    if usadas:
        citados = set(_numeros_com_ponto(readme, r"\*\*([\d.]+) imagens rotuladas\*\*"))
        if citados:
            v.checar("imagens rotuladas (soma dos splits)", {len(usadas)}, citados)
        else:
            v.pular("imagens rotuladas", f"{len(usadas)} nos splits; README nao cita")
    else:
        v.pular("imagens rotuladas", "splits nao encontrados")

    # ------------------------------------------------- 4. numero de catalogos
    try:
        from astro_classifier.data.catalogs import CATALOGS

        n_cat = len(CATALOGS)
        palavra = {6: "seis", 7: "sete", 8: "oito", 9: "nove"}.get(n_cat, str(n_cat))
        errados = []
        # O cartao do dataset entra aqui de proposito: ele e GERADO, e a
        # contagem estava escrita a mao no template - dizia "seis" enquanto a
        # tabela logo abaixo listava oito fontes, no mesmo arquivo.
        cartao = _texto("docs/dataset/dataset_card.md") or ""
        for nome, txt in (
            ("README", readme),
            ("results", results),
            ("datasets", datasets),
            ("dataset_card", cartao),
        ):
            for achado in re.findall(r"(seis|sete|oito|nove|\d+) catálogos", txt):
                if achado not in {palavra, str(n_cat)}:
                    errados.append(f"{nome}:'{achado} catálogos'")
        v.checar(
            "numero de catalogos de nebulosa",
            [],
            errados,
            f"codigo tem {n_cat} ({palavra}); docs dizem {errados}",
        )
    except Exception as exc:  # noqa: BLE001
        v.pular("numero de catalogos", f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------- 5. metricas por nivel
    nomes = {
        "level1_object": "nível 1",
        "level2_galaxy": "nível 2",
        "level3_nebula": "nível 3",
    }
    for run, rotulo in nomes.items():
        mj = raiz / "runs" / run / "metrics_test.json"
        if not mj.exists():
            v.pular(f"metricas do {rotulo}", f"{mj.name} nao existe")
            continue
        d = json.loads(mj.read_text(encoding="utf-8"))
        acc = f"{d['accuracy']:.4f}".replace(".", ",")
        n = d["n_amostras"]

        # a linha do resumo tem de citar a acuracia e o n corretos
        linha = re.search(rf"\*\*{re.escape(rotulo)}\*\*[^\n|]*\|([^\n]*)", results)
        if not linha:
            v.pular(f"metricas do {rotulo}", "linha de resumo nao encontrada em results.md")
            continue
        conteudo = linha.group(1)
        v.checar(
            f"acuracia do {rotulo} em results.md",
            True,
            acc in conteudo,
            f"esperava {acc} na linha do resumo; linha: {conteudo.strip()[:70]}",
        )
        n_fmt = f"{n:,}".replace(",", ".")
        v.checar(
            f"n do {rotulo} em results.md",
            True,
            (n_fmt in conteudo) or (str(n) in conteudo),
            f"esperava n={n_fmt} na linha do resumo",
        )

    # ------------------------------------------------- 6. cascata
    cj = raiz / "runs" / "_cascade" / "metrics.json"
    if cj.exists():
        d = json.loads(cj.read_text(encoding="utf-8"))
        folha = f"{d['folha_acuracia']:.4f}".replace(".", ",")
        v.checar(
            "acuracia da folha (cascata)",
            True,
            folha in results,
            f"results.md nao cita {folha}",
        )
    else:
        v.pular("acuracia da folha (cascata)", "metrics.json da cascata nao existe")

    # ------------------------------------------------- 7. composicao de `other`
    outros = raw / "other"
    if outros.exists():
        for sub in sorted(p for p in outros.iterdir() if p.is_dir()):
            n = sum(1 for x in sub.glob("*") if x.suffix.lower() in IMAGE_SUFFIXES)
            citado = re.search(rf"`{sub.name}/`[^|]*\|[^|]*\|\s*(\d+)\s*\|", datasets)
            if citado:
                v.checar(f"other/{sub.name} em datasets.md", n, int(citado.group(1)))
            else:
                v.pular(f"other/{sub.name}", f"{n} em disco; datasets.md nao cita numero")
    else:
        v.pular("composicao de `other`", f"{outros} nao existe")

    # ------------------------------------------------- 8. catalogo publicado
    csv = REPO / "docs" / "dataset" / "nebulae_catalog.csv"
    if csv.exists():
        cat = pd.read_csv(csv)
        n_fmt = f"{len(cat):,}".replace(",", ".")
        for nome, txt in (("README", readme), ("datasets", datasets)):
            v.checar(
                f"total de nebulosas citado no {nome}",
                True,
                n_fmt in txt,
                f"{nome} nao cita {n_fmt} nebulosas (CSV tem {len(cat)})",
            )
    else:
        v.pular("total de nebulosas", "nebulae_catalog.csv nao existe")

    return v.relatorio()


if __name__ == "__main__":
    sys.exit(main())
