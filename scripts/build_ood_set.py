"""Monta um conjunto FORA DO DOMINIO, para medir o detector de verdade.

    python scripts/build_ood_set.py

O detector de dominio existe para avisar quando alguem sobe no dashboard uma
imagem que nao se parece com nada do treino. Ate aqui ele nunca foi medido
contra dados reais - so calibrado na validacao, o que e um chute plausivel.

O QUE NAO SERVE COMO CONJUNTO FORA DO DOMINIO

As 600 fotos comuns de `non_astronomical`: foram TREINADAS como classe
`other`. O modelo as conhece, e media-las daria AUROC alta e vazia.

A IDEIA AQUI

Baixar os MESMOS objetos do conjunto de teste, nas MESMAS coordenadas e com
o MESMO enquadramento, mas de outros levantamentos:

    DSS2 color   o dominio do TREINO (placas fotograficas digitalizadas)
    Mellinger    astrofotografia AMADORA processada, colorida e saturada -
                 e literalmente o tipo de imagem que as pessoas sobem
    PanSTARRS    telescopio moderno, outra cadeia de processamento
    allWISE      INFRAVERMELHO: o objeto esta la, mas a aparencia nao tem
                 relacao com a imagem optica

Isso isola exatamente a variavel que interessa. Nao e "outro objeto", nem
"outra coisa qualquer": e o MESMO objeto astronomico, visto por outro
instrumento. Se o modelo se sai bem, aprendeu o objeto; se desaba, aprendeu
a aparencia de um levantamento especifico.

E um desenho experimental melhor do que juntar astrofotos do Hubble a mao,
porque controla objeto, enquadramento e classe.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from astro_classifier.data.cutouts import CutoutError, fetch_cutout
from astro_classifier.paths import get_paths

SURVEYS = {
    "mellinger": ("CDS/P/Mellinger/color", "astrofotografia amadora processada"),
    "panstarrs": ("CDS/P/PanSTARRS/DR1/color-i-r-g", "telescopio moderno"),
    "allwise": ("CDS/P/allWISE/color", "infravermelho"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=80, help="objetos por levantamento")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--pause", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = get_paths()
    indice = paths.catalogs / "nebulae_index.csv"
    if not indice.exists():
        print(f"{indice} nao encontrado. Rode antes: python scripts/build_nebula_dataset.py")
        return 1

    # Só objetos do conjunto de TESTE: usar os de treino mediria o detector
    # sobre imagens que o modelo ja viu (em outro levantamento, mas ainda
    # assim o mesmo objeto e o mesmo enquadramento).
    catalogo = pd.read_csv(indice)
    teste = paths.splits / "nebula_test.csv"
    if teste.exists():
        caminhos_teste = set(pd.read_csv(teste)["path"])
        antes = len(catalogo)
        catalogo = catalogo[catalogo["path"].isin(caminhos_teste)]
        print(f"{len(catalogo)} objetos do conjunto de teste (de {antes} no catalogo)")
    else:
        print("[aviso] nebula_test.csv nao encontrado; usando o catalogo inteiro")

    if catalogo.empty:
        print("Nenhum objeto para baixar.")
        return 1

    amostra = catalogo.sample(min(args.n, len(catalogo)), random_state=args.seed)
    destino_base = paths.raw.parent / "ood"
    print(f"\nDestino: {destino_base}")
    print(f"{len(amostra)} objetos x {len(SURVEYS)} levantamentos\n")

    total = 0
    for chave, (hips, descricao) in SURVEYS.items():
        pasta = destino_base / chave
        pasta.mkdir(parents=True, exist_ok=True)
        salvos = falhas = 0

        for _, r in tqdm(list(amostra.iterrows()), desc=f"{chave:<10}", total=len(amostra)):
            arquivo = pasta / Path(r["path"]).name
            if arquivo.exists():
                salvos += 1
                continue
            try:
                arquivo.write_bytes(
                    fetch_cutout(
                        float(r["ra"]), float(r["dec"]), float(r["fov_deg"]),
                        survey=hips, size_px=args.size,
                    )
                )
                salvos += 1
            except (CutoutError, ValueError, KeyError):
                falhas += 1
            import time

            time.sleep(args.pause)

        print(f"  {chave:<10} {salvos:>4} salvos, {falhas:>3} falhas   ({descricao})")
        total += salvos

    print(f"\n{total} imagens fora do dominio em {destino_base}")
    print("\nAgora mede o detector:")
    for chave in SURVEYS:
        print(f"  python scripts/evaluate_ood.py --ood-dir {destino_base / chave}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
