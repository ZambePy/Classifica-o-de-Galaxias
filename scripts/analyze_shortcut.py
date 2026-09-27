"""Mede o quanto o modelo do nivel 1 depende de CONTEXTO em vez do objeto.

    python scripts/analyze_shortcut.py

A pergunta que este script responde e a que decide se o projeto funciona de
verdade: o modelo reconhece a nebulosa, ou reconhece o pedaco de ceu onde
nebulosas costumam estar?

Nebulosas galacticas ficam no plano da Via Lactea. Olhar nessa direcao
significa ver um campo denso, alaranjado, com poeira - uma assinatura visual
facil, que correlaciona com a classe por acidente de geografia, nao por
propriedade do objeto. Um modelo pode aprender essa assinatura e parecer
otimo nas metricas, sem nunca ter olhado para a nebulosa.

TRES EVIDENCIAS INDEPENDENTES, calculadas aqui:

  1. ACERTO POR SUBPASTA DE ORIGEM
     `star_field` sao pedacos de ceu sorteados no plano galactico, SEM
     nebulosa nenhuma. Sao o grupo de controle perfeito: se o modelo os
     chama de nebulosa, ele esta decidindo pelo campo, nao pelo objeto.

  2. LATITUDE GALACTICA SOZINHA
     Treina uma arvore de decisao usando SO |b|, sem ver imagem nenhuma. A
     AUROC que ela alcanca e o quanto da tarefa se resolve por geografia -
     um piso que o modelo de visao deveria superar com folga.

  3. ACERTO POR FAIXA DE LATITUDE
     Se o desempenho cai fora do plano galactico, o modelo aprendeu o
     contexto. Se e parecido dentro e fora, aprendeu o objeto.

Rode antes e depois de mudar o dataset: a comparacao dos numeros diz se a
mudanca resolveu o problema ou so mudou a metrica.

LINHA DE BASE - medida em 2026-09-24, com 1032 nebulosas (722 de treino),
antes de acrescentar os catalogos LBN e RCW:

    acerto por origem, nivel 1
        empty_field         43   1.000
        non_astronomical    97   1.000
        planetary           41   1.000
        reflection          28   1.000
        globular_cluster    15   0.867   -> 2 viraram nebulosa
        star_field          47   0.872   -> 6 viraram NEBULOSA   <<< controle
        supernova_remnant   42   0.786   -> 9 viraram other

    geografia sozinha (so |b|, sem imagem)
        AUROC 'e supernova_remnant?'   0.795
        macro-F1 das 4 classes         0.465   (majoritaria: 0.303)

    distribuicao no ceu
        supernova_remnant   |b| mediana 0.6    99% em |b|<10
        emission            |b| mediana 1.9    86%
        planetary           |b| mediana 6.7    60%
        reflection          |b| mediana 9.4    52%

As duas piores classes eram justamente as duas que vivem no plano galactico,
e elas erravam UMA NA OUTRA. O Grad-CAM da epoca mostrava o mesmo blob de
fundo em acertos e erros de supernova_remnant.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

from astro_classifier.paths import get_paths

REPO = pathlib.Path(__file__).resolve().parents[1]


def acerto_por_origem(df: pd.DataFrame) -> pd.DataFrame:
    """Acerto do nivel 1 agrupado pela subpasta de onde a imagem veio."""
    linhas = []
    for origem, g in df.groupby("origem"):
        erros = g[g.verdadeiro != g.predito]["predito"].value_counts().to_dict()
        linhas.append(
            {
                "origem": origem,
                "n": len(g),
                "acerto": (g.verdadeiro == g.predito).mean(),
                "erros": erros,
            }
        )
    return pd.DataFrame(linhas).sort_values("acerto")


def geografia_sozinha(coords: pd.DataFrame) -> dict:
    """Quanto da tarefa se resolve so com a latitude galactica."""
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier

    if len(coords) < 50 or coords["label"].nunique() < 2:
        return {"erro": "poucos objetos com coordenada para medir"}

    X = coords[["b"]].values
    resultado = {}

    y = coords["label"].values
    resultado["macro_f1_multiclasse"] = float(
        cross_val_score(
            DecisionTreeClassifier(max_depth=3, random_state=0), X, y, cv=5, scoring="f1_macro"
        ).mean()
    )
    resultado["baseline_majoritaria"] = float(pd.Series(y).value_counts(normalize=True).iloc[0])

    for classe in sorted(set(y)):
        alvo = (y == classe).astype(int)
        if alvo.sum() < 10:
            continue
        resultado[f"auroc_{classe}"] = float(
            cross_val_score(
                DecisionTreeClassifier(max_depth=3, random_state=0), X, alvo, cv=5, scoring="roc_auc"
            ).mean()
        )
    return resultado


def _controle_limpo(df, paths) -> tuple[int, int, int] | None:
    """Recalcula o controle descartando campos que contem nebulosa catalogada.

    Devolve (n_limpo, quantos_viraram_nebula_no_limpo, n_contaminados), ou None
    se faltar coordenada para decidir.

    POR QUE ISTO E NECESSARIO

    `star_field` so funciona como controle se nao houver nebulosa no quadro. Com
    campo de 30 arcmin e sorteio livre no plano galactico, havia: 46 dos 300
    campos coletados alcancavam uma nebulosa do catalogo, e 6 dos 45 que caem no
    conjunto de teste. Como o relatorio contava 9 erros em 45, ate dois tercos
    deles podiam ser acertos.

    O criterio e o mesmo de `sky_sampling.drop_near_catalog`: o objeto entra no
    recorte se a separacao for menor que (meio-campo + raio do objeto).
    """
    import numpy as np
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    from astro_classifier.data.cutouts import cutout_filename

    indice = paths.catalogs / "other_index.csv"
    csv_neb = REPO / "docs" / "dataset" / "nebulae_catalog.csv"
    if not indice.exists() or not csv_neb.exists():
        print(
            f"\n   [aviso] sem {indice.name} ou nebulae_catalog.csv - nao da para\n"
            "           verificar se o controle esta limpo."
        )
        return None

    oi = pd.read_csv(indice)
    neb = pd.read_csv(csv_neb)
    sf = oi[oi["label"] == "star_field"].copy()
    if sf.empty or "ra" not in sf.columns:
        return None

    sf["path"] = ["other/star_field/" + cutout_filename(r) for _, r in sf.iterrows()]
    no_teste = df[df["origem"] == "star_field"]
    sf = sf[sf["path"].isin(set(no_teste["path"]))].reset_index(drop=True)
    if sf.empty:
        return None

    c = SkyCoord(ra=sf["ra"].values * u.deg, dec=sf["dec"].values * u.deg)
    cn = SkyCoord(ra=neb["ra"].values * u.deg, dec=neb["dec"].values * u.deg)
    idx, sep, _ = c.match_to_catalog_sky(cn)
    meio = sf["fov_deg"].values * 60.0 / 2.0
    raio = np.nan_to_num(
        pd.to_numeric(neb["diam_arcmin"], errors="coerce").values[idx] / 2.0, nan=0.0
    )
    sf["contaminado"] = sep.arcmin < (meio + raio)

    estado = dict(zip(sf["path"], sf["contaminado"], strict=True))
    limpos = no_teste[~no_teste["path"].map(estado).fillna(False).astype(bool)]
    nebula_limpo = int((limpos["predito"] == "nebula").sum())
    return len(limpos), nebula_limpo, int(sf["contaminado"].sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/level1_object.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from torch.utils.data import DataLoader

    from astro_classifier.config import ExperimentConfig, resolve_device
    from astro_classifier.data.datasets import AstroImageDataset
    from astro_classifier.data.transforms import eval_transforms
    from astro_classifier.models.classifier import AstroClassifier
    from astro_classifier.training.loops import evaluate

    paths = get_paths()
    config = ExperimentConfig.from_yaml(args.config)
    device = resolve_device(args.device)

    checkpoint = paths.checkpoints / "object_best.pt"
    if not checkpoint.exists():
        print(f"Checkpoint nao encontrado: {checkpoint}")
        return 1

    model, _ = AstroClassifier.load(checkpoint, device=device)
    test_ds = AstroImageDataset(
        paths.splits / "object_test.csv", paths.raw, config.level, eval_transforms()
    )
    loader = DataLoader(test_ds, batch_size=config.data.batch_size, shuffle=False, num_workers=0)
    _, y_true, y_pred, _ = evaluate(model, loader, torch.nn.CrossEntropyLoss(), device)

    df = pd.read_csv(paths.splits / "object_test.csv").copy()
    df["verdadeiro"] = [config.level.classes[i] for i in y_true]
    df["predito"] = [config.level.classes[i] for i in y_pred]
    df["origem"] = df["path"].str.split("/").str[-2]

    resultado: dict = {}

    print("=" * 74)
    print("1. ACERTO POR SUBPASTA DE ORIGEM")
    print("=" * 74)
    print("   `star_field` e o controle: ceu do plano galactico SEM nebulosa.\n")
    tabela = acerto_por_origem(df)
    print(f"   {'origem':<22}{'n':>5}{'acerto':>9}   erros")
    print("   " + "-" * 62)
    for _, r in tabela.iterrows():
        print(f"   {r['origem']:<22}{r['n']:>5}{r['acerto']:>9.3f}   {r['erros'] or ''}")
    resultado["acerto_por_origem"] = tabela.to_dict("records")

    controle = tabela[tabela["origem"] == "star_field"]
    if len(controle):
        r = controle.iloc[0]
        virou_nebula = r["erros"].get("nebula", 0)
        resultado["star_field_como_nebula"] = int(virou_nebula)
        resultado["star_field_n"] = int(r["n"])
        print(
            f"\n   >>> CONTROLE BRUTO: {virou_nebula} de {r['n']} campos estelares foram\n"
            f"       classificados como NEBULOSA ({virou_nebula / r['n']:.1%})."
        )

        # --- o controle precisa ser LIMPO, e nao era ---
        #
        # A premissa e "campo do plano galactico sem nebulosa nenhuma". Medido:
        # o recorte tem 30 arcmin, e uma fracao dos campos sorteados contem uma
        # nebulosa catalogada dentro do quadro. Nesses casos, chamar de nebulosa
        # e ACERTO, e contar como erro infla o numero do atalho.
        #
        # A coleta ja filtra isso (ver sky_sampling.drop_near_catalog), mas as
        # imagens em disco podem ter vindo de antes da correcao - entao a analise
        # confere por conta propria em vez de confiar na proveniencia.
        limpo = _controle_limpo(df, paths)
        if limpo is not None:
            n_limpo, nebula_limpo, n_sujo = limpo
            resultado["star_field_contaminados"] = int(n_sujo)
            resultado["star_field_n_limpo"] = int(n_limpo)
            resultado["star_field_como_nebula_limpo"] = int(nebula_limpo)
            if n_sujo:
                taxa = nebula_limpo / n_limpo if n_limpo else 0.0
                print(
                    f"\n   >>> {n_sujo} desses campos CONTEM uma nebulosa catalogada dentro\n"
                    f"       do recorte - para eles, 'nebulosa' e a resposta certa.\n"
                    f"   >>> CONTROLE LIMPO: {nebula_limpo} de {n_limpo} ({taxa:.1%})."
                )
            else:
                print("\n   >>> nenhum campo contem nebulosa catalogada: controle limpo.")
        print("       Quanto maior, mais o modelo decide pelo campo e nao pelo objeto.")

    # --- 2 e 3 dependem das coordenadas, que so existem para as nebulosas ---
    indice = paths.catalogs / "nebulae_index.csv"
    if indice.exists():
        cat = pd.read_csv(indice)
        gal = SkyCoord(ra=cat["ra"].values * u.deg, dec=cat["dec"].values * u.deg).galactic
        cat["b"] = np.abs(gal.b.deg)

        print("\n" + "=" * 74)
        print("2. QUANTO A GEOGRAFIA SOZINHA JA RESOLVE")
        print("=" * 74)
        print("   Arvore de decisao usando SO a latitude galactica, sem imagem.\n")
        geo = geografia_sozinha(cat[["b", "label"]])
        resultado["geografia_sozinha"] = geo
        if "erro" in geo:
            print(f"   {geo['erro']}")
        else:
            print(f"   macro-F1 das 4 classes so com |b|   {geo['macro_f1_multiclasse']:.3f}")
            print(f"   chutar a classe majoritaria         {geo['baseline_majoritaria']:.3f}")
            for k, v in sorted(geo.items()):
                if k.startswith("auroc_"):
                    print(f"   AUROC '{k[6:]}' so com |b|{'':<10}{v:>8.3f}")
            print(
                "\n   AUROC perto de 0,5 = a classe nao e previsivel pela posicao.\n"
                "   Perto de 1,0 = a posicao quase determina a classe, e o modelo\n"
                "   de visao pode estar apenas reaprendendo isso."
            )

        print("\n" + "=" * 74)
        print("3. DISTRIBUICAO NO CEU POR CLASSE")
        print("=" * 74)
        print(f"\n   {'classe':<22}{'n':>6}{'|b| mediana':>13}{'em |b|<10':>11}")
        print("   " + "-" * 52)
        dist = []
        for classe, g in cat.groupby("label"):
            linha = {
                "classe": classe,
                "n": len(g),
                "b_mediana": float(g["b"].median()),
                "fracao_plano": float((g["b"] < 10).mean()),
            }
            dist.append(linha)
            print(
                f"   {classe:<22}{len(g):>6}{linha['b_mediana']:>13.1f}"
                f"{linha['fracao_plano']:>10.0%}"
            )
        resultado["distribuicao_no_ceu"] = dist
    else:
        print(f"\n[aviso] {indice} nao encontrado; etapas 2 e 3 puladas")

    destino = paths.runs / "_shortcut" / "metrics.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultados em {destino}")
    print(
        "\nCompare com a rodada anterior. Se `star_field como nebulosa` caiu e o\n"
        "acerto por origem ficou mais uniforme, o modelo passou a olhar o objeto."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
