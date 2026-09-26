#!/usr/bin/env bash
# Bateria completa: espera a coleta, refaz os splits, retreina os tres
# niveis, avalia tudo e roda os experimentos do roteiro.
#
#   bash scripts/madrugada.sh
#
# Cada etapa grava seus proprios artefatos, entao a interrupcao no meio nao
# perde o que ja rodou. Os treinos sao SEQUENCIAIS de proposito: o nivel 2
# sozinho ja usa ~3 GB, e rodar dois ao mesmo tempo derruba a maquina.

set -u
cd "C:/Users/gabri/OneDrive/Desktop/REDE NEURAL GALAXIA"
PY="./.venv/Scripts/python.exe"
LOGS="/c/astro-data/runs/_logs"
mkdir -p "$LOGS"

etapa() { echo; echo "############ $* ############"; date '+%H:%M:%S'; }

# ---------------------------------------------------------------- 1. coleta
etapa "1/9 aguardando a coleta terminar"
until tr '\r' '\n' < "$LOGS/coleta_total.log" 2>/dev/null | grep -q "FIM DA COLETA"; do
    sleep 60
done
tr '\r' '\n' < "$LOGS/coleta_total.log" | grep -E "apos deduplicar|imagens salvas|orfas" | tail -6

# ------------------------------------------------------------ 2. inspecao
etapa "2/9 inspecao visual"
$PY scripts/inspect_dataset.py --class-dir nebulae --n 24 --cols 12 2>&1 | grep -E "imagens ->"
$PY scripts/inspect_dataset.py --class-dir other  --n 24 --cols 12 2>&1 | grep -E "imagens ->"

# -------------------------------------------------------------- 3. splits
etapa "3/9 splits consistentes"
$PY scripts/make_splits.py --all --from-index galaxies_index.csv --cap-object 2500 2>&1 \
    | grep -E "\.csv:|assignment|votos"

# -------------------------------------------------------------- 4. treinos
etapa "4/9 treino nivel 3 (nebulosa)"
$PY scripts/train.py --config configs/level3_nebula.yaml > "$LOGS/m_n3.log" 2>&1
tr '\r' '\n' < "$LOGS/m_n3.log" | grep -E "train_loss=|early stopping" | tail -2

etapa "5/9 treino nivel 1 (objeto)"
$PY scripts/train.py --config configs/level1_object.yaml > "$LOGS/m_n1.log" 2>&1
tr '\r' '\n' < "$LOGS/m_n1.log" | grep -E "train_loss=|early stopping" | tail -2

etapa "6/9 treino nivel 2 (galaxia) - o longo"
$PY scripts/train.py --config configs/level2_galaxy.yaml > "$LOGS/m_n2.log" 2>&1
tr '\r' '\n' < "$LOGS/m_n2.log" | grep -E "train_loss=|early stopping" | tail -2

# ----------------------------------------------------------- 7. avaliacoes
etapa "7/9 avaliacao no teste"
for cfg in level3_nebula level1_object level2_galaxy; do
    echo "--- $cfg ---"
    $PY scripts/evaluate.py --config "configs/$cfg.yaml" 2>&1 \
        | grep -vE "it/s\]|s/it\]" | grep -E "acuracia|macro |MCC|ROC|^ {4}[a-z_]+ +[0-9]|kappa"
done

etapa "7b/9 cascata, atalho e detector"
$PY scripts/evaluate_cascade.py 2>&1 | grep -vE "it/s\]|s/it\]" | grep -E "acuracia|erro|subtipo|^galaxy|^nebula|^other"
$PY scripts/analyze_shortcut.py 2>&1 | grep -vE "it/s\]|s/it\]" | grep -E "star_field|CONTROLE|AUROC|macro-F1"
for d in mellinger panstarrs allwise; do
    echo "--- OOD vs $d ---"
    $PY scripts/evaluate_ood.py --ood-dir "C:/astro-data/ood/$d" 2>&1 \
        | grep -vE "it/s\]|s/it\]" | grep -E "AUROC|dominio (aceito|ACEITO)"
done

# --------------------------------------------------------- 8. experimentos
etapa "8/9 bateria: TTA, backbones e sementes"
$PY scripts/run_experiments.py --only tta backbones seeds \
    --levels object nebula --seeds 1 2 > "$LOGS/m_exp.log" 2>&1
tail -25 "$LOGS/m_exp.log"

# -------------------------------------------------------------- 9. resumo
etapa "9/9 FIM"
echo "artefatos:"
echo "  /c/astro-data/runs/_experiments/resultados.md"
echo "  /c/astro-data/runs/_cascade/metrics.md"
echo "  /c/astro-data/runs/<experimento>/metrics_test.md"
date '+%H:%M:%S'
