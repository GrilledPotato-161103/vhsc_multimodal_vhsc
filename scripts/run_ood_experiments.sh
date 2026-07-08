#!/bin/bash
# OOD sensitivity experiments.
# Prerequisites: GPU pod via srun, or run on login node (CPU-only, slower).
#   srun --account=bgia-delta-gpu --partition=gpuA40x4-interactive \
#     --nodes=1 --time 1:00:00 --gpus-per-node=1 --tasks=1 \
#     --tasks-per-node=1 --cpus-per-task=16 --mem=32g --pty bash
# Then: cd /projects/bgia/duyan2/SURE && bash scripts/run_ood_experiments.sh

PY="/projects/bgia/duyan2/SURE/sure_env/bin/python"
CKPT="data/checkpoints/checkpoint.pth"
LOG="logs/experiments_input_ood.md"

log() { echo "$1" | tee -a $LOG; }

TIMESTAMP=$(date '+%Y-%m-%d %H:%M')
log ""
log "---"
log "## Run: $TIMESTAMP"
log ""
log "| # | Type | amp ratio | sps ratio | Verdict |"
log "|---|---|---|---|---|"

run_one() {
    local num=$1 ptype=$2 label=$3
    echo ""
    echo "=========================================="
    echo "Attempt $num: $label"
    echo "=========================================="
    log ""
    log "### Attempt $num — \`$label\`"
    log ""

    OUTFILE="/tmp/ood_attempt_${num}.txt"
    $PY src/eval_ood_sensitivity.py --ckpt $CKPT --type $ptype --label "$label" \
        2>&1 | tee "$OUTFILE"

    AMP=$(grep "OOD/ID amplitude ratio" $OUTFILE | grep -oP '[\d.]+$' || echo "ERR")
    SPS=$(grep "OOD/ID sps ratio"       $OUTFILE | grep -oP '[\d.]+$' || echo "ERR")
    VER=$(grep "VERDICT"                $OUTFILE | grep -oP '(SENSITIVE|NOT SENSITIVE)' | head -1 || echo "ERR")

    log ""
    log "\`\`\`"
    cat $OUTFILE >> $LOG
    log "\`\`\`"
    log ""
    log "amp ratio=$AMP | sps ratio=$SPS | **$VER**"
    log ""
    log "| $num | $label | $AMP | $SPS | $VER |"
    log ""
}

run_one 1 "sd"        "SD-Mahalanobis (baseline)"
run_one 2 "cycle"     "Cycle + Sigma_A shape"
run_one 3 "cycle_iso" "Cycle + Identity shape"
run_one 4 "gmm"       "GMM K=4 Mahalanobis"
run_one 5 "pca"       "PCA projection k=2"

echo ""
echo "Attempts 1-5 complete. Check $LOG for results."
echo "Edit this script to add attempts 6-10 based on findings."
