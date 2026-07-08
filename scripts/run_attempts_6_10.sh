#!/bin/bash
# Attempts 6-10: second-order Taylor and MC dropout propagation
# Run on GPU pod: srun --account=bgia-delta-gpu --partition=gpuA40x4-interactive \
#   --nodes=1 --time 1:00:00 --gpus-per-node=1 --tasks=1 \
#   --tasks-per-node=1 --cpus-per-task=16 --mem=32g --pty bash

PY="/projects/bgia/duyan2/SURE/sure_env/bin/python"
SCRIPT="src/eval_ood_sensitivity.py"
CKPT="data/checkpoints/checkpoint.pth"
LOG="logs/experiments_input_ood.md"

run_one() {
    local num=$1 type=$2 prop=$3 extra=$4 label=$5
    echo ""
    echo "========== Attempt $num: $label =========="
    OUTFILE="/tmp/attempt_${num}.txt"
    OMP_NUM_THREADS=4 $PY $SCRIPT --ckpt $CKPT \
        --type $type --prop $prop $extra \
        --label "$label" 2>&1 | tee $OUTFILE

    AMP=$(grep "OOD/ID amplitude ratio" $OUTFILE | grep -oP '[\d.]+$' || echo "ERR")
    SPS=$(grep "OOD/ID sps ratio"       $OUTFILE | grep -oP '[\d.]+$' || echo "ERR")
    VER=$(grep "VERDICT"                $OUTFILE | grep -oP '(SENSITIVE|NOT SENSITIVE)' | head -1 || echo "ERR")

    echo "" >> $LOG
    echo "### Attempt $num — \`$label\`" >> $LOG
    echo "" >> $LOG
    echo "\`\`\`" >> $LOG
    cat $OUTFILE >> $LOG
    echo "\`\`\`" >> $LOG
    echo "" >> $LOG
    echo "**amp $AMP** | **sps $SPS** | **$VER**" >> $LOG
    echo "" >> $LOG
    echo "| $num | $label | $AMP | $SPS | $VER |" >> $LOG
}

echo "" >> $LOG
echo "---" >> $LOG
echo "## Propagation fix experiments (Attempts 6-10) — $(date '+%Y-%m-%d %H:%M')" >> $LOG
echo "" >> $LOG
echo "**Hypothesis**: second-order Taylor and MC dropout fix Jacobian collapse (sps ratio should >> 1.5x)." >> $LOG
echo "" >> $LOG
echo "| # | Type | amp ratio | sps ratio | Verdict |" >> $LOG
echo "|---|---|---|---|---|" >> $LOG

# 6: SD + second-order
run_one 6 "sd"  "second_order" "" "SD + Second-order Taylor"

# 7: SD + MC dropout pure
run_one 7 "sd"  "mc_only" "--K 20" "SD + MC-Dropout pure (K=20)"

# 8: PCA + second-order (PCA had best sps ratio 1.62x — combine with SO)
run_one 8 "pca" "second_order" "" "PCA + Second-order Taylor"

# 9: PCA + MC dropout pure
run_one 9 "pca" "mc_only" "--K 20" "PCA + MC-Dropout pure (K=20)"

# 10: GMM + MC dropout (GMM had best amp ratio 8x)
run_one 10 "gmm" "mc_only" "--K 20" "GMM + MC-Dropout pure (K=20)"

echo ""
echo "All 10 attempts complete. See logs/experiments_input_ood.md"
