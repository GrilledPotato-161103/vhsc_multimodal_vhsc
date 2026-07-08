#!/bin/bash
#SBATCH --account=bgia-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --job-name=ood_sensitivity
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=01:00:00
#SBATCH --output=/projects/bgia/duyan2/SURE/logs/slurm_ood_%j.log
#SBATCH --error=/projects/bgia/duyan2/SURE/logs/slurm_ood_%j.err

cd /projects/bgia/duyan2/SURE

PY="/projects/bgia/duyan2/SURE/sure_env/bin/python"
SCRIPT="src/eval_ood_sensitivity.py"
CKPT="data/checkpoints/checkpoint.pth"
LOG="logs/experiments_input_ood.md"

echo "Running on: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

run_one() {
    local num=$1 type=$2 prop=$3 extra=$4 label=$5
    echo ""
    echo "========== Attempt $num: $label =========="
    OUTFILE="/tmp/slurm_attempt_${num}.txt"
    $PY $SCRIPT --ckpt $CKPT --type $type --prop $prop $extra \
        --label "$label" 2>&1 | tee $OUTFILE

    AMP=$(grep "OOD/ID amplitude ratio" $OUTFILE | grep -oP '[\d.]+$' || echo "ERR")
    SPS=$(grep "OOD/ID sps ratio"       $OUTFILE | grep -oP '[\d.]+$' || echo "ERR")
    VER=$(grep "VERDICT"                $OUTFILE | grep -oP '(SENSITIVE|NOT SENSITIVE)' | head -1 || echo "ERR")

    echo "" >> $LOG
    echo "### Attempt $num (GPU) — \`$label\`" >> $LOG
    echo "" >> $LOG
    echo "\`\`\`" >> $LOG
    cat $OUTFILE >> $LOG
    echo "\`\`\`" >> $LOG
    echo "" >> $LOG
    echo "**amp $AMP** | **sps $SPS** | **$VER**" >> $LOG
    echo "| $num | $label | $AMP | $SPS | $VER |" >> $LOG
}

echo "" >> $LOG
echo "---" >> $LOG
echo "## GPU Run (Slurm) — $(date '+%Y-%m-%d %H:%M')" >> $LOG
echo "" >> $LOG
echo "| # | Type | amp ratio | sps ratio | Verdict |" >> $LOG
echo "|---|---|---|---|---|" >> $LOG

# ---- Amplitude methods (1-5 on GPU for clean numbers) ----
run_one 1  "sd"  "first_order" ""        "SD + First-order (GPU)"
run_one 4  "gmm" "first_order" ""        "GMM K=4 (GPU)"
run_one 5  "pca" "first_order" ""        "PCA k=2 (GPU)"

# ---- Cycle attempts (decoder training is fast on GPU) ----
run_one 2  "cycle"     "first_order" "" "Cycle+SigmaA (GPU)"
run_one 3  "cycle_iso" "first_order" "" "Cycle+Identity (GPU)"

# ---- Propagation fixes ----
run_one 6  "sd"  "second_order" ""       "SD + Second-order Taylor (GPU)"
run_one 7  "pca" "second_order" ""       "PCA + Second-order Taylor (GPU)"
run_one 8  "gmm" "second_order" ""       "GMM + Second-order Taylor (GPU)"
run_one 9  "sd"  "mc_only" "--K 20"      "SD + MC-Dropout only (GPU,K=20)"
run_one 10 "pca" "mc_only" "--K 20"      "PCA + MC-Dropout only (GPU,K=20)"

echo ""
echo "All GPU attempts complete."
echo "Best combination: check sps ratio column in $LOG"
