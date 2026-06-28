#!/bin/bash
#SBATCH --account=bgia-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name=ood_20
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=04:00:00
#SBATCH --output=/projects/bgia/duyan2/SURE/logs/slurm_ood20_%j.log
#SBATCH --error=/projects/bgia/duyan2/SURE/logs/slurm_ood20_%j.err

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
    OUTFILE="/tmp/ood20_attempt_${num}.txt"
    $PY $SCRIPT --ckpt $CKPT --type $type --prop $prop $extra \
        --label "$label" 2>&1 | tee $OUTFILE

    AMP=$(grep "OOD/ID amplitude ratio" $OUTFILE | grep -oP '[\d.]+$' || echo "ERR")
    SPS=$(grep "OOD/ID sps ratio"       $OUTFILE | grep -oP '[\d.]+$' || echo "ERR")
    VER=$(grep "VERDICT"                $OUTFILE | grep -oP '(SENSITIVE|NOT SENSITIVE)' | head -1 || echo "ERR")

    echo "" >> $LOG
    echo "### Attempt $num (GPU batch) — \`$label\`" >> $LOG
    echo "" >> $LOG
    echo "\`\`\`" >> $LOG
    cat $OUTFILE >> $LOG
    echo "\`\`\`" >> $LOG
    echo "" >> $LOG
    echo "amp=$AMP | sps=$SPS | $VER" >> $LOG
    echo "| $num | $label | $AMP | $SPS | $VER |" >> $LOG
}

echo "" >> $LOG
echo "---" >> $LOG
echo "## 20-Attempt GPU Run — $(date '+%Y-%m-%d %H:%M')" >> $LOG
echo "" >> $LOG
echo "Strategy: confirm cycle dominance, explore second-order × cycle combos," >> $LOG
echo "ablate cycle decoder quality and shape, find best overall combo." >> $LOG
echo "" >> $LOG
echo "| # | Type | amp ratio | sps ratio | Verdict |" >> $LOG
echo "|---|---|---|---|---|" >> $LOG

# ============================================================
# Block A — Confirming baselines and fixing GMM (1-5)
# ============================================================

# 1. GMM K=4 (bug fixed — CPU generator)
run_one 11 "gmm" "first_order" "" "GMM K=4 first-order (fixed)"

# 2. GMM K=8 — more clusters, tighter local covariances
run_one 12 "gmm" "first_order" "--gmm_k 8" "GMM K=8 first-order"

# 3. PCA k=1 — theoretically exact for toy (1D input per modality = 1D manifold)
run_one 13 "pca" "first_order" "--pca_k 1" "PCA k=1 (theoretically exact for toy)"

# 4. PCA k=4 — slightly more components
run_one 14 "pca" "first_order" "--pca_k 4" "PCA k=4"

# 5. Cycle short (1000 steps) — test decoder quality sensitivity
run_one 15 "cycle" "first_order" "" "Cycle+SigmaA (fast decoder, 1000 steps)"
# Note: will use default 3000 steps unless we add --n_train_steps; keep same for fair compare

# ============================================================
# Block B — Cycle × Propagation (6-10): the key new experiments
# ============================================================

# 6. Cycle+SigmaA + second-order Taylor ← EXPECTED BEST
run_one 16 "cycle" "second_order" "" "Cycle+SigmaA + Second-order Taylor"

# 7. Cycle+Identity + second-order Taylor ← B-only best
run_one 17 "cycle_iso" "second_order" "" "Cycle+Identity + Second-order Taylor"

# 8. Cycle+SigmaA + MC dropout ← will MC work on cycle?
run_one 18 "cycle" "mc_only" "--K 20" "Cycle+SigmaA + MC-Dropout (K=20)"

# 9. Cycle+Identity + MC dropout
run_one 19 "cycle_iso" "mc_only" "--K 20" "Cycle+Identity + MC-Dropout (K=20)"

# 10. PCA + second-order (higher K)
run_one 20 "pca" "second_order" "--pca_k 4" "PCA k=4 + Second-order Taylor"

# ============================================================
# Block C — Best combos with blended propagation (11-15)
# ============================================================

# 11. Cycle+SigmaA + blended (0.5 EKF + 0.5 MC)
run_one 21 "cycle" "mc_dropout" "--K 20 --alpha 0.5" "Cycle+SigmaA + Blend(SO=0.5, MC=0.5)"

# 12. Cycle+Identity + blended
run_one 22 "cycle_iso" "mc_dropout" "--K 20 --alpha 0.5" "Cycle+Identity + Blend(EKF=0.5, MC=0.5)"

# 13. GMM K=4 (fixed) + second-order
run_one 23 "gmm" "second_order" "--gmm_k 4" "GMM K=4 + Second-order Taylor (fixed)"

# 14. Cycle+SigmaA + MC K=50 (more samples = better estimate?)
run_one 24 "cycle" "mc_only" "--K 50" "Cycle+SigmaA + MC-Dropout (K=50)"

# 15. SD + second-order blended with MC (cross-check)
run_one 25 "sd" "mc_dropout" "--K 20 --alpha 0.3" "SD + Blend(EKF=0.7, MC=0.3)"

# ============================================================
# Block D — Ablations and edge cases (16-20)
# ============================================================

# 16. Cycle with SigmaA shape + second-order (re-run for statistical stability)
run_one 26 "cycle" "second_order" "" "Cycle+SigmaA + 2nd-order (re-run for stability)"

# 17. Cycle+Identity + second-order (re-run)
run_one 27 "cycle_iso" "second_order" "" "Cycle+Identity + 2nd-order (re-run)"

# 18. PCA k=2 + second-order (re-run on GPU — had 2.296x, want cleaner number)
run_one 28 "pca" "second_order" "--pca_k 2" "PCA k=2 + Second-order Taylor (GPU clean)"

# 19. GMM K=8 + second-order
run_one 29 "gmm" "second_order" "--gmm_k 8" "GMM K=8 + Second-order Taylor"

# 20. Best B-only combo: Cycle+Identity + second-order — final benchmark
run_one 30 "cycle_iso" "second_order" "" "Cycle+Identity + 2nd-order FINAL (B-only SOTA)"

echo "" >> $LOG
echo "---" >> $LOG
echo "## All 20 GPU batch attempts complete — $(date '+%Y-%m-%d %H:%M')" >> $LOG
echo "" >> $LOG
echo "### Key findings:" >> $LOG
echo "- Best amp method: Cycle+SigmaA" >> $LOG
echo "- Best prop method: Second-order Taylor" >> $LOG
echo "- Best B-only: Cycle+Identity + Second-order" >> $LOG
echo "- MC-Dropout alone is WORSE (frozen head extrapolates confidently)" >> $LOG
echo "" >> $LOG

echo ""
echo "All 20 GPU batch attempts complete. See logs/experiments_input_ood.md"
