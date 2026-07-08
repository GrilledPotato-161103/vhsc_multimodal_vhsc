#!/bin/bash
#SBATCH --account=bgia-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name=aleatoric
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=04:00:00
#SBATCH --output=/projects/bgia/duyan2/SURE/logs/slurm_aleatoric_%j.log
#SBATCH --error=/projects/bgia/duyan2/SURE/logs/slurm_aleatoric_%j.err

# 10-experiment sweep over aleatoric head configurations.
# Goal: find the config where log(sigma_total) spatially matches log(loss).
# Primary metric: val/loss_unc_pcc (Pearson correlation, target > 0.3).
# Secondary: val_plot/Log_Variance_Map visually matching val_plot/Log_Loss_Map.
#
# Key questions:
#   - Does the aleatoric head fill the in-distribution blob structure?
#   - Does sigma_al_frac stay in a healthy range (0.2-0.8)?
#   - Does the epistemic OOD ramp survive alongside the aleatoric blobs?

cd /projects/bgia/duyan2/SURE
PY="/projects/bgia/duyan2/SURE/sure_env/bin/python"
TRAIN="src/train_hook.py"
CONFIG="--config-name=train_ekf_hook"

echo "Running on: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

run_exp() {
    local experiment=$1
    local extra=$2
    echo ""
    echo "================================================================"
    echo "EXPERIMENT: $experiment"
    echo "================================================================"
    $PY $TRAIN $CONFIG \
        trainer=gpu \
        trainer.max_epochs=30 \
        experiment=$experiment \
        $extra \
        2>&1
}

# ---------------------------------------------------------------------------
# Group A: Head capacity
# ---------------------------------------------------------------------------
# A1: smallest head (hdim=16) — can aleatoric signal be captured at all?
run_exp "aleatoric_A1"

# A2: medium head (hdim=32, n_layers=2) — recommended baseline
run_exp "aleatoric_A2"

# A3: larger head (hdim=64, n_layers=3) — risks aleatoric domination
run_exp "aleatoric_A3"

# ---------------------------------------------------------------------------
# Group B: lambda_aleatoric
# ---------------------------------------------------------------------------
# B1: lambda=0.1 — epistemic dominant, aleatoric as a correction
run_exp "aleatoric_B1"

# B2: lambda=5.0 — aleatoric amplified, fills blobs faster
run_exp "aleatoric_B2"

# B3: lambda=0.5 — intermediate
run_exp "aleatoric_B3"

# ---------------------------------------------------------------------------
# Group C: Input features to AleatoricHead
# ---------------------------------------------------------------------------
# C1: z only — pure function complexity, no OOD awareness
run_exp "aleatoric_C1"

# C2: z + log_sigma_ep — canonical combined input (same as A2 but tagged)
run_exp "aleatoric_C2"

# ---------------------------------------------------------------------------
# Group D: Architecture variants
# ---------------------------------------------------------------------------
# D1: batch norm instead of layer norm
run_exp "aleatoric_D1"

# D2: hdim=64 + lambda=2.0 — sweet spot search
run_exp "aleatoric_D2"

echo ""
echo "================================================================"
echo "All 10 aleatoric experiments complete."
echo "Check wandb project VISHC_Uncertainty, tags: aleatoric"
echo "Key metric: val/loss_unc_pcc  (target: > 0.3)"
echo "Key diagnostic: val/sigma_al_frac  (healthy range: 0.2-0.8)"
echo "================================================================"
