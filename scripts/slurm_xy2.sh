#!/bin/bash
#SBATCH --account=bgia-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name=xy2_fixed
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=01:00:00
#SBATCH --output=/projects/bgia/duyan2/SURE/logs/slurm_xy2_%j.log
#SBATCH --error=/projects/bgia/duyan2/SURE/logs/slurm_xy2_%j.err

cd /projects/bgia/duyan2/SURE
PY="/projects/bgia/duyan2/SURE/sure_env/bin/python"

# KEY BUG FIX: aux loss was MSE(sigma_al, residual_sq) — WRONG.
# sigma_al alone should NOT match (y-ŷ)² because sigma_ep already covers OOD residuals.
# Correct: MSE(sigma_total, residual_sq) where sigma_total = sigma_ep + lam*sigma_al.
# This means sigma_al only learns what sigma_ep misses (the in-distribution blob structure).
echo "xy2: fixed aux loss on sigma_total, xy input, lambda_al=0.3, lambda_aux=0.5"

$PY src/train_hook.py --config-name=train_ekf_hook \
    trainer=gpu \
    trainer.max_epochs=30 \
    model.ekf_net.use_aleatoric=true \
    model.ekf_net.aleatoric_input_mode=xy \
    model.ekf_net.aleatoric_xy_dim=2 \
    model.ekf_net.aleatoric_hidden_dim=64 \
    model.ekf_net.aleatoric_n_layers=3 \
    model.ekf_net.lambda_aleatoric=0.3 \
    model.lambda_aux=0.5 \
    logger.wandb.name="xy2_fixed_aux_total_laux05" \
    logger.wandb.tags='["xy_input","fixed_aux","iter_xy2"]'
