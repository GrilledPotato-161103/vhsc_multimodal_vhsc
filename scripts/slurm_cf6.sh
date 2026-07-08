#!/bin/bash
#SBATCH --account=bgia-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name=cf6
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=01:00:00
#SBATCH --output=/projects/bgia/duyan2/SURE/logs/slurm_cf6_%j.log
#SBATCH --error=/projects/bgia/duyan2/SURE/logs/slurm_cf6_%j.err

cd /projects/bgia/duyan2/SURE
PY="/projects/bgia/duyan2/SURE/sure_env/bin/python"

# cf4's smoothing was underpowered (20x20 bins, ~5 samples/cell).
# cf6: CELL-LEVEL correlation with coarse bins (8x8 ~30/cell, 12x12).
# Compares E[var|cell] vs E[err|cell] directly — the rigorous spatial-structure test.
# If CellPCC ~ 0.95 -> model captures expected error structure; per-sample is noise (WIN).
# If CellPCC ~ 0.87 -> genuine structural mismatch (need different approach).
# Uses best config: cf2 (closed_form, xy, hdim64, linear aux=0.3).
echo "cf6: best config + cell-level structural correlation"

$PY src/train_hook.py --config-name=train_ekf_hook \
    trainer=gpu \
    trainer.max_epochs=30 \
    model.ekf_net.mode=closed_form \
    model.ekf_net.use_aleatoric=true \
    model.ekf_net.aleatoric_input_mode=xy \
    model.ekf_net.aleatoric_xy_dim=2 \
    model.ekf_net.aleatoric_hidden_dim=64 \
    model.ekf_net.aleatoric_n_layers=3 \
    model.ekf_net.lambda_aleatoric=1.0 \
    model.lambda_aux=0.3 \
    model.aux_mode=linear \
    logger.wandb.name="cf6_celllevel" \
    logger.wandb.tags='["closed_form","xy","cf6","cell_pcc"]'
