#!/bin/bash
#SBATCH --account=bgia-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name=mask1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=02:00:00
#SBATCH --output=/projects/bgia/duyan2/SURE/logs/slurm_mask1_%j.log
#SBATCH --error=/projects/bgia/duyan2/SURE/logs/slurm_mask1_%j.err

cd /projects/bgia/duyan2/SURE
PY="/projects/bgia/duyan2/SURE/sure_env/bin/python"

# First full MISSING-MODALITY run. mask_rate=0.5 -> 50% both present, 25% miss-1, 25% miss-2.
# Signal-aware cycle (available modalities only) + EKF reconstruction propagation + aleatoric.
# Validation: var_miss/var_both > 1 (missing -> higher variance), tracking err ratio.
echo "mask1: mask_rate=0.5, cycle_iso + closed_form + aleatoric, 30 epochs"

$PY src/train_hook.py --config-name=train_ekf_hook \
    trainer=gpu \
    trainer.max_epochs=30 \
    model.mask_rate=0.5 \
    model.ekf_net.mode=closed_form \
    model.ekf_net.use_aleatoric=true \
    model.ekf_net.aleatoric_input_mode=xy \
    model.ekf_net.aleatoric_hidden_dim=64 \
    model.ekf_net.aleatoric_n_layers=3 \
    model.ekf_net.lambda_aleatoric=1.0 \
    model.lambda_aux=0.3 \
    logger.wandb.offline=true \
    logger.wandb.name="mask1_rate05" \
    logger.wandb.tags='["missing_modality","mask1","rate0.5"]'
echo "EXITCODE=$?"
