#!/bin/bash
#SBATCH --account=bgia-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --job-name=mask_smoke
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=00:20:00
#SBATCH --output=/projects/bgia/duyan2/SURE/logs/slurm_masksmoke_%j.log
#SBATCH --error=/projects/bgia/duyan2/SURE/logs/slurm_masksmoke_%j.err

cd /projects/bgia/duyan2/SURE
PY="/projects/bgia/duyan2/SURE/sure_env/bin/python"

# SMOKE TEST: missing-modality path. mask_rate=0.5 exercises (0,1)/(1,0) signals
# -> reconstructor ln12/ln21 active -> vmap(jacrev) through LayerNorm + EKF.
# 2 epochs, small data: just confirm it RUNS end-to-end (no vmap/BN crash).
echo "smoke: mask_rate=0.5, 2 epochs, cycle_iso + closed_form + aleatoric"

$PY src/train_hook.py --config-name=train_ekf_hook \
    trainer=gpu \
    trainer.max_epochs=2 \
    data.n_samples=8000 \
    model.mask_rate=0.5 \
    model.ekf_net.mode=closed_form \
    model.ekf_net.use_aleatoric=true \
    model.ekf_net.aleatoric_input_mode=xy \
    model.lambda_aux=0.3 \
    logger.wandb.offline=true \
    logger.wandb.name="mask_smoke" \
    logger.wandb.tags='["smoke","mask"]'
echo "EXITCODE=$?"
