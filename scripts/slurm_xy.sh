#!/bin/bash
#SBATCH --account=bgia-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name=xy_aleatoric
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=01:00:00
#SBATCH --output=/projects/bgia/duyan2/SURE/logs/slurm_xy_%j.log
#SBATCH --error=/projects/bgia/duyan2/SURE/logs/slurm_xy_%j.err

cd /projects/bgia/duyan2/SURE
PY="/projects/bgia/duyan2/SURE/sure_env/bin/python"

# Previous 5 attempts all used z or (z,sep) as input to the aleatoric head.
# Root cause of failure: z doesn't encode the error map geometry (unit circle,
# diagonals, etc.) — it compresses these away because the encoder only needs them
# for prediction, not for calibration.
# Fix: pass (x1, x2) directly. The error map IS a function of raw input space.
# Config: winner from attempt 3 (lambda_al=0.3, lambda_aux=0.3) + input_mode=xy.
echo "xy_aleatoric: raw (x1,x2) input to aleatoric head, lambda_al=0.3, lambda_aux=0.3"

$PY src/train_hook.py --config-name=train_ekf_hook \
    trainer=gpu \
    trainer.max_epochs=30 \
    model.ekf_net.use_aleatoric=true \
    model.ekf_net.aleatoric_input_mode=xy \
    model.ekf_net.aleatoric_xy_dim=2 \
    model.ekf_net.aleatoric_hidden_dim=64 \
    model.ekf_net.aleatoric_n_layers=3 \
    model.ekf_net.lambda_aleatoric=0.3 \
    model.lambda_aux=0.3 \
    logger.wandb.name="xy_aleatoric_lal03_laux03" \
    logger.wandb.tags='["xy_input","aleatoric","final"]'
