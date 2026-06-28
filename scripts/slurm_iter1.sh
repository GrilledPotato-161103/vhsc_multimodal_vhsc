#!/bin/bash
#SBATCH --account=bgia-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name=iter1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=01:00:00
#SBATCH --output=/projects/bgia/duyan2/SURE/logs/slurm_iter1_%j.log
#SBATCH --error=/projects/bgia/duyan2/SURE/logs/slurm_iter1_%j.err

cd /projects/bgia/duyan2/SURE
PY="/projects/bgia/duyan2/SURE/sure_env/bin/python"

echo "Attempt 1: z_only + lambda_aux=0.1 (direct regression supervision baseline)"

$PY src/train_hook.py --config-name=train_ekf_hook \
    trainer=gpu \
    trainer.max_epochs=30 \
    model.ekf_net.use_aleatoric=true \
    model.ekf_net.aleatoric_input_mode=z_only \
    model.ekf_net.aleatoric_hidden_dim=32 \
    model.ekf_net.lambda_aleatoric=1.0 \
    model.lambda_aux=0.1 \
    logger.wandb.name="aux_iter1_zonly_laux01" \
    logger.wandb.tags='["aux_regression","iter1","z_only"]'
