#!/bin/bash
#SBATCH --account=bgia-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name=iter2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=01:00:00
#SBATCH --output=/projects/bgia/duyan2/SURE/logs/slurm_iter2_%j.log
#SBATCH --error=/projects/bgia/duyan2/SURE/logs/slurm_iter2_%j.err

cd /projects/bgia/duyan2/SURE
PY="/projects/bgia/duyan2/SURE/sure_env/bin/python"

# Attempt 1 result: sigma_al dominated (82% fraction, 9x larger than sigma_ep).
# Aux supervision worked (NLL=-1.93 nice), but sigma_al overwhelmed sigma_ep.
# Fix: reduce lambda_aleatoric 1.0 -> 0.1 to rebalance contributions.
# Expected: sigma_total = sigma_ep + 0.1*sigma_al ~ 0.010 + 0.009 = 0.019 (50/50 split)
echo "Attempt 2: lambda_aleatoric=0.1 to rebalance epistemic/aleatoric"

$PY src/train_hook.py --config-name=train_ekf_hook \
    trainer=gpu \
    trainer.max_epochs=30 \
    model.ekf_net.use_aleatoric=true \
    model.ekf_net.aleatoric_input_mode=z_only \
    model.ekf_net.aleatoric_hidden_dim=32 \
    model.ekf_net.lambda_aleatoric=0.1 \
    model.lambda_aux=0.1 \
    logger.wandb.name="aux_iter2_lal01_laux01" \
    logger.wandb.tags='["aux_regression","iter2","lal0.1"]'
