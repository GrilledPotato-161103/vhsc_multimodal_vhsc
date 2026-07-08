#!/bin/bash
#SBATCH --account=bgia-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name=cf1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=01:00:00
#SBATCH --output=/projects/bgia/duyan2/SURE/logs/slurm_cf1_%j.log
#SBATCH --error=/projects/bgia/duyan2/SURE/logs/slurm_cf1_%j.err

cd /projects/bgia/duyan2/SURE
PY="/projects/bgia/duyan2/SURE/sure_env/bin/python"

# HYPOTHESIS: in "learned" mode, inv_alpha_net sits between sigma_total and the
# predictive variance, decoupling our decomposition from the NLL objective.
# In "closed_form" mode, var = sigma_total exactly, so Gaussian NLL MLE directly
# drives sigma_total -> (y-ŷ)². Test with PURE NLL (no aux) to isolate the effect.
echo "cf1: closed_form mode, xy aleatoric, lam=1.0, lambda_aux=0.0 (pure NLL)"

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
    model.lambda_aux=0.0 \
    logger.wandb.name="cf1_closedform_xy_pureNLL" \
    logger.wandb.tags='["closed_form","xy","cf1"]'
