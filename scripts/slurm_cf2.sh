#!/bin/bash
#SBATCH --account=bgia-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name=cf2
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=01:00:00
#SBATCH --output=/projects/bgia/duyan2/SURE/logs/slurm_cf2_%j.log
#SBATCH --error=/projects/bgia/duyan2/SURE/logs/slurm_cf2_%j.err

cd /projects/bgia/duyan2/SURE
PY="/projects/bgia/duyan2/SURE/sure_env/bin/python"

# cf1: closed_form + xy + pure NLL -> PCC_predvar=0.867, Spearman=0.399, NLL=-2.018.
# Indirection removed = huge win. But Spearman (bulk monotonic) only 0.40 because
# Gaussian NLL is tail-dominated.
# cf2: add aux MSE(sigma_total, error) back. Denser per-sample gradient on the bulk
# should lift Spearman. lambda_aux=0.3.
echo "cf2: closed_form + xy + lambda_aux=0.3 (NLL + dense regression)"

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
    logger.wandb.name="cf2_closedform_xy_aux03" \
    logger.wandb.tags='["closed_form","xy","cf2"]'
