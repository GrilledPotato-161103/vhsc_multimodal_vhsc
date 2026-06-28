#!/bin/bash
#SBATCH --account=bgia-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name=iter4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=01:00:00
#SBATCH --output=/projects/bgia/duyan2/SURE/logs/slurm_iter4_%j.log
#SBATCH --error=/projects/bgia/duyan2/SURE/logs/slurm_iter4_%j.err

cd /projects/bgia/duyan2/SURE
PY="/projects/bgia/duyan2/SURE/sure_env/bin/python"

# Attempt 3 result: sigma_al=0.0117 ~ residuals(0.011), NLL=-1.913 — the regression
# supervision found the right scale. Balance: real_al_frac~26%, ep_frac~74%.
# Attempt 4: push lambda_aux from 0.3 -> 0.5 to enforce spatial blob structure harder.
# lambda_al stays at 0.3 (good balance confirmed).
# Also train 50 epochs (more time for the spatial structure to crystallise).
echo "Attempt 4: lambda_al=0.3, lambda_aux=0.5, 50 epochs (push spatial structure)"

$PY src/train_hook.py --config-name=train_ekf_hook \
    trainer=gpu \
    trainer.max_epochs=50 \
    model.ekf_net.use_aleatoric=true \
    model.ekf_net.aleatoric_input_mode=z_only \
    model.ekf_net.aleatoric_hidden_dim=64 \
    model.ekf_net.aleatoric_n_layers=2 \
    model.ekf_net.lambda_aleatoric=0.3 \
    model.lambda_aux=0.5 \
    logger.wandb.name="aux_iter4_lal03_laux05_e50" \
    logger.wandb.tags='["aux_regression","iter4","stronger_aux"]'
