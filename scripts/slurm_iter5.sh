#!/bin/bash
#SBATCH --account=bgia-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name=iter5
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=02:00:00
#SBATCH --output=/projects/bgia/duyan2/SURE/logs/slurm_iter5_%j.log
#SBATCH --error=/projects/bgia/duyan2/SURE/logs/slurm_iter5_%j.err

cd /projects/bgia/duyan2/SURE
PY="/projects/bgia/duyan2/SURE/sure_env/bin/python"

# Summary of iterative search:
#   Att 1: lambda_al=1.0, lambda_aux=0.1, 30ep  -> NLL=-1.93, al_frac=90% (al dominates)
#   Att 2: lambda_al=0.1, lambda_aux=0.1, 30ep  -> NLL=-1.86, al_frac=27% (ep dominates)
#   Att 3: lambda_al=0.3, lambda_aux=0.3, 30ep  -> NLL=-1.913 BEST, sigma_al~residuals ✓
#   Att 4: lambda_al=0.3, lambda_aux=0.5, 50ep  -> NLL=-1.82, sigma_al=MSE (overfit aux)
#
# Winner config: lambda_al=0.3, lambda_aux=0.3, hdim=64 (attempt 3).
# Attempt 5: same winner config, 50 epochs — more time for spatial structure to emerge.
echo "Attempt 5: lambda_al=0.3, lambda_aux=0.3, 50 epochs (winner config, more epochs)"

$PY src/train_hook.py --config-name=train_ekf_hook \
    trainer=gpu \
    trainer.max_epochs=50 \
    model.ekf_net.use_aleatoric=true \
    model.ekf_net.aleatoric_input_mode=z_only \
    model.ekf_net.aleatoric_hidden_dim=64 \
    model.ekf_net.aleatoric_n_layers=2 \
    model.ekf_net.lambda_aleatoric=0.3 \
    model.lambda_aux=0.3 \
    logger.wandb.name="aux_iter5_FINAL_lal03_laux03_e50" \
    logger.wandb.tags='["aux_regression","iter5","final","winner"]'
