#!/bin/bash
#SBATCH --account=bgia-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name=iter3
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=01:00:00
#SBATCH --output=/projects/bgia/duyan2/SURE/logs/slurm_iter3_%j.log
#SBATCH --error=/projects/bgia/duyan2/SURE/logs/slurm_iter3_%j.err

cd /projects/bgia/duyan2/SURE
PY="/projects/bgia/duyan2/SURE/sure_env/bin/python"

# Attempt 1: lambda_al=1.0  -> real_al_frac=90%, NLL=-1.93 (aleatoric dominates, good NLL)
# Attempt 2: lambda_al=0.1  -> real_al_frac=27%, NLL=-1.86 (epistemic dominates, worse NLL)
# Fix: logged sigma_al_frac was wrong (missing lambda_aleatoric) — now fixed.
# Attempt 3: lambda_al=0.3, lambda_aux=0.3, hdim=64
#   Expected real_al_frac ~ 0.3*sigma_al/(sigma_ep+0.3*sigma_al) ~ 50%
#   Stronger aux (0.3) should push sigma_al toward spatial blob structure faster.
#   Bigger head (64) gives more spatial expressiveness.
echo "Attempt 3: lambda_al=0.3, lambda_aux=0.3, hdim=64 (balanced + stronger spatial)"

$PY src/train_hook.py --config-name=train_ekf_hook \
    trainer=gpu \
    trainer.max_epochs=30 \
    model.ekf_net.use_aleatoric=true \
    model.ekf_net.aleatoric_input_mode=z_only \
    model.ekf_net.aleatoric_hidden_dim=64 \
    model.ekf_net.aleatoric_n_layers=2 \
    model.ekf_net.lambda_aleatoric=0.3 \
    model.lambda_aux=0.3 \
    logger.wandb.name="aux_iter3_lal03_laux03_h64" \
    logger.wandb.tags='["aux_regression","iter3","balanced"]'
