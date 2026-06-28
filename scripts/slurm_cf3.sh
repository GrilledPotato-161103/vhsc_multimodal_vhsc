#!/bin/bash
#SBATCH --account=bgia-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name=cf3
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=01:30:00
#SBATCH --output=/projects/bgia/duyan2/SURE/logs/slurm_cf3_%j.log
#SBATCH --error=/projects/bgia/duyan2/SURE/logs/slurm_cf3_%j.err

cd /projects/bgia/duyan2/SURE
PY="/projects/bgia/duyan2/SURE/sure_env/bin/python"

# cf2: PCC=0.879, Spearman=0.410. Pearson>>Spearman gap.
# cf3: SEPARATE capacity-limit vs noise-ceiling hypotheses.
#   Big aleatoric head (hdim=128, 4 layers) + 50 epochs.
#   If Spearman jumps -> was capacity-limited.
#   If Spearman flat ~0.41 -> data-noise ceiling (30% input perturbation is irreducible).
echo "cf3: closed_form + xy + BIG head (128,4L) + aux=0.3 + 50ep (capacity test)"

$PY src/train_hook.py --config-name=train_ekf_hook \
    trainer=gpu \
    trainer.max_epochs=50 \
    model.ekf_net.mode=closed_form \
    model.ekf_net.use_aleatoric=true \
    model.ekf_net.aleatoric_input_mode=xy \
    model.ekf_net.aleatoric_xy_dim=2 \
    model.ekf_net.aleatoric_hidden_dim=128 \
    model.ekf_net.aleatoric_n_layers=4 \
    model.ekf_net.lambda_aleatoric=1.0 \
    model.lambda_aux=0.3 \
    logger.wandb.name="cf3_closedform_xy_bighead_e50" \
    logger.wandb.tags='["closed_form","xy","cf3","capacity"]'
