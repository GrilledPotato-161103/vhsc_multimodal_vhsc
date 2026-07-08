#!/bin/bash
#SBATCH --account=bgia-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name=cf5
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=01:00:00
#SBATCH --output=/projects/bgia/duyan2/SURE/logs/slurm_cf5_%j.log
#SBATCH --error=/projects/bgia/duyan2/SURE/logs/slurm_cf5_%j.err

cd /projects/bgia/duyan2/SURE
PY="/projects/bgia/duyan2/SURE/sure_env/bin/python"

# cf4 overturned the noise-ceiling hypothesis: smoothing error didn't lift PCC.
# The low Spearman (0.41) with high Pearson (0.87) = heavy-tailed target problem.
# Squared-error targets are chi-square heavy-tailed; loss is OOD-tail-dominated;
# bulk ID structure underfit -> low rank alignment.
# cf5: LOG-SPACE aux loss MSE(log sigma_total, log err). Scale-invariant ->
# equalizes gradient across error magnitudes -> should lift Spearman.
echo "cf5: closed_form + xy + LOG-space aux (lambda_aux=0.5)"

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
    model.lambda_aux=0.5 \
    model.aux_mode=log \
    logger.wandb.name="cf5_logaux" \
    logger.wandb.tags='["closed_form","xy","cf5","log_aux"]'
