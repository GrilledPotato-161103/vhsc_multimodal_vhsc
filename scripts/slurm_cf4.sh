#!/bin/bash
#SBATCH --account=bgia-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name=cf4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=01:00:00
#SBATCH --output=/projects/bgia/duyan2/SURE/logs/slurm_cf4_%j.log
#SBATCH --error=/projects/bgia/duyan2/SURE/logs/slurm_cf4_%j.err

cd /projects/bgia/duyan2/SURE
PY="/projects/bgia/duyan2/SURE/sure_env/bin/python"

# cf3 confirmed noise ceiling (big head didn't lift Spearman).
# cf4: best config (cf2: hdim64, aux0.3) + SMOOTHED-error PCC eval.
# If PCC(predvar, E[err|x]) >> PCC(predvar, raw err), the per-sample gap is
# provably irreducible data noise, not model failure. THE key validation plot.
echo "cf4: best config + smoothed-error ceiling test"

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
    logger.wandb.name="cf4_smoothceiling" \
    logger.wandb.tags='["closed_form","xy","cf4","ceiling"]'
