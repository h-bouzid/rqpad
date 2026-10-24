#!/bin/bash
# Example SLURM job: one HQ-WMCA protocol on a single GPU.
# Adapt partition / account / module lines to your cluster, then: sbatch scripts/slurm_example.sl
#SBATCH --job-name=rqpad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=%x_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
: "${HQWMCA_ROOT:?export HQWMCA_ROOT before submitting}"
# source activate rqpad

srun python train.py --train_set hqwmca --hqwmca_root "$HQWMCA_ROOT" \
  --protocol PROTOCOL-grand_test-curated.csv --num_workers "$SLURM_CPUS_PER_TASK"
