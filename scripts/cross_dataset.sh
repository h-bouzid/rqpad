#!/usr/bin/env bash
# Cross-dataset: train on one dataset (all videos), test on every video of the other.
#
#   HQWMCA_ROOT=data/HQ-WMCA SIWMV2_ROOT=data/SiW-Mv2 bash scripts/cross_dataset.sh [extra train.py arguments]
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HQWMCA_ROOT:?set HQWMCA_ROOT to the HQ-WMCA root folder}"
: "${SIWMV2_ROOT:?set SIWMV2_ROOT to the SiW-Mv2 root folder}"

python train.py --train_set hqwmca --test_set siwmv2 \
  --hqwmca_root "$HQWMCA_ROOT" --siwmv2_root "$SIWMV2_ROOT" "$@"

python train.py --train_set siwmv2 --test_set hqwmca \
  --hqwmca_root "$HQWMCA_ROOT" --siwmv2_root "$SIWMV2_ROOT" "$@"
