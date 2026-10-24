#!/usr/bin/env bash
# Ablation study on HQ-WMCA: every reduced variant on all 9 protocols.
# The full model is trained by scripts/hqwmca_protocols.sh.
#
#   HQWMCA_ROOT=data/HQ-WMCA bash scripts/ablation.sh [extra train.py arguments]
set -euo pipefail
cd "$(dirname "$0")/.."

for variant in wo_gen wo_fiqa rgb_only; do
  bash scripts/hqwmca_protocols.sh --variant "$variant" "$@"
done
