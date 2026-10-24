#!/usr/bin/env bash
# HQ-WMCA intra-dataset: grand test + the 8 leave-one-attack-out protocols.
#
#   HQWMCA_ROOT=data/HQ-WMCA bash scripts/hqwmca_protocols.sh [extra train.py arguments]
#
# Extra arguments are forwarded to train.py, e.g. "--variant wo_gen".
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HQWMCA_ROOT:?set HQWMCA_ROOT to the HQ-WMCA root folder}"

PROTOCOLS=(
  grand_test-curated
  LOO_Flexiblemask LOO_Glasses LOO_Makeup LOO_Mannequin
  LOO_Papermask LOO_Replay LOO_Rigidmask LOO_Tattoo
)

for protocol in "${PROTOCOLS[@]}"; do
  python train.py --train_set hqwmca --hqwmca_root "$HQWMCA_ROOT" \
    --protocol "PROTOCOL-${protocol}.csv" "$@"
done
