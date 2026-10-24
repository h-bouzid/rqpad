#!/usr/bin/env bash
# SiW-Mv2 intra-dataset: leave-one-attack-out, one run per unseen attack.
#
#   SIWMV2_ROOT=data/SiW-Mv2 bash scripts/siwmv2_leave_one_out.sh [extra train.py arguments]
set -euo pipefail
cd "$(dirname "$0")/.."
: "${SIWMV2_ROOT:?set SIWMV2_ROOT to the SiW-Mv2 root folder}"

ATTACKS=(
  Makeup_Cosmetic Makeup_Impersonation Makeup_Obfuscation
  Mannequin Mask_HalfMask Mask_PaperMask Mask_TransparentMask
  Paper Partial_Eye Partial_FunnyeyeGlasses Partial_Mouth Partial_PaperGlasses
  Replay Silicone
)

for attack in "${ATTACKS[@]}"; do
  python train.py --train_set siwmv2 --siwmv2_root "$SIWMV2_ROOT" \
    --unseen_attack "$attack" "$@"
done
