#!/usr/bin/env bash
# The Sec.4 ("ResNet18, following CLOSER") reading of the paper: train a
# ResNet-18 on base classes only, freeze it, then run the identical
# training-free incremental protocol on its features.
#
#   bash scripts/run_resnet_track.sh [--dataset mini_imagenet] [--gpu 0]
set -euo pipefail
DATASET="mini_imagenet"; GPU="0"; EPOCHS=200
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    *) echo "unknown arg $1" >&2; exit 1;;
  esac
done
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${REPO}"
# shellcheck disable=SC1091
source scripts/env.sh

python -m cdfscil.train_resnet --dataset "${DATASET}" --epochs "${EPOCHS}" \
    --gpu "${GPU}" --workers 16
python -m cdfscil.evaluate --dataset "${DATASET}" --clip-model "resnet18" \
    --pretrained "${DATASET}" --alpha 1.0 --zero-shot-text none \
    --tag "${DATASET}_resnet18_realonly"
