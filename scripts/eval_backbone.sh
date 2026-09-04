#!/usr/bin/env bash
# Full CD-FSCIL evaluation on an already-cached backbone feature set.
#   bash scripts/eval_backbone.sh <feature-tag> [gpu]
set -euo pipefail
TAG="$1"; GPU="${2:-0}"; DS=mini_imagenet; CLIP=ViT-B-16
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
# shellcheck disable=SC1091
source scripts/env.sh
SW="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.85 0.9 0.95 1.0"

python -m cdfscil.evaluate --dataset $DS --feature-tag "$TAG" --alpha 1.0 \
    --zero-shot-text none --tag "${DS}_${TAG}_realonly"

python -m cdfscil.feat_diffusion --dataset $DS --clip-model $CLIP \
    --feature-tag "$TAG" --text-tag "${CLIP}_openai" --text-mode llm \
    --steps 30000 --batch-size 512 --gpu "$GPU" --n-gen 64 --seed 1 \
    --run-name "${DS}_${TAG}_llm"

python -m cdfscil.evaluate --dataset $DS --feature-tag "$TAG" --zero-shot-text none \
    --gen-protos "checkpoints/${DS}_${TAG}_llm/gen_protos_feat_n64_g1.0.npz" \
    --alpha-sweep $SW --tag "${DS}_${TAG}_featdiff_llm"

python -m cdfscil.diagnose --dataset $DS --tags "$TAG" --out "results/diagnosis_${TAG}.csv"
echo "EVAL_DONE_${TAG}"
