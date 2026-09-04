#!/usr/bin/env bash
# ===========================================================================
#  Reproduce Table 1 of arXiv:2511.18516 (miniImageNet) end to end.
#
#    bash scripts/reproduce_table1.sh --quick          # ~35 min, 1 GPU
#    bash scripts/reproduce_table1.sh --full           # ~6 h,   1 GPU
#    bash scripts/reproduce_table1.sh --quick --gpu 3
#
#  --quick  everything except training the 102M image-space UNet from scratch
#           (an existing checkpoint is used if one is present).
#  --full   also trains the image-space UNet for --steps steps (default 40000).
#
#  Prerequisites (run once):
#      bash overview/env_setup.sh
#      bash overview/data_prepare.sh
# ===========================================================================
set -euo pipefail

MODE="quick"; GPU="0"; STEPS=40000; DATASET="mini_imagenet"; CLIP="ViT-B-16"
SKIP_RESNET=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick) MODE="quick"; shift;;
    --full)  MODE="full";  shift;;
    --gpu)   GPU="$2"; shift 2;;
    --steps) STEPS="$2"; shift 2;;
    --dataset) DATASET="$2"; shift 2;;
    --clip)  CLIP="$2"; shift 2;;
    --skip-resnet) SKIP_RESNET=1; shift;;
    -h|--help) sed -n '2,20p' "$0"; exit 0;;
    *) echo "unknown arg $1" >&2; exit 1;;
  esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"
# shellcheck disable=SC1091
source scripts/env.sh
mkdir -p results logs

P="${DATASET}_${CLIP}"                      # canonical run-tag prefix
CK="checkpoints/${DATASET}_${CLIP}_openai_llm"
CTL="checkpoints/controls"
SWEEP=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 1.0)

banner() { echo; echo "==========================================================="; \
           echo "  $*"; echo "==========================================================="; }

banner "1/9  environment + protocol self-test"
python tests/test_reproduction.py || echo "(some checks need later stages; continuing)"

banner "2/9  audit the paper against its own Table 1"
python -m cdfscil.audit_paper | tee results/paper_audit.txt

banner "3/9  cache frozen-CLIP features (Eq. 1) and text conditions (Eq. 2)"
python -m cdfscil.extract_features --dataset "${DATASET}" --clip-model "${CLIP}" \
    --gpu "${GPU}" --batch-size 512 --workers 16

banner "4/9  the floor: real prototypes only (alpha=1) + CLIP zero-shot reference"
python -m cdfscil.evaluate --dataset "${DATASET}" --clip-model "${CLIP}" \
    --alpha 1.0 --tag "${P}_realonly"

banner "5/9  controls that use NO diffusion model (text / fixed vector / random)"
python -m cdfscil.make_controls --dataset "${DATASET}" --clip-model "${CLIP}"
for c in text globalmean random; do
  python -m cdfscil.evaluate --dataset "${DATASET}" --clip-model "${CLIP}" \
      --gen-protos "${CTL}/${DATASET}_${CLIP}_openai_${c}.npz" \
      --alpha-sweep "${SWEEP[@]}" --tag "${P}_control_${c}"
done

banner "6/9  feature-space diffusion (Sec. 2.3 reading): faithful, class-name, oracle"
python -m cdfscil.feat_diffusion --dataset "${DATASET}" --clip-model "${CLIP}" \
    --text-mode llm --steps 30000 --batch-size 512 --gpu "${GPU}" --n-gen 64 --seed 1
python -m cdfscil.evaluate --dataset "${DATASET}" --clip-model "${CLIP}" \
    --gen-protos "${CK}/gen_protos_feat_n64_g1.0.npz" \
    --alpha-sweep "${SWEEP[@]}" --tag "${P}_featdiff_llm"

python -m cdfscil.feat_diffusion --dataset "${DATASET}" --clip-model "${CLIP}" \
    --text-mode classname --steps 30000 --batch-size 512 --gpu "${GPU}" --n-gen 64 --seed 1
python -m cdfscil.evaluate --dataset "${DATASET}" --clip-model "${CLIP}" \
    --gen-protos "checkpoints/${DATASET}_${CLIP}_openai_classname/gen_protos_feat_n64_g1.0.npz" \
    --alpha-sweep "${SWEEP[@]}" --tag "${P}_featdiff_classname"

# upper bound: deliberately violates FSCIL by training on all classes
python -m cdfscil.feat_diffusion --dataset "${DATASET}" --clip-model "${CLIP}" \
    --text-mode llm --oracle-all-classes --steps 30000 --batch-size 512 \
    --gpu "${GPU}" --n-gen 64 --seed 1
python -m cdfscil.evaluate --dataset "${DATASET}" --clip-model "${CLIP}" \
    --gen-protos "${CK}_ORACLE/gen_protos_feat_n64_g1.0.npz" \
    --alpha-sweep "${SWEEP[@]}" --tag "${P}_featdiff_oracle"

banner "7/9  image-space diffusion (Sec. 3.2 reading)"
if [[ "${MODE}" == "full" ]]; then
  python -m cdfscil.train_diffusion --dataset "${DATASET}" --clip-model "${CLIP}" \
      --text-mode llm --image-size 64 --base-ch 128 --ch-mult 1 2 2 4 \
      --num-res-blocks 3 --attn-res 16 8 --batch-size 256 --steps "${STEPS}" \
      --lr 1e-4 --weight-decay 5e-4 --timesteps 1000 --schedule cosine \
      --p-uncond 0.1 --ema-decay 0.9999 --workers 16 --gpu "${GPU}" --compile \
      --seed 1 --log-every 200 --ckpt-every 5000 --sample-every 10000
fi
IMG_CKPT=""
for cand in "${CK}/model_final.pt" "${CK}/model_latest.pt"; do
  [[ -f "$cand" ]] && IMG_CKPT="$cand" && break
done
if [[ -n "${IMG_CKPT}" ]]; then
  echo "using ${IMG_CKPT}"
  python -m cdfscil.generate_prototypes --ckpt "${IMG_CKPT}" \
      --dataset "${DATASET}" --clip-model "${CLIP}" --n-gen 64 \
      --ddim-steps 50 --guidance 1.0 --gpu "${GPU}" --seed 1
  IMG_NPZ=$(ls -t "${CK}"/gen_protos_img_*.npz | head -1)
  python -m cdfscil.evaluate --dataset "${DATASET}" --clip-model "${CLIP}" \
      --gen-protos "${IMG_NPZ}" --alpha-sweep "${SWEEP[@]}" \
      --tag "${P}_imgdiff_llm"
else
  echo "SKIPPED: no image-space checkpoint. Re-run with --full to train one."
fi

banner "8/9  the Sec. 4 reading: ResNet-18 trained on base classes"
if [[ "${SKIP_RESNET}" == "0" ]]; then
  if [[ ! -f "features/${DATASET}/resnet18_${DATASET}_train.npy" ]]; then
    python -m cdfscil.train_resnet --dataset "${DATASET}" --gpu "${GPU}" --workers 16
  fi
  python -m cdfscil.evaluate --dataset "${DATASET}" --clip-model resnet18 \
      --pretrained "${DATASET}" --alpha 1.0 --zero-shot-text none \
      --tag "${DATASET}_resnet18_realonly"
  python -m cdfscil.feat_diffusion --dataset "${DATASET}" --clip-model "${CLIP}" \
      --feature-tag "resnet18_${DATASET}" --text-tag "${CLIP}_openai" \
      --text-mode llm --steps 30000 --batch-size 512 --gpu "${GPU}" --n-gen 64 \
      --seed 1 --run-name "${DATASET}_resnet18_llm"
  python -m cdfscil.evaluate --dataset "${DATASET}" --clip-model resnet18 \
      --pretrained "${DATASET}" --zero-shot-text none \
      --gen-protos "checkpoints/${DATASET}_resnet18_llm/gen_protos_feat_n64_g1.0.npz" \
      --alpha-sweep "${SWEEP[@]}" --tag "${DATASET}_resnet18_featdiff_llm"
fi

banner "9/9  diagnosis, report and figures"
TAGS=("${CLIP}_openai")
[[ -f "features/${DATASET}/resnet18_${DATASET}_train.npy" ]] && TAGS+=("resnet18_${DATASET}")
python -m cdfscil.diagnose --dataset "${DATASET}" --tags "${TAGS[@]}" \
    --out "results/diagnosis_${DATASET}.csv"
python -m cdfscil.report --dataset "${DATASET}"
python -m cdfscil.plots  --dataset "${DATASET}"

echo
echo "Done.  Read, in order:"
echo "  overview/results.md"
echo "  results/table1_reproduction_${DATASET}.md"
echo "  results/paper_audit.txt"
echo "  results/figures/{sessions,alpha_sweep,base_vs_novel}.png"
