# Reproduction: step by step
Derive Table 1 results from paper

computed results: `	87.63, 81.75, 77.04, 72.89, 69.41, 65.81, 62.52, 60.07, 58.33`, which is comparable with the paper results

## summarization

```bash
bash scripts/env_setup.sh          # once,  ~5 min
bash scripts/data_prepare.sh       # once,  ~15 min, 5.4 GB download
```


## 1. Environment(5 min)
Set up the conda environment
```bash
bash scripts/env_setup.sh
source scripts/env.sh            # activate; do this in every new shell
# switch gpu to the actual available GPU index, e.g., 0, 1, 2, 3
python -m cdfscil.extract_features --dataset mini_imagenet --clip-model ViT-B-16 \
    --gpu 0 --batch-size 512 --workers 16 --prompt-modes llm

python -m cdfscil.recache_features --ckpt checkpoints/r12_rot_e300/model_final.pt \
    --tag r12_rot_e300_rtta --flip-tta --rot-tta --gpu 0 --workers 16

bash scripts/eval_backbone.sh r12_rot_e300_rtta 0

```


## 2. Data Preparation(20 minutes)
Downloads `miniImageNet`, `CUB-200` and `CIFAR-100` dataset

```bash
bash scripts/data_prepare.sh          # default root: ./data
```

Final self-test output:

```
  mini_imagenet  OK  train=50000 test=10000 classes=100 sessions=9  base=60  way=5  shot=5
  cifar100       OK  train=50000 test=10000 classes=100 sessions=9  base=60  way=5  shot=5
  cub200         OK  train=5994  test=5794  classes=200 sessions=11 base=100 way=10 shot=5
```

## 3. The pipeline, stage by stage

### 3.1 Cache frozen-CLIP features (Eq. 1 and Eq. 2)
extract features with ViT-B/16 OpenAI CLIP model, and save them to `features/mini_imagenet/ViT-B-16_openai_{train,test,text_llm}.npy`

```bash
# change to the actual available GPU index
python -m cdfscil.extract_features --dataset mini_imagenet --clip-model ViT-B-16 \
    --gpu 2 --batch-size 512 --workers 16 --prompt-modes llm
```


### 3.2 Cache ResNet-12 features with test-time augmentation (1-3 min)


```bash
python -m cdfscil.recache_features --ckpt checkpoints/r12_rot_e300/model_final.pt \
    --tag r12_rot_e300_rtta --flip-tta --rot-tta --gpu 2 --workers 16
```

Expected log:

```
r12_rot_e300_rtta: arch=resnet12 size=84 flip_tta=True rot_tta=True
  train: (50000, 640) -> features/mini_imagenet/r12_rot_e300_rtta_train.npy
  test: (10000, 640) -> features/mini_imagenet/r12_rot_e300_rtta_test.npy
```

## 4. Evaluate: prototype floor, feature diffusion, alpha sweep (~12 min)

```bash
bash scripts/eval_backbone.sh r12_rot_e300_rtta 2
```


```bash
grep -E "^config|^alpha=1" results/mini_imagenet_r12_rot_e300_rtt
a_featdiff_llm.csv
```

