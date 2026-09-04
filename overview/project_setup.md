# Reproduction: step by step

Everything below was executed on the machine this repository lives on
(8 × NVIDIA H100 PCIe 80 GB, 256 CPU cores, Linux 6.17, driver 580.159.04,
CUDA 13.0). One GPU is enough.

## TL;DR

```bash
bash overview/env_setup.sh          # once,  ~5 min
bash overview/data_prepare.sh       # once,  ~15 min, 5.4 GB download
bash scripts/reproduce_table1.sh --quick     # ~35 min
# or, including training the 102M image-space UNet from scratch:
bash scripts/reproduce_table1.sh --full --gpu 0     # ~6 h
```

Read `results/table1_reproduction_mini_imagenet.md`, `results/paper_audit.txt`
and `overview/results.md`.

---

## 1. Environment

```bash
bash overview/env_setup.sh
source scripts/env.sh            # activate; do this in every new shell
```

Pinned versions:

| package | version |
|---|---|
| python | 3.11 |
| torch | 2.13.0 (+cu130) |
| torchvision | 0.28.0 |
| open_clip_torch | 3.3.0 |
| nvidia-cudnn-cu13 | 9.25.1.1 |
| numpy / pillow / scipy | 2.4.6 / 12.3.0 / 1.17.1 |

### Three traps worth knowing about

**(a) cuDNN sublibrary mismatch.** `torch==2.13.0` pins
`nvidia-cudnn-cu13==9.20.0.48`, whose wheel does not contain
`libcudnn_engines_tensor_ir.so.9`. On a host that also has a system cuDNN
(here 9.22 under `/usr/lib/x86_64-linux-gnu`), the loader satisfies the missing
sublibrary from the system copy and **every convolution** fails with:

```
CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failed
cudnn_status: CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
```

`env_setup.sh` installs `nvidia-cudnn-cu13==9.25.1.1`, which ships the complete
sublibrary set, and `scripts/env.sh` puts it first on `LD_LIBRARY_PATH`.
`env_setup.sh` runs a conv2d as a self-test, so this fails at setup time rather
than an hour into training.

**(b) EMA without warm-up returns pure noise.** We hit this mid-run and it is
worth recording. With `decay = 0.9999` and no warm-up, the EMA shadow still
holds `0.9999^t` of the *random initialisation* — 37 % at 10 k steps, 1.8 % at
40 k. The training loss looks perfectly healthy (0.05, a normal `L_simple`
value) while DDIM sampling from the EMA weights returns noise; sampling from the
raw weights at the same step already produces recognisable images. `EMA.decay_at`
now ramps the decay as `min(decay, (1+t)/(10+t))`, the standard ADM/diffusers
schedule. If you resume a run whose EMA was built without warm-up, pass
`--ema-reinit` to reset the shadow to the loaded weights.

**(c) CLIP QuickGELU.** OpenAI's CLIP uses QuickGELU. `open_clip` 3.x will
happily load OpenAI weights into a plain-`nn.GELU` graph and only emit a
`UserWarning` — every feature is then silently wrong. `clip_backbone.load_clip`
maps any `pretrained="openai"` request onto the explicit `-quickgelu`
architecture and promotes that warning to an exception.

## 2. Data

```bash
bash overview/data_prepare.sh          # default root: ./data
```

Downloads miniImageNet + CUB-200 (`fscil.zip`, 4.2 GB, the mirror published by
the NC-FSCIL authors) and CIFAR-100, lays them out in the CEC structure,
installs the session index lists, and then **verifies** that the downloaded
miniImageNet split CSVs are byte-identical to the CEC ones shipped in this repo.
It aborts if they are not. See `dataset.md` for why that check exists.

Final self-test output:

```
  mini_imagenet  OK  train=50000 test=10000 classes=100 sessions=9  base=60  way=5  shot=5
  cifar100       OK  train=50000 test=10000 classes=100 sessions=9  base=60  way=5  shot=5
  cub200         OK  train=5994  test=5794  classes=200 sessions=11 base=100 way=10 shot=5
```

## 3. The pipeline, stage by stage

### 3.1 Cache frozen-CLIP features (Eq. 1 and Eq. 2)

```bash
python -m cdfscil.extract_features --dataset mini_imagenet --clip-model ViT-B-16 \
       --gpu 0 --batch-size 512 --workers 16
```

≈ 55 s. Writes `features/mini_imagenet/ViT-B-16_openai_{train,test}.npy`
(50000×512 and 10000×512) and four text-condition matrices, one per prompt mode
(`classname`, `template`, `llm`, `llm+template`).

### 3.2 The floor: real prototypes only (α = 1)

```bash
python -m cdfscil.evaluate --dataset mini_imagenet --clip-model ViT-B-16 \
       --alpha 1.0 --tag mini_b16_realonly
```

This is CD-FSCIL with the generative path switched off — the number the
diffusion model has to beat for the paper's contribution to exist.

### 3.3 Feature-space diffusion (the §2.3 reading)

```bash
python -m cdfscil.feat_diffusion --dataset mini_imagenet --clip-model ViT-B-16 \
       --text-mode llm --steps 30000 --batch-size 512 --gpu 0 --n-gen 64
```

≈ 9 min (8.4 min training + 14 s sampling all 100 classes). Trains an 87.9 M
residual-MLP denoiser on base-session CLIP features only, then emits
`gen_protos_feat_n64_g1.0.npz`.

### 3.4 Image-space diffusion (the §3.2 reading)

```bash
python -m cdfscil.train_diffusion --dataset mini_imagenet --clip-model ViT-B-16 \
    --text-mode llm --image-size 64 --base-ch 128 --ch-mult 1 2 2 4 \
    --num-res-blocks 3 --attn-res 16 8 --batch-size 256 --steps 40000 \
    --lr 1e-4 --weight-decay 5e-4 --timesteps 1000 --schedule cosine \
    --p-uncond 0.1 --ema-decay 0.9999 --gpu 0 --compile --workers 16 \
    --log-every 200 --ckpt-every 5000 --sample-every 10000
```

102.3 M UNet, 64×64, batch 256, `torch.compile`, bf16.
Measured **≈ 655 images/s** on one H100 → **≈ 4.3 h** for 40 k steps
(= 10.2 M images ≈ 341 epochs over the 30 000 base images).
Checkpoints every 5 000 steps, sample grids every 10 000.

Then turn it into prototypes (Eqs. 6–8):

```bash
python -m cdfscil.generate_prototypes \
    --ckpt checkpoints/mini_imagenet_ViT-B-16_openai_llm/model_final.pt \
    --n-gen 64 --ddim-steps 50 --guidance 1.0 --gpu 0
```

### 3.5 The α sweep — the experiment that decides the paper

```bash
python -m cdfscil.evaluate --dataset mini_imagenet --clip-model ViT-B-16 \
    --gen-protos <path>.npz \
    --alpha-sweep 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 1.0 \
    --tag mini_b16_alphasweep
```

α is the weight on the **real** prototype, so α = 1 is real-only and α = 0 is
generated-only. The paper never states its value; the sweep takes 6 s because
features are cached.

### 3.6 The ResNet-18 track (the §4 reading)

```bash
bash scripts/run_resnet_track.sh --dataset mini_imagenet --gpu 0
```

≈ 20 min for 200 epochs. This is the regime every baseline in Table 1 actually
operates in, so it is the only apples-to-apples comparison available.

### 3.7 Reports

```bash
python -m cdfscil.audit_paper                       # paper vs its own Table 1
python -m cdfscil.report --dataset mini_imagenet    # side-by-side comparison
```

## 4. Choices we had to make (because the paper does not)

| Unspecified in the paper | Our default | Flag |
|---|---|---|
| CLIP checkpoint | OpenAI **ViT-B/16** (512-d, matches `x ∈ R^512`) | `--clip-model` |
| α (Eq. 11) | swept over `{0 … 1}`, nothing assumed | `--alpha`, `--alpha-sweep` |
| N (Eq. 8) | 64 generated exemplars per class | `--n-gen` |
| UNet config | base 128, mult (1,2,2,4), 3 res blocks, attn 16/8 → **102.3 M** ("around 110M") | `--base-ch --ch-mult --num-res-blocks --attn-res` |
| Diffusion resolution | 64×64 (standard DDPM/ADM image-space resolution) | `--image-size` |
| LLM + prompt for `t_c` | Claude Opus 5, 6 descriptions/class, prompt in `descriptions.GENERATION_PROMPT`, **committed** so no API key is needed | `--text-mode`, `cdfscil.generate_descriptions` |
| Guidance scale | 1.0 = plain conditional sampler (i.e. no guidance, the faithful setting) | `--guidance` |
| EMA | decay 0.9999 **with `(1+t)/(10+t)` warm-up** — without warm-up the shadow is 37 % random init at 10 k steps and samples are noise | `--ema-decay`, `--no-ema-warmup` |
| Condition dropout | 0.1, only so guidance is *available* as an ablation | `--p-uncond` |
| Prototype normalisation | L2-normalise before averaging and after fusion | `--no-prenorm`, `--no-postnorm` |

## 5. Determinism

`utils.set_seed` seeds python/numpy/torch and pins cuDNN for evaluation.
Feature extraction, prototype construction and the α sweep are deterministic:
re-running stage 3.1 + 3.2 reproduces the reported accuracies exactly. Diffusion
*training* runs with `cudnn.benchmark=True` (speed over bitwise repeatability),
so retraining gives a slightly different model; sampling from a fixed checkpoint
is seeded and reproducible.

## 6. Wall-clock summary (1 × H100)

| stage | time |
|---|---|
| environment setup | ~5 min |
| data download + extract | ~15 min |
| CLIP feature cache | 55 s |
| evaluation / α sweep | 6 s |
| feature-space diffusion (train + sample) | 9 min |
| ResNet-18 base training (200 ep) + feature cache | ~20 min |
| image-space diffusion, 40 k steps | ~4.3 h |
| generative prototypes from the image model | ~4 min |
