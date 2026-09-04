# Operation log — what was done, in order

Chronological record of every operation, why it was done, and what it changed.
Times are from the run logs. Machine: 8 × H100 PCIe, one GPU per job.

---

## Phase 0 — situation assessment (19:48–19:55)

| # | Operation | Result |
|---|---|---|
| 0.1 | Read the existing repo: `train.py`, `models/my_vit/`, `dataloader/`, `OGDiff-code/` | `complementary/final_results_diffusion.txt` shows the existing pipeline produced CIFAR-100 session 0 = **19.90 %**, sessions 1–8 = **0.00 %**. Not in working order. |
| 0.2 | Checked hardware / network | 8×H100 (shared), 256 cores, 3.1 TB free; PyPI + HuggingFace + GitHub reachable. |
| 0.3 | Decided to write `cdfscil/` from scratch against the paper's Section 3 rather than repair the old code | The old code could not be made to match the paper's equations. |

## Phase 1 — environment (19:52–20:12)

| # | Operation | Result |
|---|---|---|
| 1.1 | `conda create -n FSCIL_env python=3.11`; install torch 2.13.0, torchvision 0.28.0, open_clip_torch 3.3.0 | env built |
| 1.2 | **Bug found:** every conv2d died with `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` | Diagnosed with `LD_DEBUG=libs`: torch pins `nvidia-cudnn-cu13==9.20.0.48`, whose wheel has **no `libcudnn_engines_tensor_ir.so.9`**; the loader silently filled it from the host's system cuDNN 9.22. |
| 1.3 | **Fix:** `pip install -U nvidia-cudnn-cu13==9.25.1.1` (complete sublibrary set) + prepend it to `LD_LIBRARY_PATH` in `scripts/env.sh` | conv2d works; a conv is now part of `env_setup.sh`'s self-test so this fails at setup, not an hour into training |
| 1.4 | **Bug found:** `open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")` emits only a `UserWarning` about QuickGELU | OpenAI CLIP uses QuickGELU; loading its weights into a plain-GELU graph silently corrupts every feature. |
| 1.5 | **Fix:** `clip_backbone._resolve_arch` maps any `pretrained="openai"` onto the explicit `-quickgelu` architecture, and the warning is promoted to an exception | features are now the real OpenAI CLIP features |

## Phase 2 — data (19:52–20:10)

| # | Operation | Result |
|---|---|---|
| 2.1 | Tried the miniImageNet link in the old README (Drive id `16V_...`) | It downloads `mini-imagenet.tar.gz` = **`mini-imagenet-cache-{train,val,test}.pkl`**, the Ravi & Larochelle *few-shot cache*. Wrong dataset — cannot reproduce Table 1. |
| 2.2 | Found the CEC/TOPIC FSCIL release published by the NC-FSCIL authors: `HarborYuan/Few-Shot-Class-Incremental-Learning` → `fscil.zip` (4.2 GB) | miniImageNet + CUB-200 in the exact CEC layout |
| 2.3 | **Verification:** `cmp` the downloaded `miniimagenet/split/{train,test}.csv` against `complementary/index_list/mini_imagenet/{train,test}.csv` | **byte-identical** (50 001 / 10 001 lines). `data_prepare.sh` now aborts if they ever differ. |
| 2.4 | Downloaded CIFAR-100 from cs.toronto.edu | ok |
| 2.5 | Wrote `cdfscil/data.py` with `sanity_check()` asserting the whole protocol | All three benchmarks pass; and session 0, built independently as "all base-class train images", was verified to be the **same set** as the shipped `session_1.txt` for all three datasets. |

## Phase 3 — implementing the method (20:00–21:00)

| # | Operation | What it implements |
|---|---|---|
| 3.1 | `clip_backbone.py` | Eq. 1/2, frozen CLIP + feature cache; `assert_frozen` rejects a trainable or `train()`-mode encoder |
| 3.2 | `build_classnames.py` → `assets/mini_imagenet_classnames.json` | wnid → class name, cross-referenced from the official ImageNet class index (asserted to be sorted-wnid order) and `open_clip.IMAGENET_CLASSNAMES`. Nothing hand-typed. |
| 3.3 | `descriptions.py` + 600 LLM-written descriptions for miniImageNet (later 600 more for CIFAR-100) | the §1.2 "multimodal semantic prior"; committed so a run needs no API key |
| 3.4 | `unet.py` — conditional UNet, 102.3 M params | Fig. 1's denoiser; `φ(p_c)` added to the timestep embedding, AdaGN scale/shift in every ResBlock |
| 3.5 | **Bug found:** 371 img/s and 33 GiB peak at batch 128 | `q,k,v` were non-contiguous, so `scaled_dot_product_attention` fell back to the math backend and materialised the full `(B,heads,HW,HW)` score matrix |
| 3.6 | **Fix:** `.contiguous()` on q/k/v, plus `torch.compile` | **899 img/s, 10.5 GiB** — 2.4× faster, 3× less memory, identical maths |
| 3.7 | `diffusion.py` | Eq. 3 forward, Eq. 5 loss, Eq. 4 DDIM sampler (generalised to work on both images and 512-d vectors) |
| 3.8 | `fscil.py` | Eqs. 8/10/11/12. Imports **only numpy** — a test parses its import graph to keep the incremental stage framework-free |
| 3.9 | `feat_diffusion.py` | the §2.3 reading (diffusion in CLIP feature space), since §2.3 and §3.2 specify different algorithms |
| 3.10 | `resnet_backbone.py` + `train_resnet.py` | the §4 reading ("ResNet18, following CLOSER") |

## Phase 4 — first results and the two regimes (20:12–21:20)

| # | Operation | Result |
|---|---|---|
| 4.1 | Cached CLIP ViT-B/16 features for all 60 000 miniImageNet images (55 s) | |
| 4.2 | Evaluated α = 1 (real prototypes only) | **92.62 → 87.26, avg 89.19** — far *above* the paper's 84.85 → 60.13 |
| 4.3 | CLIP zero-shot text classifier, no FSCIL at all | **94.82 → 91.84** |
| 4.4 | Trained ResNet-18 base session (200 ep) and evaluated | **72.05 → 44.90, avg 56.42** — far *below* the paper |
| 4.5 | Cross-checked the evaluation against a second, independently written implementation | exact agreement on all 9 sessions |
| 4.6 | `diagnose.py`: novel-only accuracy and novel→base misroute rate | ResNet-18: novel-only 25.8 %, misroute 81 %. CLIP ViT-B/16: 90.4 % / 14.8 %. The FSCIL problem is largely *pre-solved* by CLIP on miniImageNet. |

## Phase 5 — is the diffusion doing anything? (20:29–23:52)

| # | Operation | Result |
|---|---|---|
| 5.1 | Trained the feature-space diffusion on base classes only, swept α | best **+0.24** avg over switching the generative path off |
| 5.2 | Same, conditioned on bare class names instead of LLM descriptions | **+0.27** — the LLM prior makes no difference to the fused result |
| 5.3 | **Oracle**: trained the diffusion on *all 100 classes* (deliberately violating FSCIL) to bound the ceiling | **+0.10** |
| 5.4 | **Control**: fused with one fixed vector, identical for every class, zero class information | **+0.16** |
| 5.5 | **Control**: random vector per class | +0.00 (α = 1 optimal) |
| 5.6 | **Control**: frozen CLIP *text* embedding (free, no model, no sampling) | **+4.29** |
| 5.7 | **Control**: TEEN-style calibration (paper's ref [13]) | +0.22 |
| 5.8 | α_base = 1 (only novel prototypes fused — where shrinkage should pay off) | diffusion **+0.00**; fixed vector **+0.39** |
| 5.9 | Swept N (Eq. 8) over 4/16/64/256 | the generative prototype alone improves 51 → 64, but the fused result saturates at N ≈ 64 |
| 5.10 | Repeated on CIFAR-100 and CUB-200 | diffusion **+0.81** and **+0.84** vs controls +0.14 and +0.00 — here it *does* carry real class information |

## Phase 6 — image-space diffusion (20:26 → ongoing)

| # | Operation | Result |
|---|---|---|
| 6.1 | Trained the 102.3 M UNet at 64×64, batch 256, `torch.compile`, bf16 | ~650 img/s |
| 6.2 | **Bug found at 10 k steps:** loss healthy (0.05) but DDIM samples were pure noise | EMA had **no warm-up**: at decay 0.9999 the shadow still held `0.9999^10000` = **37 %** of the random initialisation. Sampling from the raw weights at the same step already gave recognisable images. |
| 6.3 | **Fix:** `EMA.decay_at = min(decay, (1+t)/(10+t))` (ADM/diffusers schedule) + `--ema-reinit`; resumed from the 10 k checkpoint | samples at 15 k and 20 k are recognisable birds/objects |

## Phase 7 — closing the gap to Table 1 (23:45 → ongoing)

Motivated by a specific observation: the Table 1 rows that reach ~84 at session 0
(NC-FSCIL 84.02, Tri-WE 84.13) do **not** use a torchvision ResNet-18. They use
the few-shot-standard **ResNet-12** and stronger base-session recipes. Our 72.05
was a correct *ResNet-18* number (CEC's own is 72.00) but the wrong architecture
for that accuracy band.

| # | Operation |
|---|---|
| 7.1 | Implemented **ResNet-12** (64-160-320-640, 3×conv3×3 blocks, LeakyReLU 0.1, 12.4 M params) — the standard few-shot backbone |
| 7.2 | Added rotation "fantasy" virtual classes (SAVC / S3C style): the base head predicts `class × rotation` = 240 labels; base session only, encoder still frozen afterwards |
| 7.3 | Added cosine LR schedule and label smoothing |
| 7.4 | Launched three configs in parallel: ResNet-12/300 ep, ResNet-12+rotation/120 ep, ResNet-18+rotation/200 ep |

**Important:** raising session-0 this way means using a *stronger backbone than
CLOSER's*, which the paper says it follows. See `results.md` for the honest
accounting of what that does and does not demonstrate.

## Phase 7 results (00:14–01:20)

| # | Operation | Base acc | FSCIL s0 / last / avg |
|---|---|---|---|
| 7.5 | ResNet-18 + rotation, cosine LR, ls 0.1, 200 ep | 73.72 | 74.10 / 44.71 / 57.30 |
| 7.6 | + flip-TTA (`recache_features --flip-tta`) | — | 75.57 / 46.04 / 58.67 |
| 7.7 | **ResNet-12**, cosine LR, ls 0.1, 300 ep | **85.20** | 84.67 / 53.62 / 66.74 |
| 7.8 | + flip-TTA | — | 85.12 / 54.26 / 67.29 |
| 7.9 | **ResNet-12 + rotation**, 120 ep | **85.63** | 85.73 / 56.23 / 68.62 |
| 7.10 | + flip-TTA | — | 86.73 / 57.78 / 69.82 |
| 7.11 | + flip **and** rotation TTA | — | 86.20 / 58.19 / 69.88 |
| 7.12 | + CD-FSCIL Eq. 11 fusion (feature diffusion, α = 0.8) | — | **86.15 / 58.72 / 70.14** |
| | *paper Table 1* | | *84.85 / 60.13 / 71.07* |

Average-accuracy gap closed from **−14.65 to −0.93**; session 0 matched (86.15
vs 84.85). Attribution: ResNet-12 +9.44, rotation +2.21, TTA +1.81, cosine
LR/label smoothing +0.88, **diffusion fusion +0.26** (total +13.72).

| # | Operation | Outcome |
|---|---|---|
| 7.13 | Added `cdfscil/recache_features.py` (flip / rotation TTA on a frozen encoder — encoder forward only, no test labels, no transduction) | +0.55 to +1.37 avg per backbone |
| 7.14 | Added the TEEN-style calibration control (`make_controls --> teen`, the paper's ref [13]) | +0.33 on the best backbone, i.e. on par with the diffusion's +0.26 |
| 7.15 | **Tried and rejected:** leave-one-out prototype calibration to fix base-class bias | Mean LOO self-similarity is 0.9699 (base) vs 0.3995 (novel). That 2.4× ratio is real cluster tightness, not estimation noise; dividing it out drops avg 67.29 → 62.37. The residual last-session gap is a calibration problem the method has no mechanism for. |
| 7.16 | Wrote `scripts/reproduce_best.sh` | one command for the best configuration, with the caveat in its header |

## Phase 6 completion (01:03–01:20)

| # | Operation | Result |
|---|---|---|
| 6.4 | Image-space UNet finished 40 k steps (3.40 h wall clock, ~635 img/s) | samples at 40 k are recognisable birds / objects; grids under `checkpoints/.../samples/` |
| 6.5 | Generated 64 exemplars × 100 classes (DDIM 50 steps), re-encoded with frozen CLIP, swept α | best **+0.25** avg over no diffusion — matching the feature-space variant (+0.24) and barely above the content-free constant control (+0.16). The finding holds with the fully-trained image-space model the paper actually describes. |

| 6.6 | Swept classifier-free guidance (1.0 / 2.0 / 3.0) on the fully-trained image model | guidance lifts the generated prototype alone by **+13.3** (47.4 → 60.7) but the fused result by **+0.05** (+0.25 → +0.30). Independent confirmation that the bottleneck is not conditioning strength. |

## Phase 8 — the 300-epoch rotation run (01:00–04:40)

Launched because the 120-epoch rotation run was clearly cut short by the cosine
schedule (the no-rotation ResNet-12 gained from 120 → 300 epochs).

| # | Operation | Base acc | FSCIL s0 / last / avg |
|---|---|---|---|
| 8.1 | ResNet-12 + rotation, **300 ep** cosine, ls 0.1 (3.23 h) | **85.95** | 85.55 / 54.89 / 67.73 |
| 8.2 | + flip-TTA | — | 86.70 / 56.43 / 69.06 |
| 8.3 | + flip **and** rotation TTA | — | 87.63 / 58.33 / 70.61 |
| 8.4 | + CD-FSCIL Eq. 11 fusion (feature diffusion, α = 0.7) | — | **87.58 / 58.56 / 70.76** |
| 8.5 | + TEEN control instead of the diffusion (α = 0.8) | — | 87.63 / 58.62 / 70.72 |
| | *paper Table 1* | | *84.85 / 60.13 / 71.07* |

Average-accuracy gap closed to **−0.31**; session 0 exceeded by +2.73.
Attribution along a → d → f → h → i: ResNet-12 (+ cosine LR/ls) **+10.32**,
rotation virtual classes +0.99, flip/rot TTA +2.88, **diffusion fusion +0.15**
(total +14.34).

Non-obvious observation worth recording: the 120-epoch rotation model is *better*
than the 300-epoch one without TTA (68.62 vs 67.73) and *worse* with full TTA
(69.88 vs 70.61). Longer rotation training makes the encoder more
rotation-specialised, so rotation TTA recovers more from it.

| # | Operation | Outcome |
|---|---|---|
| 8.6 | Rewrote `overview/results.md` §2bis (ten-configuration ladder, attribution, caveats), `README.md` Bottom line, `scripts/reproduce_best.sh` (now 300 ep, updated expected numbers) | — |
| 8.7 | **Found and fixed a documentation drift:** the CIFAR-100 feature-diffusion row had gone stale (avg 68.94 / Δ+0.81 in the docs vs 68.87 / Δ+0.74 in `results/`) because `reproduce_table1.sh --dataset cifar100` re-ran and overwrote that experiment during script validation | 4 places corrected in `results.md` + `README.md` |
| 8.8 | Added **`tests/test_doc_numbers.py`** — a manifest of every headline number quoted in the docs, checked against `results/*.json`, so this drift class fails loudly instead of surviving a review | **32/32 numbers verified** |

## Final state (04:40)

```
tests/test_reproduction.py   9/9 passed     (protocol, splits, freezing,
                                             independent-eval cross-check, audit)
tests/test_doc_numbers.py    2/2 passed     (32 documented numbers vs results/)
cdfscil.audit_paper          8 inconsistencies in the paper's own Tables 1 and 2
scripts/reproduce_table1.sh  9/9 stages, validated end to end on CIFAR-100
```
