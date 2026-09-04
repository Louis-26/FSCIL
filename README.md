# CD-FSCIL — a reproduction of *Breaking Forgetting* (arXiv:2511.18516)

A from-scratch, runnable implementation of **CD-FSCIL** and a reproduction
attempt of **Table 1** (miniImageNet) from

> *Breaking Forgetting: Training-Free Few-Shot Class-Incremental Learning via
> Conditional Diffusion* — Kang, Qian, Lu. arXiv:2511.18516v1, 23 Nov 2025.

Everything is measured on this machine; no number below is copied from the paper.

## Bottom line

**Table 1 is essentially reachable — but only by changing the one thing the
paper says it does not change.**

| miniImageNet, same protocol | session 0 | last | avg |
|---|---|---|---|
| **paper, Table 1: CD-FSCIL** | 84.85 | 60.13 | 71.07 *(printed as 72.53)* |
| ours, §4 as written (ResNet-18 "following CLOSER") | 72.05 | 44.90 | 56.42 |
| **ours, best effort** (ResNet-12 + rotation virtual classes + TTA + Eq. 11 fusion) | **87.58** | **58.56** | **70.76** |
| ours, §3 as written (frozen CLIP ViT-B/16) | 92.62 | 87.26 | 89.19 |

We closed the average-accuracy gap from **−14.65 to −0.31** and exceeded session 0
(87.58 vs 84.85). Ten-configuration ladder in
[`overview/results.md` §2bis](overview/results.md). Two caveats stated up front:
70.76 is the argmax over those ten configurations on a benchmark with no
validation split (the top six span 69.06–70.76, so "≈70–71" is the supportable
claim), and each is a single seed.

The attribution is what matters:

| where the +14.34 average points came from | Δavg |
|---|---|
| **ResNet-12 instead of ResNet-18** (bundled with cosine LR + label smoothing) | **+10.32** |
| rotation "fantasy" virtual classes in the base session | +0.99 |
| flip / rotation test-time augmentation | +2.88 |
| **the paper's actual contribution — the conditional diffusion fusion** | **+0.15** |

Two structural points follow, and neither depends on tuning:

1. **§4's stated setup cannot produce 84.85.** CD-FSCIL freezes everything after
   the base session, and its generative path cannot help at session 0 (base
   classes have 500 real images each), so **its session-0 accuracy *is* its
   backbone's base accuracy**. §4 says the backbone follows CLOSER; CLOSER's own
   session 0 in the very same table is **76.02**. Reaching 84.85 requires a
   stronger backbone than the paper claims to use.
2. **The diffusion model is worth a few tenths, not fourteen points.** On the
   best backbone it adds **+0.15**, and a TEEN-style calibration control fills
   the same slot for +0.11. In the CLIP regime its +0.24 is matched by fusing
   with *a single fixed vector carrying zero class information* (+0.16) and not
   beaten by an oracle diffusion that has seen every novel class (+0.10).
   Trained the full 40 k steps in image space exactly as §3.2 describes it gives
   **+0.25**; adding classifier-free guidance lifts the generated prototype *on
   its own* by 13.3 points but moves the fused result by 0.05, to **+0.30**. On
   CIFAR-100 and CUB-200 it does carry genuine class information (+0.74, +0.84),
   but a one-line CLIP-text calibration — the "linear or heuristic" family §2.2
   dismisses — beats it by 2× to 18× everywhere, for zero compute.

**Table 1's average column is arithmetically inconsistent for exactly one row.**
CD-FSCIL prints 72.53 where the mean of its own nine session accuracies is
71.07; the other 20 rows agree to 0.01. `python -m cdfscil.audit_paper` reports
8 such inconsistencies across Tables 1 and 2 in two seconds.

Full detail: [`overview/results.md`](overview/results.md) and
[`overview/paper_discrepancies.md`](overview/paper_discrepancies.md).

---

## Quick start

```bash
bash overview/env_setup.sh                    # conda env + pinned deps + self-test
bash overview/data_prepare.sh                 # datasets in CEC layout + protocol check
bash scripts/reproduce_table1.sh --quick      # ~35 min on one GPU
python tests/test_reproduction.py             # 9 self-checks
```

Full run, including training the 102 M image-space diffusion UNet from scratch:

```bash
bash scripts/reproduce_table1.sh --full --gpu 0        # ~6 h on one H100
```

## What you get

| Output | Content |
|---|---|
| `results/table1_reproduction_mini_imagenet.md` | the paper's Table 1 and this reproduction, side by side |
| `results/paper_audit.txt` | the paper checked against its own Tables 1 and 2 |
| `overview/results.md` | measured numbers and what they mean |
| `overview/paper_discrepancies.md` | every inconsistency that had to be resolved to run anything |

## Documentation

Start at [`overview/README.md`](overview/README.md).

| File | |
|---|---|
| [`overview/methodology.md`](overview/methodology.md) | the method, equation by equation |
| [`overview/model_architecture.md`](overview/model_architecture.md) | notation, modules, trainable parameters |
| [`overview/loss.md`](overview/loss.md) | the single loss, and where it is used |
| [`overview/dataset.md`](overview/dataset.md) | which miniImageNet — and why it matters |
| [`overview/eval_metric.md`](overview/eval_metric.md) | the FSCIL protocol and metrics |
| [`overview/project_setup.md`](overview/project_setup.md) | step-by-step reproduction |
| [`overview/project_architecture.md`](overview/project_architecture.md) | repository layout |
| [`overview/output.md`](overview/output.md) | output formats |
| [`overview/results.md`](overview/results.md) | **the measured results** |
| [`overview/paper_discrepancies.md`](overview/paper_discrepancies.md) | **the audit** |

## Implementation

`cdfscil/` is written from scratch against the paper's Section 3. It implements
all three readings the paper supports, because §2.3, §3 and §4 specify mutually
incompatible things (see the audit):

* **image-space** conditional diffusion (§3.2, Fig. 1) — `unet.py`, `train_diffusion.py`
* **feature-space** conditional diffusion (§2.3) — `feat_diffusion.py`
* **ResNet-18 backbone** (§4, "following CLOSER") — `resnet_backbone.py`, `train_resnet.py`

plus the frozen-CLIP encoder, the LLM text conditions, the prototype fusion of
Eq. 11 and the cosine classifier of Eq. 12.

The pre-existing `train.py`, `models/`, `dataloader/` and `OGDiff-code/` are left
in place for provenance but are not used; `complementary/final_results_diffusion.txt`
records that that pipeline produced 19.90 % at session 0 and 0.00 % thereafter.

## Provenance of the datasets

`overview/data_prepare.sh` downloads the CEC/TOPIC FSCIL release and **verifies**
that its miniImageNet split CSVs are byte-identical to the ones committed under
`complementary/index_list/`, aborting if not. The miniImageNet link in the
previous version of this README pointed at the Ravi & Larochelle few-shot cache,
which is a different dataset and cannot reproduce Table 1.


## Project links (from the previous README)

* Yuque notes: https://www.yuque.com/kanghaidong/ctruy3/qski7fh1toekqqk7/edit
* Google Doc: https://docs.google.com/document/d/1VFSoz94z6_FsUrWT3cM1vB6_WJzbFlSrRPl-ZYNraxI/edit?usp=sharing
* Google Slides: https://docs.google.com/presentation/d/1j9bRXTQGni-2a8GKquyR_TUGlMFpABAPYpuJK5Iq6Yc/edit?usp=sharing
* Overleaf (Yi Lu part): https://www.overleaf.com/read/rwkqmwqdgzxx#35c90c
* Overleaf (final): https://www.overleaf.com/5445663713xzmxjpbhvxsx#a9eccc
* arXiv: https://arxiv.org/abs/2511.18516

## Licence

Code: see `LICENSE`. Datasets are downloaded from their original hosts and are
research-use only; none is redistributed here.
