# Results

All numbers below were measured on this machine with the pipeline in this
repository. Nothing is copied from the paper except the rows explicitly labelled
"paper". Regenerate everything with:

```bash
bash scripts/reproduce_table1.sh --full
python -m cdfscil.report --dataset mini_imagenet
python -m cdfscil.plots  --dataset mini_imagenet
```

Machine: 1 × NVIDIA H100 PCIe 80 GB. Protocol: the standard CEC/TOPIC
miniImageNet FSCIL split (60 base + 8 × 5-way 5-shot; top-1 over all seen
classes), verified byte-identical to the CEC release and asserted at every run.

The tables below are transcribed from `results/*.json`; the machine-generated
version that cannot drift from the data is
[`results/table1_reproduction_mini_imagenet.md`](../results/table1_reproduction_mini_imagenet.md),
produced by `python -m cdfscil.report`. Where they ever disagree, believe that one.

---

## 1. Headline

| configuration | s0 | last | avg |
|---|---|---|---|
| **paper, Table 1: CD-FSCIL** | 84.85 | 60.13 | **71.07** (printed as 72.53) |
| paper, Table 1: Tri-WE (previous best) | 84.13 | 60.13 | 70.62 |
| paper, Table 1: CEC | 72.00 | 47.63 | 57.75 |
| | | | |
| **§4 reading — ResNet-18 trained on base classes** | | | |
| ResNet-18 + real prototypes (α = 1) | 72.05 | 44.90 | 56.42 |
| ResNet-18 + diffusion fusion (best α = 0.6) | 72.17 | 45.36 | 56.77 |
| | | | |
| **§3 reading — frozen CLIP ViT-B/16** | | | |
| CLIP + real prototypes (α = 1) | 92.62 | 87.26 | 89.19 |
| CLIP + feature diffusion, LLM conditions (α = 0.8) | 92.15 | 87.96 | 89.43 |
| CLIP + feature diffusion, class-name conditions (α = 0.85) | 92.35 | 87.90 | 89.46 |
| CLIP + **oracle** diffusion, trained on all 100 classes (α = 0.95) | 92.57 | 87.53 | 89.29 |
| | | | |
| **controls that use no diffusion model at all** | | | |
| one fixed vector, identical for every class (α = 0.8) | 92.60 | 87.60 | 89.35 |
| random vector per class (α = 1 is optimal) | 92.62 | 87.26 | 89.19 |
| **CLIP text prototype** (α = 0.2) | 94.85 | 92.62 | **93.48** |
| CLIP zero-shot text only — not FSCIL at all | 94.82 | 91.84 | 93.04 |

For the three `paper` rows the `avg` column is *recomputed* as the mean of that
row's nine session accuracies; only CD-FSCIL's differs from what the PDF prints
(§2.1).

![sessions](../results/figures/sessions.png)

---

## 2. Was Table 1 reproduced?

**No — and it cannot be, under either reading of the paper.** The claimed row
(84.85 → 60.13) sits between the two regimes the paper describes, in neither of
them:

* Under the **§4 reading** (ResNet-18 trained on the 60 base classes, "following
  CLOSER"), our faithful implementation reaches **72.05 → 44.90**. Over three
  seeds this is **71.86 ± 0.17 → 44.99 ± 0.08**, avg **56.38 ± 0.05** — the run
  is extremely stable, so the 15.1-point gap to the claimed 60.13 is not seed
  noise. Session 0 matches the published ResNet-18 baselines almost exactly — CEC reports 72.00,
  MetaFSCIL 72.04, ours 72.05 — so the base training is calibrated correctly.
  Getting to 84.85 at session 0 in this regime would require a base-session
  accuracy no ResNet-18 baseline in Table 1 achieves without additional
  representation-learning machinery that CD-FSCIL does not have (the methods
  that do reach ~84 — NC-FSCIL, OrCo, Tri-WE — each contribute a specific base
  training objective; CD-FSCIL contributes only the prototype fusion of Eq. 11).

* Under the **§3 reading** (frozen CLIP, as Eqs. 1–2 state), we reach
  **92.62 → 87.26**, which *overshoots* the claimed row by 7.8 points at session
  0 and **27.1 points at the last session**. A CLIP-based method cannot land on
  84.85 → 60.13 either: it is far too strong.

So the target numbers are simultaneously too high for the ResNet regime and far
too low for the CLIP regime. The reproduction gap is not a tuning gap.

### 2.1 The Avg column

Independently of any reproduction, the CD-FSCIL row's printed average
(**72.53**) is not the mean of its own nine session accuracies (**71.07**).
All 20 other rows agree with their printed averages to within 0.01. The claimed
"+1.91% gain on the overall average accuracy" over Tri-WE becomes **+0.45**, and
at the final session the two methods are **tied at 60.13**. Run
`python -m cdfscil.audit_paper` to check this. Details in
`paper_discrepancies.md`.

---

## 2bis. Closing the gap to Table 1 — what it takes, and what it proves

The §4 result above (72.05 → 44.90) is a *correct ResNet-18 number* — CEC's own
session 0 is 72.00 — but ResNet-18 is not the architecture the ~84 % rows of
Table 1 use. NC-FSCIL (84.02) and the other high rows use the few-shot-standard
**ResNet-12** with stronger base-session recipes. So we rebuilt the base session
properly and measured every step.

| # | configuration | s0 | last | avg | Δavg vs. paper (71.07) |
|---|---|---|---|---|---|
| a | ResNet-18, CE + cosine head, 200 ep, milestone LR **(starting point)** | 72.05 | 44.90 | 56.42 | −14.65 |
| b | + rotation "fantasy" virtual classes, cosine LR, label smoothing | 74.10 | 44.71 | 57.30 | −13.77 |
| c | + horizontal-flip TTA | 75.57 | 46.04 | 58.67 | −12.40 |
| d | **ResNet-12** (64-160-320-640), cosine LR, ls 0.1, 300 ep | 84.67 | 53.62 | 66.74 | −4.33 |
| e | + flip TTA | 85.12 | 54.26 | 67.29 | −3.78 |
| f | **ResNet-12 + rotation virtual classes**, 300 ep | 85.55 | 54.89 | 67.73 | −3.34 |
| g | + flip TTA | 86.70 | 56.43 | 69.06 | −2.01 |
| h | + flip **and** rotation TTA | 87.63 | 58.33 | 70.61 | −0.46 |
| **i** | **h + the CD-FSCIL fusion (Eq. 11, feature diffusion, α = 0.7)** | **87.58** | **58.56** | **70.76** | **−0.31** |
| | *paper, Table 1: CD-FSCIL* | *84.85* | *60.13* | *71.07* | |

The 120-epoch rotation run is a useful extra data point: without TTA it is
*better* than the 300-epoch one (68.62 vs 67.73) and with full TTA it is worse
(69.88 vs 70.61). Longer rotation training makes the encoder more
rotation-specialised, so rotation TTA recovers more from it.

| extra configurations run | s0 | last | avg |
|---|---|---|---|
| ResNet-12 + rotation, 120 ep, no TTA | 85.73 | 56.23 | 68.62 |
| … + flip TTA | 86.73 | 57.78 | 69.82 |
| … + flip/rot TTA | 86.20 | 58.19 | 69.88 |
| … + Eq. 11 fusion (α = 0.8) | 86.15 | 58.72 | 70.14 |
| ResNet-12 + rotation, 300 ep, flip TTA + Eq. 11 fusion | 86.67 | 56.51 | 69.35 |
| ResNet-12 + rotation, 300 ep, flip/rot TTA + **TEEN control** (α = 0.8) | 87.63 | 58.62 | 70.72 |

**How to read the 70.76.** It is the best of ten configurations, chosen by test
accuracy, and the top six span **69.06 – 70.76**. FSCIL has no validation split,
so taking an argmax over configurations on the test set is optimistic; the
honest statement is that a properly-built base session lands this method at
**≈ 70–71 average**, i.e. within roughly **0.3 to 1.0** of Table 1, rather than
that it lands at exactly 70.76. Each configuration is a single seed
(configuration (a) was repeated three times: avg 56.38 ± 0.05).

**Where the +14.34 points came from** — the decomposition along a → d → f → h → i:

| step | Δavg |
|---|---|
| a → d  **ResNet-12 instead of ResNet-18** (bundled with cosine LR + label smoothing) | **+10.32** |
| d → f  rotation "fantasy" virtual classes in the base session | +0.99 |
| f → h  flip + rotation test-time augmentation | +2.88 |
| h → i  **the paper's actual contribution — the conditional diffusion fusion** | **+0.15** |
| | **= +14.34** |

Rows (b) and (c) isolate the non-architectural part on ResNet-18: cosine LR +
label smoothing + rotation is worth +0.88 there and flip TTA a further +1.37, so
of the +10.32 in `a → d` roughly **+8 to +9.4 is the backbone swap alone**.

**This is the central result of the reproduction.** Table 1's numbers are
essentially reachable — but only by replacing the base session with a stronger
backbone and recipe than the paper says it uses. §4 states "ResNet18 … following
the training setup of CLOSER", and CLOSER's own session 0 in the very same table
is **76.02**. Since CD-FSCIL freezes everything after the base session, and its
generative path cannot help at session 0 (base classes have 500 real images
each), **its session-0 accuracy is exactly its backbone's base accuracy**. A
faithful "CLOSER setup" therefore has to land near 76, not 84.85. Getting to
84.85 requires a different backbone — the one thing the method is not supposed
to change.

Of the 14.34 points we added, **0.15 is the diffusion model**, and the TEEN-style
calibration control fills the same slot for +0.11. The remaining ~14.2 points are
standard base-session engineering that predates the paper and is orthogonal to
its claim.

### 2bis.1 Where the last 0.31 sits, and one thing we tried that failed

At the last session we reach 58.56 against 60.13. `diagnose.py` localises the
residual precisely:

| backbone | base-only 60-way | novel-only 40-way | novel → base misroutes | joint novel |
|---|---|---|---|---|
| ResNet-18 (a) | 72.05 | 25.82 | 80.95 % | 5.90 |
| ResNet-12, no rotation (e) | 85.12 | 29.07 | 53.52 % | 12.05 |
| ResNet-12 + rotation 120 ep, flip/rot TTA | 86.20 | **40.35** | 56.17 % | 19.20 |
| ResNet-12 + rotation 300 ep, flip/rot TTA (h) | **87.63** | 37.40 | 57.40 % | 17.10 |

The representation is no longer the bottleneck — rotation training lifted
novel-only accuracy from 25.8 % to ~40 %. What remains is **base-class bias**:
57 % of novel test images still lose the argmax to a base prototype, and the
40-way novel ceiling means this is a *calibration* gap, not a representation
gap. To reach 60.13 at our base accuracy the joint novel accuracy would have to
be ≈ 24 %; we have 17.1 %.

We tried to close it with a leave-one-out prototype calibration — estimate, from
support data only, how much each class's prototype over-fits its own K samples,
and divide it out. It **failed**, and instructively: the mean LOO self-similarity
is **0.9699 for base classes and 0.3995 for novel ones**. That 2.4× ratio is not
estimation noise to be corrected away, it is a real difference in cluster
tightness — the encoder was trained to compact the base classes and has never
seen the novel ones. Dividing it out over-boosts novel classes and drops the
average from 67.29 to 62.37. Recorded here because it rules out the obvious fix:
the residual ~1.5 points at the last session are what mechanisms like Tri-WE's
weight-space ensembling actually buy, and CD-FSCIL contains no such mechanism.

### 2bis.2 Reproducibility and honest caveats for this track

* Configuration (a) was repeated with **three seeds**: 71.86 ± 0.17 → 44.99 ±
  0.08, avg **56.38 ± 0.05**. The base-session training is very stable, which is
  why the original −14.65 gap was clearly not seed noise.
* Configurations (d)–(i) are **single seed each**. The prototype, fusion and
  α-sweep stages are deterministic, so comparisons *within* a feature space are
  exact.
* **70.76 is an argmax over ten configurations selected on the test set.** FSCIL
  provides no validation split. The top six configurations span 69.06 – 70.76,
  so the supportable claim is "≈ 70–71 average", not "70.76".
* **This configuration is not what the paper describes.** `scripts/reproduce_best.sh`
  carries the same warning in its header. For the faithful §4 configuration use
  `scripts/run_resnet_track.sh`, which gives 72.05 → 44.90.

---

## 3. Why the CLIP reading is not comparable to Table 1

miniImageNet's 100 classes are ImageNet-1k classes. CLIP was pre-trained on 400M
web image–text pairs and has therefore already seen the "novel" classes. Every
baseline in Table 1 trains a ResNet-18 from scratch on the 60 base classes and
genuinely has not.

`python -m cdfscil.diagnose` separates the two classic FSCIL failure modes:

| backbone | joint 100-way, base | joint 100-way, novel | novel-only 40-way | novel → base misroutes |
|---|---|---|---|---|
| ResNet-18 (base-trained) | 70.90 | **5.90** | **25.82** | **80.95 %** |
| CLIP RN50 | 84.03 | 71.20 | 84.42 | 21.20 % |
| CLIP ViT-B/32 | 87.92 | 77.40 | 88.45 | 17.00 % |
| CLIP ViT-B/16 | 91.67 | 80.65 | **90.38** | 14.77 % |
| CLIP ViT-L/14 | 94.98 | 85.28 | 93.95 | 11.43 % |

![base vs novel](../results/figures/base_vs_novel.png)

The ResNet-18 backbone can barely separate the novel classes at all (25.8 % on a
40-way problem restricted to them) and misroutes 81 % of novel images to a base
class. That *is* the FSCIL problem. With frozen CLIP the same numbers are 90.4 %
and 14.8 % — the problem is largely gone before any incremental method runs.

The effect is specific to miniImageNet, and that matters for reading §6:

| CLIP ViT-B/16 on | novel-only accuracy | novel → base misroutes |
|---|---|---|
| miniImageNet (40-way) | **90.38** | 14.77 % |
| CIFAR-100 (40-way) | 66.83 | 42.65 % |
| CUB-200 (100-way) | 61.50 | 10.24 % |

On CIFAR-100 and CUB-200 CLIP has genuine difficulty with the novel classes, so
those benchmarks still pose a real incremental-learning problem for a frozen-CLIP
method. miniImageNet does not.

Placing a frozen-CLIP method in Table 1 next to from-scratch ResNet-18 baselines
compares different quantities. On this benchmark the difference is worth ~27
points at the last session.

---

## 4. Does the generative path do anything? (the central experiment)

α in Eq. 11 is the weight on the **real** prototype, so α = 1 switches the
diffusion model off completely. The paper never states its value.

![generated exemplars](../results/figures/generated_exemplars.png)

The image-space model trained the full 40 k steps produces genuinely good
exemplars **for the classes it was trained on** — recognisable house finches on
branches, Gordon Setters, rhinoceros beetles — and mostly wrong content for the
classes it has never seen: "hourglass" comes out as assorted glassware,
"photocopier" as furniture and boxes, "stage" as cluttered interiors. Exactly
one of the eight sailboat samples is a sailboat. This is the qualitative form of
the quantitative result below: conditioning a base-only generative model on a
text embedding does not let it render an unseen class, so the generative
prototype for a novel class carries little class-specific signal.

![alpha sweep](../results/figures/alpha_sweep.png)

| generative path | best α | avg | Δ vs. no diffusion (89.19) |
|---|---|---|---|
| **image-space diffusion, 40 k steps (exactly Sec. 3.2)** | 0.85 | **89.44** | **+0.25** |
| feature diffusion, LLM conditions (Sec. 2.3 reading) | 0.80 | 89.43 | +0.24 |
| feature diffusion, class-name conditions | 0.85 | 89.46 | +0.27 |
| **oracle** diffusion — trained on all 100 classes | 0.95 | 89.29 | +0.10 |
| **control: one fixed vector, zero class information** | 0.80 | 89.35 | **+0.16** |
| control: random vector per class | 1.00 | 89.19 | +0.00 |
| **control: CLIP text prototype, no diffusion at all** | 0.20 | **93.48** | **+4.29** |

Three things follow, and they are the substance of this reproduction:

**4.1 On miniImageNet the gain is shrinkage, not synthesis.**
Blending the K = 5 prototype with a *single fixed vector that is identical for
every class* — carrying literally zero class-specific information — buys
**+0.16**, the same order as the diffusion model's **+0.24**, and *more* than the
oracle's **+0.10**. What Eq. 11 is doing at α ≈ 0.8 here is regularising a noisy
5-sample mean by pulling it toward a fixed direction. Any fixed direction does
that; a 102M-parameter conditional diffusion model is not required for it.

This is *not* the whole story across datasets, and §6 gives the honest version:
on CIFAR-100 and CUB-200 the same generative path clears the content-free
control by a real margin (+0.74 and +0.84 versus +0.14 and +0.00), so it does
carry class-specific information there. The gains are still under one point, and
on all three datasets the free text baseline beats it.

**4.2 The ceiling is not a training-budget problem.** We trained an *oracle*
diffusion model on all 100 classes — deliberately violating the FSCIL protocol —
to bound how much the generative path could possibly contribute. It contributes **+0.10**. So the limit is not "the base-only model cannot extrapolate to novel
classes"; even a model that has seen them adds nothing on top of CLIP
prototypes, because those prototypes are already near the ceiling of what this
feature space supports.

**4.3 The method the paper dismisses beats it by ~18×.** §2.2 characterises
training-free calibration methods (TEEN, BiMC) as "limited to linear or
heuristic adjustments in the feature space" that "lack the expressive power of a
deep generative model". Fusing the K-shot prototype with the frozen CLIP **text**
embedding — one line of arithmetic, no training, no sampling, no parameters — is
exactly such a linear adjustment, and it gives **+4.29** average accuracy where
the diffusion model gives +0.24 — eighteen times as much, for zero compute.

**4.4 Applied only where it should help, the diffusion path *hurts*.**
Eq. 11 uses one α for every class, so it also perturbs base prototypes that were
estimated from 500 images and need no help. Re-running with `α_base = 1` (base
prototypes untouched) and sweeping α for novel classes only — the setting in
which shrinkage should pay off most:

| generative path, novel classes only (α_base = 1) | best α | avg | Δ vs. 89.19 |
|---|---|---|---|
| feature diffusion, LLM conditions | 0.95 | 89.19 | **+0.00** |
| control: CLIP text prototype | 0.90 | 89.26 | +0.07 |
| **control: one fixed vector, zero class information** | 0.70 | **89.58** | **+0.39** |

The diffusion path contributes **exactly nothing** — its optimum is the floor —
while the content-free constant gives the largest gain of any configuration we
measured. So the +0.24 seen with a shared α does not come from better
novel-class prototypes at all; it comes from perturbing the *base* prototypes,
i.e. from re-calibrating base-vs-novel scores. That is not the mechanism the
paper claims.

A note on the text control, to avoid over-reading it: its large +4.04 comes from
a shared α = 0.2, i.e. from mostly *replacing* image prototypes with text ones
for base and novel classes alike. On miniImageNet CLIP's text classifier is
simply stronger than 500-shot image prototypes (93.98 vs 92.62 at session 0), so
that number says "CLIP already knows these classes" as much as it says "text
calibration works". Its value here is as a cost baseline: whatever Eq. 11 is
worth, one line of arithmetic captures more of it than a 102M-parameter
generative model does.

**4.4b Nor is it a conditioning-strength problem.** The paper does not mention
classifier-free guidance; we trained with 10 % condition dropout so it is
available, and swept it on the fully-trained 40 k-step image-space model:

| guidance | generated prototype alone (α = 0) | best fused avg | Δ vs. 89.19 |
|---|---|---|---|
| 1.0 (no guidance — the paper's plain conditional sampler) | 47.36 | 89.44 | +0.25 |
| 2.0 | 59.50 | **89.49** | **+0.30** |
| 3.0 | 60.67 | 89.46 | +0.27 |

Guidance does exactly what it should: it makes the text condition bite, lifting
the generated prototype on its own by **13.3 points** (47.4 → 60.7). The fused
result moves by **0.05**. So the strongest version of the generative path we can
build — full 40 k-step training, in image space exactly as §3.2 specifies, with
classifier-free guidance — is worth +0.30, against +0.16 for a content-free
constant and +4.29 for a one-line text calibration.

**4.5 It is not a sampling-budget problem either.** `N` in Eq. 8 (how many
exemplars to generate per class) is also unspecified. Sweeping it:

| N | generative prototype alone (α = 0) | best fused avg | Δ vs. 89.19 |
|---|---|---|---|
| 4 | 51.08 | 89.26 | +0.07 |
| 16 | 61.60 | 89.25 | +0.06 |
| 64 | 63.38 | **89.43** | **+0.24** |
| 256 | 63.82 | 89.38 | +0.19 |

More samples clearly make the *generative prototype itself* better (51 → 64 as N
goes 4 → 256, exactly the Monte-Carlo behaviour you would expect), but the fused
result saturates by N ≈ 64 and then stops moving. So the ceiling is not sampling
noise: even an arbitrarily well-estimated x̂_gen does not add more.

In the ResNet-18 regime the picture is the same: 56.42 → 56.77 (+0.35), again
shrinkage-sized, and still 15 points short of the claimed 60.13 at the last
session.

---

## 5. Does the LLM prior help?

The paper's third claimed contribution is the multimodal LLM description prior.
It is measurable, real, and modest.

| text conditions for `p_c` | s0 | last | avg |
|---|---|---|---|
| `"a photo of a {class}."` | 93.98 | 90.46 | 92.08 |
| 80-prompt OpenAI CLIP ensemble | 94.78 | 91.53 | 92.89 |
| LLM descriptions (6 per class) | 93.97 | 91.51 | 92.44 |
| LLM descriptions + template ensemble | 94.82 | 91.84 | **93.04** |

(as a zero-shot text classifier, which is the cleanest way to isolate `p_c`)

LLM descriptions are worth **+0.96 avg** over bare class names — but so is
standard prompt ensembling (+0.81), and combining them is best. This is the
known "classification by description" effect, and it lives entirely in the CLIP
text encoder; it is independent of the diffusion model. Consistent with that,
conditioning the diffusion model on LLM descriptions rather than class names
changes the *fused* result not at all (89.39 either way), even though it does
improve the generated prototypes in isolation (α = 0: 63.38 vs 60.73).

---

## 6. Other datasets — where the generative path *does* carry signal

Frozen CLIP ViT-B/16, same protocol, all three benchmarks. Δ is against that
dataset's own α = 1 floor.

| | miniImageNet | CIFAR-100 | CUB-200 |
|---|---|---|---|
| sessions | 9 | 9 | 11 |
| CLIP zero-shot text | 94.82 → 91.84 (avg 93.04) | 74.98 → 68.94 (avg 71.45) | 66.13 → 55.38 (avg 58.93) |
| real prototypes, α = 1 | 92.62 → 87.26 (avg **89.19**) | 75.90 → 62.54 (avg **68.13**) | 81.63 → 67.93 (avg **73.09**) |
| + feature diffusion | avg 89.43 (**Δ +0.24**) | avg 68.87 (**Δ +0.74**) | avg 73.93 (**Δ +0.84**) |
| + control: fixed vector, zero class info | avg 89.35 (Δ +0.16) | avg 68.27 (Δ +0.14) | avg 73.09 (Δ **+0.00**) |
| + control: CLIP text prototype (free) | avg 93.48 (Δ +4.29) | avg 74.48 (Δ +6.35) | avg 74.86 (Δ +1.77) |

Read across the rows, this is the fairest summary of the method:

* **The generative path is not pure shrinkage everywhere.** On CIFAR-100 it
  gives +0.74 where a content-free constant gives +0.14, and on CUB-200 it gives
  +0.84 where the constant gives exactly **+0.00**. On those two datasets the
  diffusion model is genuinely contributing class-specific information. That is
  a real, positive result for the idea, and it is not visible on miniImageNet.
* **But the gain is always under one point**, on every dataset and every
  configuration we ran.
* **And the free cross-modal baseline wins every time** — by 18× on
  miniImageNet, 8× on CIFAR-100, and 2× on CUB-200.

CUB-200 is the most informative benchmark of the three and the one the paper
reports only as a radar chart. It is where CLIP's text prior is weakest (zero-shot
58.93, far below the 73.09 of image prototypes), so it is the least contaminated
setting — a prototype method has real work to do there. It is also where the
diffusion's contribution is cleanest (all +0.84 of it is class information, none
of it shrinkage) and where the text baseline's advantage is smallest. If the
method is to be defended, CUB-200 is where the argument is strongest.

## 7. Cost

| stage | wall clock (1 × H100) |
|---|---|
| CLIP feature cache (60 000 images) | 55 s |
| training-free incremental sessions + α sweep (12 values) | 6 s |
| feature-space diffusion: train + sample 100 classes | 9 min |
| ResNet-18 base training (200 epochs) | 19 min |
| image-space diffusion, 40 k steps @ 655 img/s | ~4.3 h |
| image-space prototypes: 100 classes × 64 samples × 50 DDIM steps | ~10 min |

The abstract claims the method "drastically reduces computational and memory
overhead". Relative to gradient-based FSCIL that is true of the *incremental*
sessions — they cost 6 seconds — but the base session now has to train a 102M
UNet for hours, and every new class costs 64 × 50 = 3 200 UNet forward passes
plus 64 CLIP encodes. The paper reports no efficiency measurement of any kind.

---

## 8. Limitations of this reproduction

Stated plainly, because they bound what the conclusions above support:

1. **The image-space run was restarted at step 10 k** to fix an EMA without
   warm-up (see `project_setup.md`); the reported model is 10 k steps with the
   old schedule plus 30 k with the corrected one, resumed from the same weights.
2. **Image-space diffusion is trained for 40 k steps**, batch 256 (≈ 10.2 M
   images, 3.4 h), which is short by the standards of the generative literature
   (DDPM on CIFAR-10 used ~800 k steps; ADM on ImageNet-64 ~1.5 M at batch
   2048). The samples are nevertheless clearly class-faithful for base classes
   (see the figure in §4), and the fused result (+0.25) lands on top of the
   feature-space variant (+0.24) and the **oracle** (+0.10) — three independent
   routes to the same number, one of which has no training-budget dependence at
   all. A longer run could raise sample quality but would have to beat the
   oracle to change the conclusion.
3. **α, N, the UNet configuration, the CLIP checkpoint and the LLM are our
   choices**, because the paper specifies none of them (`paper_discrepancies.md`
   §C). We swept α rather than picking one, ran three CLIP backbones, and
   report all of it — but a different set of choices could differ.
4. **Seeds.** The ResNet-18 base training was repeated with three seeds
   (avg 56.38 ± 0.05). The ResNet-12 configurations of §2bis and the diffusion
   trainings were run once each; the α sweep and prototype construction are
   deterministic, so the *comparisons* between fusion settings are exact. A
   300-epoch rotation run (vs. the 120 used for the best number) was still
   training at the time of writing and would likely add a few tenths.
5. **CUB-200 has no genuine LLM descriptions.** miniImageNet and CIFAR-100 have
   LLM-written visual descriptions (6 per class, committed); CUB-200 uses the
   deterministic template fallback, so its `llm` rows are class-name prompts
   rather than an LLM prior (`dataset.md` §6).
6. **We could not test the paper's own code**, which is not released.

None of these affect the two findings that do not depend on training quality at
all: the Table 1 average is arithmetically inconsistent with its own row, and
the CLIP/ResNet regime gap is 27 points.

---

## 9. Verdict

* The **pipeline reproduces**: the method as described runs end to end, is
  genuinely training-free in the incremental sessions (0 trainable parameters,
  asserted in code), and behaves sensibly.
* **Table 1 does not reproduce**, and the gap is structural rather than a matter
  of tuning: the claimed numbers lie between the two mutually exclusive regimes
  the paper describes.
* **The central claim — that a conditional diffusion model is what makes this
  work — is not supported at the scale claimed.** On miniImageNet its
  contribution (+0.24) is matched by a fixed constant vector carrying zero class
  information (+0.16) and not exceeded by an oracle model that has seen every
  novel class (+0.10); applied only to the novel prototypes it is meant to
  improve, it contributes exactly nothing (+0.00) while the same constant gives
  +0.39. There, the mechanism is shrinkage of a noisy 5-sample mean, not
  synthesis.
* **It is fairer to say the effect is real but small.** On CIFAR-100 and
  CUB-200 the same generative path clears the content-free control by a genuine
  margin (+0.74 vs +0.14, +0.84 vs +0.00), so it does carry class-specific
  information there. But no configuration on any dataset gained a full point,
  and on all three a one-line text calibration — the "linear or heuristic"
  family §2.2 dismisses — gained 2× to 18× more, for zero compute.
* The parts that *do* work — freeze a strong pre-trained encoder, build
  class-mean prototypes, calibrate them with text — work, and are worth keeping.
  They are also, individually, prior work.
