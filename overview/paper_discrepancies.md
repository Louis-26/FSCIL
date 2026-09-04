# Internal inconsistencies in arXiv:2511.18516v1

This file records everything that has to be resolved before the paper can be
reproduced at all, plus the checks that can be run mechanically. It is written
to be checkable, not rhetorical: each item quotes the paper and, where possible,
is verified by a script.

```bash
python -m cdfscil.audit_paper      # checks the paper against Tables 1 and 2
```

As of this writing it reports **8** inconsistencies, all computed from the
paper's own printed numbers.

Categories: **[A]** arithmetic that disagrees with the paper's own tables,
**[B]** two sections of the paper specifying incompatible things,
**[C]** information required to reproduce that is simply absent,
**[D]** bibliography.

---

## A. Arithmetic

### A1. The Avg column of Table 1 (miniImageNet)

`Avg` is the mean of the nine session accuracies on the same row. Recomputing it
for all 21 rows:

* **20 of 21 rows agree with their printed value to within 0.01.**
* The **CD-FSCIL** row does not:

| | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| printed | 84.85 | 82.05 | 77.18 | 74.05 | 70.58 | 65.54 | 63.79 | 61.49 | 60.13 | **72.53** |
| mean of that row | | | | | | | | | | **71.07** |

The printed average is **1.46 points higher** than the row it summarises. Since
the second-best method (Tri-WE) has `Avg = 70.62`, this is the difference
between a headline gain of **+1.91** and an actual gain of **+0.45**.

### A2. The "+1.91%" claim in §4.1

> "maintains stable superiority up to Session 7, leading to a **+1.91% gain on
> the overall average accuracy**."

`72.53 − 70.62 = 1.91`, so the sentence is consistent with the *printed* Avg but
not with the underlying numbers. Using the correct average: `71.07 − 70.62 =
+0.45`.

### A3. The "Replay by 12.92%" claim in §4.1

> "at Session 8, CD-FSCIL outperforms FACT by 9.64%, MetaFSCIL by 10.94%, and
> **Replay by 12.92%**."

From Table 1 at session 8: FACT `60.13 − 50.49 = 9.64` ✓, MetaFSCIL
`60.13 − 49.19 = 10.94` ✓, Replay `60.13 − 48.21 = **11.92**` ✗. The table's own
"Last sess. impro." column also says `+11.92` for Replay.

### A4. The last session is a tie, not a win

CD-FSCIL session 8 = **60.13**; Tri-WE session 8 = **60.13**. Identical to two
decimals. The abstract claims state-of-the-art performance and §4.1 says the
method "consistently surpasses all existing approaches throughout the entire
training trajectory"; at the final and most-cited session it draws. The
"Last session improvement" column is left blank for both rows.

### A5. Table 2's Avg column is also inconsistent — for three of four rows

Recomputing the mean of each row's ten session accuracies:

| row | printed Avg | mean of that row | delta |
|---|---|---|---|
| CLOSER | 77.87 | **78.17** | −0.30 |
| + Diffusion | 75.77 | **75.57** | +0.20 |
| + LLMs | 75.73 | **75.53** | +0.20 |
| Diffusion + LLMs (CD-FSCIL) | 81.79 | 81.79 | +0.00 |

Note the pattern: the three comparison rows are all off, in the direction that
narrows the gap for the baseline and widens it for the two partial ablations,
while the full-method row is exact. As in Table 1, the row that matters is the
one that is arithmetically self-consistent.

### A6. Table 2's prose contradicts Table 2's numbers

> "(1) Adding the Diffusion module alone **improves** the average accuracy from
> 77.87% (CLOSER) to 75.77% ... Similarly, incorporating CLIP alone also
> **boosts** performance, yielding an Avg of 75.73%"

75.77 < 77.87 and 75.73 < 77.87. Both ablation rows are **below** the CLOSER
baseline, i.e. each module *alone* hurts. The text describes both as
improvements.

### A7. Table 2 has the wrong number of CUB-200 sessions

CUB-200 under the stated protocol (§4: "100 base classes are followed by ten
10-way 5-shot sessions") has **11** sessions (0 … 10). Table 2 has **10**
columns (0 … 9), and Fig. 2 plots S1 … S10. Fig. 3 plots CIFAR-100 as S1 … S8,
but CIFAR-100 has **9** sessions.

### A8. Table 2 appears to report base-class accuracy, not total accuracy

The CD-FSCIL row of Table 2 is `82.39, 82.31, 82.19, 82.02, 81.85, 81.67,
81.56, 81.38, 81.31, 81.18` (Avg 81.79). Those are, to a rounding error, the
**"Base Class Accuracy"** curve of Fig. 2 (82.4 … 81.2), not its **"Total
Accuracy"** curve (79.5 … 67.2). Table 2's caption says "session-wise Acc (%)".

---

## B. Sections that specify incompatible things

### B1. Which encoder? Frozen CLIP or a trained ResNet-18?

§3.1 (Eq. 1–2), unambiguous:
> "given a frozen CLIP encoder `E_CLIP`, a clean image `v_0` is encoded as
> `x = E_CLIP^img(v_0)` ... where `x ∈ R^512`"

§4 Implementation details, equally unambiguous:
> "Our implementation follows the training setup of CLOSER, with **ResNet18** as
> the backbone encoder."

These are different feature extractors with different training regimes and
wildly different accuracy on this benchmark. **This is the single most important
ambiguity**: it decides whether the method should be compared against Table 1's
baselines at all (see §C below and `results.md`). This reproduction implements
both.

### B2. Where does diffusion happen? Image space or CLIP feature space?

§2.3 "Our Positioning":
> "We surpass the gradient-update dilemma by employing a diffusion model to
> **directly synthesize high-fidelity feature prototypes in the CLIP embedding
> space**, diverging from prior works like MetaDiff that generate weights."

§3.2 and §3.5:
> "Let `v_0` denote a clean image sample. The forward diffusion process
> gradually adds Gaussian noise ..." / "Operating in the **image domain** allows
> the diffusion model to produce semantically faithful visual exemplars that are
> re-embedded into CLIP's aligned feature space."

Fig. 1 draws image-space diffusion. Eqs. 6–7 sample images and then re-encode
them. So §2.3 describes a different algorithm from §3. Both are implemented here
(`cdfscil/unet.py` vs `cdfscil/feat_diffusion.py`).

### B3. Does CD-FSCIL generate weights, like MetaDiff, or not?

§2.3 says it *diverges* from MetaDiff, "that generate weights". §4 Implementation
details says:

> "our CD-FSCIL framework learns a **diffusion-based meta-optimizer** ... This
> allows CD-FSCIL to learn **parameter evolution** without inner-loop
> backpropagation"

"meta-optimizer" and "parameter evolution" are descriptions of MetaDiff's
weight-space method, and match nothing in §3.

### B4. The training budget is not consistent with the dataset

> "we optimize our CD-FSCIL module for 30 epochs (**10,000 iterations per
> epoch**)"

300,000 iterations total. The miniImageNet base session has 30,000 images, so an
"epoch" is 117 iterations at batch 256 (or 234 at 128). 10,000 iterations per
epoch would require a batch size of 3, or a dataset 40–85× larger.

### B5. miniImageNet resolution

§4 says miniImageNet is "60,000 images of size 84×84". The CEC release that
defines the benchmark splits stores images at original ImageNet resolution
(~500×375); 84×84 is a transform applied by the dataloader, and is the size of
the *historical Ravi & Larochelle cache*, which is a different dataset. Minor,
but it is one more sign that the described setup and the cited splits were not
checked against each other.

---

## C. Missing information required to reproduce

| Symbol | Where | Why it matters |
|---|---|---|
| **α** (Eq. 11) | never given a value anywhere | It is the *only* inference hyper-parameter and it controls the entire contribution. `α = 1` reduces CD-FSCIL to a plain nearest-class-mean classifier with no diffusion at all. |
| **N** (Eq. 8) | never given a value | Number of generated exemplars per class. |
| UNet configuration | "around 110M parameters" only | No channel widths, depths, attention resolutions. |
| Which CLIP checkpoint | "CLIP [8]" and `R^512` | ViT-B/16, ViT-B/32 and RN50 all differ by several points here. |
| LLM used for descriptions | "Large Language Models (LLMs)" | Neither model, prompt, nor number of descriptions per class is given. |
| Guidance scale | not mentioned | Text conditioning is weak without classifier-free guidance. |
| Efficiency numbers | claimed, never measured | The abstract claims it "drastically reduces computational and memory overhead"; no timing, FLOP or memory measurement appears in the paper, and §3.5 concedes a ~110M UNet plus 50-step DDIM sampling per class. |

Our choices for all of these are documented in `project_setup.md` and are
exposed as command-line flags, so any of them can be changed and re-run.

---

## D. Bibliography

Duplicated entries, several with different author lists for the same paper:

| Same work, cited twice | Note |
|---|---|
| [1] and [20] — Tao et al., *Few-shot class-incremental learning*, CVPR 2020 | [1] lists "Xing Wei", [20] lists "Xiaoyan Wei" |
| [5] and [28] — *Few-shot incremental learning with continually evolved classifiers* (CEC), CVPR 2021 | [28] has the correct authors (Chi Zhang, Nan Song, Guosheng Lin, Yun Zheng, Pan Pan, Yinghui Xu). [5]'s author list — "Chenyang Zhang, Meng Song, Yue Liu, Yunhe Gao, Zhihua Zhang" — does not correspond to that paper. The text cites CEC as [5] and Table 1 cites it as [28]. |
| [6] and [31] — Zhou et al., *Forward compatible few-shot class-incremental learning* (FACT), CVPR 2022 | different author lists; [31] is correct |
| [19] and [46] — Ho et al., *Denoising diffusion probabilistic models* | [46] lists "UC Berkeley" as an author and has no venue |

Also: Table 1 dates IDLVQ (Chen & Lee) to ICLR 2020; it is ICLR 2021.

---

## E. What this means for reproduction

None of the above is fatal on its own, but together they mean **the paper does
not determine a single experiment**. To reproduce it you must choose:

1. CLIP or ResNet-18 (§3 vs §4);
2. image-space or feature-space diffusion (§3.2 vs §2.3);
3. a value for α and N.

This repository runs **all** of those choices rather than picking one silently,
and reports what each produces. See `results.md`.

The most consequential consequence is comparability. Every method in Table 1 —
iCaRL through Tri-WE — trains a ResNet-18 **from scratch on the 60 base
classes** and has never seen the 40 novel classes. A frozen-CLIP method has seen
all 100 of them: miniImageNet's classes are ImageNet-1k classes, and CLIP
zero-shot alone scores **94.82 % → 91.84 %** on this exact protocol. Placing a
CLIP-based method in that table without noting the change of regime compares
quantities that are not the same quantity — and, as measured here, the
difference is worth roughly 27 points at the final session.
