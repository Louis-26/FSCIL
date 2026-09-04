# Overview — reproducing CD-FSCIL (arXiv:2511.18516)

## What this is

A from-scratch implementation of **CD-FSCIL** and a reproduction attempt of
**Table 1** (miniImageNet) from *Breaking Forgetting: Training-Free Few-Shot
Class-Incremental Learning via Conditional Diffusion* (Kang, Qian, Lu;
arXiv:2511.18516v1).

The paper's proposal: after the base session, freeze everything, and handle each
new class by (a) generating exemplars for it with a class-conditional diffusion
model trained only on base-session data and conditioned on an LLM-written text
description, (b) averaging their CLIP features into a *generative prototype*,
(c) blending that with the mean of the 5 real support images, and (d) classifying
by cosine similarity. No gradients in any incremental session, so forgetting is
claimed to be eliminated by construction rather than mitigated.

The paper's own code is not released, and the previous pipeline in this
repository was not in working order (`complementary/final_results_diffusion.txt`:
19.90 % at session 0, 0.00 % thereafter). So `cdfscil/` is written from scratch
against Section 3.

## The short version of what happened

**The method runs. Table 1 does not reproduce, and the reason is structural.**

1. The paper specifies two mutually exclusive encoders — a **frozen CLIP**
   encoder in §3 (Eqs. 1–2, `x ∈ R^512`) and a **ResNet-18 trained following
   CLOSER** in §4. We built both.

2. Under the **ResNet-18** reading we get **72.05 → 44.90** (avg 56.42). Our
   session-0 accuracy lands on the published ResNet-18 baselines almost exactly
   (CEC 72.00, MetaFSCIL 72.04, ours 72.05), so the base training is calibrated.
   The claimed 84.85 → 60.13 is far above this.

3. Under the **frozen CLIP** reading we get **92.62 → 87.26** (avg 89.19) —
   *27 points above* the claimed final accuracy. A CLIP-based method cannot land
   on 60.13 either; it is much too strong, because miniImageNet's "novel"
   classes are ImageNet classes CLIP already knows. CLIP zero-shot alone, with
   no FSCIL machinery at all, scores 94.82 → 91.84.

   So the claimed row is simultaneously too high for one regime and far too low
   for the other.

4. **The diffusion model — the paper's actual contribution — contributes ~0.2
   points, and it is shrinkage rather than synthesis.** Sweeping α (Eq. 11; the
   paper never gives its value), the best fusion is **+0.24** avg over switching
   the generative path off. Blending instead with *a single fixed vector,
   identical for every class and carrying zero class information*, gives
   **+0.16**. An **oracle** diffusion trained on all 100 classes — deliberately
   breaking the FSCIL protocol to bound the ceiling — gives **+0.10**. And when
   the fusion is applied only to the novel prototypes it is supposed to improve
   (α_base = 1), the diffusion path contributes **exactly nothing** (+0.00)
   while the same content-free constant gives the best result we measured
   anywhere (**+0.39**). Meanwhile fusing with the frozen CLIP **text**
   embedding — one line of arithmetic, no model, no sampling — gives **+4.29**:
   the "linear or heuristic" baseline family that §2.2 dismisses as lacking
   expressive power.

   To be fair to the method: this is a miniImageNet result. On **CIFAR-100** and
   **CUB-200** the same generative path does clear the content-free control by a
   real margin (+0.80 vs +0.14, and +0.84 vs +0.00), so it is carrying genuine
   class information there. But no configuration on any dataset gained a full
   point, and the free text baseline won on all three.

5. Independently of any reproduction, **Table 1's Avg column is arithmetically
   inconsistent for exactly one row**: CD-FSCIL prints 72.53 where the mean of
   its own nine session accuracies is 71.07. The other 20 rows agree to within
   0.01. The claimed "+1.91 %" average gain over Tri-WE is **+0.45**, and at the
   final session the two are **tied at 60.13**. In Table 2 the pattern inverts:
   the three comparison rows' averages are off by 0.20–0.30 while the CD-FSCIL
   row is exact, and the prose calls two ablations "improvements" when both are
   *below* the baseline they are compared against.
   `python -m cdfscil.audit_paper` checks all of this in two seconds — it
   currently reports **8** inconsistencies, all derived from the paper's own
   printed numbers.

Full numbers, figures and the reasoning: **[`results.md`](results.md)**.
Everything that had to be resolved before the code could be written at all:
**[`paper_discrepancies.md`](paper_discrepancies.md)**.

## Reproduce it

```bash
bash overview/env_setup.sh                    # conda env, pinned deps, self-test
bash overview/data_prepare.sh                 # datasets in CEC layout + protocol check
bash scripts/reproduce_table1.sh --quick      # ~35 min on one GPU
python tests/test_reproduction.py             # 7 self-checks, all must pass
```

`--full` additionally trains the 102 M image-space UNet from scratch (~6 h).

## Documents

| | |
|---|---|
| [`project_setup.md`](project_setup.md) | step-by-step reproduction, versions, and the two environment traps |
| [`project_architecture.md`](project_architecture.md) | repository layout and pipeline dependency graph |
| [`dataset.md`](dataset.md) | which miniImageNet — three incompatible datasets share the name |
| [`methodology.md`](methodology.md) | the method, equation by equation |
| [`model_architecture.md`](model_architecture.md) | notation, modules, trainable-parameter count |
| [`loss.md`](loss.md) | the single loss, and where it is used |
| [`eval_metric.md`](eval_metric.md) | FSCIL protocol and metrics |
| [`output.md`](output.md) | output formats and how to read them |
| [`results.md`](results.md) | **the measured results** |
| [`paper_discrepancies.md`](paper_discrepancies.md) | **the audit** |
| [`env_setup.sh`](env_setup.sh), [`data_prepare.sh`](data_prepare.sh) | the one-time scripts |

## How this reproduction was kept honest

* **The protocol is asserted, not assumed.** `FSCILBenchmark.sanity_check()`
  runs at every entry point and fails if session 0 is not exactly the base-class
  train split, if any incremental session does not hold exactly `way × shot`
  samples of exactly the right classes, or if the final test set is not the
  complete test split.
* **The dataset identity is proven.** `data_prepare.sh` aborts unless the
  downloaded miniImageNet split CSVs are byte-identical to the CEC ones
  committed here. (The miniImageNet link in the old top-level README pointed at
  the Ravi & Larochelle few-shot cache — a different dataset that cannot
  reproduce Table 1.)
* **Freezing is enforced.** `assert_frozen` rejects an encoder that is trainable
  or in `train()` mode; `PrototypeBank` refuses to overwrite an existing class
  prototype, so an old class cannot be silently updated. `cdfscil/fscil.py`,
  which implements every incremental-session computation, imports only `numpy` —
  a test parses its import graph to keep it that way.
* **The evaluation is cross-checked** against a second, independently written
  implementation that shares no code with it — agreement is exact on all 9
  sessions (`tests/test_reproduction.py`).
* **Both readings of the paper are implemented and reported.** Nothing ambiguous
  was resolved silently in the method's favour.
* **Controls were run before conclusions were drawn.** The fixed-vector,
  random-vector, oracle and text-prototype controls exist specifically so that
  "the diffusion model helps" could be falsified rather than assumed.
