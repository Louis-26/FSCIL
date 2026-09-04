# Outputs: formats, locations, and how to read them

## 1. Where everything lands

| Path | Written by | Content |
|---|---|---|
| `features/<ds>/<tag>_train.npy` | `extract_features`, `train_resnet` | `(N_train, D)` float32, row `i` = feature of train image `i` |
| `features/<ds>/<tag>_test.npy` | same | `(N_test, D)` float32 |
| `features/<ds>/<tag>_text_<mode>.npy` | `extract_features` | `(C, D)` float32, **L2-normalised**, row `c` = `p_c` (Eq. 2) |
| `checkpoints/<run>/model_{step,final,latest}.pt` | `train_diffusion` | `{model, ema, opt, step, args, cond_dim}` |
| `checkpoints/<run>/feat_diffusion.pt` | `feat_diffusion` | `{model, ema, whitener, args, dim, cond_dim}` |
| `checkpoints/<run>/gen_protos_*.npz` | `generate_prototypes`, `feat_diffusion` | `classes (C,)`, `protos (C, D)` — the `x̂_gen_c` of Eq. 8 |
| `checkpoints/<run>/samples/step*.png` | `train_diffusion` | qualitative grid, 4 base + 4 novel classes |
| `checkpoints/<run>/generated_g<g>/class*.png` | `generate_prototypes` | 8 generated images per class |
| `results/<tag>.json` | `evaluate` | full per-session metrics, every config |
| `results/<tag>.csv` | `evaluate` | flat table, one row per config |
| `results/table1_reproduction_<ds>.md` | `report` | paper vs. reproduction, side by side |
| `results/paper_audit.txt` | `audit_paper` | the paper checked against its own Table 1 |
| `logs/*.log` | all stages | timestamped progress |

`<tag>` for CLIP runs is `<model>_<pretrained>`, e.g. `ViT-B-16_openai`; for the
ResNet track it is `resnet18_<dataset>`.

## 2. `results/<tag>.json`

```jsonc
{
  "dataset":   "mini_imagenet",
  "clip":      "ViT-B-16_openai",
  "prenorm":   true,          // L2-normalise features before averaging
  "postnorm":  true,          // L2-normalise the fused prototype
  "gen_protos": "checkpoints/.../gen_protos_feat_n64_g1.0.npz",  // or null
  "runs": {
    "zeroshot_text[llm+template]": { ... },
    "alpha=0.9": {
      "sessions": [
        { "session": 0, "n_classes": 60, "n": 6000,
          "acc": 92.40,        // top-1 over ALL classes seen so far
          "base_acc": 92.40,   // restricted to base-class test images
          "novel_acc": NaN }   // NaN at session 0: no novel classes yet
        // ... one entry per session
      ],
      "per_session": [92.40, 91.72, ...],   // the Table-1 row
      "avg":  89.39,                        // mean of per_session
      "last": 87.80,                        // final session
      "pd":   4.60                          // per_session[0] - per_session[-1]
    }
  }
}
```

Run keys:

* `alpha=<a>` — Eq. 11 with `α = a` on the **real** prototype.
  `alpha=1` is real-only; `alpha=0` is generated-only.
* `alpha=<a>,alpha_base=<b>` — different α for base and novel classes.
* `zeroshot_text[<mode>]` — reference row: classify against CLIP text embeddings
  instead of prototypes. Not part of CD-FSCIL; included because it bounds how
  much of the accuracy is CLIP's prior knowledge rather than the method.

## 3. Reading the metrics

* **`acc` is over all seen classes**, so the test set grows every session
  (miniImageNet: 6000 → 10000 images). A rising `acc` with a growing class
  count would be surprising; a gently falling one is normal.
* **`base_acc` vs `novel_acc` is the diagnostic that matters.** Overall accuracy
  is dominated by base classes early on (60 of 65 classes at session 1), so a
  method can look stable while learning nothing new. In this reproduction the
  gap is large — see `results.md`.
* **`pd` (performance drop)** is the cleanest single forgetting proxy. A truly
  frozen-prototype method has *zero* forgetting on old classes by construction,
  so its `pd` comes entirely from the task getting harder (more classes) and
  from novel classes being weaker, not from drift.
* **`avg`** is the mean of the session accuracies — nothing more. The audit in
  `paper_discrepancies.md` §A1 exists because that is not what Table 1's Avg
  column contains for the CD-FSCIL row.

## 4. `gen_protos_*.npz`

```python
z = np.load("gen_protos_feat_n64_g1.0.npz")
z["classes"]   # (C,)   int   class labels
z["protos"]    # (C, D) float32, each row L2-normalised: the x̂_gen_c of Eq. 8
```

File-name fields: `n<N>` exemplars per class, `s<T_sample>` DDIM steps,
`g<guidance>` guidance scale, `step<k>` training step of the source checkpoint.

## 5. Units and conventions

* All accuracies are **top-1 percentages** in `[0, 100]`, rounded to 2 dp in the
  summaries and stored at full precision in `sessions`.
* Sessions are **0-indexed** (`s0 … s8` for miniImageNet). The paper's Table 1
  uses the same 0-indexed columns.
* `NaN` in `novel_acc` at session 0 means "no novel classes exist yet", not a
  failure.
* Features are stored **unnormalised**; normalisation happens in `fscil.py` so
  that the pre/post-normalisation choice stays ablatable.
* Prototype rows are indexed by **class label**, and `cosine_predict` returns
  class labels rather than row indices, so results stay correct even when the
  seen-class set is not `0..C-1`.
