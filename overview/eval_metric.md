# Evaluation metric

## 1. Protocol

Standard CEC/TOPIC FSCIL evaluation, identical across every method in Table 1.

| Dataset | classes | base | incremental sessions | shots | total sessions |
|---|---|---|---|---|---|
| miniImageNet | 100 | 60 | 8 × 5-way | 5 | 9 |
| CIFAR-100 | 100 | 60 | 8 × 5-way | 5 | 9 |
| CUB-200 | 200 | 100 | 10 × 10-way | 5 | 11 |

After session `s` the classifier covers `C_s = base + s·way` classes and is
evaluated on **the full test split of every one of those classes** — not just
the new ones, and not a subsample.

miniImageNet test-set sizes per session:
`6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000`.

The exact 5-shot samples for each incremental session come from
`index_list/<dataset>/session_{s+1}.txt` (shipped with CEC). They are read
verbatim, never resampled — resampling changes the numbers by ±1–2 points and
makes cross-paper comparison meaningless.

## 2. Metrics reported

Let `A_s` be top-1 accuracy over all seen classes after session `s`, and
`S` the number of sessions.

| Metric | Definition | Meaning |
|---|---|---|
| `A_s` | top-1 over all seen classes | the per-session row of Table 1 |
| `Avg` | `(1/S) Σ_s A_s` | the paper's "Avg" column |
| `Last` | `A_{S-1}` | final accuracy — the hardest number |
| `PD` | `A_0 − A_{S-1}` | performance drop; a direct forgetting proxy |
| `Base_s` | top-1 restricted to base-class test images | stability |
| `Novel_s` | top-1 restricted to novel-class test images | plasticity |

`Base_s` / `Novel_s` are what Figs. 2–3 of the paper plot, and they are the only
way to tell "high accuracy because base classes dominate the test set" from
"actually learned the new classes".

## 3. Definition of `Avg` — and an audit

`Avg` is the arithmetic mean of the `S` session accuracies on the same row.
`cdfscil/report.py` recomputes it for every row of the paper's Table 1 and
compares against the printed value:

```
20 of 21 rows agree to within 0.01.
The CD-FSCIL row does not: printed 72.53, actual mean of its own nine
session accuracies 71.07 (overstated by +1.46).
```

Reproduce that check with:

```bash
python -m cdfscil.report --dataset mini_imagenet
```

Details in `paper_discrepancies.md`.

## 4. Implementation

`cdfscil/fscil.py`:

* `run_sessions` walks sessions in order, adds only that session's classes to
  the prototype bank, and asserts every seen class has a prototype before
  testing.
* `cosine_predict` returns *class labels*, not row indices — a subtle but common
  source of off-by-one errors once the label set stops being `0..C-1` contiguous
  from the classifier's point of view.
* `session_metrics` splits accuracy into overall / base / novel.
* `summarize` produces `per_session`, `avg`, `last`, `pd`.

No test image is ever used to fit anything: prototypes come from train-split
images only, and the diffusion model sees session-0 train images only.
