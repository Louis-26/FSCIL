# Methodology — CD-FSCIL (arXiv:2511.18516)

> Source: *Breaking Forgetting: Training-Free Few-Shot Class-Incremental Learning
> via Conditional Diffusion*, Kang, Qian, Lu — arXiv:2511.18516v1, 23 Nov 2025.

## 1. The problem the paper attacks

Few-Shot Class-Incremental Learning (FSCIL) presents a model with a large **base
session** (many classes, many images) followed by a stream of **incremental
sessions**, each introducing a handful of new classes with only `K = 5` labelled
images each. After every session the model is tested on *all classes seen so
far*. Two failure modes dominate:

* **catastrophic forgetting** — gradient updates on the new classes distort the
  base feature space;
* **weak plasticity** — 5 images per class produce noisy gradients, so the new
  classes are underfitted.

The paper's thesis is that both failures share one root cause — *the gradient
update itself* — and that the fix is to remove gradient optimisation from the
incremental sessions entirely.

## 2. The proposed answer

> "freeze all network parameters after the base session and reformulate
> incremental learning as a generative inference problem"

Concretely, CD-FSCIL is built from three frozen components and one arithmetic
step:

| # | Component | Trained? | Role |
|---|---|---|---|
| 1 | CLIP image encoder `E_img` | frozen, never trained | image → 512-d feature (Eq. 1) |
| 2 | CLIP text encoder `E_text` | frozen, never trained | LLM description → condition `p_c` (Eq. 2) |
| 3 | Conditional diffusion UNet `ε_θ` | trained **once**, on base-session images only, then frozen | generate exemplars for any class from `p_c` |
| 4 | Prototype fusion + cosine classifier | no parameters | Eqs. 8–12 |

Because nothing is updated after session 0, forgetting is claimed to be removed
*by construction* rather than mitigated.

## 3. The pipeline, equation by equation

### 3.1 Encoding (Eqs. 1–2)

```
x   = E_img (v0)                     x   ∈ R^512      visual feature
p_c = E_text(t_c)                    p_c ∈ R^512      class condition
```

`t_c` is a natural-language description of class `c`. The paper's contribution
here is that `t_c` is **not** the bare class name but a rich, LLM-written visual
description — "a seagull with a red beak and gray feathers" rather than
"seagull" — which is meant to compensate for the scarcity of image data.

### 3.2 Diffusion in image space (Eqs. 3–4)

Forward process, cosine schedule, `T = 1000`:

```
q(v_t | v_0) = N( v_t ; sqrt(ᾱ_t)·v_0 , (1-ᾱ_t)·I )                      (Eq. 3)
```

Reverse process, deterministic DDIM with `T_sample = 50`:

```
v_{t-1} = sqrt(ᾱ_{t-1}) · ( v_t - sqrt(1-ᾱ_t)·ε_θ(v_t,t,c) ) / sqrt(ᾱ_t)
          + sqrt(1-ᾱ_{t-1}) · ε_θ(v_t,t,c)                              (Eq. 4)
```

with `c = φ(p_c)`, i.e. the CLIP text embedding pushed through a learned
projection and fused into the denoiser.

### 3.3 Base session (Eq. 5)

The only training in the whole method:

```
L_diff = E_{t, v0, ε} ‖ ε − ε_θ( v_t , t , φ(p_c) ) ‖²                   (Eq. 5)
```

evaluated **only on session-0 images**. CLIP stays frozen throughout.

### 3.4 Incremental sessions — training-free (Eqs. 6–11)

For each class `c` (base or novel), two prototypes are built and blended:

*Generative path*
```
ṽ_i^(c) = DiffusionSampler(c)                     N samples, DDIM-50    (Eq. 6)
x̃_i^(c) = E_img( ṽ_i^(c) )                                             (Eq. 7)
x̂_gen_c = (1/N) Σ_i x̃_i^(c)                                            (Eq. 8)
```

*Real path* — the `K = 5` support images
```
x_k^real = E_img( v_k )                                                 (Eq. 9)
x̂_real_c = (1/K) Σ_k x_k^real                                          (Eq. 10)
```

*Fusion*
```
x̂_c = (1 − α) · x̂_gen_c  +  α · x̂_real_c                              (Eq. 11)
```

`α ∈ [0,1]` is the paper's only inference hyper-parameter; **its value is never
stated**. `α = 1` collapses the method to a plain CLIP nearest-class-mean
classifier; `α = 0` uses generated data only.

### 3.5 Inference (Eq. 12)

```
ŷ = argmax_c   ⟨ x_q , x̂_c ⟩ / (‖x_q‖ ‖x̂_c‖)                          (Eq. 12)
```

A non-parametric cosine nearest-prototype classifier. No logits are learned, so
adding a class is literally appending a row to a matrix.

## 4. Why this should work (the paper's argument)

1. **Stability.** Prototypes of old classes are written once and never touched
   again, so old-class accuracy cannot drift. Forgetting is structurally zero.
2. **Plasticity.** The 5 real shots are augmented by `N` synthetic exemplars
   drawn from a generative model, so the novel-class prototype is estimated from
   `N + K` rather than `K` samples.
3. **Cost.** No backward pass in any incremental session.

## 5. Where the argument is fragile — and what this reproduction measures

The generative path is the entire contribution: without it (`α = 1`) CD-FSCIL is
a textbook prototype classifier that predates the paper by years. But the
diffusion model is trained **only on base classes** (§3.3) and is asked to
generate **novel** classes it has never seen, steered only by a text embedding.
Whether that produces prototypes good enough to improve on the real 5 shots is
an empirical question the paper does not isolate — Table 2's ablation reports
numbers that contradict its own prose (see `paper_discrepancies.md`).

This reproduction therefore treats **the α-sweep as the central experiment**:

* `α = 1` — real prototypes only (the floor the method must beat)
* `α = 0` — generated prototypes only (how good is the generative path alone?)
* `α ∈ (0,1)` — is there any blend that beats both?

and reports the answer for both readings of "where the diffusion lives"
(image space per §3.2, feature space per §2.3).

## 6. Related-work positioning (paper §2)

* vs. **PEFT / prompt methods** (L2P, DualPrompt, CODA-Prompt, Tip-Adapter):
  those still compute gradients on a small parameter subset, so parameter drift
  persists.
* vs. **training-free calibration** (TEEN, BiMC): genuinely gradient-free but
  restricted to linear/heuristic prototype adjustment; CD-FSCIL claims a deep
  generative model is strictly more expressive.
* vs. **MetaDiff**: MetaDiff diffuses *model weights*; CD-FSCIL diffuses
  *images* (§3.2) — or *features* (§2.3), depending on which section you read.
