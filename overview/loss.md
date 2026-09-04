# Loss functions

CD-FSCIL has exactly **one** loss, and it is used in exactly **one** session.

## 1. Diffusion denoising loss (Eq. 5) — base session only

```
L_diff = E_{t ~ U[0,T), v0 ~ D_base, ε ~ N(0,I)}  ‖ ε − ε_θ( v_t , t , φ(p_c) ) ‖²₂
```

where `v_t = sqrt(ᾱ_t)·v_0 + sqrt(1−ᾱ_t)·ε` (Eq. 3).

* **ε-parameterisation** (predict the noise), not `v_0`- or `v`-prediction.
* **Uniform timestep sampling** over `[0, T)` with `T = 1000`.
* **Unweighted** across timesteps — the plain `L_simple` of Ho et al., which is
  what Eq. 5 literally writes.
* `D_base` is the session-0 image set **only**. `train_diffusion.py` asserts
  `(train_labels[base_ids] < base_class).all()` before the first step, so a
  novel-class image can never enter the loss.

Implementation: `cdfscil/diffusion.py::GaussianDiffusion.training_loss`.

### Condition dropout (implementation addition)

With probability `p_uncond = 0.1` the condition is replaced by a learned null
embedding. The paper does not mention this. It is included so that
classifier-free guidance is *available* as an ablation; the default sampler uses
`guidance = 1.0`, which is algebraically identical to no guidance, so the
faithful setting is what runs unless you ask otherwise.

## 2. Incremental sessions — no loss at all

Sessions `1 … S` involve no objective, no optimiser and no backward pass. The
prototype update is closed-form arithmetic:

```
x̂_gen_c = mean_i  E_img( DiffusionSampler(c)_i )      (Eq. 8)
x̂_real_c = mean_k  E_img( v_k )                        (Eq. 10)
x̂_c      = (1−α)·x̂_gen_c + α·x̂_real_c                 (Eq. 11)
```

This is the paper's central structural claim, and the reproduction enforces it
rather than trusting it:

* `PrototypeBank.add_real` raises if a class prototype is written twice, so an
  old class can never be silently updated.
* `clip_backbone.assert_frozen` raises if the CLIP encoder is in `train()` mode
  or has any parameter with `requires_grad=True`.

## 3. Auxiliary loss for the §4 (ResNet-18) reading

The Sec. 4 reading of the paper says the backbone is a ResNet-18 trained
"following CLOSER". That base-session training needs its own objective, which is
the standard FSCIL one:

```
L_base = CrossEntropy( τ · cos(f(v), W) , y )        τ = 16
```

a cosine classifier with temperature 16 over the base classes, SGD
(momentum 0.9, nesterov, weight-decay 5e-4), MultiStep LR decay. Used **only**
in session 0; the encoder is frozen afterwards and the head discarded.

## 4. Losses that are *not* used

For the record, since several are common in this literature and might be
expected: no knowledge distillation, no replay/rehearsal loss, no contrastive or
supervised-contrastive term, no prototype-alignment or orthogonality
regulariser, no classifier fine-tuning in incremental sessions. The paper claims
none of them, and none are used.
