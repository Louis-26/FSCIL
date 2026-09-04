# Model architecture, notation and trainable parameters

## 1. Notation

| Symbol | Meaning | Shape |
|---|---|---|
| `v_0` | clean image | `3 × H × W` |
| `v_t` | noisy image at diffusion step `t` | `3 × H × W` |
| `x`   | CLIP visual feature, `E_img(v_0)` | `512` |
| `t_c` | text description of class `c` (LLM-written) | string |
| `p_c` | CLIP text embedding, `E_text(t_c)` | `512` |
| `c = φ(p_c)` | condition embedding fed to the denoiser | `512 → 512` |
| `ε_θ` | denoiser (predicts the added noise) | — |
| `β_t, α_t, ᾱ_t` | cosine noise schedule, `T = 1000` | scalars |
| `N` | generated exemplars per class (Eq. 8) | int, default 64 |
| `K` | real shots per novel class | 5 |
| `α` | fusion weight on the **real** prototype (Eq. 11) | `[0,1]` |
| `x̂_c` | final prototype of class `c` | `512` |

Session indexing: session `0` is the base session; session `s ≥ 1` introduces
classes `[base + (s-1)·way , base + s·way)`.

## 2. Component inventory

| Component | Module | Params | Trained | When |
|---|---|---|---|---|
| CLIP ViT-B/16 image encoder | `clip_backbone.load_clip` | 86.2 M | **never** | — |
| CLIP ViT-B/16 text encoder | `clip_backbone.load_clip` | 37.8 M | **never** | — |
| Conditional UNet `ε_θ` (image space) | `cdfscil/unet.py` | **102.3 M** | yes | base session only |
| Feature denoiser `ε_θ` (feature space) | `cdfscil/feat_diffusion.py` | 87.9 M | yes | base session only |
| ResNet-18 backbone (§4 reading) | `cdfscil/resnet_backbone.py` | 11.2 M | yes | base session only |
| Prototype bank + cosine classifier | `cdfscil/fscil.py` | **0** | no | — |

**Trainable parameters during any incremental session: 0.** This is asserted in
code, not just claimed. In fact `cdfscil/fscil.py` — which implements Eqs. 8–12,
i.e. *every* incremental-session computation — imports only `numpy`; a test
(`tests/test_reproduction.py::test_incremental_stage_uses_no_neural_network`)
parses its import graph and fails if a deep-learning framework ever appears
there. In addition: `PrototypeBank.add_real` refuses to overwrite a class
that already has a prototype, and `clip_backbone.assert_frozen` fails if the
CLIP encoder has any tensor with `requires_grad=True` or is not in `eval()`.

## 3. Conditional UNet (image space, §3.2 / Fig. 1)

The paper says "around 110M parameters". Our default is the closest standard
ADM-style configuration:

```
image_size        64
in/out channels   3
base channels     128
channel mult      (1, 2, 2, 4)        ->  resolutions 64, 32, 16, 8
res blocks        3 per resolution
attention at      16, 8
attention heads   4
dropout           0.1
--------------------------------------------------
parameters        102,320,387   (102.3 M)
```

Adding attention at resolution 32 as well (`--attn-res 32 16 8`) gives 104.2 M,
marginally closer to the paper's figure, but costs ~20 % throughput for no
measurable benefit at this training budget. The trained checkpoint reported in
`results.md` is the 102.3 M configuration.

Structure (`cdfscil/unet.py`):

```
v_t ──> Conv3x3 ──┐
                  │
 t ──> sinusoidal(128) ──> MLP ──> temb(512) ──┐
                                               ├─(+)──> emb ──┐
 p_c ──> φ: Linear(512→512) SiLU Linear ───────┘              │  "Condition Fusion"
                                                              │
        ┌── down: 3× [ResBlock(emb) (+Attn @16,8)] per stage, Downsample ──┐
        │                                                                  │
        ├── mid : ResBlock(emb) → Attn → ResBlock(emb)                     │
        │                                                                  │
        └── up  : 4× [ResBlock(emb ⊕ skip) (+Attn)] per stage, Upsample ───┘
                                        │
                              GroupNorm → SiLU → Conv3x3 ──> ε̂
```

* **Conditioning** follows Fig. 1's "Time Embedding → Condition Fusion →
  Residual Blocks": `φ(p_c)` is *added* to the timestep embedding, and every
  `ResBlock` applies it as an adaptive-GroupNorm scale/shift. This is the
  standard AdaGN mechanism.
* `null_cond` is a learned unconditional embedding, used only if
  classifier-free guidance is switched on at sampling time.
* Every output projection is zero-initialised (`zero_module`), the usual DDPM
  stabilisation.

### Implementation detail that matters

`AttentionBlock` makes `q,k,v` contiguous before
`F.scaled_dot_product_attention`. Without it PyTorch silently falls back to the
math backend and materialises the full `(B, heads, HW, HW)` score matrix — in
our measurements 33.1 GiB peak and 371 img/s, versus 10.5 GiB and 899 img/s once
fixed (batch 128, H100, `torch.compile`). It is a correctness-neutral but
5×-cost bug that is easy to ship by accident.

## 4. Feature denoiser (feature space, §2.3)

The 1-D analogue used to test the paper's other self-description. Residual MLP
with adaptive-LayerNorm conditioning:

```
dim 512 → width 1024 → 8 × ResMLPBlock(adaLN(t, p_c)) → LayerNorm → 512
parameters  87.9 M
```

CLIP features are L2-normalised and standardised with **base-session-only**
statistics before diffusion; the transform is inverted at sampling time.

## 5. ResNet-18 backbone (§4 reading)

Standard torchvision ResNet-18 (7×7 stride-2 stem + max-pool) for miniImageNet
(84×84) and CUB (224×224); CIFAR-style 3×3 stem with no max-pool for CIFAR-100
(32×32). Trained with a cosine classifier (temperature 16) on base classes only,
then frozen and used purely as a feature extractor — exactly the decoupled
recipe shared by CEC / FACT / SAVC / CLOSER.

```
encoder  11.18 M   (512-d output)
head     0.03 M    (cosine classifier, 60 base classes; discarded after training)
```

## 6. What is frozen when

```
session 0   ┌─ CLIP                     frozen
            ├─ UNet ε_θ                 TRAINED  (Eq. 5, base images only)
            └─ prototypes (base)        written once

session s≥1 ┌─ CLIP                     frozen
            ├─ UNet ε_θ                 frozen   <- no gradients ever again
            ├─ prototypes (old)         untouched
            └─ prototypes (new)         written once from Eqs. 8/10/11
```
