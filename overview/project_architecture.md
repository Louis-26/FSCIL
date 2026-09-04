# Project architecture

## 1. Tree

```
📁 FSCIL/
├── 📁 overview/                      # this documentation set
│   ├── 📄 README.md                  # abstract + what was found
│   ├── 📄 project_architecture.md    # (this file)
│   ├── 📄 project_setup.md           # step-by-step reproduction
│   ├── 📜 env_setup.sh               # one-time conda env + pinned deps + self-test
│   ├── 📄 dataset.md                 # which miniImageNet, and why it matters
│   ├── 📜 data_prepare.sh            # one-time download + layout + protocol check
│   ├── 📄 methodology.md             # the paper's method, equation by equation
│   ├── 📄 model_architecture.md      # notation, modules, trainable parameters
│   ├── 📄 loss.md                    # the single loss, and where it is used
│   ├── 📄 eval_metric.md             # FSCIL protocol and metrics
│   ├── 📄 output.md                  # output formats and how to read them
│   ├── 📄 results.md                 # measured numbers + analysis
│   └── 📄 paper_discrepancies.md     # audit of the paper's internal consistency
│
├── 📁 cdfscil/                       # the implementation (all of it)
│   ├── 📄 data.py                    # FSCIL benchmarks + protocol assertions
│   ├── 📄 clip_backbone.py           # frozen CLIP (Eq. 1-2) + feature cache
│   ├── 📄 descriptions.py            # text conditions: classname / template / LLM
│   ├── 📄 build_classnames.py        # wnid -> class name, from canonical sources
│   ├── 📄 generate_descriptions.py   # (re)generate the LLM description asset
│   ├── 📄 extract_features.py        # CLI: cache image + text features
│   ├── 📄 unet.py                    # conditional UNet eps_theta  (image space)
│   ├── 📄 diffusion.py               # Eq. 3-5 forward/loss + Eq. 4 DDIM sampler
│   ├── 📄 train_diffusion.py         # CLI: base-session training (image space)
│   ├── 📄 generate_prototypes.py     # CLI: Eq. 6-8 from the image-space model
│   ├── 📄 feat_diffusion.py          # feature-space variant (Sec. 2.3 reading)
│   ├── 📄 resnet_backbone.py         # ResNet-18 encoder (Sec. 4 reading)
│   ├── 📄 train_resnet.py            # CLI: base-session ResNet-18 training
│   ├── 📄 fscil.py                   # prototypes, fusion (Eq. 11), eval (Eq. 12)
│   ├── 📄 evaluate.py                # CLI: session-wise eval + alpha sweep
│   ├── 📄 make_controls.py           # CLI: no-diffusion control prototypes
│   ├── 📄 diagnose.py                # CLI: novel-only acc vs base-class misroutes
│   ├── 📄 report.py                  # CLI: build the Table-1 comparison
│   ├── 📄 audit_paper.py             # CLI: check the paper against its own table
│   ├── 📄 plots.py                   # CLI: the four report figures
│   ├── 📄 utils.py                   # seeding, logging, io, timers
│   └── 📁 assets/
│       ├── 📄 mini_imagenet_classnames.json      # generated, committed
│       ├── 📁 descriptions/mini_imagenet.json    # LLM class descriptions
│       └── 📁 paper/table1_miniimagenet.json     # Table 1, transcribed verbatim
│
├── 📁 scripts/
│   ├── 📜 env.sh                     # source this before anything
│   ├── 📜 reproduce_table1.sh        # ONE COMMAND, end to end
│   └── 📜 run_resnet_track.sh        # the Sec. 4 (ResNet-18) reading
│
├── 📁 tests/
│   └── 📄 test_reproduction.py       # 9 self-checks (protocol, splits, freezing,
│                                     #  independent-eval cross-check, paper audit)
│
├── 📁 data/                          # created by data_prepare.sh (not in git)
│   ├── miniimagenet/{images,split}
│   ├── CUB_200_2011/
│   ├── cifar-100-python/
│   └── index_list/{mini_imagenet,cifar100,cub200}/session_*.txt
│
├── 📁 features/<dataset>/            # cached backbone features (.npy)
├── 📁 checkpoints/<run>/             # trained diffusion / ResNet weights + samples
├── 📁 results/                       # session tables (.json/.csv/.md), audit
│   ├── table1_reproduction_<ds>.md   # paper vs reproduction, machine-generated
│   ├── paper_audit.txt               # the paper checked against its own Table 1
│   ├── diagnosis_<ds>.csv            # novel-only accuracy / misroute rates
│   └── figures/*.png                 # sessions, alpha sweep, base-vs-novel, samples
├── 📁 logs/                          # one log file per stage
│
├── 📁 complementary/index_list/      # the canonical CEC session splits (in git)
├── 📁 literature/                    # reference PDFs
├── 📁 dataloader/, models/, OGDiff-code/   # pre-existing code, superseded
└── 📄 README.md
```

## 2. Dependency graph of the pipeline

```
overview/env_setup.sh          overview/data_prepare.sh
        │                              │
        └──────────────┬───────────────┘
                       ▼
        cdfscil.extract_features            (frozen CLIP -> features/*.npy)
                       │
        ┌──────────────┼───────────────────────────┬─────────────────────┐
        ▼              ▼                           ▼                     ▼
 cdfscil.evaluate   cdfscil.feat_diffusion   cdfscil.train_diffusion  cdfscil.train_resnet
  (alpha=1 floor)     (Sec 2.3 reading)        (Sec 3.2 reading)       (Sec 4 reading)
                       │                           │                     │
                       │                  cdfscil.generate_prototypes    │
                       │                           │                     │
                       └──────────────┬────────────┘                     │
                                      ▼                                  │
                             cdfscil.evaluate  (alpha sweep) ◄───────────┘
                                      │
                                      ▼
                             cdfscil.report  +  cdfscil.audit_paper
                                      │
                                      ▼
                   results/table1_reproduction_mini_imagenet.md
```

## 3. Design decisions worth knowing

**Features are cached once, globally.** Every image in a benchmark gets a stable
integer id; backbone features for the *entire* train and test split are computed
once into a single `.npy`, and sessions only slice into that matrix. This is why
the incremental stage is genuinely free of forward passes, why the alpha sweep
over 12 values takes 6 seconds, and why swapping backbones is a one-flag change.

**The protocol is asserted, not assumed.** `FSCILBenchmark.sanity_check()` runs
at the start of every entry point and fails loudly if session 0 is not exactly
the base-class train split, if any incremental session does not have exactly
`way × shot` samples of exactly the right classes, or if the final test set is
not the complete test split. A silently mis-prepared dataset is the single most
common way FSCIL numbers go wrong, and it produces *plausible* wrong numbers.

**Freezing is enforced.** `clip_backbone.assert_frozen` raises if the encoder is
in `train()` mode or has any parameter with `requires_grad=True`.
`PrototypeBank.add_real` raises if a class prototype is written twice, so an
old-class prototype cannot be silently updated by a later session.

**Both readings of the paper are implemented, not one.** Where §3 and §4 (or
§2.3 and §3.2) specify incompatible things, both are built and both are
reported. Nothing is silently resolved in the method's favour.

**Everything the paper leaves unspecified is a flag with a documented default**
(α, N, guidance scale, CLIP checkpoint, UNet width, prompt mode), so any choice
can be changed and re-run rather than argued about.

## 4. Pre-existing code in this repository

`train.py`, `models/my_vit/`, `dataloader/`, `OGDiff-code/` predate this work.
They are left untouched for provenance. They are not used by the reproduction:
`complementary/final_results_diffusion.txt` records what that pipeline produced
(CIFAR-100: session 0 = 19.90 %, sessions 1–8 = 0.00 %), i.e. it was not in
working order. The `cdfscil/` package is written from scratch against the paper.
