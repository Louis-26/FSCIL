Breaking Forgetting: Training-Free Few-Shot Class-Incremental Learning via Conditional Diffusion


# Quick start

```bash
bash scripts/env_setup.sh                    # conda env + pinned deps + self-test
bash scripts/data_prepare.sh                 # datasets in CEC layout + protocol check
bash scripts/reproduce_table1.sh --quick      # ~35 min on one GPU
python tests/test_reproduction.py             # 9 self-checks
```

Full run, including training the 102 M image-space diffusion UNet from scratch:

```bash
bash scripts/reproduce_table1.sh --full --gpu 0        # ~6 h on one H100
```

# model performance


# Documentation

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




## Project links (from the previous README)

* Yuque notes: https://www.yuque.com/kanghaidong/ctruy3/qski7fh1toekqqk7/edit
* Google Doc: https://docs.google.com/document/d/1VFSoz94z6_FsUrWT3cM1vB6_WJzbFlSrRPl-ZYNraxI/edit?usp=sharing
* Google Slides: https://docs.google.com/presentation/d/1j9bRXTQGni-2a8GKquyR_TUGlMFpABAPYpuJK5Iq6Yc/edit?usp=sharing
* Overleaf (Yi Lu part): https://www.overleaf.com/read/rwkqmwqdgzxx#35c90c
* Overleaf (final): https://www.overleaf.com/5445663713xzmxjpbhvxsx#a9eccc
* arXiv: https://arxiv.org/abs/2511.18516

