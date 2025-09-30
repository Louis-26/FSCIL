
# OGDiff

This is the official repository for the paper **"Rethinking Open Set Domain Generalization: A Conditional Diffusion Perspective"**, which is currently **under review at AAAI**. The codebase provides a comprehensive implementation for conditional diffusion-based domain generalization in open set scenarios.

## 1. Introduction

This repository implements a novel approach for open set domain generalization (DG) using conditional diffusion models. Our method is designed to enhance the generalization ability of models across unseen domains, especially under open set conditions where unknown classes may appear during testing.

## 2. Features

* **Conditional Diffusion for Feature Space**: Diffusion models are applied in the feature space to improve robustness and generalization.
* **Optimizations for Diffusion Methods**: A concise and efficient feature concatenation method with learnable weights.
* **Data Augmentation**: Domain-aware data augmentation methods for certain tasks.
* **Efficient Training**: Supports large batch sizes, mixed precision (AMP), and advanced learning rate scheduling (CosineAnnealingLR).

## 3. Training

Update the dataset paths in `main_diff_eval.py` if needed. Then run:

```bash
python main_diff_eval.py --source-domain photo cartoon art_painting --target-domain sketch --save-name my_exp --gpu 0
```

**Key options:**

* `--batch-size`: Batch size per iteration (default: 512, adjust based on GPU memory)
* `--num-epoch`: Number of training epochs (default: 6000)
* `--lr`: Learning rate (default: 8e-4 for large batch)
* `--eval-step`: Evaluation frequency (default: 3)
* `--save-later`: Save model in the last 15% of iterations
* `--random-split`: If no validation set, use this to split from training data

All logs and metrics will be saved in `./experiment/log/` and `./experiment/metrics/` with timestamped filenames.

## 4. Evaluation

Section to be filled.

## 5. Citation

If you use this codebase or ideas from our work, please cite our paper (currently under review). Citation details will be provided upon publication.

## 6. Contact

For questions or issues, please open an issue on GitHub or contact the authors.

