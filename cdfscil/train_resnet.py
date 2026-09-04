"""Base-session training of the ResNet-18 backbone (Sec. 4 reading).

    python -m cdfscil.train_resnet --dataset mini_imagenet --epochs 200

Trains ONLY on session-0 data, then freezes the encoder and caches 512-d
features for every image so that `cdfscil.evaluate` can run the identical
training-free incremental protocol on top of them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from .clip_backbone import extract_split_features
from .data import ImageListDataset, build_benchmark
from .extract_features import load_classname_map
from .resnet_backbone import BaseSessionModel
from .utils import (Timer, count_params, get_logger, set_seed, worker_init_fn)

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def transforms_for(dataset: str, size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.5, 1.0),
                                         interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    resize = int(size * 8 / 7)                       # 84 -> 96, 224 -> 256
    return transforms.Compose([
        transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


DEFAULTS = {
    "mini_imagenet": dict(size=84,  small_input=False, pretrained=False,
                          epochs=200, lr=0.1,   milestones=[120, 160]),
    "cifar100":      dict(size=32,  small_input=True,  pretrained=False,
                          epochs=200, lr=0.1,   milestones=[120, 160]),
    "cub200":        dict(size=224, small_input=False, pretrained=True,
                          epochs=120, lr=0.01,  milestones=[60, 90]),
}


def rotate_batch(x, y, n_virtual):
    """Expand a batch with 0/90/180/270 rotations; label becomes y*4 + r."""
    if n_virtual == 1:
        return x, y
    xs, ys = [], []
    for r in range(4):
        xs.append(torch.rot90(x, r, dims=(2, 3)))
        ys.append(y * 4 + r)
    return torch.cat(xs), torch.cat(ys)


@torch.no_grad()
def evaluate_base(model, loader, device):
    """Top-1 over the real classes: with virtual labels, scores are max-pooled
    over the rotation group so the metric stays comparable."""
    model.eval()
    nv = model.n_virtual
    correct = total = 0
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        if nv > 1:
            logits = logits.view(logits.shape[0], -1, nv).max(-1).values
        correct += (logits.argmax(1) == y).sum().item(); total += y.numel()
    model.train()
    return 100.0 * correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mini_imagenet")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--features", default="features")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--milestones", type=int, nargs="*", default=None)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=16.0)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet12"])
    ap.add_argument("--schedule", default="milestone", choices=["milestone", "cosine"])
    ap.add_argument("--rotation", action="store_true",
                    help="SAVC/S3C-style rotation 'fantasy': predict class x rotation "
                         "as base_class*4 virtual labels during the base session only")
    ap.add_argument("--drop-rate", type=float, default=0.1)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    d = DEFAULTS[args.dataset]
    epochs = args.epochs or d["epochs"]
    lr = args.lr if args.lr is not None else d["lr"]
    milestones = args.milestones if args.milestones is not None else d["milestones"]
    size = args.size or d["size"]

    tag = args.tag or (f"{args.arch}_{args.dataset}"
                       + ("_rot" if args.rotation else "")
                       + (f"_seed{args.seed}" if args.seed != 1 else ""))
    log = get_logger("resnet", f"logs/train_resnet_{tag}.log")
    set_seed(args.seed, deterministic=False)
    device = torch.device(f"cuda:{args.gpu}")
    log.info(f"{tag}: arch={args.arch} epochs={epochs} lr={lr} sched={args.schedule} "
             f"milestones={milestones} size={size} rotation={args.rotation} "
             f"ls={args.label_smoothing} pretrained={d['pretrained']}")

    bench = build_benchmark(args.dataset, args.data_root,
                            load_classname_map(args.dataset))
    bench.sanity_check()
    base_ids = bench.session_train_ids[0]
    assert (bench.train_labels[base_ids] < bench.base_class).all()

    tr = ImageListDataset(bench, base_ids, "train", transforms_for(args.dataset, size, True))
    base_test_ids = bench.test_ids(0)
    te = ImageListDataset(bench, base_test_ids, "test", transforms_for(args.dataset, size, False))
    dtr = DataLoader(tr, batch_size=args.batch_size, shuffle=True, drop_last=True,
                     num_workers=args.workers, pin_memory=True,
                     persistent_workers=True, worker_init_fn=worker_init_fn)
    dte = DataLoader(te, batch_size=256, shuffle=False, num_workers=args.workers,
                     pin_memory=True)
    log.info(f"base train {len(tr)} / base test {len(te)}")

    n_virtual = 4 if args.rotation else 1
    model = BaseSessionModel(bench.base_class, d["pretrained"], d["small_input"],
                             args.temperature, arch=args.arch,
                             n_virtual=n_virtual, drop_rate=args.drop_rate).to(device)
    log.info(f"params {count_params(model)/1e6:.2f}M")
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum,
                          weight_decay=args.weight_decay, nesterov=True)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
             if args.schedule == "cosine"
             else torch.optim.lr_scheduler.MultiStepLR(opt, milestones, args.gamma))
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    outdir = Path(args.out) / tag
    outdir.mkdir(parents=True, exist_ok=True)
    t = Timer(); best = 0.0
    model.train()
    for ep in range(epochs):
        run, n = 0.0, 0
        for x, y, _ in dtr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            x, y = rotate_batch(x, y, n_virtual)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = crit(model(x), y)
            loss.backward(); opt.step()
            run += loss.item() * y.numel(); n += y.numel()
        sched.step()
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            acc = evaluate_base(model, dte, device)
            best = max(best, acc)
            log.info(f"epoch {ep+1}/{epochs} loss {run/n:.4f} "
                     f"base-test top1 {acc:.2f} (best {best:.2f}) lr "
                     f"{sched.get_last_lr()[0]:.4f} [{t.human()}]")
    torch.save({"model": model.state_dict(), "args": vars(args),
                "defaults": d, "base_test_acc": best}, outdir / "model_final.pt")
    log.info(f"saved {outdir/'model_final.pt'} | best base-test {best:.2f} "
             f"| {t.human()}")

    # ---- freeze and cache features for the whole benchmark ---------------- #
    enc = model.encoder.eval()
    enc.requires_grad_(False)
    pre = transforms_for(args.dataset, size, False)
    for split in ("train", "test"):
        f = extract_split_features(bench, split, enc, pre, device, args.features,
                                   tag, 256, args.workers, overwrite=True)
        log.info(f"cached {split} features {f.shape}")


if __name__ == "__main__":
    main()
