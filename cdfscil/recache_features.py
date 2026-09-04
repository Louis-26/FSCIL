"""Re-cache features from a trained base-session checkpoint, optionally with
horizontal-flip test-time augmentation (average of the image and its mirror).

Flip-TTA is a standard, transduction-free trick: it touches only the *encoder
forward pass*, uses no test labels and no cross-sample information, so it stays
inside the training-free protocol.

    python -m cdfscil.recache_features --ckpt checkpoints/r12_e300/model_final.pt \
        --tag r12_e300_tta --flip-tta --gpu 0
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from .clip_backbone import assert_frozen, feature_cache_path
from .data import ImageListDataset, build_benchmark
from .extract_features import load_classname_map
from .resnet_backbone import BaseSessionModel
from .train_resnet import DEFAULTS, transforms_for
from .utils import get_logger, worker_init_fn


@torch.no_grad()
def encode(model, loader, device, flip_tta: bool, rot_tta: bool = False) -> np.ndarray:
    assert_frozen(model)
    views = [lambda x: x]
    if flip_tta:
        views.append(lambda x: torch.flip(x, dims=(3,)))
    if rot_tta:
        base = list(views)
        for r in (1, 2, 3):
            for v in base:
                views.append(lambda x, v=v, r=r: torch.rot90(v(x), r, dims=(2, 3)))
    feats, order = [], []
    for imgs, _, gids in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = sum(model.encode_image(v(imgs)).float() for v in views)
        feats.append(f.cpu()); order.append(gids)
    feats = torch.cat(feats).numpy().astype(np.float32)
    order = torch.cat(order).numpy()
    return feats[np.argsort(order, kind="stable")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dataset", default="mini_imagenet")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--features", default="features")
    ap.add_argument("--flip-tta", action="store_true")
    ap.add_argument("--rot-tta", action="store_true",
                    help="also average the 4 rotations (natural for a model trained "
                         "with the rotation 'fantasy' virtual classes)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()

    log = get_logger("recache")
    device = torch.device(f"cuda:{args.gpu}")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a, d = ck["args"], ck["defaults"]
    bench = build_benchmark(args.dataset, args.data_root,
                            load_classname_map(args.dataset))
    nv = 4 if a.get("rotation") else 1
    model = BaseSessionModel(bench.base_class, d["pretrained"], d["small_input"],
                             a["temperature"], arch=a.get("arch", "resnet18"),
                             n_virtual=nv, drop_rate=a.get("drop_rate", 0.1))
    model.load_state_dict(ck["model"])
    enc = model.encoder.to(device).eval()
    enc.requires_grad_(False)
    size = a.get("size") or d["size"]
    pre = transforms_for(args.dataset, size, False)
    log.info(f"{args.tag}: arch={a.get('arch')} size={size} "
             f"flip_tta={args.flip_tta} rot_tta={args.rot_tta}")

    for split in ("train", "test"):
        n = len(bench.train_labels) if split == "train" else len(bench.test_labels)
        ds = ImageListDataset(bench, np.arange(n), split, pre)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True,
                        worker_init_fn=worker_init_fn)
        f = encode(enc, dl, device, args.flip_tta, args.rot_tta)
        p = feature_cache_path(args.features, args.dataset, args.tag, split)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, f)
        log.info(f"  {split}: {f.shape} -> {p}")


if __name__ == "__main__":
    main()
