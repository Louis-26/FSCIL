"""One-time CLIP feature cache for a whole benchmark (Eq. 1) + text conditions (Eq. 2).

    python -m cdfscil.extract_features --dataset mini_imagenet --clip-model ViT-B-16

Writes
    features/<dataset>/<tag>_train.npy      (N_train, D) float32
    features/<dataset>/<tag>_test.npy       (N_test,  D) float32
    features/<dataset>/<tag>_text_<mode>.npy (C, D)     float32, L2-normalised
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .clip_backbone import (class_text_embeddings, extract_split_features,
                            load_clip, model_tag)
from .data import build_benchmark
from .descriptions import build_prompts
from .utils import Timer, get_logger, set_seed


def load_classname_map(dataset: str):
    if dataset != "mini_imagenet":
        return None
    p = Path(__file__).parent / "assets" / "mini_imagenet_classnames.json"
    return {w: v["clip_name"] for w, v in json.load(open(p)).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mini_imagenet",
                    choices=["mini_imagenet", "cifar100", "cub200"])
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--clip-model", default="ViT-B-16")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--out", default="features")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--prompt-modes", nargs="+",
                    default=["classname", "template", "llm", "llm+template"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    log = get_logger("extract", f"logs/extract_{args.dataset}.log")
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    tag = model_tag(args.clip_model, args.pretrained)
    log.info(f"dataset={args.dataset} clip={tag} device={device}")

    bench = build_benchmark(args.dataset, args.data_root, load_classname_map(args.dataset))
    info = bench.sanity_check()
    log.info(f"protocol OK: {info}")

    model, preprocess, tokenizer = load_clip(args.clip_model, args.pretrained, device)
    log.info(f"CLIP preprocess: {preprocess}")

    t = Timer()
    for split in ("train", "test"):
        f = extract_split_features(bench, split, model, preprocess, device,
                                   args.out, tag, args.batch_size, args.workers,
                                   args.overwrite)
        log.info(f"  {split}: {f.shape}  [{t.human()}]")

    keys = bench.wnids if args.dataset == "mini_imagenet" else None
    outdir = Path(args.out) / args.dataset
    outdir.mkdir(parents=True, exist_ok=True)
    for mode in args.prompt_modes:
        path = outdir / f"{tag}_text_{mode.replace('+', '-')}.npy"
        if path.exists() and not args.overwrite:
            log.info(f"  text[{mode}]: cached")
            continue
        try:
            prompts = build_prompts(mode, bench.class_names, args.dataset, keys)
        except (FileNotFoundError, KeyError) as e:
            log.warning(f"  text[{mode}]: skipped ({e})")
            continue
        emb = class_text_embeddings(model, tokenizer, prompts, device)
        np.save(path, emb.numpy().astype(np.float32))
        log.info(f"  text[{mode}]: {tuple(emb.shape)} from "
                 f"{sum(len(p) for p in prompts)} prompts")

    log.info(f"done in {t.human()}")


if __name__ == "__main__":
    main()
