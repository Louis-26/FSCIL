"""Frozen CLIP backbone (Eq. 1 / Eq. 2 of the paper) plus a feature cache.

The paper states x = E_CLIP^img(v0) with x in R^512 and p_c = E_CLIP^text(t_c)
with p_c in R^512.  Every OpenAI CLIP checkpoint with a 512-d joint space
(ViT-B/16, ViT-B/32, RN50) satisfies that; ViT-B/16 is our default.

Weights are loaded through open_clip with `pretrained="openai"`, i.e. the exact
OpenAI CLIP parameters -- open_clip is used only because it is pip-installable
and version-pinnable, which the original `openai/CLIP` git checkout is not.

Nothing here is ever trained: `requires_grad_(False)` + `.eval()` are enforced
at load time and asserted again before feature extraction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys

from tqdm import tqdm

from .data import FSCILBenchmark, ImageListDataset
from .utils import worker_init_fn

SUPPORTED = {
    "ViT-B-16": 512,
    "ViT-B-32": 512,
    "RN50":     1024,   # note: RN50's joint space is 1024-d, not 512
    "ViT-L-14": 768,
}


def model_tag(model_name: str, pretrained: str) -> str:
    return f"{model_name}_{pretrained}".replace("/", "-")


def _resolve_arch(model_name: str, pretrained: str) -> str:
    """OpenAI CLIP uses QuickGELU, open_clip's plain names use nn.GELU.

    Loading the OpenAI weights into a plain-GELU graph silently degrades every
    feature (open_clip only emits a UserWarning), so map onto the explicit
    `-quickgelu` architecture whenever the OpenAI checkpoints are requested.
    """
    if pretrained == "openai" and not model_name.endswith("-quickgelu"):
        return model_name + "-quickgelu"
    return model_name


def load_clip(model_name: str = "ViT-B-16", pretrained: str = "openai",
              device: torch.device | str = "cuda"):
    """Returns (model, preprocess, tokenizer).  Model is frozen and in eval()."""
    import open_clip
    arch = _resolve_arch(model_name, pretrained)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)   # QuickGELU mismatch -> hard fail
        model, _, preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(arch)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model, preprocess, tokenizer


def assert_frozen(model: torch.nn.Module) -> None:
    assert not model.training, "CLIP backbone must be in eval() mode"
    n_train = sum(p.requires_grad for p in model.parameters())
    assert n_train == 0, f"CLIP backbone has {n_train} trainable tensors"


# --------------------------------------------------------------------------- #
# image features
# --------------------------------------------------------------------------- #


@torch.no_grad()
def encode_images(model, loader, device, amp: bool = True) -> np.ndarray:
    assert_frozen(model)
    feats, order = [], []
    ctx = torch.autocast("cuda", dtype=torch.float16) if (amp and str(device).startswith("cuda")) \
        else torch.autocast("cpu", enabled=False)
    bar = tqdm(loader, desc="encode", leave=False,
           disable=not sys.stderr.isatty())
    for imgs, _, gids in bar:
        imgs = imgs.to(device, non_blocking=True)
        with ctx:
            f = model.encode_image(imgs)
        feats.append(f.float().cpu())
        order.append(gids)
    feats = torch.cat(feats).numpy().astype(np.float32)
    order = torch.cat(order).numpy()
    # restore benchmark id order (DataLoader keeps order with shuffle=False, but
    # be explicit so this is safe under any sampler)
    inv = np.argsort(order, kind="stable")
    return feats[inv]


def feature_cache_path(cache_dir, dataset: str, tag: str, split: str) -> Path:
    return Path(cache_dir) / dataset / f"{tag}_{split}.npy"


def extract_split_features(bench: FSCILBenchmark, split: str, model, preprocess,
                           device, cache_dir, tag: str,
                           batch_size: int = 256, workers: int = 16,
                           overwrite: bool = False) -> np.ndarray:
    """Encode *every* image of `split` once and memoise to .npy."""
    path = feature_cache_path(cache_dir, bench.name, tag, split)
    if path.exists() and not overwrite:
        return np.load(path)

    n = len(bench.train_labels) if split == "train" else len(bench.test_labels)
    ds = ImageListDataset(bench, np.arange(n), split, preprocess)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers,
                    pin_memory=True, worker_init_fn=worker_init_fn,
                    persistent_workers=False)
    feats = encode_images(model, dl, device)
    assert feats.shape[0] == n, f"{feats.shape[0]} != {n}"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, feats)
    return feats


# --------------------------------------------------------------------------- #
# text features (Eq. 2)
# --------------------------------------------------------------------------- #


@torch.no_grad()
def encode_texts(model, tokenizer, texts: list[str], device,
                 batch_size: int = 256) -> torch.Tensor:
    """Encode a flat list of prompts -> (len(texts), D) float32 on cpu."""
    assert_frozen(model)
    out = []
    for i in range(0, len(texts), batch_size):
        toks = tokenizer(texts[i:i + batch_size]).to(device)
        f = model.encode_text(toks)
        out.append(f.float().cpu())
    return torch.cat(out)


@torch.no_grad()
def class_text_embeddings(model, tokenizer, prompts_per_class: list[list[str]],
                          device, normalize_each: bool = True) -> torch.Tensor:
    """p_c for every class.

    `prompts_per_class[c]` is the list of prompts describing class c (a single
    template, or the LLM descriptions).  Following the CLIP zero-shot recipe the
    per-prompt embeddings are L2-normalised, averaged, then re-normalised.
    """
    embs = []
    for prompts in prompts_per_class:
        f = encode_texts(model, tokenizer, prompts, device)
        if normalize_each:
            f = F.normalize(f, dim=-1)
        m = f.mean(0)
        embs.append(F.normalize(m, dim=-1))
    return torch.stack(embs)
