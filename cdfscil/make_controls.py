"""Control 'generative prototypes' that use no diffusion model at all.

These exist so that "the diffusion model helps" can be falsified rather than
assumed.  Each writes an npz in exactly the format `cdfscil.evaluate
--gen-protos` expects, so it is plugged into Eq. 11 in place of x_gen_c.

    text        the frozen CLIP text embedding p_c.  Free: no training, no
                sampling, no parameters.  This is the "linear / heuristic
                cross-modal adjustment" family (TEEN, BiMC) that Sec. 2.2
                dismisses as lacking expressive power.
    globalmean  ONE vector, the mean of all base-session features, used for
                EVERY class.  Carries zero class-specific information, so any
                accuracy gain from fusing it is pure shrinkage of the noisy
                K-shot mean.  This is the control that tells you whether a
                generative model is doing anything a constant cannot.
    random      a fixed random unit vector per class.  Sanity check: should
                only ever hurt.
    teen        TEEN-style calibration (Wang et al., NeurIPS 2023 -- reference
                [13] of the paper): each class's slot is filled with a
                softmax-similarity-weighted mixture of the BASE prototypes.
                This is exactly the "linear or heuristic adjustment in the
                feature space" that Sec. 2.2 says lacks the expressive power of
                a deep generative model, so it is the sharpest available
                comparison for the generative path.  It is NOT CD-FSCIL.

    python -m cdfscil.make_controls --dataset mini_imagenet --clip-model ViT-B-16
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .clip_backbone import model_tag
from .data import build_benchmark
from .extract_features import load_classname_map
from .fscil import l2norm
from .utils import get_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mini_imagenet")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--features", default="features")
    ap.add_argument("--clip-model", default="ViT-B-16")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--text-mode", default="llm")
    ap.add_argument("--out", default="checkpoints/controls")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feature-tag", default=None,
                    help="override the cached-feature tag (e.g. a ResNet run name)")
    ap.add_argument("--teen-temp", type=float, default=16.0)
    args = ap.parse_args()

    log = get_logger("controls")
    tag = args.feature_tag or model_tag(args.clip_model, args.pretrained)
    bench = build_benchmark(args.dataset, args.data_root,
                            load_classname_map(args.dataset))
    fdir = Path(args.features) / args.dataset
    Xtr = np.load(fdir / f"{tag}_train.npy")
    classes = np.arange(bench.num_classes)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tpath = fdir / f"{tag}_text_{args.text_mode.replace('+', '-')}.npy"
    if tpath.exists():
        text = np.load(tpath)
        np.savez(out / f"{args.dataset}_{tag}_text.npz",
                 classes=classes, protos=l2norm(text).astype(np.float32))
    else:
        log.info(f"  (no text embeddings for {tag}; skipping the text control)")

    base_ids = bench.session_train_ids[0]
    assert (bench.train_labels[base_ids] < bench.base_class).all()
    gm = l2norm(l2norm(Xtr[base_ids]).mean(0))
    np.savez(out / f"{args.dataset}_{tag}_globalmean.npz",
             classes=classes,
             protos=np.tile(gm, (bench.num_classes, 1)).astype(np.float32))

    dim = Xtr.shape[1]
    rng = np.random.default_rng(args.seed)
    np.savez(out / f"{args.dataset}_{tag}_random.npz", classes=classes,
             protos=l2norm(rng.standard_normal((bench.num_classes, dim))).astype(np.float32))

    # ---- TEEN-style calibration ---------------------------------------- #
    # real prototype of every class, built exactly as the FSCIL protocol allows
    real = np.zeros((bench.num_classes, dim), np.float32)
    for s_ in range(bench.sessions):
        ids = bench.session_train_ids[s_]
        for c in sorted(set(bench.train_labels[ids].tolist())):
            sel = ids[bench.train_labels[ids] == c]
            real[c] = l2norm(l2norm(Xtr[sel]).mean(0))
    base_p = real[:bench.base_class]                       # base prototypes only
    logits = (real @ base_p.T) * args.teen_temp
    w = np.exp(logits - logits.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    np.savez(out / f"{args.dataset}_{tag}_teen.npz", classes=classes,
             protos=l2norm(w @ base_p).astype(np.float32))

    log.info(f"wrote control prototype sets (text/globalmean/random/teen) to {out}")


if __name__ == "__main__":
    main()
