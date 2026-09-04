"""Why does an FSCIL method fail (or not)?  Two orthogonal causes, measured.

    python -m cdfscil.diagnose --dataset mini_imagenet --tags resnet18_mini_imagenet ViT-B-16_openai

Joint top-1 over all 100 classes hides *which* of the two classic failures is
happening:

  (a) the backbone cannot separate the novel classes at all
      -> measured by `novel_only`: a 40-way problem among novel classes only.
  (b) the backbone separates them fine, but novel prototypes lose to base ones
      -> measured by `misroute`: the fraction of novel test images assigned to a
         BASE class in the joint problem.

A method whose novel_only is high but whose misroute is also high has a
calibration problem.  A method whose novel_only is low has a representation
problem, and no prototype trick will save it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import build_benchmark
from .extract_features import load_classname_map
from .fscil import l2norm
from .utils import write_csv


def diagnose(bench, Xtr, Xte, gen=None, alpha=1.0):
    proto = {}
    for s in range(bench.sessions):
        ids = bench.session_train_ids[s]
        for c in sorted(set(bench.train_labels[ids].tolist())):
            sel = ids[bench.train_labels[ids] == c]
            p = l2norm(l2norm(Xtr[sel]).mean(0))
            if gen is not None and c in gen:
                p = l2norm((1 - alpha) * l2norm(gen[c]) + alpha * p)
            proto[c] = p
    allc = np.arange(bench.num_classes)
    P = np.stack([proto[int(c)] for c in allc])
    Q = l2norm(Xte)
    novel_m = bench.test_labels >= bench.base_class
    base_m = ~novel_m

    pred = allc[(Q @ P.T).argmax(1)]
    nc = np.arange(bench.base_class, bench.num_classes)
    Pn = np.stack([proto[int(c)] for c in nc])
    pn = nc[(Q[novel_m] @ Pn.T).argmax(1)]
    bc = np.arange(bench.base_class)
    Pb = np.stack([proto[int(c)] for c in bc])
    pb = bc[(Q[base_m] @ Pb.T).argmax(1)]

    return {
        "joint_all": float(100 * (pred == bench.test_labels).mean()),
        "joint_base": float(100 * (pred[base_m] == bench.test_labels[base_m]).mean()),
        "joint_novel": float(100 * (pred[novel_m] == bench.test_labels[novel_m]).mean()),
        "base_only": float(100 * (pb == bench.test_labels[base_m]).mean()),
        "novel_only": float(100 * (pn == bench.test_labels[novel_m]).mean()),
        "misroute_novel_to_base": float(100 * (pred[novel_m] < bench.base_class).mean()),
        "misroute_base_to_novel": float(100 * (pred[base_m] >= bench.base_class).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mini_imagenet")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--features", default="features")
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--gen-protos", default=None)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", default="results/diagnosis.csv")
    args = ap.parse_args()

    bench = build_benchmark(args.dataset, args.data_root,
                            load_classname_map(args.dataset))
    gen = None
    if args.gen_protos:
        z = np.load(args.gen_protos)
        gen = {int(c): z["protos"][i] for i, c in enumerate(z["classes"])}

    hdr = ["tag", "joint_all", "joint_base", "joint_novel", "base_only",
           "novel_only", "misroute_novel_to_base", "misroute_base_to_novel"]
    rows = []
    print(f"{'tag':28s} {'all':>7s} {'base':>7s} {'novel':>7s} "
          f"{'base40':>8s} {'novel40':>8s} {'n->b%':>7s}")
    for tag in args.tags:
        d = Path(args.features) / args.dataset
        Xtr = np.load(d / f"{tag}_train.npy")
        Xte = np.load(d / f"{tag}_test.npy")
        r = diagnose(bench, Xtr, Xte, gen, args.alpha)
        rows.append([tag] + [f"{r[k]:.2f}" for k in hdr[1:]])
        print(f"{tag:28s} {r['joint_all']:7.2f} {r['joint_base']:7.2f} "
              f"{r['joint_novel']:7.2f} {r['base_only']:8.2f} "
              f"{r['novel_only']:8.2f} {r['misroute_novel_to_base']:7.2f}")
    write_csv(rows, args.out, hdr)
    print(f"\nwrote {args.out}")
    print("base40/novel40 = accuracy when only base / only novel classes compete;\n"
          "n->b%          = novel test images assigned to a base class in the joint problem.")


if __name__ == "__main__":
    main()
