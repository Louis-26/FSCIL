"""Build the wnid -> human-readable class name table for miniImageNet.

Two canonical sources are combined so the mapping is verifiable rather than
hand-typed:

  * `imagenet_class_index.json` (Keras/Amazon mirror) gives  index -> (wnid, name).
    ImageNet-1k indices are defined as *sorted wnid order*, which the script
    asserts.
  * `open_clip.IMAGENET_CLASSNAMES` gives the CLIP-paper class names in the same
    index order (the names OpenAI used for the reported zero-shot numbers).

Result is written to cdfscil/assets/mini_imagenet_classnames.json and committed,
so the pipeline never needs network access at run time.

    python -m cdfscil.build_classnames
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

ASSET = Path(__file__).parent / "assets" / "mini_imagenet_classnames.json"
IN_INDEX_URL = ("https://s3.amazonaws.com/deep-learning-models/image-models/"
                "imagenet_class_index.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default=str(ASSET))
    args = ap.parse_args()

    with urllib.request.urlopen(IN_INDEX_URL, timeout=60) as r:
        idx = json.load(r)
    wnids_1k = [idx[str(i)][0] for i in range(1000)]
    raw_names = [idx[str(i)][1].replace("_", " ") for i in range(1000)]
    assert wnids_1k == sorted(wnids_1k), "ImageNet-1k indices must be sorted-wnid order"

    from open_clip import IMAGENET_CLASSNAMES
    clip_names = list(IMAGENET_CLASSNAMES)
    assert len(clip_names) == 1000

    split = Path(args.data_root) / "miniimagenet" / "split" / "train.csv"
    rows = [l.strip().split(",") for l in open(split).readlines()[1:] if l.strip()]
    wnids = []
    for _, w in rows:
        if w not in wnids:
            wnids.append(w)
    assert len(wnids) == 100, f"expected 100 miniImageNet wnids, got {len(wnids)}"

    w2i = {w: i for i, w in enumerate(wnids_1k)}
    out = {}
    for w in wnids:
        i = w2i[w]
        out[w] = {"clip_name": clip_names[i], "imagenet_name": raw_names[i],
                  "imagenet_index": i}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out} with {len(out)} classes")
    for w in wnids[:3] + wnids[-3:]:
        print(" ", w, "->", out[w]["clip_name"])


if __name__ == "__main__":
    main()
