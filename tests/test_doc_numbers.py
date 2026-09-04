"""Guard against documentation drift.

Every headline number quoted in README.md / overview/results.md is listed here
against the results file it came from.  Re-running an experiment overwrites its
json; this test then fails instead of letting the docs quietly go stale.
(That is not hypothetical -- the CIFAR-100 feature-diffusion row drifted by 0.07
when the pipeline was re-run during script validation, and this test is what
caught it.)

    python tests/test_doc_numbers.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

# (label, results stem, run key or "best", expected (s0, last, avg); None = don't check)
CLAIMS = [
    # --- overview/results.md section 2bis: the a -> i ladder ---
    ("a  ResNet-18, as Sec. 4 written", "mini_imagenet_resnet18_realonly", "alpha=1", (72.05, 44.90, 56.42)),
    ("b  + rotation + cosine + ls", "mini_imagenet_r18_rot_e200_realonly", "alpha=1", (74.10, 44.71, 57.30)),
    ("c  + flip-TTA", "mini_imagenet_r18_rot_e200_tta_realonly", "alpha=1", (75.57, 46.04, 58.67)),
    ("d  ResNet-12, 300 ep", "mini_imagenet_r12_e300_realonly", "alpha=1", (84.67, 53.62, 66.74)),
    ("e  + flip-TTA", "mini_imagenet_r12_e300_tta_realonly", "alpha=1", (85.12, 54.26, 67.29)),
    ("f  ResNet-12 + rotation, 300 ep", "mini_imagenet_r12_rot_e300_realonly", "alpha=1", (85.55, 54.89, 67.73)),
    ("g  + flip-TTA", "mini_imagenet_r12_rot_e300_tta_realonly", "alpha=1", (86.70, 56.43, 69.06)),
    ("h  + flip/rot-TTA", "mini_imagenet_r12_rot_e300_rtta_realonly", "alpha=1", (87.63, 58.33, 70.61)),
    ("i  + Eq.11 fusion  [BEST]", "mini_imagenet_r12_rot_e300_rtta_featdiff_llm", "best", (87.58, 58.56, 70.76)),
    ("   TEEN control on (h)", "mini_imagenet_r12_rot_e300_rtta_control_teen", "best", (87.63, 58.62, 70.72)),
    ("   120ep: rotation, no TTA", "mini_imagenet_r12_rot_e120_realonly", "alpha=1", (85.73, 56.23, 68.62)),
    ("   120ep: + flip-TTA", "mini_imagenet_r12_rot_e120_tta_realonly", "alpha=1", (86.73, 57.78, 69.82)),
    ("   120ep: + flip/rot-TTA", "mini_imagenet_r12_rot_e120_rtta_realonly", "alpha=1", (86.20, 58.19, 69.88)),
    ("   120ep: + Eq.11 fusion", "mini_imagenet_r12_rot_e120_rtta_featdiff_llm", "best", (86.15, 58.72, 70.14)),
    ("   300ep: flip-TTA + fusion", "mini_imagenet_r12_rot_e300_tta_featdiff_llm", "best", (86.67, 56.51, 69.35)),
    # --- section 1 / 4: the CLIP regime ---
    ("CLIP ViT-B/16 floor", "mini_imagenet_ViT-B-16_realonly", "alpha=1", (92.62, 87.26, 89.19)),
    ("CLIP image-space diffusion 40k", "mini_imagenet_ViT-B-16_imgdiff_llm", "best", (92.73, 87.60, 89.44)),
    ("CLIP image-space, guidance 2.0", "mini_imagenet_ViT-B-16_imgdiff_g2.0", "best", (None, None, 89.49)),
    ("CLIP feature diffusion", "mini_imagenet_ViT-B-16_featdiff_llm", "best", (92.15, 87.96, 89.43)),
    ("CLIP oracle diffusion", "mini_imagenet_ViT-B-16_featdiff_oracle", "best", (92.57, 87.53, 89.29)),
    ("CONTROL fixed vector", "mini_imagenet_ViT-B-16_control_globalmean", "best", (92.60, 87.60, 89.35)),
    ("CONTROL CLIP text", "mini_imagenet_ViT-B-16_control_text", "best", (94.85, 92.62, 93.48)),
    ("CLIP ViT-L/14 floor", "mini_imagenet_ViT-L-14_realonly", "alpha=1", (95.52, 91.10, 92.73)),
    ("CLIP ViT-B/32 floor", "mini_imagenet_ViT-B-32_realonly", "alpha=1", (89.23, 83.71, 85.61)),
    ("CLIP RN50 floor", "mini_imagenet_RN50_realonly", "alpha=1", (85.78, 78.90, 81.31)),
    # --- section 6: other datasets ---
    ("CIFAR-100 floor", "cifar100_ViT-B-16_realonly", "alpha=1", (75.90, 62.54, 68.13)),
    ("CIFAR-100 feature diffusion", "cifar100_ViT-B-16_featdiff_llm", "best", (75.65, 63.78, 68.87)),
    ("CIFAR-100 CONTROL fixed vector", "cifar100_ViT-B-16_control_globalmean", "best", (75.62, 62.93, 68.27)),
    ("CIFAR-100 CONTROL text", "cifar100_ViT-B-16_control_text", "best", (78.97, 71.26, 74.48)),
    ("CUB-200 floor", "cub200_ViT-B-16_realonly", "alpha=1", (81.63, 67.93, 73.09)),
    ("CUB-200 feature diffusion", "cub200_ViT-B-16_featdiff_llm", "best", (81.56, 69.68, 73.93)),
    ("CUB-200 CONTROL text", "cub200_ViT-B-16_control_text", "best", (81.91, 71.09, 74.86)),
]

TOL = 0.005


def _get(stem: str, key: str):
    f = Path("results") / f"{stem}.json"
    if not f.exists():
        return None
    runs = json.load(open(f))["runs"]
    if key == "best":
        cand = [(k, v) for k, v in runs.items() if k.startswith("alpha=")]
        return max(cand, key=lambda kv: kv[1]["avg"])[1] if cand else None
    return runs.get(key)


def test_documented_numbers_match_results():
    bad = []
    for label, stem, key, exp in CLAIMS:
        r = _get(stem, key)
        if r is None:
            bad.append((label, "MISSING", stem))
            continue
        got = (r["per_session"][0], r["last"], r["avg"])
        if any(e is not None and abs(e - g) > TOL for e, g in zip(exp, got)):
            bad.append((label, exp, tuple(round(g, 2) for g in got)))
    for b in bad:
        print(f"  [DRIFT] {b[0]:34s} docs={b[1]} results={b[2]}")
    assert not bad, f"{len(bad)}/{len(CLAIMS)} documented numbers no longer match results/"
    print(f"  [ok] all {len(CLAIMS)} documented numbers match results/*.json")


def test_paper_table1_avg_claim():
    """The two numbers the whole audit hangs on."""
    d = json.load(open("cdfscil/assets/paper/table1_miniimagenet.json"))
    cd = next(m for m in d["methods"] if m["name"] == "CD-FSCIL")
    assert abs(cd["avg_printed"] - 72.53) < 1e-9
    assert abs(float(np.mean(cd["acc"])) - 71.07) < 0.005
    print("  [ok] paper Table 1 CD-FSCIL: printed 72.53, true mean 71.07")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    print(f"running {len(tests)} documentation checks\n")
    for t in tests:
        try:
            t()
        except Exception as e:                                    # noqa: BLE001
            failed += 1
            print(f"  [FAIL] {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
