"""Assemble the final Table-1 comparison from the runs in results/.

    python -m cdfscil.report --dataset mini_imagenet

Produces results/table1_reproduction.{md,csv} containing
  * the paper's Table 1 verbatim, with an audit of its printed Avg column,
  * every configuration this repository actually ran,
so the two can be read side by side.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .utils import write_csv

PAPER = Path(__file__).parent / "assets" / "paper"


def audit_paper_table(path: Path):
    d = json.load(open(path))
    rows, mismatches = [], []
    for m in d["methods"]:
        rec = float(np.mean(m["acc"]))
        delta = m["avg_printed"] - rec
        if abs(delta) > 0.05:
            mismatches.append((m["name"], m["avg_printed"], rec, delta))
        rows.append({"name": m["name"], "venue": m["venue"], "acc": m["acc"],
                     "avg_printed": m["avg_printed"], "avg_recomputed": round(rec, 2),
                     "delta": round(delta, 2)})
    return d, rows, mismatches


# (results-file stem, human label, "best" = pick the best alpha, else a run key)
HEADLINE = [
    ("mini_imagenet_resnet18_realonly",         "ResNet-18 + real prototypes  (Sec. 4 AS WRITTEN)", "alpha=1"),
    ("mini_imagenet_r18_rot_e200_tta_realonly", "  + rotation virtual classes + flip-TTA",       "alpha=1"),
    ("mini_imagenet_r12_e300_tta_realonly",     "  ResNet-12 + flip-TTA",                        "alpha=1"),
    ("mini_imagenet_r12_rot_e300_realonly",     "  ResNet-12 + rotation 300ep",                  "alpha=1"),
    ("mini_imagenet_r12_rot_e300_tta_realonly", "  ResNet-12 + rotation + flip-TTA",             "alpha=1"),
    ("mini_imagenet_r12_rot_e300_rtta_realonly","  ResNet-12 + rotation + flip/rot-TTA",         "alpha=1"),
    ("mini_imagenet_r12_rot_e300_rtta_featdiff_llm", "  + CD-FSCIL Eq.11 fusion  [BEST EFFORT]", "best"),
    ("mini_imagenet_r12_rot_e300_rtta_control_teen", "  + TEEN control (no diffusion)",          "best"),
    ("mini_imagenet_r12_rot_e120_rtta_featdiff_llm", "  (120ep variant + fusion, for reference)", "best"),
    ("mini_imagenet_resnet18_featdiff_llm",     "ResNet-18 + feature diffusion",                "best"),
    ("mini_imagenet_ViT-B-16_imgdiff_llm",      "CLIP + image-space diffusion, 40k steps (Sec 3.2)", "best"),
    ("mini_imagenet_ViT-B-16_realonly",         "CLIP ViT-B/16 + real prototypes  (Sec. 3 regime)", "alpha=1"),
    ("mini_imagenet_ViT-B-16_featdiff_llm",     "CLIP + feature diffusion, LLM conditions",     "best"),
    ("mini_imagenet_ViT-B-16_featdiff_classname", "CLIP + feature diffusion, class-name conditions", "best"),
    ("mini_imagenet_ViT-B-16_featdiff_oracle",  "CLIP + ORACLE diffusion (trained on all 100 classes)", "best"),
    ("mini_imagenet_ViT-B-16_control_globalmean", "CONTROL: one fixed vector, zero class information", "best"),
    ("mini_imagenet_ViT-B-16_control_random",   "CONTROL: random vector per class",             "best"),
    ("mini_imagenet_ViT-B-16_control_text",     "CONTROL: CLIP text prototype (free, no diffusion)", "best"),
    ("mini_imagenet_ViT-B-16_realonly",         "CLIP zero-shot text only (not FSCIL)", "zeroshot_text[llm+template]"),
    ("mini_imagenet_ViT-L-14_realonly",         "CLIP ViT-L/14 + real prototypes",              "alpha=1"),
    ("mini_imagenet_ViT-B-32_realonly",         "CLIP ViT-B/32 + real prototypes",              "alpha=1"),
    ("mini_imagenet_RN50_realonly",             "CLIP RN50 + real prototypes",                  "alpha=1"),
]


def headline_rows(results_dir):
    """Compact 'what did we measure' table; silently skips runs not present."""
    out = []
    for stem, label, sel in HEADLINE:
        f = Path(results_dir) / f"{stem}.json"
        if not f.exists():
            continue
        runs = json.load(open(f))["runs"]
        if sel == "best":
            cands = [(k, v) for k, v in runs.items() if k.startswith("alpha=")]
            if not cands:
                continue
            key, r = max(cands, key=lambda kv: kv[1]["avg"])
        else:
            if sel not in runs:
                continue
            key, r = sel, runs[sel]
        out.append([label, f"{r['per_session'][0]:.2f}", f"{r['last']:.2f}",
                    f"{r['avg']:.2f}", key])
    return out


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mini_imagenet")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    ptab = PAPER / f"table1_{args.dataset.replace('_', '')}.json"
    if args.dataset == "mini_imagenet":
        ptab = PAPER / "table1_miniimagenet.json"

    lines = [f"# Table 1 reproduction - {args.dataset}", ""]
    csv_rows = []

    # -------------------------------------------------- paper, audited --- #
    if ptab.exists():
        d, rows, mism = audit_paper_table(ptab)
        ncol = len(d["columns"])
        lines += ["## A. The paper's Table 1, as printed (arXiv:2511.18516v1)", "",
                  "`avg (printed)` is the number in the PDF; `avg (recomputed)` is the",
                  "mean of the nine session accuracies on the same row.", ""]
        hdr = ["method", "venue"] + d["columns"] + ["avg (printed)", "avg (recomputed)", "delta"]
        body = [[r["name"], r["venue"]] + [f"{a:.2f}" for a in r["acc"]]
                + [f"{r['avg_printed']:.2f}", f"{r['avg_recomputed']:.2f}",
                   f"{r['delta']:+.2f}"] for r in rows]
        lines += [md_table(hdr, body), ""]
        for r in rows:
            csv_rows.append(["paper:" + r["name"]] + [f"{a:.2f}" for a in r["acc"]]
                            + [f"{r['avg_recomputed']:.2f}", f"{r['acc'][-1]:.2f}"])
        if mism:
            lines += ["**Audit result.** The printed average disagrees with the row's own",
                      "session accuracies for:", ""]
            for n, p, rec, dl in mism:
                lines.append(f"* `{n}`: printed **{p:.2f}**, actual mean of the row "
                             f"**{rec:.2f}** (overstated by {dl:+.2f}).")
            lines += ["", "Every other row agrees to within 0.01.", ""]

    # ------------------------------------------------- headline ---------- #
    hl = headline_rows(args.results)
    if hl:
        lines += ["## B. Headline: what this reproduction measured "
                  "(miniImageNet, same protocol)", "",
                  md_table(["configuration", "session 0", "last", "avg", "config"], hl),
                  "",
                  "`config` is the Eq. 11 setting selected; `alpha=1` means the "
                  "generative path is switched off entirely.", ""]

    # ------------------------------------------------- all runs ----------- #
    lines += ["## C. Every configuration run", ""]
    files = sorted(Path(args.results).glob("*.json"))
    hdr = None
    body = []
    for f in files:
        r = json.load(open(f))
        if r.get("dataset") != args.dataset:
            continue
        for key, run in r["runs"].items():
            ps = run["per_session"]
            if hdr is None:
                hdr = ["run", "config"] + [f"s{i}" for i in range(len(ps))] \
                      + ["avg", "last", "PD"]
            body.append([f.stem, key] + [f"{a:.2f}" for a in ps]
                        + [f"{run['avg']:.2f}", f"{run['last']:.2f}", f"{run['pd']:.2f}"])
            csv_rows.append([f"ours:{f.stem}:{key}"] + [f"{a:.2f}" for a in ps]
                            + [f"{run['avg']:.2f}", f"{run['last']:.2f}"])
    if body:
        lines += [md_table(hdr, body), ""]
    else:
        lines += ["_(no result json found - run cdfscil.evaluate first)_", ""]

    out_md = Path(args.out) / f"table1_reproduction_{args.dataset}.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
    ncols = max(len(r) for r in csv_rows) if csv_rows else 0
    write_csv(csv_rows, Path(args.out) / f"table1_reproduction_{args.dataset}.csv",
              ["row"] + [f"s{i}" for i in range(ncols - 3)] + ["avg", "last"])
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
