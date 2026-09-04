"""Figures for the reproduction report.

    python -m cdfscil.plots --dataset mini_imagenet

Writes results/figures/*.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

PAPER = Path(__file__).parent / "assets" / "paper" / "table1_miniimagenet.json"
C = {"paper": "#b5341f", "prev": "#d98c1f", "ours_clip": "#1f6fb5",
     "ours_rn": "#3f8f4f", "zs": "#7a5ea8", "grey": "#8a8a8a"}


def load_runs(results_dir, dataset):
    out = {}
    for f in sorted(Path(results_dir).glob("*.json")):
        r = json.load(open(f))
        if r.get("dataset") != dataset:
            continue
        for k, v in r["runs"].items():
            out[f"{f.stem}::{k}"] = v
    return out


def fig_sessions(runs, out, dataset):
    d = json.load(open(PAPER))
    by = {m["name"]: m for m in d["methods"]}
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    x = np.arange(9)

    def pick(sub, key):
        for k, v in runs.items():
            if sub in k and k.endswith(key):
                return v
        return None

    ax.plot(x, by["CD-FSCIL"]["acc"], "o--", color=C["paper"], lw=2.2, ms=6,
            label="CD-FSCIL, as printed in Table 1")
    ax.plot(x, by["Tri-WE"]["acc"], "s--", color=C["prev"], lw=1.6, ms=5,
            label="Tri-WE (prev. best in Table 1)")
    ax.plot(x, by["CEC"]["acc"], "^--", color=C["grey"], lw=1.4, ms=5,
            label="CEC (ResNet-18 reference)")

    rn = pick("resnet18", "alpha=1")
    if rn:
        ax.plot(x, rn["per_session"], "^-", color=C["ours_rn"], lw=2.2, ms=6,
                label="ours - ResNet-18 + prototypes (§4 regime)")
    cl = pick("mini_imagenet_ViT-B-16_realonly", "alpha=1")
    if cl:
        ax.plot(x, cl["per_session"], "o-", color=C["ours_clip"], lw=2.2, ms=6,
                label="ours - frozen CLIP + prototypes (§3 regime)")
    zs = None
    for k, v in runs.items():
        if "mini_imagenet_ViT-B-16_realonly" in k and "zeroshot" in k:
            zs = v
    if zs:
        ax.plot(x, zs["per_session"], ":", color=C["zs"], lw=2.0,
                label="ours - CLIP zero-shot text only (no FSCIL at all)")

    ax.set_xlabel("incremental session"); ax.set_ylabel("top-1 accuracy over all seen classes (%)")
    ax.set_title(f"{dataset}: what regime does Table 1 live in?", fontsize=12)
    ax.set_xticks(x); ax.grid(alpha=.3); ax.set_ylim(35, 100)
    ax.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout(); fig.savefig(out / "sessions.png", dpi=160)
    plt.close(fig)


def fig_alpha(runs, out):
    """Only the curves that answer the question, with readable names."""
    SERIES = [
        ("mini_imagenet_ViT-B-16_featdiff_llm", "feature diffusion, LLM conditions (faithful)", "#1f6fb5", "-"),
        ("mini_imagenet_ViT-B-16_featdiff_classname",  "feature diffusion, class-name conditions",     "#5ba3d9", "-"),
        ("mini_imagenet_ViT-B-16_featdiff_oracle",     "ORACLE diffusion (saw all 100 classes)",       "#b5341f", "-"),
        ("mini_imagenet_ViT-B-16_imgdiff_llm",  "image-space diffusion (Sec. 3.2)",             "#8a5ea8", "-"),
        ("mini_imagenet_ViT-B-16_control_globalmean",  "control: one fixed vector, zero class info",   "#3f8f4f", "--"),
        ("mini_imagenet_ViT-B-16_control_text",    "control: CLIP text prototype (free)",          "#d98c1f", "--"),
        ("mini_imagenet_ViT-B-16_control_random",      "control: random vector per class",             "#8a8a8a", ":"),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    floor = None
    plotted = 0
    for stem, label, col, ls in SERIES:
        pts = []
        for k, v in runs.items():
            if not k.startswith(stem + "::") or "alpha=" not in k:
                continue
            try:
                a = float(k.split("alpha=")[1].split(",")[0])
            except ValueError:
                continue
            pts.append((a, v["avg"]))
            if a == 1.0:
                floor = v["avg"]
        if len(pts) < 3:
            continue
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, color=col,
                marker="o", ms=4, lw=2, label=label)
        plotted += 1
    if not plotted:
        plt.close(fig); return
    if floor is not None:
        ax.axhline(floor, color="k", lw=1, ls=":")
        ax.annotate(f"no diffusion at all ({floor:.2f})", (0.02, floor + 0.4),
                    fontsize=8.5)
    ax.set_xlabel(r"$\alpha$ — weight on the REAL prototype  "
                  r"($\alpha=1$: generative path switched off)")
    ax.set_ylabel("average accuracy over 9 sessions (%)")
    ax.set_title("Eq. 11 fusion on miniImageNet / CLIP ViT-B/16:\n"
                 "does the generative path contribute anything?", fontsize=11.5)
    ax.set_ylim(55, 95); ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(out / "alpha_sweep.png", dpi=160)
    plt.close(fig)


def fig_base_novel(runs, out):
    cand = [(k, v) for k, v in runs.items()
            if k.endswith("alpha=1") and "mini_imagenet_ViT-B-16_realonly" in k]
    cand += [(k, v) for k, v in runs.items()
             if k.endswith("alpha=1") and "resnet18" in k]
    if not cand:
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for (k, v), col in zip(cand, [C["ours_clip"], C["ours_rn"]]):
        s = v["sessions"]
        x = np.arange(len(s))
        lbl = "CLIP ViT-B/16" if "ViT-B-16" in k else "ResNet-18"
        ax.plot(x, [r["base_acc"] for r in s], "-", color=col, lw=2, label=f"{lbl} — base classes")
        ax.plot(x, [r["novel_acc"] for r in s], "--", color=col, lw=2, label=f"{lbl} — novel classes")
    ax.set_xlabel("incremental session"); ax.set_ylabel("top-1 accuracy (%)")
    ax.set_title("stability (base) vs plasticity (novel)", fontsize=12)
    ax.grid(alpha=.3); ax.legend(fontsize=8.5); ax.set_ylim(0, 100)
    fig.tight_layout(); fig.savefig(out / "base_vs_novel.png", dpi=160)
    plt.close(fig)


def fig_generated(gen_dir, bench_names, base_class, out, n_base=4, n_novel=4):
    """Montage of what the frozen diffusion model actually produces, for base
    classes (which it trained on) and novel classes (which it never saw)."""
    from pathlib import Path as _P
    import matplotlib.image as mpimg
    d = _P(gen_dir)
    if not d.is_dir():
        return
    grids = sorted(d.glob("class*.png"))
    if not grids:
        return

    def pick(lo, hi, k):
        cand = [g for g in grids if lo <= int(g.name[5:8]) < hi]
        if not cand:
            return []
        step = max(1, len(cand) // k)
        return cand[::step][:k]

    rows = ([("base (seen in training)", g) for g in pick(0, base_class, n_base)]
            + [("novel (never seen)", g) for g in pick(base_class, len(bench_names), n_novel)])
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 1, figsize=(9, 1.05 * len(rows)))
    if len(rows) == 1:
        axes = [axes]
    for ax, (kind, g) in zip(axes, rows):
        ax.imshow(mpimg.imread(str(g)))
        cid = int(g.name[5:8])
        ax.set_ylabel(f"{bench_names[cid]}\n[{kind.split()[0]}]", rotation=0,
                      ha="right", va="center", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle("Exemplars sampled from the frozen base-session diffusion model\n"
                 "(DDIM 50 steps, guidance 1.0, conditioned on the class text embedding)",
                 fontsize=10)
    fig.tight_layout(rect=(0.10, 0, 1, 0.93))
    fig.savefig(out / "generated_exemplars.png", dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mini_imagenet")
    ap.add_argument("--results", default="results")
    ap.add_argument("--generated-dir", default=None,
                    help="checkpoints/<run>/generated_g1.0 to montage")
    args = ap.parse_args()
    out = Path(args.results) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.results, args.dataset)
    print(f"loaded {len(runs)} runs")
    fig_sessions(runs, out, args.dataset)
    fig_alpha(runs, out)
    fig_base_novel(runs, out)
    if args.generated_dir:
        import json as _j
        from .data import build_benchmark
        from .extract_features import load_classname_map
        b = build_benchmark(args.dataset, "data", load_classname_map(args.dataset))
        fig_generated(args.generated_dir, b.class_names, b.base_class, out)
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
