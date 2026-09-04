"""Session-wise FSCIL evaluation -> the Table-1 row.

    python -m cdfscil.evaluate --dataset mini_imagenet --clip-model ViT-B-16 \
        --gen-protos checkpoints/mini_imagenet/gen_protos.npz --alpha 0.8

Without --gen-protos this evaluates the pure real-prototype configuration
(alpha = 1), which is the training-free baseline the generative path has to beat.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .clip_backbone import model_tag
from .data import build_benchmark
from .extract_features import load_classname_map
from .fscil import cosine_predict, l2norm, run_sessions, session_metrics, summarize
from .utils import get_logger, set_seed, write_csv, write_json


def load_gen_protos(path) -> dict:
    z = np.load(path)
    classes, protos = z["classes"], z["protos"]
    return {int(c): protos[i] for i, c in enumerate(classes)}


def zero_shot_sessions(bench, test_feats, text_emb):
    """CLIP zero-shot reference: classify against text embeddings only."""
    out = []
    for s in range(bench.sessions):
        seen = bench.seen_classes(s)
        tids = bench.test_ids(s)
        preds = cosine_predict(test_feats[tids], text_emb[seen], seen)
        m = session_metrics(preds, bench.test_labels[tids], bench.base_class)
        m["session"] = s
        out.append(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mini_imagenet")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--features", default="features")
    ap.add_argument("--clip-model", default="ViT-B-16")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--feature-tag", default=None,
                    help="override the cached-feature tag (e.g. a ResNet run name)")
    ap.add_argument("--gen-protos", default=None,
                    help=".npz with arrays `classes` and `protos`")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="Eq. 11 weight on the REAL prototype (1.0 = real only)")
    ap.add_argument("--alpha-base", type=float, default=None,
                    help="separate alpha for base classes (default: --alpha)")
    ap.add_argument("--alpha-sweep", nargs="*", type=float, default=None)
    ap.add_argument("--no-prenorm", action="store_true")
    ap.add_argument("--no-postnorm", action="store_true")
    ap.add_argument("--zero-shot-text", default="llm+template",
                    help="prompt mode for the CLIP zero-shot reference row")
    ap.add_argument("--tag", default=None, help="name for the output files")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    set_seed(args.seed)
    tag = args.feature_tag or model_tag(args.clip_model, args.pretrained)
    run_tag = args.tag or f"{args.dataset}_{tag}"
    log = get_logger("eval", f"logs/eval_{run_tag}.log")

    bench = build_benchmark(args.dataset, args.data_root,
                            load_classname_map(args.dataset))
    bench.sanity_check()

    fdir = Path(args.features) / args.dataset
    train_feats = np.load(fdir / f"{tag}_train.npy")
    test_feats = np.load(fdir / f"{tag}_test.npy")
    log.info(f"features: train{train_feats.shape} test{test_feats.shape}")

    gen = load_gen_protos(args.gen_protos) if args.gen_protos else None
    if gen:
        log.info(f"generative prototypes for {len(gen)} classes "
                 f"from {args.gen_protos}")

    prenorm, postnorm = not args.no_prenorm, not args.no_postnorm
    report = {"dataset": args.dataset, "clip": tag, "prenorm": prenorm,
              "postnorm": postnorm, "gen_protos": args.gen_protos,
              "runs": {}}

    # ---- CLIP zero-shot reference ---------------------------------------- #
    zs_path = fdir / f"{tag}_text_{args.zero_shot_text.replace('+', '-')}.npy"
    if zs_path.exists():
        zs = zero_shot_sessions(bench, test_feats, np.load(zs_path))
        report["runs"][f"zeroshot_text[{args.zero_shot_text}]"] = {
            "sessions": zs, **summarize(zs)}
        log.info(f"CLIP zero-shot [{args.zero_shot_text}]: {summarize(zs)}")

    # ---- main configuration(s) ------------------------------------------- #
    alphas = args.alpha_sweep if args.alpha_sweep else [args.alpha]
    for a in alphas:
        ab = args.alpha_base if args.alpha_base is not None else a
        res = run_sessions(bench, train_feats, test_feats, gen,
                           alpha_base=ab, alpha_novel=a,
                           prenorm=prenorm, postnorm=postnorm)
        key = f"alpha={a:g}" + (f",alpha_base={ab:g}" if ab != a else "")
        report["runs"][key] = {"sessions": res, **summarize(res)}
        s = summarize(res)
        log.info(f"{key:28s} per-session={s['per_session']} "
                 f"avg={s['avg']} last={s['last']} PD={s['pd']}")

    outdir = Path(args.out)
    write_json(report, outdir / f"{run_tag}.json")

    rows = []
    for key, r in report["runs"].items():
        rows.append([key] + [f"{x:.2f}" for x in r["per_session"]]
                    + [f"{r['avg']:.2f}", f"{r['last']:.2f}", f"{r['pd']:.2f}"])
    header = (["config"] + [f"s{i}" for i in range(bench.sessions)]
              + ["avg", "last", "PD"])
    write_csv(rows, outdir / f"{run_tag}.csv", header)
    log.info(f"wrote {outdir / (run_tag + '.json')} and .csv")


if __name__ == "__main__":
    main()
