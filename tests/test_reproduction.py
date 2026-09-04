"""Self-checks for the CD-FSCIL reproduction.

    python tests/test_reproduction.py          # plain python, no pytest needed
    pytest tests/test_reproduction.py -v       # also works

These guard the three things that most often go silently wrong in FSCIL
reproductions: the wrong dataset variant, a leaky/incorrect session protocol,
and a backbone that is not actually frozen.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from cdfscil.data import build_benchmark                       # noqa: E402
from cdfscil.fscil import l2norm, run_sessions                 # noqa: E402

CMAP = {w: v["clip_name"] for w, v in json.load(
    open(ROOT / "cdfscil/assets/mini_imagenet_classnames.json")).items()}


# --------------------------------------------------------------------------- #
def test_protocol_all_datasets():
    """Session composition, shot counts and test growth for all 3 benchmarks."""
    expect = {
        "mini_imagenet": dict(n_train=50000, n_test=10000, n_cls=100, sessions=9),
        "cifar100":      dict(n_train=50000, n_test=10000, n_cls=100, sessions=9),
        "cub200":        dict(n_train=5994,  n_test=5794,  n_cls=200, sessions=11),
    }
    for ds, e in expect.items():
        b = build_benchmark(ds, "data", CMAP if ds == "mini_imagenet" else None)
        info = b.sanity_check()                       # raises on any violation
        assert info["n_train_total"] == e["n_train"], (ds, info)
        assert info["n_test_total"] == e["n_test"], (ds, info)
        assert info["n_classes"] == e["n_cls"], (ds, info)
        assert b.sessions == e["sessions"], (ds, b.sessions)
        print(f"  [ok] {ds}: protocol matches CEC/TOPIC")


def test_mini_imagenet_split_is_the_cec_one():
    """The downloaded split must be byte-identical to the CEC index lists,
    otherwise the session_*.txt files index into the wrong file-name space."""
    for f in ("train", "test"):
        a = (ROOT / f"data/miniimagenet/split/{f}.csv").read_bytes()
        b = (ROOT / f"complementary/index_list/mini_imagenet/{f}.csv").read_bytes()
        assert a == b, f"{f}.csv differs from the CEC split"
    print("  [ok] miniImageNet split is byte-identical to the CEC release")


def test_session0_matches_the_shipped_cec_list():
    """We build session 0 as 'every train image of every base class'. The CEC
    release also ships it explicitly as session_1.txt. They must be the same
    set, for all three datasets and all three id encodings (int indices for
    CIFAR, file names for miniImageNet, relative paths for CUB)."""
    from pathlib import Path as _P
    idx = ROOT / "complementary/index_list"

    b = build_benchmark("cifar100", "data")
    ids = {int(l) for l in open(idx / "cifar100/session_1.txt") if l.strip()}
    assert ids == set(np.asarray(b.session_train_ids[0]).tolist())

    b = build_benchmark("mini_imagenet", "data", CMAP)
    fn2i = {_P(p).name: i for i, p in enumerate(b.train_paths)}
    ids = {fn2i[_P(l.strip()).name]
           for l in open(idx / "mini_imagenet/session_1.txt") if l.strip()}
    assert ids == set(np.asarray(b.session_train_ids[0]).tolist())

    b = build_benchmark("cub200", "data")
    rel2id = {"/".join(_P(p).parts[-3:]): i for i, p in enumerate(b.train_paths)}
    ids = {rel2id["/".join(_P(l.strip()).parts[-3:])]
           for l in open(idx / "cub200/session_1.txt") if l.strip()}
    assert ids == set(np.asarray(b.session_train_ids[0]).tolist())
    print("  [ok] session 0 equals the shipped CEC session_1.txt on all 3 datasets")


def test_no_novel_data_in_base_session():
    """The diffusion model may only ever see base-class images."""
    for ds in ("mini_imagenet", "cifar100", "cub200"):
        b = build_benchmark(ds, "data", CMAP if ds == "mini_imagenet" else None)
        ids = b.session_train_ids[0]
        assert (b.train_labels[ids] < b.base_class).all(), ds
    print("  [ok] session 0 contains base-class images only")


def test_prototypes_are_write_once():
    """An old class's prototype must never be updated by a later session."""
    from cdfscil.fscil import PrototypeBank
    bank = PrototypeBank(8, 4)
    f = np.random.randn(4, 8).astype(np.float32)
    y = np.array([0, 0, 1, 1])
    bank.add_real(f, y, [0, 1])
    try:
        bank.add_real(f, y, [0])
        raise SystemExit("FAIL: prototype was silently overwritten")
    except AssertionError:
        pass
    print("  [ok] PrototypeBank refuses to overwrite an existing prototype")


def test_clip_is_frozen():
    """assert_frozen must reject a trainable or train()-mode encoder."""
    import torch
    from cdfscil.clip_backbone import assert_frozen
    m = torch.nn.Linear(4, 4)
    m.eval(); m.requires_grad_(False)
    assert_frozen(m)
    m.requires_grad_(True)
    try:
        assert_frozen(m)
        raise SystemExit("FAIL: assert_frozen accepted a trainable module")
    except AssertionError:
        pass
    m.requires_grad_(False); m.train()
    try:
        assert_frozen(m)
        raise SystemExit("FAIL: assert_frozen accepted a train()-mode module")
    except AssertionError:
        pass
    print("  [ok] assert_frozen rejects trainable / train()-mode encoders")


def test_eval_matches_independent_implementation():
    """Cross-check run_sessions against a from-scratch reimplementation that
    shares no code with cdfscil.fscil."""
    feat = ROOT / "features/mini_imagenet/ViT-B-16_openai_train.npy"
    if not feat.exists():
        print("  [skip] features not cached; run cdfscil.extract_features first")
        return
    b = build_benchmark("mini_imagenet", "data", CMAP)
    Xtr = np.load(feat)
    Xte = np.load(ROOT / "features/mini_imagenet/ViT-B-16_openai_test.npy")

    ours = [r["acc"] for r in run_sessions(b, Xtr, Xte, None, 1.0, 1.0)]

    # --- independent version ------------------------------------------------
    proto, accs = {}, []
    for s in range(b.sessions):
        for c in sorted(set(b.train_labels[b.session_train_ids[s]].tolist())):
            ids = b.session_train_ids[s]
            sel = ids[b.train_labels[ids] == c]
            proto[c] = l2norm(l2norm(Xtr[sel]).mean(0))
        seen = np.arange(b.base_class + s * b.way)
        P = np.stack([proto[int(c)] for c in seen])
        m = b.test_labels < len(seen)
        pred = seen[(l2norm(Xte[m]) @ P.T).argmax(1)]
        accs.append(100.0 * (pred == b.test_labels[m]).mean())

    for i, (a, c) in enumerate(zip(ours, accs)):
        assert abs(a - c) < 1e-6, f"session {i}: {a} vs {c}"
    print(f"  [ok] evaluation matches an independent implementation "
          f"on all {len(ours)} sessions (last={ours[-1]:.2f})")


def test_incremental_stage_uses_no_neural_network():
    """The whole point of the paper is that incremental sessions are
    training-free.  cdfscil/fscil.py -- which implements Eqs. 8-12, i.e. every
    incremental-session computation -- must therefore not even import a deep
    learning framework."""
    import ast as _ast
    src = (ROOT / "cdfscil" / "fscil.py").read_text()
    mods = set()
    for node in _ast.walk(_ast.parse(src)):
        if isinstance(node, _ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, _ast.ImportFrom) and node.level == 0 and node.module:
            mods.add(node.module.split(".")[0])
    banned = mods & {"torch", "tensorflow", "jax", "open_clip"}
    assert not banned, f"fscil.py imports {banned}"
    assert "numpy" in mods
    print(f"  [ok] incremental stage is pure numpy (imports: {sorted(mods)})")


def test_paper_audit_finds_the_known_inconsistencies():
    from cdfscil.report import audit_paper_table
    _, _, mism = audit_paper_table(
        ROOT / "cdfscil/assets/paper/table1_miniimagenet.json")
    names = {m[0] for m in mism}
    assert names == {"CD-FSCIL"}, names
    _, printed, recomputed, delta = mism[0]
    assert abs(printed - 72.53) < 1e-6 and abs(recomputed - 71.07) < 0.01
    print(f"  [ok] Table 1 audit: only CD-FSCIL's Avg is inconsistent "
          f"({printed:.2f} printed vs {recomputed:.2f} actual)")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(tests)} checks\n")
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:                                   # noqa: BLE001
            failed += 1
            print(f"  [FAIL] {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
