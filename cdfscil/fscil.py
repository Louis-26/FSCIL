"""Training-free prototype construction and session-wise FSCIL evaluation.

Implements Section 3.3 / 3.4 of the paper:

    Eq.  8   x_gen_c  = (1/N) sum_i  E_img(v~_i^(c))          generative path
    Eq. 10   x_real_c = (1/K) sum_k  E_img(v_k)               real path
    Eq. 11   x_c      = (1-alpha) * x_gen_c + alpha * x_real_c
    Eq. 12   y_hat    = argmax_c  cos(x_q, x_c)

Normalisation is an implementation choice the paper leaves open.  We L2-normalise
each feature before averaging (the standard CLIP prototype recipe) and again
after fusion so that `alpha` interpolates on the unit sphere rather than being
dominated by whichever path happens to have the larger norm.  `--no-prenorm`
switches to raw averaging so the choice can be ablated instead of assumed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def l2norm(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def class_means(feats: np.ndarray, labels: np.ndarray, classes,
                prenorm: bool = True) -> dict:
    """Mean feature per class (Eq. 10 / Eq. 8 depending on what `feats` are).

    With `prenorm`, features are L2-normalised before averaging AND the mean is
    re-normalised.  The second normalisation matters: the mean of K unit vectors
    has norm < 1 (0.876 for a 500-image base class here), so without it the real
    and generative prototypes enter Eq. 11 at different scales and `alpha` stops
    being a clean interpolation weight.
    """
    out = {}
    f = l2norm(feats) if prenorm else feats
    for c in classes:
        m = labels == c
        if not m.any():
            continue
        mu = f[m].mean(0)
        out[int(c)] = l2norm(mu) if prenorm else mu
    return out


# --------------------------------------------------------------------------- #
# prototype bank
# --------------------------------------------------------------------------- #


@dataclass
class PrototypeBank:
    """Accumulates one prototype per class as sessions arrive.

    Prototypes of previously-seen classes are *never* touched again, which is
    what makes the incremental stage training-free and forgetting-free by
    construction: session s only writes rows for the classes it introduces.
    """
    dim: int
    num_classes: int
    prenorm: bool = True
    postnorm: bool = True

    def __post_init__(self):
        self.real = np.zeros((self.num_classes, self.dim), np.float32)
        self.gen = np.zeros((self.num_classes, self.dim), np.float32)
        self.has_real = np.zeros(self.num_classes, bool)
        self.has_gen = np.zeros(self.num_classes, bool)

    def add_real(self, feats: np.ndarray, labels: np.ndarray, classes) -> None:
        for c, m in class_means(feats, labels, classes, self.prenorm).items():
            assert not self.has_real[c], f"class {c} already has a real prototype"
            self.real[c] = m
            self.has_real[c] = True

    def add_gen(self, protos: dict) -> None:
        for c, v in protos.items():
            self.gen[c] = l2norm(v) if self.prenorm else v
            self.has_gen[c] = True

    def fused(self, classes, alpha_base: float, alpha_novel: float,
              base_class: int) -> np.ndarray:
        """Eq. 11, evaluated only for `classes`."""
        classes = np.asarray(classes)
        a = np.where(classes < base_class, alpha_base, alpha_novel)[:, None]
        g, r = self.gen[classes], self.real[classes]
        # a class with no generative prototype falls back to the real one, so
        # that alpha never silently mixes in a zero vector.
        no_gen = ~self.has_gen[classes]
        p = (1.0 - a) * g + a * r
        p[no_gen] = r[no_gen]
        return l2norm(p) if self.postnorm else p


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #

def cosine_predict(query: np.ndarray, protos: np.ndarray,
                   classes: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """Eq. 12.  Returns predicted *class labels* (not row indices)."""
    q = l2norm(query)
    p = l2norm(protos)
    preds = np.empty(len(q), np.int64)
    for i in range(0, len(q), chunk):
        sim = q[i:i + chunk] @ p.T
        preds[i:i + chunk] = classes[sim.argmax(1)]
    return preds


def session_metrics(preds: np.ndarray, labels: np.ndarray,
                    base_class: int) -> dict:
    """Overall / base-subset / novel-subset top-1 accuracy for one session."""
    correct = preds == labels
    base_m = labels < base_class
    novel_m = ~base_m
    return {
        "acc": float(correct.mean() * 100),
        "base_acc": float(correct[base_m].mean() * 100) if base_m.any() else float("nan"),
        "novel_acc": float(correct[novel_m].mean() * 100) if novel_m.any() else float("nan"),
        "n": int(len(labels)),
    }


def run_sessions(bench, train_feats: np.ndarray, test_feats: np.ndarray,
                 gen_protos_by_class: dict | None = None,
                 alpha_base: float = 1.0, alpha_novel: float = 1.0,
                 prenorm: bool = True, postnorm: bool = True) -> list[dict]:
    """Walk the incremental protocol and return one metrics dict per session.

    `gen_protos_by_class[c]` is the diffusion-generated prototype for class c
    (already in CLIP feature space).  Pass None to run the real-prototype-only
    configuration, i.e. alpha = 1.
    """
    bank = PrototypeBank(train_feats.shape[1], bench.num_classes, prenorm, postnorm)
    if gen_protos_by_class:
        bank.add_gen(gen_protos_by_class)

    results = []
    for s in range(bench.sessions):
        ids = bench.session_train_ids[s]
        bank.add_real(train_feats[ids], bench.train_labels[ids],
                      bench.session_classes(s))

        seen = bench.seen_classes(s)
        assert bank.has_real[seen].all(), f"session {s}: missing real prototypes"

        protos = bank.fused(seen, alpha_base, alpha_novel, bench.base_class)
        tids = bench.test_ids(s)
        preds = cosine_predict(test_feats[tids], protos, seen)
        m = session_metrics(preds, bench.test_labels[tids], bench.base_class)
        m["session"] = s
        m["n_classes"] = int(len(seen))
        results.append(m)
    return results


def summarize(results: list[dict]) -> dict:
    accs = [r["acc"] for r in results]
    return {
        "per_session": [round(a, 2) for a in accs],
        "avg": round(float(np.mean(accs)), 2),
        "last": round(accs[-1], 2),
        "pd": round(accs[0] - accs[-1], 2),      # performance drop, session0 -> last
    }
