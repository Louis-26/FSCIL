"""FSCIL benchmarks (miniImageNet / CIFAR-100 / CUB-200) under the standard
CEC-TOPIC protocol.

Protocol (identical to Tao et al. CVPR'20 and every method in Table 1):

    miniImageNet : 100 classes, 60 base + 8 sessions x (5-way, 5-shot)   ->  9 sessions
    CIFAR-100    : 100 classes, 60 base + 8 sessions x (5-way, 5-shot)   ->  9 sessions
    CUB-200      : 200 classes, 100 base + 10 sessions x (10-way, 5-shot) -> 11 sessions

After session s the model is evaluated on the *full test split of every class
seen so far* and we report top-1 accuracy.  The exact few-shot samples of every
incremental session are fixed by the `index_list/<dataset>/session_*.txt` files
shipped with CEC -- they are what makes numbers comparable across papers, so we
read them verbatim rather than resampling.

The classes are laid out so that label c < base_class  <=>  c is a base class,
and session s (>=1) introduces labels [base + (s-1)*way, base + s*way).

Design note
-----------
Every image in the benchmark gets a stable integer id in `train_index` /
`test_index`.  Backbone features are extracted **once** for all ids and cached;
sessions then only slice into that matrix.  This keeps the training-free
incremental stage genuinely free of any forward pass beyond the one-time cache.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# benchmark configuration
# --------------------------------------------------------------------------- #

DATASET_CONFIG = {
    "mini_imagenet": dict(num_classes=100, base_class=60,  way=5,  shot=5, sessions=9),
    "cifar100":      dict(num_classes=100, base_class=60,  way=5,  shot=5, sessions=9),
    "cub200":        dict(num_classes=200, base_class=100, way=10, shot=5, sessions=11),
}


@dataclass
class FSCILBenchmark:
    """Holds the full image inventory plus the per-session index bookkeeping."""

    name: str
    root: Path
    num_classes: int
    base_class: int
    way: int
    shot: int
    sessions: int

    # inventory -------------------------------------------------------------
    train_paths: list = field(default_factory=list)   # absolute paths OR None for cifar
    train_labels: np.ndarray = None
    test_paths: list = field(default_factory=list)
    test_labels: np.ndarray = None
    train_images: np.ndarray = None                   # cifar only (N,32,32,3) uint8
    test_images: np.ndarray = None

    # per-session train ids (indices into train_paths / train_images) --------
    session_train_ids: list = field(default_factory=list)

    class_names: list = field(default_factory=list)   # human readable, index == label
    wnids: list = field(default_factory=list)         # mini_imagenet only

    # ------------------------------------------------------------------ #
    def seen_classes(self, session: int) -> np.ndarray:
        n = self.base_class + session * self.way
        return np.arange(n)

    def session_classes(self, session: int) -> np.ndarray:
        if session == 0:
            return np.arange(self.base_class)
        lo = self.base_class + (session - 1) * self.way
        return np.arange(lo, lo + self.way)

    def test_ids(self, session: int) -> np.ndarray:
        """Ids of every test image whose label has been seen by `session`."""
        n = self.base_class + session * self.way
        return np.where(self.test_labels < n)[0]

    def get_train_image(self, i: int) -> Image.Image:
        if self.train_images is not None:
            return Image.fromarray(self.train_images[i])
        return Image.open(self.train_paths[i]).convert("RGB")

    def get_test_image(self, i: int) -> Image.Image:
        if self.test_images is not None:
            return Image.fromarray(self.test_images[i])
        return Image.open(self.test_paths[i]).convert("RGB")

    # ------------------------------------------------------------------ #
    def sanity_check(self) -> dict:
        """Assert the protocol invariants; returns a dict for the report."""
        info = {}
        assert len(self.session_train_ids) == self.sessions, (
            f"{self.name}: expected {self.sessions} sessions, "
            f"got {len(self.session_train_ids)}")

        # session 0 == every train image of every base class
        base_expected = np.where(self.train_labels < self.base_class)[0]
        s0 = np.asarray(self.session_train_ids[0])
        assert set(s0.tolist()) == set(base_expected.tolist()), (
            f"{self.name}: session 0 is not exactly the base-class train split "
            f"({len(s0)} vs {len(base_expected)})")
        info["session0_train"] = int(len(s0))

        # incremental sessions: way*shot samples, exactly the right novel classes
        for s in range(1, self.sessions):
            ids = np.asarray(self.session_train_ids[s])
            labs = self.train_labels[ids]
            expect_cls = set(self.session_classes(s).tolist())
            assert len(ids) == self.way * self.shot, (
                f"{self.name} s{s}: {len(ids)} samples, expected "
                f"{self.way * self.shot}")
            assert set(labs.tolist()) == expect_cls, (
                f"{self.name} s{s}: labels {sorted(set(labs.tolist()))} != "
                f"{sorted(expect_cls)}")
            for c in expect_cls:
                assert (labs == c).sum() == self.shot, (
                    f"{self.name} s{s}: class {c} has {(labs == c).sum()} shots")
        info["incremental_train_per_session"] = self.way * self.shot

        # test split grows monotonically and ends at the full test set
        counts = [len(self.test_ids(s)) for s in range(self.sessions)]
        assert counts == sorted(counts), f"{self.name}: test set not monotone"
        assert counts[-1] == len(self.test_labels), (
            f"{self.name}: final test set {counts[-1]} != {len(self.test_labels)}")
        info["test_per_session"] = counts
        info["n_train_total"] = int(len(self.train_labels))
        info["n_test_total"] = int(len(self.test_labels))
        info["n_classes"] = int(self.num_classes)
        assert len(self.class_names) == self.num_classes
        return info


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #

def _index_list_dir(root: Path, dataset: str) -> Path:
    """`index_list` lives either under data/ or in the repo's complementary/."""
    key = {"mini_imagenet": "mini_imagenet",
           "cifar100": "cifar100",
           "cub200": "cub200"}[dataset]
    for cand in (root / "index_list" / key,
                 root.parent / "complementary" / "index_list" / key):
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        f"index_list for {dataset} not found under {root}/index_list or "
        f"{root.parent}/complementary/index_list")


def build_mini_imagenet(root: Path, class_names_map: dict | None = None) -> FSCILBenchmark:
    cfg = DATASET_CONFIG["mini_imagenet"]
    img_dir = root / "miniimagenet" / "images"
    split_dir = root / "miniimagenet" / "split"
    if not img_dir.is_dir():
        raise FileNotFoundError(f"missing {img_dir} -- run scripts/01_prepare_data.sh")

    def _read(split):
        lines = [l.strip() for l in open(split_dir / f"{split}.csv")][1:]
        return [l.split(",") for l in lines if l]

    train_rows, test_rows = _read("train"), _read("test")

    # label id == position of the wnid in sorted order, which is exactly the
    # order the CEC csv files use and the order of index_list/session_*.txt.
    wnids = []
    for _, w in train_rows:
        if w not in wnids:
            wnids.append(w)
    assert len(wnids) == cfg["num_classes"], f"{len(wnids)} wnids found"
    assert wnids == sorted(wnids), "CEC csv is expected to be in sorted-wnid order"
    w2l = {w: i for i, w in enumerate(wnids)}

    train_paths = [str(img_dir / fn) for fn, _ in train_rows]
    train_labels = np.array([w2l[w] for _, w in train_rows], dtype=np.int64)
    test_paths = [str(img_dir / fn) for fn, _ in test_rows]
    test_labels = np.array([w2l[w] for _, w in test_rows], dtype=np.int64)

    fn2id = {Path(p).name: i for i, p in enumerate(train_paths)}

    idx_dir = _index_list_dir(root, "mini_imagenet")
    session_ids = [np.where(train_labels < cfg["base_class"])[0]]
    for s in range(1, cfg["sessions"]):
        lines = [l.strip() for l in open(idx_dir / f"session_{s + 1}.txt") if l.strip()]
        ids = [fn2id[Path(l).name] for l in lines]
        session_ids.append(np.array(ids, dtype=np.int64))

    names = [(class_names_map or {}).get(w, w) for w in wnids]
    return FSCILBenchmark(
        name="mini_imagenet", root=root, **cfg,
        train_paths=train_paths, train_labels=train_labels,
        test_paths=test_paths, test_labels=test_labels,
        session_train_ids=session_ids, class_names=names, wnids=wnids)


def build_cifar100(root: Path) -> FSCILBenchmark:
    cfg = DATASET_CONFIG["cifar100"]
    base = root / "cifar-100-python"
    if not base.is_dir():
        raise FileNotFoundError(f"missing {base} -- run scripts/01_prepare_data.sh")

    def _load(fn):
        with open(base / fn, "rb") as f:
            return pickle.load(f, encoding="latin1")

    tr, te, meta = _load("train"), _load("test"), _load("meta")
    train_images = tr["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_images = te["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    train_labels = np.array(tr["fine_labels"], dtype=np.int64)
    test_labels = np.array(te["fine_labels"], dtype=np.int64)
    names = [n.replace("_", " ") for n in meta["fine_label_names"]]

    idx_dir = _index_list_dir(root, "cifar100")
    session_ids = [np.where(train_labels < cfg["base_class"])[0]]
    for s in range(1, cfg["sessions"]):
        ids = [int(l) for l in open(idx_dir / f"session_{s + 1}.txt") if l.strip()]
        session_ids.append(np.array(ids, dtype=np.int64))

    return FSCILBenchmark(
        name="cifar100", root=root, **cfg,
        train_paths=[None] * len(train_labels), train_labels=train_labels,
        test_paths=[None] * len(test_labels), test_labels=test_labels,
        train_images=train_images, test_images=test_images,
        session_train_ids=session_ids, class_names=names)


def build_cub200(root: Path) -> FSCILBenchmark:
    cfg = DATASET_CONFIG["cub200"]
    base = root / "CUB_200_2011"
    if not base.is_dir():
        raise FileNotFoundError(f"missing {base} -- run scripts/01_prepare_data.sh")

    id2rel = {}
    for line in open(base / "images.txt"):
        i, rel = line.strip().split(" ", 1)
        id2rel[int(i)] = rel
    id2lab = {}
    for line in open(base / "image_class_labels.txt"):
        i, c = line.strip().split()
        id2lab[int(i)] = int(c) - 1                      # -> 0-based
    is_train = {}
    for line in open(base / "train_test_split.txt"):
        i, t = line.strip().split()
        is_train[int(i)] = int(t) == 1

    tr_paths, tr_labels, te_paths, te_labels = [], [], [], []
    for i, rel in sorted(id2rel.items()):
        p = str(base / "images" / rel)
        if is_train[i]:
            tr_paths.append(p); tr_labels.append(id2lab[i])
        else:
            te_paths.append(p); te_labels.append(id2lab[i])
    train_labels = np.array(tr_labels, dtype=np.int64)
    test_labels = np.array(te_labels, dtype=np.int64)

    # session txt lines look like  CUB_200_2011/images/<cls>/<file>.jpg
    rel2id = {"/".join(Path(p).parts[-3:]): i for i, p in enumerate(tr_paths)}
    idx_dir = _index_list_dir(root, "cub200")
    session_ids = [np.where(train_labels < cfg["base_class"])[0]]
    for s in range(1, cfg["sessions"]):
        ids = []
        for l in open(idx_dir / f"session_{s + 1}.txt"):
            l = l.strip()
            if not l:
                continue
            ids.append(rel2id["/".join(Path(l).parts[-3:])])
        session_ids.append(np.array(ids, dtype=np.int64))

    names = []
    for line in open(base / "classes.txt"):
        _, nm = line.strip().split(" ", 1)
        names.append(nm.split(".", 1)[1].replace("_", " "))

    return FSCILBenchmark(
        name="cub200", root=root, **cfg,
        train_paths=tr_paths, train_labels=train_labels,
        test_paths=te_paths, test_labels=test_labels,
        session_train_ids=session_ids, class_names=names)


def build_benchmark(dataset: str, root: str | os.PathLike,
                    class_names_map: dict | None = None) -> FSCILBenchmark:
    root = Path(root)
    if dataset == "mini_imagenet":
        return build_mini_imagenet(root, class_names_map)
    if dataset == "cifar100":
        return build_cifar100(root)
    if dataset == "cub200":
        return build_cub200(root)
    raise ValueError(f"unknown dataset {dataset}")


# --------------------------------------------------------------------------- #
# torch datasets
# --------------------------------------------------------------------------- #


class ImageListDataset(Dataset):
    """Wraps a subset of the benchmark inventory with an arbitrary transform."""

    def __init__(self, bench: FSCILBenchmark, ids, split: str, transform):
        self.bench, self.ids, self.split, self.transform = bench, np.asarray(ids), split, transform
        self.labels = (bench.train_labels if split == "train" else bench.test_labels)[self.ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        gid = int(self.ids[i])
        img = (self.bench.get_train_image(gid) if self.split == "train"
               else self.bench.get_test_image(gid))
        return self.transform(img), int(self.labels[i]), gid
