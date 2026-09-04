"""Small shared helpers: determinism, logging, json/csv io, timers."""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG we touch.

    `deterministic=True` also pins cuDNN so that two runs of the evaluation
    pipeline on the same machine give bit-identical numbers.  Diffusion
    *training* is left on the fast kernels (see train_diffusion.py) because the
    reproduction only requires the trained model, not bit-identical weights.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding, derived from torch's base seed."""
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #


def get_logger(name: str = "cdfscil", logfile: str | os.PathLike | None = None,
               level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:                      # already configured
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s",
                            datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if logfile is not None:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# io
# --------------------------------------------------------------------------- #


def read_json(path: str | os.PathLike):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path: str | os.PathLike, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def write_csv(rows, path: str | os.PathLike, header=None) -> None:
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if header:
            w.writerow(header)
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# misc
# --------------------------------------------------------------------------- #


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0

    def human(self) -> str:
        s = int(self.elapsed())
        if s >= 3600:
            return f"{s / 3600:.2f}h"
        if s >= 60:
            return f"{s / 60:.1f}m"
        return f"{s}s"


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def pick_device(gpu: str | int | None = None) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if gpu is None or gpu == "auto":
        return torch.device("cuda:0")
    return torch.device(f"cuda:{gpu}")


def free_gpu_ranking():
    """Return cuda indices sorted by free memory (most free first)."""
    if not torch.cuda.is_available():
        return []
    stats = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        stats.append((free, i))
    stats.sort(reverse=True)
    return [i for _, i in stats]
