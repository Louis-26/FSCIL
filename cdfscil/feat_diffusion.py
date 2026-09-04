"""Feature-space conditional diffusion -- the Section 2.3 reading of the paper.

Sec. 2.3 ("Our Positioning") states the model synthesises
    "high-fidelity feature prototypes in the CLIP embedding space",
whereas Sec. 3.2 / 3.5 describe an image-space UNet whose samples are then
re-encoded by CLIP.  The two readings are mutually exclusive, so we implement
both and report both (see overview/06_discrepancies.md).

This module is the feature-space one: a conditional DDPM over the 512-d CLIP
vector itself.  The denoiser is a residual MLP with adaptive-LayerNorm
conditioning on (t, p_c) -- the direct 1-D analogue of the image UNet.  It
trains in minutes rather than hours, which makes the alpha ablation cheap.

Data whitening: CLIP features are L2-normalised and then standardised with the
*base-session* mean/std before diffusion, and the transform is inverted at
sampling time.  Statistics come only from session-0 data, so no novel-class
information leaks in.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .clip_backbone import model_tag
from .data import build_benchmark
from .diffusion import GaussianDiffusion
from .extract_features import load_classname_map
from .fscil import l2norm
from .unet import timestep_embedding, zero_module
from .utils import Timer, count_params, get_logger, set_seed


class ResMLPBlock(nn.Module):
    """Pre-norm residual MLP with adaLN (scale/shift) conditioning."""

    def __init__(self, width: int, emb_ch: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(emb_ch, 2 * width))
        self.ff = nn.Sequential(nn.Linear(width, 4 * width), nn.SiLU(),
                                nn.Dropout(dropout),
                                zero_module(nn.Linear(4 * width, width)))

    def forward(self, x, emb):
        scale, shift = self.mod(emb).chunk(2, dim=-1)
        h = self.norm(x) * (1 + scale) + shift
        return x + self.ff(h)


class FeatureDenoiser(nn.Module):
    """eps_theta(x_t, t, phi(p_c)) for x in R^D (D = CLIP embedding dim)."""

    def __init__(self, dim=512, width=1024, depth=8, cond_dim=512,
                 time_ch=256, dropout=0.0):
        super().__init__()
        self.time_ch = time_ch
        emb_ch = width
        self.time_embed = nn.Sequential(nn.Linear(time_ch, emb_ch), nn.SiLU(),
                                        nn.Linear(emb_ch, emb_ch))
        self.cond_embed = nn.Sequential(nn.Linear(cond_dim, emb_ch), nn.SiLU(),
                                        nn.Linear(emb_ch, emb_ch))
        self.null_cond = nn.Parameter(torch.zeros(cond_dim))
        self.in_proj = nn.Linear(dim, width)
        self.blocks = nn.ModuleList([ResMLPBlock(width, emb_ch, dropout)
                                     for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(width),
                                 zero_module(nn.Linear(width, dim)))

    def forward(self, x, t, cond=None, drop_mask=None):
        emb = self.time_embed(timestep_embedding(t, self.time_ch))
        if cond is None:
            cond = self.null_cond.expand(x.shape[0], -1)
        elif drop_mask is not None:
            cond = torch.where(drop_mask[:, None],
                               self.null_cond.expand_as(cond), cond)
        emb = emb + self.cond_embed(cond)
        h = self.in_proj(x)
        for b in self.blocks:
            h = b(h, emb)
        return self.out(h)


# --------------------------------------------------------------------------- #

class FeatureWhitener:
    """L2-normalise, then standardise with base-session statistics."""

    def __init__(self, mean: np.ndarray, std: np.ndarray, scale: float = 1.0):
        self.mean, self.std, self.scale = mean, std, scale

    @classmethod
    def fit(cls, feats: np.ndarray, scale: float = 1.0):
        f = l2norm(feats)
        return cls(f.mean(0), f.std(0) + 1e-6, scale)

    def forward(self, feats: np.ndarray) -> np.ndarray:
        return ((l2norm(feats) - self.mean) / self.std) * self.scale

    def inverse(self, z: np.ndarray) -> np.ndarray:
        return l2norm(z / self.scale * self.std + self.mean)

    def state(self):
        return {"mean": self.mean, "std": self.std, "scale": self.scale}

    @classmethod
    def load(cls, d):
        return cls(d["mean"], d["std"], float(d["scale"]))


# --------------------------------------------------------------------------- #

def _tags(args):
    base = model_tag(args.clip_model, args.pretrained)
    return (args.feature_tag or base), (args.text_tag or base)


def _run_dir(args):
    base = model_tag(args.clip_model, args.pretrained)
    name = args.run_name or f"{args.dataset}_{args.feature_tag or base}_{args.text_mode}"
    if args.oracle_all_classes:
        name += "_ORACLE"
    return Path(args.out) / name


def train(args, log):
    device = torch.device(f"cuda:{args.gpu}")
    tag, ttag = _tags(args)
    bench = build_benchmark(args.dataset, args.data_root,
                            load_classname_map(args.dataset))
    bench.sanity_check()

    fdir = Path(args.features) / args.dataset
    train_feats = np.load(fdir / f"{tag}_train.npy")
    text_emb = np.load(fdir / f"{ttag}_text_{args.text_mode.replace('+', '-')}.npy")

    if args.oracle_all_classes:
        log.warning("ORACLE MODE: training on ALL classes. This VIOLATES the FSCIL "
                    "protocol and is only an upper bound on the generative path.")
        ids = np.arange(len(bench.train_labels))
    else:
        ids = bench.session_train_ids[0]
        assert (bench.train_labels[ids] < bench.base_class).all(), \
            "the diffusion model may only see base-session images"
    X = train_feats[ids]
    Y = bench.train_labels[ids]
    log.info(f"training features {X.shape} over {len(np.unique(Y))} classes "
             f"({'ORACLE all-class' if args.oracle_all_classes else 'base session only'})")

    wh = FeatureWhitener.fit(X, args.scale)
    Z = torch.from_numpy(wh.forward(X)).float().to(device)
    Yt = torch.from_numpy(Y).long().to(device)
    C = torch.from_numpy(text_emb).float().to(device)

    net = FeatureDenoiser(dim=X.shape[1], width=args.width, depth=args.depth,
                          cond_dim=C.shape[1], dropout=args.dropout).to(device)
    log.info(f"FeatureDenoiser params: {count_params(net)/1e6:.2f}M")
    diff = GaussianDiffusion(args.timesteps, args.schedule, device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    ema = {k: v.clone() for k, v in net.state_dict().items()}
    t = Timer(); net.train()
    for step in range(args.steps):
        idx = torch.randint(0, Z.shape[0], (args.batch_size,), device=device)
        x0, cond = Z[idx], C[Yt[idx]]
        opt.zero_grad(set_to_none=True)
        loss = diff.training_loss(net, x0, cond, args.p_uncond)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            for k, v in net.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(args.ema_decay).add_(v, alpha=1 - args.ema_decay)
                else:
                    ema[k].copy_(v)
        if (step + 1) % args.log_every == 0:
            log.info(f"step {step+1}/{args.steps} loss {loss.item():.4f} "
                     f"lr {sched.get_last_lr()[0]:.2e} [{t.human()}]")

    out = _run_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "ema": ema,
                "whitener": wh.state(), "args": vars(args),
                "dim": X.shape[1], "cond_dim": C.shape[1]},
               out / "feat_diffusion.pt")
    log.info(f"saved {out/'feat_diffusion.pt'} in {t.human()}")
    return out / "feat_diffusion.pt"


@torch.no_grad()
def sample_prototypes(ckpt_path, args, log):
    """Generate N exemplars per class and average them -> x_gen_c (Eq. 6-8)."""
    device = torch.device(f"cuda:{args.gpu}")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    _, ttag = _tags(args)
    bench = build_benchmark(args.dataset, args.data_root,
                            load_classname_map(args.dataset))
    text_emb = np.load(Path(args.features) / args.dataset /
                       f"{ttag}_text_{args.text_mode.replace('+', '-')}.npy")

    a = ck["args"]
    net = FeatureDenoiser(dim=ck["dim"], width=a["width"], depth=a["depth"],
                          cond_dim=ck["cond_dim"]).to(device)
    net.load_state_dict(ck["ema" if args.use_ema else "model"])
    net.eval()
    wh = FeatureWhitener.load(ck["whitener"])
    diff = GaussianDiffusion(a["timesteps"], a["schedule"], device)

    g = torch.Generator(device=device).manual_seed(args.seed)
    classes = np.arange(bench.num_classes)
    protos = np.zeros((len(classes), ck["dim"]), np.float32)
    t = Timer()
    for c in classes:
        cond = torch.from_numpy(text_emb[c]).float().to(device)[None].repeat(args.n_gen, 1)
        z = diff.ddim_sample(net, (args.n_gen, ck["dim"]), cond,
                             steps=args.ddim_steps, guidance=args.guidance,
                             device=device, generator=g, clip_denoised=False)
        x = wh.inverse(z.float().cpu().numpy())
        protos[c] = l2norm(x).mean(0)
        if (c + 1) % 25 == 0:
            log.info(f"  sampled {c+1}/{len(classes)} classes [{t.human()}]")
    out = Path(ckpt_path).parent / f"gen_protos_feat_n{args.n_gen}_g{args.guidance}.npz"
    np.savez(out, classes=classes, protos=protos)
    log.info(f"saved {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mini_imagenet")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--features", default="features")
    ap.add_argument("--clip-model", default="ViT-B-16")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--text-mode", default="llm")
    ap.add_argument("--feature-tag", default=None,
                    help="override the cached-feature tag, e.g. resnet18_mini_imagenet")
    ap.add_argument("--text-tag", default=None,
                    help="override the tag used for the text conditions")
    ap.add_argument("--oracle-all-classes", action="store_true",
                    help="UPPER BOUND ONLY - train on every class, violating the "
                         "FSCIL protocol, to measure the ceiling of the generative path")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--out", default="checkpoints")
    # model / optim
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--schedule", default="cosine")
    ap.add_argument("--p-uncond", type=float, default=0.1)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--log-every", type=int, default=2000)
    # sampling
    ap.add_argument("--n-gen", type=int, default=64)
    ap.add_argument("--ddim-steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--use-ema", action="store_true", default=True)
    ap.add_argument("--sample-only", default=None, help="path to a checkpoint")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    _rn = _run_dir(args).name
    log = get_logger("featdiff", f"logs/feat_diffusion_{_rn}.log")
    set_seed(args.seed)
    ckpt = Path(args.sample_only) if args.sample_only else train(args, log)
    sample_prototypes(ckpt, args, log)


if __name__ == "__main__":
    main()
