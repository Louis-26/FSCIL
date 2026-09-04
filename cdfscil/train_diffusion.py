"""Base-session training of the conditional diffusion model (Sec. 3.3, Eq. 5).

The model sees ONLY session-0 images (the base classes).  After this script it is
frozen forever -- that is the paper's structural argument against forgetting.

    python -m cdfscil.train_diffusion --dataset mini_imagenet --steps 120000

Implementation choices the paper does not pin down, all exposed as flags:
  * EMA of the weights (decay 0.9999). Standard for DDPM sampling quality;
    both the raw and the EMA weights are checkpointed.
  * 10% condition dropout so classifier-free guidance is *available* at sampling
    time. Sampling defaults to guidance=1.0, i.e. the paper's plain conditional
    reverse process.
  * 64x64 training resolution (the standard DDPM/ADM image-space resolution).
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from .clip_backbone import model_tag
from .data import ImageListDataset, build_benchmark
from .diffusion import GaussianDiffusion, to_pil_batch
from .extract_features import load_classname_map
from .unet import ConditionalUNet
from .utils import Timer, count_params, get_logger, set_seed, worker_init_fn


def train_transform(size: int):
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),      # -> [-1, 1]
    ])


class EMA:
    """Exponential moving average of the weights, WITH warm-up.

    Without warm-up the shadow retains decay**step of the random initialisation:
    at decay=0.9999 that is 37% after 10k steps and 1.8% after 40k, which is
    enough to make DDIM sampling return pure noise even while the training loss
    looks healthy.  (We hit exactly this: at step 10k the raw weights produced
    recognisable images and the EMA weights produced noise.)  The standard fix,
    used by ADM and diffusers, ramps the decay in:

        decay_t = min(decay, (1 + t) / (10 + t))
    """

    def __init__(self, model, decay=0.9999, warmup=True):
        self.decay, self.warmup = decay, warmup
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def decay_at(self, step: int) -> float:
        if not self.warmup:
            return self.decay
        return min(self.decay, (1.0 + step) / (10.0 + step))

    @torch.no_grad()
    def reinit_from(self, model):
        """Reset the shadow to the current weights (used when resuming a run
        whose EMA was polluted by a warm-up-free schedule)."""
        self.shadow.load_state_dict(model.state_dict())

    @torch.no_grad()
    def update(self, model, step: int = 0):
        d = self.decay_at(step)
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(d).add_(p.detach(), alpha=1 - d)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mini_imagenet")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--features", default="features")
    ap.add_argument("--clip-model", default="ViT-B-16")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--text-mode", default="llm",
                    choices=["classname", "template", "llm", "llm+template"])
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--base-ch", type=int, default=128)
    ap.add_argument("--ch-mult", type=int, nargs="+", default=[1, 2, 2, 4])
    ap.add_argument("--num-res-blocks", type=int, default=3)
    ap.add_argument("--attn-res", type=int, nargs="+", default=[16, 8])
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the UNet (~1.7x faster, ~2x less memory)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--schedule", default="cosine", choices=["cosine", "linear"])
    ap.add_argument("--p-uncond", type=float, default=0.1)
    ap.add_argument("--ema-decay", type=float, default=0.9999)
    ap.add_argument("--no-ema-warmup", action="store_true",
                    help="disable the (1+t)/(10+t) EMA warm-up ramp")
    ap.add_argument("--ema-reinit", action="store_true",
                    help="on --resume, reset the EMA shadow to the loaded weights")
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=10000)
    ap.add_argument("--sample-every", type=int, default=20000)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    args = ap.parse_args()

    run = f"{args.dataset}_{model_tag(args.clip_model, args.pretrained)}_{args.text_mode}"
    outdir = Path(args.out) / run
    outdir.mkdir(parents=True, exist_ok=True)
    log = get_logger("diff", f"logs/train_diffusion_{run}.log")
    set_seed(args.seed, deterministic=False)          # speed > bitwise repeatability
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(f"cuda:{args.gpu}")
    log.info(f"run={run} device={device} args={vars(args)}")

    # ---- data: base session only ----------------------------------------- #
    bench = build_benchmark(args.dataset, args.data_root,
                            load_classname_map(args.dataset))
    bench.sanity_check()
    base_ids = bench.session_train_ids[0]
    assert (bench.train_labels[base_ids] < bench.base_class).all(), \
        "diffusion must only ever see base-session images"
    ds = ImageListDataset(bench, base_ids, "train", train_transform(args.image_size))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True,
                    persistent_workers=True, worker_init_fn=worker_init_fn)
    log.info(f"base-session images: {len(ds)} ({len(dl)} iters/epoch)")

    # ---- class conditions p_c (Eq. 2) ------------------------------------ #
    tpath = (Path(args.features) / args.dataset /
             f"{model_tag(args.clip_model, args.pretrained)}"
             f"_text_{args.text_mode.replace('+', '-')}.npy")
    text_emb = torch.from_numpy(np.load(tpath)).float().to(device)
    log.info(f"class conditions {tuple(text_emb.shape)} from {tpath}")

    # ---- model ------------------------------------------------------------ #
    net = ConditionalUNet(image_size=args.image_size, base=args.base_ch,
                          ch_mult=tuple(args.ch_mult),
                          num_res_blocks=args.num_res_blocks,
                          attn_resolutions=tuple(args.attn_res),
                          dropout=args.dropout,
                          cond_dim=text_emb.shape[1]).to(device)
    log.info(f"UNet params: {count_params(net)/1e6:.1f}M")
    ema = EMA(net, args.ema_decay, warmup=not args.no_ema_warmup)  # tracks the uncompiled module
    if args.compile:
        net = torch.compile(net)
        log.info("torch.compile enabled")
    diff = GaussianDiffusion(args.timesteps, args.schedule, device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp == "fp16"))

    step0 = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        _raw = net._orig_mod if hasattr(net, "_orig_mod") else net
        _raw.load_state_dict(ck["model"])
        if args.ema_reinit:
            ema.reinit_from(_raw)
            log.info("EMA shadow re-initialised from the loaded weights")
        else:
            ema.shadow.load_state_dict(ck["ema"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        step0 = ck["step"]
        log.info(f"resumed from {args.resume} at step {step0}")

    def _raw_state():
        """Strip the _orig_mod prefix torch.compile adds, so checkpoints load
        into a plain ConditionalUNet."""
        sd = net.state_dict()
        return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

    def save(step, final=False):
        payload = {"model": _raw_state(), "ema": ema.shadow.state_dict(),
                   "step": step, "args": vars(args),
                   "cond_dim": text_emb.shape[1]}
        p = outdir / ("model_final.pt" if final else f"model_step{step}.pt")
        torch.save({**payload, "opt": opt.state_dict()}, p)
        torch.save(payload, outdir / "model_latest.pt")   # stable pointer
        log.info(f"saved {p}")

    # ---- train ------------------------------------------------------------ #
    base_net = net._orig_mod if hasattr(net, "_orig_mod") else net
    net.train()
    t = Timer()
    it = iter(dl)
    losses, tick = [], time.time()
    for step in range(step0, args.steps):
        try:
            imgs, labels, _ = next(it)
        except StopIteration:
            it = iter(dl); imgs, labels, _ = next(it)
        imgs = imgs.to(device, non_blocking=True)
        cond = text_emb[labels.to(device)]

        lr = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            loss = diff.training_loss(net, imgs, cond, args.p_uncond)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        ema.update(base_net, step + 1)
        losses.append(loss.item())

        if (step + 1) % args.log_every == 0:
            ips = args.log_every * args.batch_size / (time.time() - tick)
            log.info(f"step {step+1}/{args.steps} loss {np.mean(losses[-args.log_every:]):.4f} "
                     f"lr {lr:.2e} {ips:.0f} img/s elapsed {t.human()}")
            tick = time.time()
        if (step + 1) % args.ckpt_every == 0:
            save(step + 1)
        if (step + 1) % args.sample_every == 0:
            _dump_samples(ema.shadow, diff, text_emb, bench, outdir, step + 1,
                          args.image_size, device, log)

    save(args.steps, final=True)
    json.dump({"loss_curve": losses[::50], "steps": args.steps,
               "wall_clock_s": t.elapsed()},
              open(outdir / "train_log.json", "w"))
    log.info(f"training done in {t.human()}")


@torch.no_grad()
def _dump_samples(model, diff, text_emb, bench, outdir, step, size, device, log):
    """Small qualitative grid: 4 base classes and 4 novel classes."""
    from PIL import Image
    model.eval()          # the EMA shadow is always in eval mode
    cls = list(range(0, 4)) + list(range(bench.base_class, bench.base_class + 4))
    cond = text_emb[torch.tensor(cls, device=device)]
    v = diff.ddim_sample(model, (len(cls), 3, size, size), cond, steps=50,
                         guidance=1.0, device=device)
    arr = to_pil_batch(v)
    grid = np.concatenate(list(arr), axis=1)
    d = outdir / "samples"; d.mkdir(exist_ok=True)
    Image.fromarray(grid).save(d / f"step{step}.png")
    names = [bench.class_names[c] for c in cls]
    log.info(f"sample grid @ step {step}: {names}")


if __name__ == "__main__":
    main()
