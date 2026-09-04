"""Generative path: sample exemplars from the frozen diffusion model and turn
them into prototypes (Eqs. 6-8).

    v~_i^(c) = DiffusionSampler(c)            Eq. 6   (DDIM, 50 steps)
    x~_i^(c) = E_CLIP^img( v~_i^(c) )         Eq. 7   (frozen CLIP)
    x_gen_c  = (1/N) sum_i x~_i^(c)           Eq. 8

Nothing is trained here: the UNet and CLIP are both frozen, so this is part of
the "training-free" incremental stage.

    python -m cdfscil.generate_prototypes \
        --ckpt checkpoints/mini_imagenet_ViT-B-16_openai_llm/model_final.pt \
        --n-gen 64 --ddim-steps 50 --guidance 1.0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .clip_backbone import load_clip, model_tag
from .data import build_benchmark
from .diffusion import GaussianDiffusion, to_pil_batch
from .extract_features import load_classname_map
from .fscil import l2norm
from .unet import ConditionalUNet
from .utils import Timer, get_logger, set_seed


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="mini_imagenet")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--features", default="features")
    ap.add_argument("--clip-model", default="ViT-B-16")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--text-mode", default=None,
                    help="defaults to the text mode the model was trained with")
    ap.add_argument("--n-gen", type=int, default=64, help="N in Eq. 8")
    ap.add_argument("--ddim-steps", type=int, default=50, help="T_sample")
    ap.add_argument("--guidance", type=float, default=1.0,
                    help="1.0 = plain conditional sampler (the paper's setting)")
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--use-raw-weights", action="store_true",
                    help="use the non-EMA weights")
    ap.add_argument("--save-grids", type=int, default=8,
                    help="how many images per class to dump as a PNG grid")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    log = get_logger("genproto", f"logs/generate_prototypes_{args.dataset}.log")
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ta = ck["args"]
    text_mode = args.text_mode or ta["text_mode"]
    log.info(f"checkpoint step={ck.get('step')} trained_text_mode={ta['text_mode']} "
             f"sampling_text_mode={text_mode}")

    net = ConditionalUNet(image_size=ta["image_size"], base=ta.get("base_ch", 128),
                          ch_mult=tuple(ta.get("ch_mult", (1, 2, 2, 4))),
                          num_res_blocks=ta.get("num_res_blocks", 3),
                          attn_resolutions=tuple(ta.get("attn_res", (16, 8))),
                          dropout=ta.get("dropout", 0.1),
                          cond_dim=ck["cond_dim"]).to(device)
    net.load_state_dict(ck["model" if args.use_raw_weights else "ema"])
    net.eval()
    diff = GaussianDiffusion(ta["timesteps"], ta["schedule"], device)

    bench = build_benchmark(args.dataset, args.data_root,
                            load_classname_map(args.dataset))
    tag = model_tag(args.clip_model, args.pretrained)
    text_emb = torch.from_numpy(
        np.load(Path(args.features) / args.dataset /
                f"{tag}_text_{text_mode.replace('+', '-')}.npy")).float().to(device)

    clip_model, preprocess, _ = load_clip(args.clip_model, args.pretrained, device)

    # CLIP's own preprocessing, applied to generated tensors without a PIL round-trip
    import torchvision.transforms as T
    norm = T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                       std=(0.26862954, 0.26130258, 0.27577711))

    def clip_encode(imgs01: torch.Tensor) -> torch.Tensor:
        """imgs01 in [0,1], (B,3,h,w) -> CLIP features."""
        x = torch.nn.functional.interpolate(imgs01, size=224, mode="bicubic",
                                            align_corners=False, antialias=True)
        return clip_model.encode_image(norm(x.clamp(0, 1))).float()

    outdir = Path(args.out) if args.out else Path(args.ckpt).parent
    outdir.mkdir(parents=True, exist_ok=True)
    grid_dir = outdir / f"generated_g{args.guidance}"
    grid_dir.mkdir(exist_ok=True)

    g = torch.Generator(device=device).manual_seed(args.seed)
    classes = np.arange(bench.num_classes)
    protos = np.zeros((len(classes), text_emb.shape[1]), np.float32)
    t = Timer()
    for c in classes:
        feats, keep = [], []
        remaining = args.n_gen
        while remaining > 0:
            b = min(args.batch_size, remaining)
            cond = text_emb[c][None].repeat(b, 1)
            v = diff.ddim_sample(net, (b, 3, ta["image_size"], ta["image_size"]),
                                 cond, steps=args.ddim_steps, eta=args.eta,
                                 guidance=args.guidance, device=device, generator=g)
            imgs01 = (v.clamp(-1, 1) + 1) / 2
            feats.append(clip_encode(imgs01).cpu())
            if len(keep) == 0:
                keep = to_pil_batch(v[:args.save_grids])
            remaining -= b
        f = torch.cat(feats).numpy()
        protos[c] = l2norm(f).mean(0)
        if args.save_grids and (c % 5 == 0 or c >= bench.base_class):
            Image.fromarray(np.concatenate(list(keep), axis=1)).save(
                grid_dir / f"class{c:03d}_{bench.class_names[c].replace('/', '-')}.png")
        if (c + 1) % 10 == 0:
            log.info(f"  {c+1}/{len(classes)} classes [{t.human()}]")

    out = outdir / (f"gen_protos_img_n{args.n_gen}_s{args.ddim_steps}"
                    f"_g{args.guidance}_step{ck.get('step')}.npz")
    np.savez(out, classes=classes, protos=protos)
    log.info(f"saved {out}  ({t.human()})")
    print(out)


if __name__ == "__main__":
    main()
