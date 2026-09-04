"""Class-conditional image-space UNet, eps_theta(v_t, t, c)  (Sec. 3.2 / Fig. 1).

A standard DDPM/ADM denoiser: GroupNorm + SiLU residual blocks, self-attention
at the low-resolution stages, sinusoidal timestep embedding.  The class
condition is the CLIP text embedding p_c (Eq. 2) mapped by phi (an MLP) into the
timestep-embedding space and *added* to it -- the "Condition Fusion" box of
Fig. 1 -- so every residual block is modulated by (t, c) through adaptive
group-norm scale/shift.

The paper reports "around 110M parameters".  The default config below
(base=128, mult=(1,2,2,4), 3 res blocks, attention at 16 and 8) is 102.3M at
64x64 -- the closest standard ADM-style configuration to that figure that also
trains in a reasonable time here.  Adding attention at resolution 32 as well
(`--attn-res 32 16 8`) gives 104.2M but costs ~20% throughput for no measurable
benefit at this budget.  `python -m cdfscil.unet` prints the exact count.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period)
                      * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(a), torch.sin(a)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def zero_module(m: nn.Module) -> nn.Module:
    for p in m.parameters():
        p.detach().zero_()
    return m


def norm(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(32 if ch % 32 == 0 else 8, ch)


class ResBlock(nn.Module):
    """Residual block with adaptive group-norm conditioning on emb=(t,c)."""

    def __init__(self, in_ch, out_ch, emb_ch, dropout=0.0):
        super().__init__()
        self.in_layers = nn.Sequential(norm(in_ch), nn.SiLU(),
                                       nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_ch, 2 * out_ch))
        self.out_norm = norm(out_ch)
        self.out_layers = nn.Sequential(nn.SiLU(), nn.Dropout(dropout),
                                        zero_module(nn.Conv2d(out_ch, out_ch, 3, padding=1)))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.in_layers(x)
        scale, shift = self.emb_layers(emb)[:, :, None, None].chunk(2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return self.skip(x) + h


class AttentionBlock(nn.Module):
    def __init__(self, ch, num_heads=4):
        super().__init__()
        assert ch % num_heads == 0
        self.num_heads = num_heads
        self.norm = norm(ch)
        self.qkv = nn.Conv1d(ch, ch * 3, 1)
        self.proj = zero_module(nn.Conv1d(ch, ch, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.norm(x).reshape(b, c, h * w)
        qkv = self.qkv(y)                                    # b, 3c, hw
        q, k, v = qkv.reshape(b, 3, self.num_heads, c // self.num_heads,
                              h * w).unbind(1)               # b, heads, d, hw
        # .contiguous() matters: without a contiguous last dim SDPA silently
        # falls back to the math backend and materialises the full
        # (b, heads, hw, hw) score matrix -- 10x the memory and ~5x slower.
        q, k, v = (t.permute(0, 1, 3, 2).contiguous() for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v)          # b, heads, hw, d
        a = a.permute(0, 1, 3, 2).reshape(b, c, h * w)
        return x + self.proj(a).reshape(b, c, h, w)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class ConditionalUNet(nn.Module):
    """eps_theta(v_t, t, phi(p_c)) -> predicted noise, same shape as v_t."""

    def __init__(self, image_size=64, in_ch=3, base=128, ch_mult=(1, 2, 2, 4),
                 num_res_blocks=3, attn_resolutions=(16, 8), dropout=0.1,
                 cond_dim=512, num_heads=4):
        super().__init__()
        self.image_size = image_size
        self.cond_dim = cond_dim
        emb_ch = base * 4

        self.time_embed = nn.Sequential(nn.Linear(base, emb_ch), nn.SiLU(),
                                        nn.Linear(emb_ch, emb_ch))
        self.base = base
        # phi: CLIP text embedding -> timestep-embedding space (Fig. 1 "Condition Fusion")
        self.cond_embed = nn.Sequential(nn.Linear(cond_dim, emb_ch), nn.SiLU(),
                                        nn.Linear(emb_ch, emb_ch))
        # learned null condition for classifier-free guidance
        self.null_cond = nn.Parameter(torch.zeros(cond_dim))

        self.in_conv = nn.Conv2d(in_ch, base, 3, padding=1)

        self.down = nn.ModuleList()
        skip_chs = [base]
        ch, res = base, image_size
        for i, m in enumerate(ch_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, base * m, emb_ch, dropout)]
                ch = base * m
                if res in attn_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                self.down.append(nn.ModuleList(layers))
                skip_chs.append(ch)
            if i != len(ch_mult) - 1:
                self.down.append(nn.ModuleList([Downsample(ch)]))
                skip_chs.append(ch)
                res //= 2

        self.mid = nn.ModuleList([ResBlock(ch, ch, emb_ch, dropout),
                                  AttentionBlock(ch, num_heads),
                                  ResBlock(ch, ch, emb_ch, dropout)])

        self.up = nn.ModuleList()
        for i, m in reversed(list(enumerate(ch_mult))):
            for j in range(num_res_blocks + 1):
                layers = [ResBlock(ch + skip_chs.pop(), base * m, emb_ch, dropout)]
                ch = base * m
                if res in attn_resolutions:
                    layers.append(AttentionBlock(ch, num_heads))
                if i and j == num_res_blocks:
                    layers.append(Upsample(ch))
                    res *= 2
                self.up.append(nn.ModuleList(layers))

        self.out = nn.Sequential(norm(ch), nn.SiLU(),
                                 zero_module(nn.Conv2d(ch, in_ch, 3, padding=1)))

    # ------------------------------------------------------------------ #
    def forward(self, x, t, cond=None, drop_mask=None):
        """`cond` is p_c (B, cond_dim).  `drop_mask` (B,) True -> use null cond."""
        emb = self.time_embed(timestep_embedding(t, self.base))
        if cond is None:
            cond = self.null_cond.expand(x.shape[0], -1)
        elif drop_mask is not None:
            cond = torch.where(drop_mask[:, None],
                               self.null_cond.expand_as(cond), cond)
        emb = emb + self.cond_embed(cond)

        h = self.in_conv(x)
        hs = [h]
        for block in self.down:
            for layer in block:
                h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
            hs.append(h)
        for layer in self.mid:
            h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
        for block in self.up:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in block:
                h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
        return self.out(h)


if __name__ == "__main__":
    net = ConditionalUNet()
    n = sum(p.numel() for p in net.parameters())
    print(f"ConditionalUNet params: {n/1e6:.1f}M")
    x = torch.randn(2, 3, 64, 64)
    t = torch.randint(0, 1000, (2,))
    c = torch.randn(2, 512)
    print("out:", tuple(net(x, t, c).shape))
