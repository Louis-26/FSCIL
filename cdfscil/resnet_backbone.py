"""ResNet-18 base-session backbone -- the Section 4 reading of the paper.

Sec. 4 states: "Our implementation follows the training setup of CLOSER, with
ResNet18 as the backbone encoder."  That is a completely different feature
extractor from the frozen CLIP encoder of Sec. 3, and it is the regime in which
*every* competitor in Table 1 operates: a ResNet-18 trained from scratch on the
60 base classes only.  We implement it so the reproduction can be judged in the
same regime the reference numbers come from.

The recipe is the standard FSCIL base-session one shared by CEC / FACT / SAVC /
CLOSER: ResNet-18 encoder + cosine classifier with temperature, SGD with
milestone decay, and a nearest-class-mean classifier for the incremental
sessions.  For CUB-200 all FSCIL papers start from ImageNet-pretrained weights;
miniImageNet and CIFAR-100 are trained from scratch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18 as tv_resnet18


# --------------------------------------------------------------------------- #
# ResNet-12 -- the backbone the ~84% rows of Table 1 actually use
# --------------------------------------------------------------------------- #

class _Block12(nn.Module):
    """Standard few-shot ResNet-12 block: 3x conv3x3 + BN + LeakyReLU(0.1),
    1x1 residual projection, 2x2 max-pool, optional DropBlock-style dropout."""

    def __init__(self, inp, out, drop_rate=0.0):
        super().__init__()
        self.c1 = nn.Conv2d(inp, out, 3, padding=1, bias=False); self.b1 = nn.BatchNorm2d(out)
        self.c2 = nn.Conv2d(out, out, 3, padding=1, bias=False); self.b2 = nn.BatchNorm2d(out)
        self.c3 = nn.Conv2d(out, out, 3, padding=1, bias=False); self.b3 = nn.BatchNorm2d(out)
        self.down = nn.Sequential(nn.Conv2d(inp, out, 1, bias=False), nn.BatchNorm2d(out))
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, x):
        r = self.down(x)
        h = self.act(self.b1(self.c1(x)))
        h = self.act(self.b2(self.c2(h)))
        h = self.b3(self.c3(h))
        h = self.act(h + r)
        return self.drop(self.pool(h))


class ResNet12Encoder(nn.Module):
    """640-d encoder. Channels 64-160-320-640; 84x84 -> 42 -> 21 -> 10 -> 5."""

    def __init__(self, drop_rate: float = 0.1):
        super().__init__()
        chans = [64, 160, 320, 640]
        blocks, inp = [], 3
        for i, c in enumerate(chans):
            blocks.append(_Block12(inp, c, drop_rate if i >= 2 else 0.0))
            inp = c
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 640

    def forward(self, x):
        return self.pool(self.blocks(x)).flatten(1)

    def encode_image(self, x):
        return self.forward(x)


class ResNet18Encoder(nn.Module):
    """512-d encoder.  `encode_image` mirrors the CLIP API so the same feature
    extraction / caching code works for both backbones."""

    def __init__(self, imagenet_pretrained: bool = False, small_input: bool = False):
        super().__init__()
        net = tv_resnet18(weights="IMAGENET1K_V1" if imagenet_pretrained else None)
        if small_input:
            # CIFAR-style stem for 32x32 inputs
            net.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
            net.maxpool = nn.Identity()
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 512

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.pool(x).flatten(1)

    # CLIP-compatible alias used by clip_backbone.encode_images
    def encode_image(self, x):
        return self.forward(x)


class CosineClassifier(nn.Module):
    """Normalised-weight classifier, temperature 16 (the FSCIL default)."""

    def __init__(self, dim: int, num_classes: int, temperature: float = 16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, dim))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.temperature = temperature

    def forward(self, feats):
        return self.temperature * F.normalize(feats, dim=-1) @ F.normalize(self.weight, dim=-1).T


def build_encoder(arch: str, imagenet_pretrained=False, small_input=False,
                  drop_rate: float = 0.1):
    if arch == "resnet18":
        return ResNet18Encoder(imagenet_pretrained, small_input)
    if arch == "resnet12":
        return ResNet12Encoder(drop_rate)
    raise ValueError(f"unknown arch {arch}")


class BaseSessionModel(nn.Module):
    """Encoder + cosine head.

    `n_virtual` > 1 turns on the rotation "fantasy" trick used by SAVC / S3C:
    the head predicts `base_classes * n_virtual` labels, where the virtual index
    is the rotation applied to the image.  It is a base-session-only
    self-supervision signal -- the encoder is still frozen afterwards and
    prototypes are still built from unrotated images.
    """

    def __init__(self, num_base_classes: int, imagenet_pretrained=False,
                 small_input=False, temperature: float = 16.0,
                 arch: str = "resnet18", n_virtual: int = 1, drop_rate: float = 0.1):
        super().__init__()
        self.encoder = build_encoder(arch, imagenet_pretrained, small_input, drop_rate)
        self.n_virtual = n_virtual
        self.head = CosineClassifier(self.encoder.out_dim,
                                     num_base_classes * n_virtual, temperature)

    def forward(self, x):
        return self.head(self.encoder(x))
