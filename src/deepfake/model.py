"""Paper-inspired MH-FFNet. See MODEL_NOTES.md for explicit adaptations."""
import math
from functools import lru_cache
import torch
from torch import nn
from torchvision.models import convnext_tiny, convnext_base
from torchvision.models import ConvNeXt_Tiny_Weights, ConvNeXt_Base_Weights


@lru_cache(maxsize=16)
def dct_basis(n):
    k = torch.arange(n, dtype=torch.float32)[:, None]
    x = torch.arange(n, dtype=torch.float32)[None, :]
    basis = torch.cos(math.pi * (x + .5) * k / n) * math.sqrt(2 / n)
    basis[0] /= math.sqrt(2)
    return basis


def frequency_residual(x, cutoff=.25):
    """Orthonormal DCT-II, triangular low-band removal, inverse DCT.

    Operates on the last two dimensions. Float32 avoids unstable half-precision
    frequency operations during AMP. The entire operation is differentiable.
    """
    if not 0 <= cutoff < 1:
        raise ValueError('cutoff must be in [0, 1)')
    with torch.autocast(device_type=x.device.type, enabled=False):
        value = x.float()
        h, w = value.shape[-2:]
        dh = dct_basis(h).to(x.device)
        dw = dct_basis(w).to(x.device)
        u = torch.arange(h, device=x.device)[:, None] / max(h - 1, 1)
        v = torch.arange(w, device=x.device)[None, :] / max(w - 1, 1)
        mask = ((u + v) / 2 > cutoff).float()
        coeff = dh @ value @ dw.T
        out = dh.T @ (coeff * mask) @ dw
    return out.to(x.dtype)


class FGEM(nn.Module):
    def __init__(self, channels, patch_size=None, cutoff=.25):
        super().__init__()
        self.patch_size, self.cutoff = patch_size, cutoff
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.out = nn.Conv2d(channels, channels, 1)
        self.dropout = nn.Dropout2d(p=0.1)
        # Start the final projection as identity; the added frequency fusion is learned.
        nn.init.eye_(self.out.weight[:, :, 0, 0])
        nn.init.zeros_(self.out.bias)

    def frequency(self, x):
        if self.patch_size is None:
            return frequency_residual(x, self.cutoff)
        b, c, h, w = x.shape
        p = self.patch_size
        if h % p or w % p:
            raise ValueError('Feature dimensions must be divisible by patch size')
        patches = x.reshape(b, c, h//p, p, w//p, p).permute(0, 1, 2, 4, 3, 5)
        filtered = frequency_residual(patches, self.cutoff)
        return filtered.permute(0, 1, 2, 4, 3, 5).reshape(b, c, h, w)

    def forward(self, x):
        fine = self.frequency(x)
        q, k, v = self.q(x).flatten(2), self.k(fine).flatten(2), self.v(fine).flatten(2)
        with torch.autocast(device_type=x.device.type, enabled=False):
            attention = torch.softmax(q.float() @ k.float().transpose(1, 2)
                                      / math.sqrt(x.shape[1]*x.shape[2]*x.shape[3]), dim=-1)
            fusion = (attention @ v.float()).reshape_as(x)
        return self.dropout(self.out(x + fusion.to(x.dtype)))


class Detector(nn.Module):
    def __init__(self, backbone='tiny', pretrained=False, mode='both', cutoff=.25, patch_size=7):
        super().__init__()
        if backbone not in ('tiny', 'base') or mode not in ('cnn', 'patch', 'global', 'both'):
            raise ValueError('Unsupported backbone or frequency mode')
        if backbone == 'tiny':
            network = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
            channels = (96, 192)
        else:
            network = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT if pretrained else None)
            channels = (128, 256)
        network.classifier[2] = nn.Linear(network.classifier[2].in_features, 2)
        self.features = network.features
        self.pool, self.classifier = network.avgpool, network.classifier
        self.patch = FGEM(channels[0], patch_size, cutoff) if mode in ('patch', 'both') else nn.Identity()
        self.global_frequency = FGEM(channels[1], None, cutoff) if mode in ('global', 'both') else nn.Identity()
        self.config = dict(backbone=backbone, mode=mode, cutoff=cutoff, patch_size=patch_size)

    @property
    def cam_layer(self):
        return self.features[7]

    def forward(self, x):
        x = self.features[1](self.features[0](x))
        x = self.patch(x)
        x = self.features[3](self.features[2](x))
        x = self.global_frequency(x)
        for layer in self.features[4:]:
            x = layer(x)
        return self.classifier(self.pool(x))
