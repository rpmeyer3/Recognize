import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.insert(3, nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.W_gate = nn.Sequential(nn.Conv2d(gate_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.W_skip = nn.Sequential(nn.Conv2d(skip_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip):
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        if g.shape[2:] != s.shape[2:]:
            g = F.interpolate(g, size=s.shape[2:], mode="bilinear", align_corners=False)
        return skip * self.psi(self.relu(g + s))


class ChannelAttention(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        mid = max(ch // reduction, 8)
        self.mlp = nn.Sequential(nn.Linear(ch, mid, bias=False), nn.ReLU(inplace=True), nn.Linear(mid, ch, bias=False))

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = x.mean(dim=(2, 3))
        mx = x.amax(dim=(2, 3))
        attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * attn.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAMBlock(nn.Module):
    def __init__(self, ch, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attn = ChannelAttention(ch, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x):
        return self.spatial_attn(self.channel_attn(x))


class BlurPool(nn.Module):
    def __init__(self, channels, kernel_size=4, stride=2):
        super().__init__()
        self.stride = stride
        self.channels = channels
        kernels = {
            1: [1.0], 2: [1.0, 1.0], 3: [1.0, 2.0, 1.0],
            4: [1.0, 3.0, 3.0, 1.0], 5: [1.0, 4.0, 6.0, 4.0, 1.0],
        }
        if kernel_size not in kernels:
            raise ValueError(f"Unsupported kernel size: {kernel_size}")
        k = np.array(kernels[kernel_size])
        kernel = k[:, None] * k[None, :]
        kernel = kernel / kernel.sum()
        kernel = torch.tensor(kernel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        self.register_buffer("kernel", kernel)
        self.pad = kernel_size // 2

    def forward(self, x):
        return F.conv2d(F.pad(x, (self.pad,) * 4, mode="reflect"), self.kernel, stride=self.stride, groups=self.channels)
