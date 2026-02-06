"""
Custom layers for the Attention U-Net architecture.

Includes:
- ConvBlock: Standard double-conv with BatchNorm + ReLU
- AttentionGate: Gated attention for skip connections
- CBAMBlock: Channel + Spatial Attention Module
- BlurPool: Anti-aliased downsampling (Zhang, 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    """Double convolution block: (Conv2d → BN → ReLU) × 2."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.insert(3, nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """
    Attention Gate for skip connections (Oktay et al., 2018).

    Learns to suppress irrelevant (noise) activations in the skip
    connection by computing a spatial attention map conditioned on the
    gating signal from the decoder.

    Args:
        gate_ch: Channels in the gating signal (from decoder).
        skip_ch: Channels in the skip connection (from encoder).
        inter_ch: Intermediate channel dimension for attention computation.
    """

    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int):
        super().__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gate: Decoder feature map (lower resolution, upsampled to match skip).
            skip: Encoder feature map (higher resolution).
        Returns:
            Attended skip connection.
        """
        g = self.W_gate(gate)
        s = self.W_skip(skip)

        # Align spatial dimensions (gate may be slightly off after upsampling)
        if g.shape[2:] != s.shape[2:]:
            g = F.interpolate(g, size=s.shape[2:], mode="bilinear", align_corners=False)

        attn = self.relu(g + s)
        attn = self.psi(attn)
        return skip * attn


class ChannelAttention(nn.Module):
    """Channel Attention sub-module of CBAM."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        # Global average & max pooling → shared MLP
        avg = x.mean(dim=(2, 3))  # (B, C)
        mx = x.amax(dim=(2, 3))   # (B, C)
        attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx))  # (B, C)
        return x * attn.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    """Spatial Attention sub-module of CBAM."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size % 2 == 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        mx = x.amax(dim=1, keepdim=True)   # (B, 1, H, W)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module (Woo et al., 2018).

    Applies channel attention followed by spatial attention, improving
    feature selectivity — critical for distinguishing signal from noise.
    """

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class BlurPool(nn.Module):
    """
    Anti-aliased downsampling (Zhang, 2019).

    Applies a fixed low-pass filter before strided operation to suppress
    aliasing artifacts — particularly beneficial when input contains
    high-frequency noise.

    Args:
        channels: Number of input channels.
        kernel_size: Size of the blur kernel (default 4).
        stride: Downsampling stride (default 2).
    """

    def __init__(self, channels: int, kernel_size: int = 4, stride: int = 2):
        super().__init__()
        self.stride = stride
        self.channels = channels

        # Construct binomial (Pascal triangle) blur kernel
        if kernel_size == 1:
            k = np.array([1.0])
        elif kernel_size == 2:
            k = np.array([1.0, 1.0])
        elif kernel_size == 3:
            k = np.array([1.0, 2.0, 1.0])
        elif kernel_size == 4:
            k = np.array([1.0, 3.0, 3.0, 1.0])
        elif kernel_size == 5:
            k = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        else:
            raise ValueError(f"Unsupported blur kernel size: {kernel_size}")

        # Outer product → 2D kernel
        kernel = k[:, None] * k[None, :]
        kernel = kernel / kernel.sum()
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)

        self.register_buffer("kernel", kernel)
        self.pad = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Depthwise blur convolution + stride
        return F.conv2d(
            F.pad(x, (self.pad,) * 4, mode="reflect"),
            self.kernel,
            stride=self.stride,
            groups=self.channels,
        )
