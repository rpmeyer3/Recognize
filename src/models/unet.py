"""
Vanilla U-Net baseline for comparison.

Standard encoder-decoder with skip connections but no attention mechanisms.
Serves as a performance baseline against the Attention U-Net.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvBlock


class UNet(nn.Module):
    """
    Standard U-Net for binary segmentation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        base_filters: First stage filter count.
        depth: Number of encoder/decoder stages.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_filters: int = 64,
        depth: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.depth = depth

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        ch_in = in_channels
        encoder_channels = []
        for i in range(depth):
            ch_out = base_filters * (2 ** i)
            self.encoders.append(ConvBlock(ch_in, ch_out, dropout=dropout))
            if i < depth - 1:
                self.pools.append(nn.MaxPool2d(2))
            encoder_channels.append(ch_out)
            ch_in = ch_out

        # Decoder
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in range(depth - 2, -1, -1):
            ch_dec = encoder_channels[i + 1]
            ch_skip = encoder_channels[i]
            self.upsamplers.append(
                nn.ConvTranspose2d(ch_dec, ch_skip, kernel_size=2, stride=2)
            )
            self.decoders.append(ConvBlock(ch_skip * 2, ch_skip, dropout=dropout))

        self.out_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i in range(self.depth):
            x = self.encoders[i](x)
            if i < self.depth - 1:
                skips.append(x)
                x = self.pools[i](x)

        for i in range(self.depth - 1):
            x = self.upsamplers[i](x)
            skip = skips[self.depth - 2 - i]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.decoders[i](x)

        return self.out_conv(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_unet(cfg: dict) -> UNet:
    """Factory function to build UNet from config dict."""
    model_cfg = cfg.get("model", cfg)
    return UNet(
        in_channels=model_cfg.get("in_channels", 1),
        out_channels=model_cfg.get("out_channels", 1),
        base_filters=model_cfg.get("base_filters", 64),
        depth=model_cfg.get("depth", 5),
        dropout=model_cfg.get("dropout", 0.0),
    )
