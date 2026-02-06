"""
Attention U-Net with optional CBAM and BlurPool.

Primary architecture for noise-robust pattern delineation. Attention gates
on skip connections suppress noise-activated features, while CBAM blocks
improve channel/spatial selectivity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import ConvBlock, AttentionGate, CBAMBlock, BlurPool


class AttentionUNet(nn.Module):
    """
    Attention U-Net for binary segmentation in noisy conditions.

    Args:
        in_channels: Number of input channels (1 for grayscale).
        out_channels: Number of output channels (1 for binary mask).
        base_filters: Number of filters in the first encoder stage.
        depth: Number of encoder/decoder levels.
        use_cbam: Whether to attach CBAM blocks to encoder stages.
        use_blur_pool: Use anti-aliased downsampling instead of MaxPool.
        dropout: Dropout probability in conv blocks.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_filters: int = 64,
        depth: int = 5,
        use_cbam: bool = True,
        use_blur_pool: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.depth = depth

        # ---- Encoder ----
        self.encoders = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        self.cbam_blocks = nn.ModuleList() if use_cbam else None

        ch_in = in_channels
        encoder_channels = []
        for i in range(depth):
            ch_out = base_filters * (2 ** i)
            self.encoders.append(ConvBlock(ch_in, ch_out, dropout=dropout))
            if use_cbam:
                self.cbam_blocks.append(CBAMBlock(ch_out))

            if i < depth - 1:  # No downsampling after bottleneck
                if use_blur_pool:
                    self.downsamplers.append(BlurPool(ch_out))
                else:
                    self.downsamplers.append(nn.MaxPool2d(2))

            encoder_channels.append(ch_out)
            ch_in = ch_out

        # ---- Decoder ----
        self.upsamplers = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in range(depth - 2, -1, -1):
            ch_dec = encoder_channels[i + 1]  # Decoder input channels
            ch_skip = encoder_channels[i]     # Skip connection channels

            self.upsamplers.append(
                nn.ConvTranspose2d(ch_dec, ch_skip, kernel_size=2, stride=2)
            )
            self.attention_gates.append(
                AttentionGate(
                    gate_ch=ch_skip,    # After upsampling
                    skip_ch=ch_skip,    # Encoder skip
                    inter_ch=ch_skip // 2,
                )
            )
            self.decoders.append(
                ConvBlock(ch_skip * 2, ch_skip, dropout=dropout)
            )

        # ---- Output ----
        self.out_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W) input image tensor.
        Returns:
            (B, C_out, H, W) raw logits (apply sigmoid for probabilities).
        """
        # Encoder path — collect skip connections
        skips = []
        for i in range(self.depth):
            x = self.encoders[i](x)
            if self.cbam_blocks is not None:
                x = self.cbam_blocks[i](x)
            if i < self.depth - 1:
                skips.append(x)
                x = self.downsamplers[i](x)

        # Decoder path — attend to skips
        for i in range(self.depth - 1):
            x = self.upsamplers[i](x)
            skip = skips[self.depth - 2 - i]

            # Handle size mismatches from odd dimensions
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

            skip = self.attention_gates[i](gate=x, skip=skip)
            x = torch.cat([x, skip], dim=1)
            x = self.decoders[i](x)

        return self.out_conv(x)

    def count_parameters(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_attention_unet(cfg: dict) -> AttentionUNet:
    """Factory function to build AttentionUNet from config dict."""
    model_cfg = cfg.get("model", cfg)
    return AttentionUNet(
        in_channels=model_cfg.get("in_channels", 1),
        out_channels=model_cfg.get("out_channels", 1),
        base_filters=model_cfg.get("base_filters", 64),
        depth=model_cfg.get("depth", 5),
        use_cbam=model_cfg.get("use_cbam", True),
        use_blur_pool=model_cfg.get("use_blur_pool", True),
        dropout=model_cfg.get("dropout", 0.1),
    )
