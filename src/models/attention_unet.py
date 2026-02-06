import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import ConvBlock, AttentionGate, CBAMBlock, BlurPool


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=64,
                 depth=5, use_cbam=True, use_blur_pool=True, dropout=0.1):
        super().__init__()
        self.depth = depth
        self.encoders = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        self.cbam_blocks = nn.ModuleList() if use_cbam else None

        ch_in = in_channels
        enc_channels = []
        for i in range(depth):
            ch = base_filters * (2 ** i)
            self.encoders.append(ConvBlock(ch_in, ch, dropout=dropout))
            if use_cbam:
                self.cbam_blocks.append(CBAMBlock(ch))
            if i < depth - 1:
                self.downsamplers.append(BlurPool(ch) if use_blur_pool else nn.MaxPool2d(2))
            enc_channels.append(ch)
            ch_in = ch

        self.upsamplers = nn.ModuleList()
        self.attn_gates = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            ch_dec = enc_channels[i + 1]
            ch_skip = enc_channels[i]
            self.upsamplers.append(nn.ConvTranspose2d(ch_dec, ch_skip, kernel_size=2, stride=2))
            self.attn_gates.append(AttentionGate(ch_skip, ch_skip, ch_skip // 2))
            self.decoders.append(ConvBlock(ch_skip * 2, ch_skip, dropout=dropout))

        self.out_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for i in range(self.depth):
            x = self.encoders[i](x)
            if self.cbam_blocks is not None:
                x = self.cbam_blocks[i](x)
            if i < self.depth - 1:
                skips.append(x)
                x = self.downsamplers[i](x)

        for i in range(self.depth - 1):
            x = self.upsamplers[i](x)
            skip = skips[self.depth - 2 - i]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            skip = self.attn_gates[i](gate=x, skip=skip)
            x = self.decoders[i](torch.cat([x, skip], dim=1))

        return self.out_conv(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_attention_unet(cfg):
    m = cfg.get("model", cfg)
    return AttentionUNet(
        in_channels=m.get("in_channels", 1), out_channels=m.get("out_channels", 1),
        base_filters=m.get("base_filters", 64), depth=m.get("depth", 5),
        use_cbam=m.get("use_cbam", True), use_blur_pool=m.get("use_blur_pool", True),
        dropout=m.get("dropout", 0.1),
    )
