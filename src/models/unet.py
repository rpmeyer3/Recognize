import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import ConvBlock


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=64, depth=5, dropout=0.0):
        super().__init__()
        self.depth = depth
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        ch_in = in_channels
        enc_channels = []
        for i in range(depth):
            ch = base_filters * (2 ** i)
            self.encoders.append(ConvBlock(ch_in, ch, dropout=dropout))
            if i < depth - 1:
                self.pools.append(nn.MaxPool2d(2))
            enc_channels.append(ch)
            ch_in = ch

        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            ch_dec, ch_skip = enc_channels[i + 1], enc_channels[i]
            self.upsamplers.append(nn.ConvTranspose2d(ch_dec, ch_skip, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(ch_skip * 2, ch_skip, dropout=dropout))

        self.out_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
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
            x = self.decoders[i](torch.cat([x, skip], dim=1))
        return self.out_conv(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_unet(cfg):
    m = cfg.get("model", cfg)
    return UNet(
        in_channels=m.get("in_channels", 1), out_channels=m.get("out_channels", 1),
        base_filters=m.get("base_filters", 64), depth=m.get("depth", 5),
        dropout=m.get("dropout", 0.0),
    )
