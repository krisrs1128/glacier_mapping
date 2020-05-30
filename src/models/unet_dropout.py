#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, inchannels, outchannels, dropout, spatial, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=padding)
        self.conv2 = nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=padding)
        if spatial:
            self.dropout = nn.Dropout2d(p=dropout)
        else:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = F.relu(self.dropout(self.conv1(x)))
        x = F.relu(self.conv2(x))
        return x


class UpBlock(nn.Module):
    def __init__(self, inchannels, outchannels, dropout, spatial):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(
            inchannels, outchannels, kernel_size=2, stride=2
        )
        self.conv = ConvBlock(inchannels, outchannels, dropout, spatial)

    def forward(self, x, skips):
        x = self.upconv(x)
        x = torch.cat([skips, x], 1)
        return self.conv(x)


class UnetDropout(nn.Module):
    """
    This model initializes U-Net with dropout for the downsampling layers (default 0.2)
    Initialization:
        model = UnetDropout(12, 1, 4, dropout = (dropout_probability))
    Forward pass as image/numpy array:
        x = torch.from_numpy(np.random.uniform(0,1,(1,10,512,512))).float()
        model(x)
    """

    def __init__(
        self,
        inchannels,
        outchannels,
        net_depth,
        dropout=0.2,
        spatial=False,
        channel_layer=16,
    ):
        super().__init__()
        self.downblocks = nn.ModuleList()
        self.upblocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        in_channels = inchannels
        out_channels = channel_layer
        for _ in range(net_depth):
            conv = ConvBlock(in_channels, out_channels, dropout, spatial)
            self.downblocks.append(conv)
            in_channels, out_channels = out_channels, 2 * out_channels

        self.middle_conv = ConvBlock(in_channels, out_channels, dropout, spatial)

        in_channels, out_channels = out_channels, int(out_channels / 2)
        for _ in range(net_depth):
            upconv = UpBlock(in_channels, out_channels, dropout, spatial)
            self.upblocks.append(upconv)
            in_channels, out_channels = out_channels, int(out_channels / 2)

        self.seg_layer = nn.Conv2d(2 * out_channels, outchannels, kernel_size=1)

    def forward(self, x):
        decoder_outputs = []

        for op in self.downblocks:
            decoder_outputs.append(op(x))
            x = self.pool(decoder_outputs[-1])

        x = self.middle_conv(x)

        for op in self.upblocks:
            x = op(x, decoder_outputs.pop())
        return self.seg_layer(x)
