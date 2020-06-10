#!/usr/bin/env python
"""
UNet Model Class

This is a segmentation model to use for training and inference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Single Encoder Block

    Transforms large image with small inchannels into smaller image with larger
    outchannels, via two convolution / relu pairs.
    """
    def __init__(self, inchannels, outchannels, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=padding)
        self.conv2 = nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=padding)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class UpBlock(nn.Module):
    """
    Single Decoder Block

    Transforms small image with large inchannels into larger image with smaller
    outchannels, via two convolution / relu pairs.
    """
    def __init__(self, inchannels, outchannels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(
            inchannels, outchannels, kernel_size=2, stride=2
        )
        self.conv = ConvBlock(inchannels, outchannels)

    def forward(self, x, skips):
        x = self.upconv(x)
        x = torch.cat([skips, x], 1)
        return self.conv(x)


class Unet(nn.Module):
    """
    U-Net Model

    Combines the encoder and decoder blocks with skip connections, to arrive at
    a U-Net model.
    """
    def __init__(self, inchannels, outchannels, net_depth, channel_layer=16):
        super().__init__()
        self.downblocks = nn.ModuleList()
        self.upblocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        in_channels = inchannels
        out_channels = channel_layer
        for _ in range(net_depth):
            conv = ConvBlock(in_channels, out_channels)
            self.downblocks.append(conv)
            in_channels, out_channels = out_channels, 2 * out_channels

        self.middle_conv = ConvBlock(in_channels, out_channels)

        in_channels, out_channels = out_channels, int(out_channels / 2)
        for _ in range(net_depth):
            upconv = UpBlock(in_channels, out_channels)
            self.upblocks.append(upconv)
            in_channels, out_channels = out_channels, int(out_channels / 2)

        self.seg_layer = nn.Conv2d(2 * out_channels, outchannels, kernel_size=1)

    def forward(self, x):
        decoder_outputs = []

        for layer in self.downblocks:
            decoder_outputs.append(layer(x))
            x = self.pool(decoder_outputs[-1])

        x = self.middle_conv(x)

        for layer in self.upblocks:
            x = layer(x, decoder_outputs.pop())
        return self.seg_layer(x)
