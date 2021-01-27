#!/usr/bin/env python
"""
Functions to simplify transfer data preparation
"""
from pathlib import Path
from rasterio.windows import from_bounds
from scipy import interpolate
import numpy as np


def patch_fractions(x_shape, out_size=(512, 512)):
    """
    Cut points for (512, 512) patches

    To get aligned x & y patches, we crop tiles according to the fractions of the
    original image dimensions (this is assuming that, even though x and y have
    different dimensions, they share the same lat / long bounding boxes). We use this
    function to ensure that the fractions we crop at provide 512 x 512 images.
    """
    a = out_size[0] / x_shape[0]
    b = out_size[1] / x_shape[1]
    a_grid = np.arange(0, 1, a)
    b_grid = np.arange(0, 1, b)

    fractions = []
    for i in range(len(a_grid) - 1):
        for j in range(len(b_grid) - 1):
            fractions.append((a_grid[i], a_grid[i + 1], b_grid[j], b_grid[j + 1]))

    return fractions


def crop_fraction(u, pixel_fraction = [0.4, 0.5, 0.4, 0.5]):
    """
    Crop a W x H x C image

    This crops images according to the pixel fraction boundaries. E.g., if we want
    to get the pixels between the 10 and 20% widths and heights of the image, we
    would use [0.1, 0.2, 0.1, 0.2] as the pixel fraction argument.
    """
    a, b, c, d = pixel_fraction
    ix = [int(a * u.shape[0]), int(b * u.shape[0]), int(c * u.shape[1]), int(d * u.shape[1])]
    return u[ix[0]:ix[1], ix[2]:ix[3], :]


def resize_mask(x_shape, y):
    """
    Linearly interpolate Y to X shape

    This resizes the Y data to overlap exactly with X. Otherwise, different
    resolutions in the mask and input images would cause issues.
    """
    lin_grid = lambda s: [
        np.linspace(0, s[1], s[1], endpoint=False),
        np.linspace(0, s[0], s[0], endpoint=False)
    ]

    x_grid = lin_grid(x_shape)
    y_grid = lin_grid(y.shape)
    y_resized = []

    for j in range(y.shape[2]):
        f_interp = interpolate.interp2d(*y_grid, y[:, :, j])
        y_resized.append(f_interp(*x_grid))

    return np.stack(y_resized, axis=2)


def patch_tile(tilef, landcover):
    # read and reorient data
    tile = tilef.read()
    tile = np.transpose(tile, (1, 2, 0))
    landcover_bounds = from_bounds(*tilef.bounds, landcover.transform)
    y = landcover.read(window = landcover_bounds)
    y = np.transpose(y, (1, 2, 0))

    # extract pairs
    pixel_frac = patch_fractions(tile.shape[:2])
    pairs = []
    for pf in pixel_frac:
        x = crop_fraction(tile, pf)
        y_ = crop_fraction(y, pf)
        y_ = resize_mask(x.shape, y_)

        pairs.append([x, y_])

    return pairs
