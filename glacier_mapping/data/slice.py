#!/usr/bin/env python
"""
Convert Large Tiff and Mask files to Slices (512 x 512 subtiles)
"""
from pathlib import Path
import argparse
import os
import numpy as np
from geopandas.geodataframe import GeoDataFrame
from skimage.util.shape import view_as_windows
import pandas as pd
import rasterio
import shapely.geometry
from tqdm import tqdm
import matplotlib.pyplot as plt


def squash(x):
    return (x - x.min()) / x.ptp()


def slice_tile(img, size=(512, 512), overlap=6):
    """Slice an image into overlapping patches
    Args:
        img (np.array): image to be sliced
        size tuple(int, int, int): size of the slices
        overlap (int): how much the slices should overlap
    Returns:
        list of slices [np.array]"""
    size_ = (size[0], size[1], img.shape[2])
    patches = view_as_windows(img, size_, step=size[0] - overlap)
    result = []
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            result.append(patches[i, j, 0])
    return result


def slices_metadata(imgf, img_path, mask_path, size=(512, 512), overlap=6):
    """
    Write geometry and source information to metadata
    """
    meta = slice_polys(imgf, size, overlap)
    meta["img_source"] = img_path
    meta["mask_source"] = mask_path
    return meta


def slice_polys(imgf, size=(512, 512), overlap=6):
    """
    Get Polygons Corresponding to Slices
    """
    ix_row = np.arange(0, imgf.meta["height"], size[0] - overlap)
    ix_col = np.arange(0, imgf.meta["width"], size[1] - overlap)
    lats = np.linspace(imgf.bounds.bottom, imgf.bounds.top, imgf.meta["height"])
    longs = np.linspace(imgf.bounds.left, imgf.bounds.right, imgf.meta["width"])

    polys = []
    for i in range(len(ix_row) - 1):
        for j in range(len(ix_col) - 1):
            box = shapely.geometry.box(
                longs[ix_col[j]],
                lats[ix_row[i]],
                longs[ix_col[j + 1]],
                lats[ix_row[i + 1]],
            )
            polys.append(box)

    return GeoDataFrame(geometry=polys, crs=imgf.meta["crs"].to_string())


def slice_pair(img, mask, **kwargs):
    """
    Slice an image / mask pair
    """
    # maskout areas with nans
    nan_mask = np.isnan(img[:, :, 0])
    nan_mask = np.expand_dims(nan_mask, axis=2)
    nan_mask = np.repeat(nan_mask, mask.shape[-1], axis=2)
    mask[nan_mask] = 0

    img_slices = slice_tile(img, **kwargs)
    mask_slices = slice_tile(mask, **kwargs)
    return img_slices, mask_slices


def write_pair_slices(img_path, mask_path, out_dir, border_path='',
                      out_base="slice",**kwargs):
    """ Write sliced images and masks to numpy arrays

    Args:
        img_path(String): the path to the raw image tiff
        mask_path(String): the paths to the mask array
        border_path(String): teh path to the border array
        output_base(String): The basenames for all the output numpy files
        out_dir(String): The directory to which all the results will be stored
    Returns:
        Writes a csv to metadata path
    """
    imgf = rasterio.open(img_path)
    img = imgf.read().transpose(1, 2, 0)
    mask = np.load(mask_path)
    if border_path:
        img = clip_image(img, border_path)
    img_slices, mask_slices = slice_pair(img, mask, **kwargs)
    metadata = slices_metadata(imgf, img_path, mask_path, **kwargs)

    # loop over slices for individual tile / mask pairs
    slice_stats = []
    for k in tqdm(range(len(img_slices))):
        img_slice_path = Path(out_dir, f"{out_base}_img_{k:03}.npy")
        mask_slice_path = Path(out_dir, f"{out_base}_mask_{k:03}.npy")
        np.save(img_slice_path, img_slices[k])
        np.save(mask_slice_path, mask_slices[k])

        # update metadata
        stats = {"img_slice": str(img_slice_path), "mask_slice": str(mask_slice_path)}
        img_slice_mean = np.nan_to_num(img_slices[k]).mean()
        mask_mean = mask_slices[k].mean(axis=(0, 1))
        stats.update({f"mask_mean_{i}": v for i, v in enumerate(mask_mean)})
        stats.update({"img_mean": img_slice_mean})
        slice_stats.append(stats)

    slice_stats = pd.DataFrame(slice_stats)
    return pd.concat([metadata, slice_stats], axis=1)

def clip_image(img, shp_path):
    """Clip an image to the extent of an mask.
    Args:
        img(numpy.array): Image to clip
        shp_path(String): A path to the mask to clip with
    Returns:
        The clipped images, with non-valid points as numpy.nan
    """
    mask = np.load(shp_path)
    mask = np.repeat(mask, img.shape[-1], axis=2)
    img[mask == 0] = np.nan
    return img

def plot_slices(slice_dir, processed=False, n_cols=3, div=3000, n_examples=5):
    """Helper to plot slices in a directory
    """
    files = list(Path(slice_dir).glob("*img*npy"))
    _, ax = plt.subplots(n_examples, n_cols, figsize=(15,15))
    for i in range(n_examples):
        index = np.random.randint(0, len(files))
        img = np.load(files[index])
        mask = np.load(str(files[index]).replace("img", "mask"))

        if not processed:
            ax[i, 0].imshow(np.nan_to_num(img[:, :, [0, 1, 2]]) / div)
        else:
            ax[i, 0].imshow(squash(img))

        for j in range(mask.shape[2]):
            ax[i, j + 1].imshow(mask[:, :, j])

    return ax
