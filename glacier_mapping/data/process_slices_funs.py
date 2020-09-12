#!/usr/bin/env python
"""
Functions to support slice processing
"""
from pathlib import Path
from shutil import copyfile
import json
import os
import random
import sys
import numpy as np


def filter_directory(slice_meta, filter_perc=0.2, filter_channel=1):
    """ Return Paths for Pairs passing Filter Criteria

    Args:
        filter_perc(float): The minimum percentage 1's in the filter_channel needed to pass the filter.
        filter_channel(int): The channel to do the filtering on.

    Return:
        img and mask

    """
    slice_meta = slice_meta[slice_meta[f"mask_mean_{filter_channel}"] > filter_perc]
    slice_meta = slice_meta[slice_meta["img_mean"] > 0]
    return [
        {"img": d["img_slice"], "mask": d["mask_slice"]}
        for _, d in slice_meta.iterrows()
    ]


def random_split(ids, split_ratio, **kwargs):
    """ Randomly split a list of paths into train / dev / test

    Args:
        ids(int): IDs of data to split
        split_ratio: Ratio of split among train:dev:test

    Return:
        Train/Test/Dev splits
    """
    random.shuffle(ids)
    sizes = len(ids) * np.array(split_ratio)
    ix = [int(s) for s in np.cumsum(sizes)]
    return {
        "train": ids[: ix[0]],
        "dev": ids[ix[0] : ix[1]],
        "test": ids[ix[1] : ix[2]],
    }

def geographic_split(ids, split_ratio, geojsons, slice_meta, **kwargs):
    """
    Warning: Does not use the split ratio. Only refers to the geojsons for
    defining the split.
    """
    splits = {"train": [], "test": []}

    for slice_id in ids:
        cur_meta = slice_meta.where(slice_meta.ids == slice_id) # get the row of the pandas with the current slice id
        geo = cur_meta["geometry"]
        if geojsons[0].contains(geo):
            splits["train"].append(slice_id)
        else:
            if geojsons[1].contains(geo):
                splits["test"].append(slice_id)

    return splits


def reshuffle(split_ids, output_dir="output/", n_cpu=3):
    """ Reshuffle Data for Training, given a dictionary specifying train / dev / test split, copy into train / dev / test folders.

    Args:
        split_ids(int): IDs of files to split
        output_dir(str): Directory to place the split dataset
    Return:
        Target locations
    """
    for split_type in split_ids:
        path = Path(output_dir, split_type)
        os.makedirs(path, exist_ok=True)

    target_locs = {k: [] for k in split_ids}
    for split_type in split_ids:
        for i in range(len(split_ids[split_type])):
            cur_locs = {}
            for im_type in ["img", "mask"]:
                print(f"shuffling image {i} - {im_type}")
                source = split_ids[split_type][i][im_type]
                target = Path(
                    output_dir, split_type, os.path.basename(source)
                ).resolve()
                copyfile(source, target)
                cur_locs[im_type] = target

            target_locs[split_type].append(cur_locs)
    return target_locs


def generate_stats(image_paths, sample_size, outpath="stats.json"):
    """ Function to generate statistics of the input image channels

    Args:
        image_paths: List of Paths to images in directory
        sample_size(int): integer giving the size of the sample from which to compute the statistics
        outpath(str): The path to the output json file containing computed statistics

    Return:
         Dictionary with keys for means and stds across the channels in input images
    """
    sample_size = min(sample_size, len(image_paths))
    image_paths = np.random.choice(image_paths, sample_size, replace=False)
    images = [np.load(image_path) for image_path in image_paths]
    batch = np.stack(images)
    means = np.nanmean(batch, axis=(0, 1, 2))
    stds = np.nanstd(batch, axis=(0, 1, 2))

    with open(outpath, "w+") as f:
        stats = {"means": means.tolist(), "stds": stds.tolist()}

        json.dump(stats, f)
    return stats


def normalize_(img, means, stds):
    """
    Args:
        img: Input image to normalize
        means: Computed mean of the input channels
        stds: Computed standard deviation of the input channels

    Return:
        img: Normalized img
    """
    for i in range(img.shape[2]):
        img[:, :, i] -= means[i]
        if stds[i] > 0:
            img[:, :, i] /= stds[i]
        else:
            img[:, :, i] = 0

    return img


def normalize(img, mask, stats_path):
    """wrapper for postprocess

    Args:
        img: image to normalize
        mask: mask
        stats_path: path to dataset statistics

    Return:
        Normalized image and corresponding mask
    """
    stats = json.load(open(stats_path, "r"))
    img = normalize_(img, stats["means"], stats["stds"])
    return img, mask


def impute(img, mask, value=0):
    """Replace NAs with value

    Args:
        img: image to impute
        mask: mask to impute
        value: imputation value

    Return:
        image and corresponding mask after imputation
    """
    img = np.nan_to_num(img, nan=value)
    return img, mask


def extract_channel(img, mask, mask_channels=None, img_channels=None):
    """Subset specific channels from raster

    Args:
        img: Image to extract
        mask:  Mask to extract
        mask_channels: Mask channels to extract
        img_channels: Image channels to extract

    Return:
        Image and corresponding mask with specified channels
    """
    if mask_channels is None:
        mask_channels = np.arange(mask.shape[2])

    if img_channels is None:
        img_channels = np.arange(img.shape[2])

    return img[:, :, img_channels], mask[:, :, mask_channels]


def postprocess_tile(img, process_funs):
    """Apply a list of processing functions

    Args:
        img: Image to postprocess
        process_funs: Specified process functions

    Return:
        Image, mask and specified process functions
    """
    # create fake mask input
    process_funs.extract_channel.mask_channels = 0
    mask = np.zeros((img.shape[0], img.shape[1], 1))

    return postprocess_(img, mask, process_funs)


def postprocess_(img, mask, process_funs):
    """Internal helper for postprocess_tile

    Args:
        img: Image to postprocess
        mask: Mask to postprocess
        process_funs: Specified post process functions

    Return:
        Post processed images and masks
    """
    for fun_name, fun_args in process_funs.items():
        f = getattr(sys.modules[__name__], fun_name)
        img, mask = f(img, mask, **fun_args)

    return img, mask


def postprocess(img_path, mask_path, process_funs):
    """process a single image / mask pair

    Args:
        img_path(str): Path to single image
        mask_path(str): Path to single mask
        process_funs: Specified process functions

    Return:
        Postprocess image, mask and postprocess function

    """
    img, mask = np.load(img_path), np.load(mask_path)
    return postprocess_(img, mask, process_funs)
